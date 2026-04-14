"""Trust-SSL: additive-residual selective invariance.

This module implements the model described in Section III of the paper.
The same class is used for three configurations:

  - Full Trust-SSL:       num_factors=6, gate_kind="evidential"
  - Scalar uncertainty:   num_factors=1, gate_kind="evidential"
  - Cosine gate:          num_factors=6, gate_kind="cosine"

The one-class design is deliberate: it makes the ablations controlled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone, MLPProjector
from .simclr import info_nce


# ─────────────────────────────────────────────────────────────────
# Factorization head
# ─────────────────────────────────────────────────────────────────
class FactorHead(nn.Module):
    """Shared nonlinear stem followed by T linear factor projections.

    Equation (2) in the paper:
        z_v^t = normalize(W^t * g(h_v))
    """

    def __init__(self, feat_dim: int, num_factors: int, factor_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        self.stem = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_dim, factor_dim, bias=False) for _ in range(num_factors)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return tensor of shape (B, T, d), L2-normalized along the last axis."""
        g = self.stem(h)
        z = torch.stack([proj(g) for proj in self.projections], dim=1)  # (B, T, d)
        return F.normalize(z, dim=-1)


# ─────────────────────────────────────────────────────────────────
# Evidential head
# ─────────────────────────────────────────────────────────────────
class EvidentialHead(nn.Module):
    """Per-factor softplus heads producing non-negative evidence.

    Equations (3)–(5): evidence, Dirichlet, belief / ignorance.
    """

    def __init__(self, factor_dim: int, num_prototypes: int, num_factors: int, prior: float = 0.05):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_factors = num_factors
        self.prior = prior
        self.heads = nn.ModuleList(
            [nn.Linear(factor_dim, num_prototypes) for _ in range(num_factors)]
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Input z: (B, T, d). Returns (belief, ignorance), both (B, T, M) or (B, T).

        belief:     (B, T, M)  -- per-prototype belief mass b_{v,m}^t
        ignorance:  (B, T)     -- total ignorance u_v^t
        """
        per_factor_evidence = []
        for t, head in enumerate(self.heads):
            ev_t = F.softplus(head(z[:, t, :]))           # (B, M)
            per_factor_evidence.append(ev_t)
        evidence = torch.stack(per_factor_evidence, dim=1)    # (B, T, M)

        alpha = evidence + self.prior                         # (B, T, M)
        S = alpha.sum(dim=-1, keepdim=True)                   # (B, T, 1)
        belief = evidence / S                                 # (B, T, M)
        ignorance = (self.prior * self.num_prototypes) / S.squeeze(-1)  # (B, T)
        return belief, ignorance


# ─────────────────────────────────────────────────────────────────
# Dempster–Shafer fusion
# ─────────────────────────────────────────────────────────────────
def dempster_fusion(
    b1: torch.Tensor,
    b2: torch.Tensor,
    u1: torch.Tensor,
    u2: torch.Tensor,
    epsilon: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-factor conflict K and fused ignorance I.

    Equations (6)–(7) from the paper.

    b1, b2: (B, T, M) belief tensors for the two views
    u1, u2: (B, T)    ignorance scalars for the two views

    Returns:
        K: (B, T) conflict mass
        I: (B, T) fused ignorance
    """
    # K = sum_{i != j} b1_i * b2_j = (sum b1)(sum b2) - sum_i b1_i b2_i
    inner = (b1 * b2).sum(dim=-1)                       # (B, T)
    outer = b1.sum(dim=-1) * b2.sum(dim=-1)             # (B, T)
    K = (outer - inner).clamp(min=0.0, max=0.999)       # (B, T)

    # I = u1 u2 / (1 - K) + epsilon * |u1 - u2|, clamped to 1.
    I = (u1 * u2) / (1.0 - K) + epsilon * (u1 - u2).abs()
    I = I.clamp(min=0.0, max=1.0)
    return K, I


# ─────────────────────────────────────────────────────────────────
# Trust gate
# ─────────────────────────────────────────────────────────────────
def evidential_gate(K: torch.Tensor, I: torch.Tensor,
                    alpha: float, gamma: float, lambda_min: float) -> torch.Tensor:
    """Equation (8): w = lambda_min + (1 - lambda_min) * exp(-alpha K - gamma I).

    Returns (B, T) in [lambda_min, 1].
    """
    return lambda_min + (1.0 - lambda_min) * torch.exp(-alpha * K - gamma * I)


# ─────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────
class TrustSSL(nn.Module):
    """Trust-SSL with configurable gate.

    Configuration keys (see configs/trust_ssl.yaml):
        num_factors:         int (T)        default 6
        factor_dim:          int (d)        default 128
        num_prototypes:      int (M)        default 64
        prior_strength:      float (beta)   default 0.05
        gate_alpha:          float          default 2.0
        gate_gamma:          float          default 3.0
        temperature:         float          default 0.1
        gate_kind:           "evidential" | "cosine"
        num_corrupt_classes: int for the auxiliary predictor, default 9
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.num_factors = int(cfg.get("num_factors", 6))
        self.factor_dim = int(cfg.get("factor_dim", 128))
        self.num_prototypes = int(cfg.get("num_prototypes", 64))
        self.prior_strength = float(cfg.get("prior_strength", 0.05))
        self.gate_alpha = float(cfg.get("gate_alpha", 2.0))
        self.gate_gamma = float(cfg.get("gate_gamma", 3.0))
        self.temperature = float(cfg.get("temperature", 0.1))
        self.gate_kind = cfg.get("gate_kind", "evidential")
        num_corrupt_classes = int(cfg.get("num_corrupt_classes", 9))

        # Backbone + global projector (for base contrastive loss)
        self.backbone, feat_dim = build_backbone(pretrained=False)
        self.projector = MLPProjector(feat_dim, hidden_dim=2048, out_dim=256)

        # Factorization
        self.factor_head = FactorHead(feat_dim, self.num_factors, self.factor_dim)

        # Gate branch
        if self.gate_kind == "evidential":
            self.evidential = EvidentialHead(
                self.factor_dim, self.num_prototypes, self.num_factors,
                prior=self.prior_strength,
            )
        elif self.gate_kind == "cosine":
            # Per-factor learned temperature in a log-parameterized scalar
            self.cosine_log_tau = nn.Parameter(
                torch.full((self.num_factors,), fill_value=float(torch.log(torch.tensor(0.5)).item()))
            )
        else:
            raise ValueError(f"gate_kind must be 'evidential' or 'cosine', got {self.gate_kind}")

        # Auxiliary corruption-family predictor on top of backbone features
        self.aux_classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_corrupt_classes),
        )

    # ──────────────────────────────────────────────────────────────
    # Gating
    # ──────────────────────────────────────────────────────────────
    def compute_gate_and_signals(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> dict:
        """Return dict with fields:
            w:  (B, T)  trust weights in [lambda_min, 1]
            K:  (B, T)  conflict signal (None if no evidential head)
            I:  (B, T)  ignorance signal (None if no evidential head)

        lambda_min is injected by the caller via `_lambda_min`.
        """
        if self.gate_kind == "evidential":
            b1, u1 = self.evidential(z1)
            b2, u2 = self.evidential(z2)
            K, I = dempster_fusion(b1, b2, u1, u2)
            w = evidential_gate(K, I, self.gate_alpha, self.gate_gamma, self._lambda_min)
            return {"w": w, "K": K, "I": I}

        # Cosine gate: w = sigmoid((cos / tau) - b), bounded to [lambda_min, 1]
        cos = (z1 * z2).sum(dim=-1)                                  # (B, T)
        tau = self.cosine_log_tau.exp().clamp(min=1e-3)              # (T,)
        logits = cos / tau                                           # (B, T)
        w_raw = torch.sigmoid(logits)
        w = self._lambda_min + (1.0 - self._lambda_min) * w_raw
        return {"w": w, "K": None, "I": None}

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────
    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        corrupt_labels: torch.Tensor | None = None,
        lambda_sel: float = 0.0,
        lambda_min: float = 0.5,
    ) -> dict:
        """Run forward pass and compute the additive-residual loss.

        Equation (10): L = L_simclr + lambda_sel * L_add + aux terms.
        """
        self._lambda_min = lambda_min

        # Backbone
        h1 = self.backbone(view1)
        h2 = self.backbone(view2)

        # Base contrastive on global projector
        g1 = F.normalize(self.projector(h1), dim=-1)
        g2 = F.normalize(self.projector(h2), dim=-1)
        base = info_nce(g1, g2, self.temperature)

        # Factor projections (B, T, d)
        z1 = self.factor_head(h1)
        z2 = self.factor_head(h2)

        # Gate and signals
        gate = self.compute_gate_and_signals(z1, z2)
        w = gate["w"]                                                # (B, T)

        # Additive-residual alignment loss, Equation (9)
        # Stop-gradient on w so the backbone gradient is a clean rescaling.
        w_sg = w.detach()
        cos = (z1 * z2).sum(dim=-1)                                  # (B, T)
        align_term = (w_sg * (1.0 - cos)).mean()

        # Auxiliary corruption classifier (applied to mean of the two views)
        aux_loss = torch.zeros((), device=view1.device)
        if corrupt_labels is not None:
            logits = self.aux_classifier(0.5 * (h1 + h2))
            aux_loss = F.cross_entropy(logits, corrupt_labels)

        # Total
        total = base + lambda_sel * align_term + 0.5 * aux_loss

        return {
            "loss": total,
            "base": base.detach(),
            "align": align_term.detach(),
            "aux": aux_loss.detach(),
            "w_mean": w.detach().mean(),
            "K_mean": gate["K"].detach().mean() if gate["K"] is not None else None,
            "I_mean": gate["I"].detach().mean() if gate["I"] is not None else None,
            "h": (h1, h2),
        }

    # ──────────────────────────────────────────────────────────────
    # Inference helpers
    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features, the standard input for downstream probes."""
        return self.backbone(x)

    @torch.no_grad()
    def compute_ki(self, view1: torch.Tensor, view2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (K, I) averaged over factors for each pair in the batch.

        Used by the K–I trajectory analysis (Figure 4 in the paper).
        Only works when `gate_kind == 'evidential'`.
        """
        if self.gate_kind != "evidential":
            raise RuntimeError("compute_ki is only available for the evidential gate")
        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.factor_head(h1)
        z2 = self.factor_head(h2)
        b1, u1 = self.evidential(z1)
        b2, u2 = self.evidential(z2)
        K, I = dempster_fusion(b1, b2, u1, u2)
        return K.mean(dim=-1), I.mean(dim=-1)
