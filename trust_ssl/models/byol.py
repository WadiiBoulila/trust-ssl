"""BYOL baseline (predictor + momentum target network)."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone, MLPProjector, Predictor


class BYOL(nn.Module):
    """BYOL with a momentum target network and a predictor head.

    No negative samples; loss is regression of the online predictor
    output to the target projector output, in both directions.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone, feat_dim = build_backbone(pretrained=False)
        self.projector = MLPProjector(feat_dim, hidden_dim=4096, out_dim=256)
        self.predictor = Predictor(in_dim=256, hidden_dim=4096)

        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_backbone.parameters():
            p.requires_grad_(False)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

        self.base_tau = cfg.get("byol_tau", 0.996)
        self._tau = self.base_tau

    def set_momentum(self, tau: float) -> None:
        """Update the momentum coefficient (cosine schedule in training loop)."""
        self._tau = tau

    @torch.no_grad()
    def _update_target(self) -> None:
        for p_o, p_t in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            p_t.data = self._tau * p_t.data + (1.0 - self._tau) * p_o.data
        for p_o, p_t in zip(self.projector.parameters(), self.target_projector.parameters()):
            p_t.data = self._tau * p_t.data + (1.0 - self._tau) * p_o.data

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> dict:
        # Online path, both views
        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        p1 = self.predictor(self.projector(h1))
        p2 = self.predictor(self.projector(h2))

        with torch.no_grad():
            t1 = self.target_projector(self.target_backbone(view1))
            t2 = self.target_projector(self.target_backbone(view2))

        loss = _byol_mse(p1, t2) + _byol_mse(p2, t1)
        self._update_target()
        return {"loss": loss, "h": (h1, h2)}

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _byol_mse(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1).detach()
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()
