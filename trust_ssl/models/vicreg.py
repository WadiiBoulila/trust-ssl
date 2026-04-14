"""VICReg baseline (variance-invariance-covariance regularization)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone, MLPProjector


class VICReg(nn.Module):
    """VICReg: three regularizers on the embedding batch.

    sim: mean squared error between the two views.
    std: hinge on the per-feature standard deviation (variance term).
    cov: Frobenius norm of the off-diagonal entries of the embedding
         covariance matrix, summed over both views.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone, feat_dim = build_backbone(pretrained=False)
        self.projector = MLPProjector(feat_dim, hidden_dim=8192, out_dim=8192)
        self.sim_coeff = cfg.get("sim_coeff", 25.0)
        self.std_coeff = cfg.get("std_coeff", 25.0)
        self.cov_coeff = cfg.get("cov_coeff", 1.0)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> dict:
        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # Invariance
        sim_loss = F.mse_loss(z1, z2)

        # Variance: hinge toward std >= 1
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        std_loss = 0.5 * (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean())

        # Covariance off-diagonal
        batch_size, embed_dim = z1.shape
        z1c = z1 - z1.mean(dim=0, keepdim=True)
        z2c = z2 - z2.mean(dim=0, keepdim=True)
        cov_z1 = (z1c.T @ z1c) / (batch_size - 1)
        cov_z2 = (z2c.T @ z2c) / (batch_size - 1)
        cov_loss = (
            _off_diag(cov_z1).pow(2).sum() / embed_dim
            + _off_diag(cov_z2).pow(2).sum() / embed_dim
        )

        loss = self.sim_coeff * sim_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
        return {"loss": loss, "h": (h1, h2), "z": (z1, z2)}

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _off_diag(mat: torch.Tensor) -> torch.Tensor:
    n = mat.shape[0]
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
