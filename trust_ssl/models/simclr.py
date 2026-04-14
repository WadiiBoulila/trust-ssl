"""SimCLR baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone, MLPProjector


class SimCLR(nn.Module):
    """SimCLR with ResNet-50 backbone and 2048-2048-256 projector.

    Loss is InfoNCE with in-batch negatives.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.backbone, feat_dim = build_backbone(pretrained=False)
        self.projector = MLPProjector(feat_dim, hidden_dim=2048, out_dim=256)
        self.temperature = cfg.get("temperature", 0.1)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> dict:
        h1 = self.backbone(view1)
        h2 = self.backbone(view2)
        z1 = F.normalize(self.projector(h1), dim=-1)
        z2 = F.normalize(self.projector(h2), dim=-1)

        loss = info_nce(z1, z2, self.temperature)
        return {"loss": loss, "h": (h1, h2), "z": (z1, z2)}

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard NT-Xent loss with in-batch negatives.

    z1, z2 are (B, D) and already L2-normalized.
    """
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                # (2B, D)
    sim = z @ z.t() / temperature                 # (2B, 2B)

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)

    return F.cross_entropy(sim, targets)
