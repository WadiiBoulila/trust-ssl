"""ResNet-50 backbone used by all methods in the paper."""

import torch
import torch.nn as nn
from torchvision.models import resnet50


def build_backbone(pretrained: bool = False) -> tuple[nn.Module, int]:
    """Return a ResNet-50 feature extractor and its output dimensionality.

    The final fully connected layer is replaced with an identity so that
    the module returns the 2048-dim pooled features.
    """
    net = resnet50(weights="DEFAULT" if pretrained else None)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat_dim


class MLPProjector(nn.Module):
    """Three-layer MLP projector used by SimCLR / BYOL / VICReg / Trust-SSL.

    Default sizes follow the paper: 2048 -> 2048 -> 256.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """Two-layer MLP predictor used by BYOL."""

    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
