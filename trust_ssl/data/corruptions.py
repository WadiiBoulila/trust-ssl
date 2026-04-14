"""Corruption transforms used for the robustness evaluation and the K–I study.

Nine corruption types × five severities, grouped into three families
(erasure, contradiction, weather) as reported in Section V of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _as_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float()


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def _severity_scale(severity: int, lo: float, hi: float) -> float:
    """Linearly interpolate (lo, hi) by severity in {1,...,5}."""
    severity = int(max(1, min(5, severity)))
    t = (severity - 1) / 4.0
    return lo + t * (hi - lo)


# ──────────────────────────────────────────────────────────────────────
# Corruption primitives
# Input / output convention:
#   x: (C, H, W) torch.Tensor in [0, 1]
#   severity: int in {1, 2, 3, 4, 5}
# ──────────────────────────────────────────────────────────────────────
def gaussian_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    sigma = _severity_scale(severity, 1.0, 6.0)
    k = int(2 * round(3 * sigma) + 1)
    coords = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    kernel = (g.transpose(-1, -2) @ g).view(1, 1, k, k).to(x.device)
    x_b = x.unsqueeze(0)
    pad = k // 2
    c = x_b.shape[1]
    kernel_full = kernel.expand(c, 1, k, k)
    return F.conv2d(x_b, kernel_full, padding=pad, groups=c).squeeze(0)


def motion_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    length = int(_severity_scale(severity, 5.0, 25.0))
    kernel = torch.zeros(length, length, device=x.device)
    kernel[length // 2, :] = 1.0 / length
    kernel = kernel.view(1, 1, length, length).expand(x.shape[0], 1, length, length)
    return F.conv2d(x.unsqueeze(0), kernel, padding=length // 2, groups=x.shape[0]).squeeze(0)


def haze(x: torch.Tensor, severity: int) -> torch.Tensor:
    beta = _severity_scale(severity, 0.1, 0.6)
    A = 0.9
    out = x * (1.0 - beta) + A * beta
    return _clamp01(out)


def occlusion(x: torch.Tensor, severity: int) -> torch.Tensor:
    c, h, w = x.shape
    frac = _severity_scale(severity, 0.08, 0.30)
    bh = int(h * (frac ** 0.5))
    bw = int(w * (frac ** 0.5))
    y0 = torch.randint(0, max(1, h - bh), (1,)).item()
    x0 = torch.randint(0, max(1, w - bw), (1,)).item()
    out = x.clone()
    out[:, y0:y0 + bh, x0:x0 + bw] = 0.0
    return out


def color_distortion(x: torch.Tensor, severity: int) -> torch.Tensor:
    s = _severity_scale(severity, 0.3, 1.5)
    gain = 1.0 + s * (torch.rand(3, 1, 1, device=x.device) - 0.5)
    bias = s * 0.2 * (torch.rand(3, 1, 1, device=x.device) - 0.5)
    return _clamp01(x * gain + bias)


def brightness_inversion(x: torch.Tensor, severity: int) -> torch.Tensor:
    alpha = _severity_scale(severity, 0.2, 1.0)
    return _clamp01((1.0 - alpha) * x + alpha * (1.0 - x))


def contrast_reversal(x: torch.Tensor, severity: int) -> torch.Tensor:
    alpha = _severity_scale(severity, 0.2, 1.0)
    m = x.mean(dim=(-1, -2), keepdim=True)
    return _clamp01(m + (1.0 - 2.0 * alpha) * (x - m))


def channel_dropout(x: torch.Tensor, severity: int) -> torch.Tensor:
    frac = _severity_scale(severity, 0.1, 0.9)
    c = x.shape[0]
    n_drop = max(1, int(round(c * frac)))
    idx = torch.randperm(c, device=x.device)[:n_drop]
    out = x.clone()
    out[idx] = 0.0
    return out


def rain(x: torch.Tensor, severity: int) -> torch.Tensor:
    intensity = _severity_scale(severity, 0.05, 0.5)
    n_streaks = int(20 * severity)
    c, h, w = x.shape
    overlay = torch.zeros_like(x)
    for _ in range(n_streaks):
        y0 = torch.randint(0, h, (1,)).item()
        x0 = torch.randint(0, w, (1,)).item()
        length = int(_severity_scale(severity, 4.0, 16.0))
        length = min(length, h - y0)
        overlay[:, y0:y0 + length, x0] = 1.0
    return _clamp01(x * (1.0 - intensity) + overlay * intensity)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Corruption:
    name: str
    fn: Callable[[torch.Tensor, int], torch.Tensor]
    family: str


CORRUPTIONS: list[Corruption] = [
    Corruption("gaussian_blur",       gaussian_blur,       "erasure"),
    Corruption("motion_blur",         motion_blur,         "erasure"),
    Corruption("haze",                haze,                "erasure"),
    Corruption("occlusion",           occlusion,           "erasure"),
    Corruption("color_distortion",    color_distortion,    "contradiction"),
    Corruption("brightness_inversion",brightness_inversion,"contradiction"),
    Corruption("contrast_reversal",   contrast_reversal,   "contradiction"),
    Corruption("channel_dropout",     channel_dropout,     "contradiction"),
    Corruption("rain",                rain,                "weather"),
]

FAMILY_OF: dict[str, str] = {c.name: c.family for c in CORRUPTIONS}


def apply_corruption(name: str, x: torch.Tensor, severity: int) -> torch.Tensor:
    """Apply the named corruption at the given severity to a (C, H, W) tensor.

    The input is assumed to be in [0, 1]. Mean/std normalization should
    be applied by the caller after corruption.
    """
    for c in CORRUPTIONS:
        if c.name == name:
            return c.fn(x, severity)
    raise ValueError(f"unknown corruption: {name}")
