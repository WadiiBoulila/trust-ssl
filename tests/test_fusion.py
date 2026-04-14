"""Unit tests for the Dempster-Shafer fusion implementation.

These tests check the mathematical properties of the K and I signals
used in Equations (6)-(7) of the paper:

  - K is zero when both belief vectors agree on a single prototype.
  - K is large when the two belief vectors concentrate on disjoint
    prototypes (the canonical conflict case).
  - I is near zero when both views are highly confident (evidence >> prior).
  - I approaches one as both views approach total ignorance.
  - K is monotone non-decreasing as the two belief vectors are rotated
    to be more disjoint.
"""

import torch

from trust_ssl.models.trust_ssl import dempster_fusion


def _one_hot(dim: int, idx: int) -> torch.Tensor:
    v = torch.zeros(dim)
    v[idx] = 1.0
    return v


def test_conflict_zero_on_agreement() -> None:
    """Two delta-function beliefs on the same prototype give K = 0."""
    b1 = _one_hot(10, 3).view(1, 1, 10)
    b2 = _one_hot(10, 3).view(1, 1, 10)
    u1 = torch.zeros(1, 1)
    u2 = torch.zeros(1, 1)
    K, I = dempster_fusion(b1, b2, u1, u2)
    assert K.item() < 1e-5, f"expected K ~= 0, got {K.item()}"
    assert I.item() < 1e-5


def test_conflict_high_on_disjoint_beliefs() -> None:
    """Two delta-function beliefs on different prototypes give K -> 1."""
    b1 = _one_hot(10, 2).view(1, 1, 10)
    b2 = _one_hot(10, 7).view(1, 1, 10)
    u1 = torch.zeros(1, 1)
    u2 = torch.zeros(1, 1)
    K, _ = dempster_fusion(b1, b2, u1, u2)
    assert K.item() > 0.9, f"expected K near 1 for disjoint beliefs, got {K.item()}"


def test_ignorance_small_when_confident() -> None:
    """When both views have u -> 0, fused ignorance should be small."""
    b1 = _one_hot(5, 0).view(1, 1, 5)
    b2 = _one_hot(5, 0).view(1, 1, 5)
    u1 = torch.tensor([[0.001]])
    u2 = torch.tensor([[0.001]])
    _, I = dempster_fusion(b1, b2, u1, u2)
    assert I.item() < 1e-3


def test_ignorance_large_when_uncertain() -> None:
    """When both views have u -> 1, fused ignorance should be large."""
    b1 = (torch.ones(5) / 5.0).view(1, 1, 5)
    b2 = (torch.ones(5) / 5.0).view(1, 1, 5)
    u1 = torch.tensor([[0.9]])
    u2 = torch.tensor([[0.9]])
    _, I = dempster_fusion(b1, b2, u1, u2)
    assert I.item() > 0.5


def test_conflict_monotone_in_angle() -> None:
    """As two belief vectors rotate from aligned to disjoint, K rises."""
    b_aligned = torch.tensor([0.8, 0.1, 0.05, 0.05]).view(1, 1, 4)
    b_medium = torch.tensor([0.4, 0.4, 0.1, 0.1]).view(1, 1, 4)
    b_disjoint = torch.tensor([0.05, 0.05, 0.4, 0.5]).view(1, 1, 4)
    u = torch.tensor([[0.0]])

    K1, _ = dempster_fusion(b_aligned, b_aligned, u, u)
    K2, _ = dempster_fusion(b_aligned, b_medium, u, u)
    K3, _ = dempster_fusion(b_aligned, b_disjoint, u, u)
    assert K1.item() < K2.item() < K3.item(), (
        f"expected monotone K: {K1.item()} < {K2.item()} < {K3.item()}"
    )
