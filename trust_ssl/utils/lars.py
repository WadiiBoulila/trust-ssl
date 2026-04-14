"""LARS optimizer (layer-wise adaptive rate scaling).

Follows the formulation of You et al., 2017. Used for pretraining all
six methods in the paper with base lr 0.3 scaled by batch size.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.3,
        weight_decay: float = 1e-6,
        momentum: float = 0.9,
        eta: float = 0.001,
        epsilon: float = 1e-8,
        exclude_from_adaptation=None,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if momentum < 0.0:
            raise ValueError(f"invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            epsilon=epsilon,
        )
        super().__init__(params, defaults)
        self.exclude_from_adaptation = exclude_from_adaptation or (lambda p: False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            eps = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                if not self.exclude_from_adaptation(p):
                    w_norm = p.norm(p=2)
                    g_norm = grad.norm(p=2)
                    trust_ratio = torch.where(
                        (w_norm > 0) & (g_norm > 0),
                        eta * w_norm / (g_norm + eps),
                        torch.ones_like(w_norm),
                    )
                    grad = grad * trust_ratio

                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(p)
                    state["momentum_buffer"] = buf
                buf.mul_(momentum).add_(grad)

                p.add_(buf, alpha=-lr)

        return loss
