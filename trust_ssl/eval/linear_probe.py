"""Linear-probe training on frozen backbone features.

This module provides a single function `train_linear_head` that is
called by both the linear-eval script and the robustness-eval script.
The linear head is a single `nn.Linear(feat_dim, num_classes)` trained
with SGD and a cosine schedule on the training split of the given
dataset, and selected by validation accuracy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(model, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen backbone on `loader` and return concatenated (features, labels)."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model.extract(x) if hasattr(model, "extract") else model(x)
        feats.append(f.detach().cpu())
        labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def train_linear_head(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 100,
    batch_size: int = 256,
    base_lr: float = 0.1,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> tuple[nn.Linear, dict]:
    """Train a single linear classifier on frozen features.

    Returns (best_head, summary), where `best_head` is the linear head
    with the highest validation accuracy and `summary` is a dict of
    training statistics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = train_feats.shape[1]

    head = nn.Linear(feat_dim, num_classes).to(device)
    nn.init.zeros_(head.bias)
    nn.init.normal_(head.weight, std=0.01)

    opt = torch.optim.SGD(head.parameters(), lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    best_val = -1.0
    best_state = None
    best_epoch = -1

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)

    n = train_feats.shape[0]
    steps_per_epoch = max(1, math.ceil(n / batch_size))

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs - 1)))
        for g in opt.param_groups:
            g["lr"] = lr

        for step in range(steps_per_epoch):
            sl = perm[step * batch_size: (step + 1) * batch_size]
            if sl.numel() == 0:
                continue
            xb = train_feats[sl]
            yb = train_labels[sl]
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Validation
        head.eval()
        with torch.no_grad():
            val_pred = head(val_feats).argmax(dim=-1)
            val_acc = (val_pred == val_labels).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    return head, {
        "best_val_accuracy": best_val,
        "best_epoch": best_epoch,
        "epochs": epochs,
    }


@torch.no_grad()
def evaluate_head(head: nn.Linear, feats: torch.Tensor, labels: torch.Tensor,
                  device: torch.device | None = None) -> float:
    """Return top-1 accuracy of `head` on the given features."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head.eval()
    feats = feats.to(device)
    labels = labels.to(device)
    pred = head(feats).argmax(dim=-1)
    return (pred == labels).float().mean().item()
