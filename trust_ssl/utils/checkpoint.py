"""Checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, epoch: int | None = None,
                    extra: dict | None = None) -> None:
    """Save a checkpoint containing the model state, and optionally optimizer/epoch."""
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = int(epoch)
    if extra:
        state.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a checkpoint file. Returns the raw dict produced by save_checkpoint."""
    return torch.load(path, map_location=map_location, weights_only=False)
