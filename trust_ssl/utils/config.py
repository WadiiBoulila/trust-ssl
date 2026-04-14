"""Simple YAML config loader with override support."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Read a YAML file and return a plain dict."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} must be a mapping, got {type(data).__name__}")
    return data


def merge_configs(base: dict, override: dict | None) -> dict:
    """Shallow merge two config dicts; values in `override` win."""
    out = dict(base)
    if override:
        out.update(override)
    return out
