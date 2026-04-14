"""Learning-rate and coefficient schedules."""

from __future__ import annotations

import math


def cosine_schedule(base: float, current_step: int, total_steps: int,
                    warmup_steps: int = 0, min_value: float = 0.0) -> float:
    """Cosine annealing with linear warmup.

    Returns a value in [min_value, base] depending on the progress of
    `current_step` through `total_steps`.
    """
    if total_steps <= 0:
        return base
    if current_step < warmup_steps:
        return base * (current_step + 1) / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return min_value + 0.5 * (base - min_value) * (1.0 + math.cos(math.pi * progress))


def linear_ramp(value: float, current_step: int, start_step: int, end_step: int) -> float:
    """Piecewise-linear ramp from 0 -> value between start_step and end_step."""
    if current_step <= start_step:
        return 0.0
    if current_step >= end_step:
        return value
    frac = (current_step - start_step) / max(1, end_step - start_step)
    return value * frac
