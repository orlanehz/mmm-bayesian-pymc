"""Transformations (adstock, saturation, etc.)."""
from __future__ import annotations

import numpy as np


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Simple geometric adstock.
    - x must be a 1D non-negative array
    - decay in [0, 1)
    """
    if not (0.0 <= decay < 1.0):
        raise ValueError("decay must be in [0, 1)")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if np.any(x < 0):
        raise ValueError("x must be non-negative")

    out = np.zeros_like(x)
    for t in range(len(x)):
        out[t] = x[t] + (out[t - 1] * decay if t > 0 else 0.0)
    return out


def log_saturation(x: np.ndarray) -> np.ndarray:
    """Simple diminishing returns: log(1+x), robust to negatives by clipping at 0."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.maximum(x, 0.0))