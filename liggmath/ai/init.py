"""Weight initialization schemes for neural network layers."""

from __future__ import annotations

import numpy as np


def xavier_uniform(fan_in: int, fan_out: int, seed: int | None = None) -> np.ndarray:
    """Glorot/Xavier uniform initialization, shape (fan_in, fan_out)."""
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def xavier_normal(fan_in: int, fan_out: int, seed: int | None = None) -> np.ndarray:
    """Glorot/Xavier normal initialization, shape (fan_in, fan_out)."""
    rng = np.random.default_rng(seed)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def he_uniform(fan_in: int, fan_out: int, seed: int | None = None) -> np.ndarray:
    """He uniform initialization (good default for ReLU networks)."""
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def he_normal(fan_in: int, fan_out: int, seed: int | None = None) -> np.ndarray:
    """He normal initialization (good default for ReLU networks)."""
    rng = np.random.default_rng(seed)
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))
