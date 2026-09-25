"""Numerical helpers: gradient checking, one-hot encoding, dataset utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central-difference numerical gradient of a scalar function f at point x.

    Useful for verifying analytic gradients (gradient checking).
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + h
        f_plus = f(x)

        x[idx] = orig - h
        f_minus = f(x)

        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * h)
        it.iternext()
    return grad


def clip_gradients(grad: np.ndarray, max_norm: float) -> np.ndarray:
    """Rescale grad so its L2 norm does not exceed max_norm."""
    grad = np.asarray(grad, dtype=float)
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad


def one_hot(labels: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Convert integer class labels to one-hot vectors."""
    labels = np.asarray(labels, dtype=int)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    out = np.zeros((labels.size, num_classes))
    out[np.arange(labels.size), labels] = 1.0
    return out


def shuffle_data(*arrays: np.ndarray, seed: int | None = None) -> tuple[np.ndarray, ...]:
    """Shuffle multiple arrays together along the first axis."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    perm = rng.permutation(n)
    return tuple(np.asarray(a)[perm] for a in arrays)


def train_test_split(
    *arrays: np.ndarray, test_size: float = 0.2, seed: int | None = None
) -> tuple[np.ndarray, ...]:
    """Split arrays into train/test sets. Returns (train_0, test_0, train_1, test_1, ...)."""
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")
    n = len(arrays[0])
    shuffled = shuffle_data(*arrays, seed=seed)
    n_test = int(round(n * test_size))
    result = []
    for arr in shuffled:
        result.append(arr[n_test:])
        result.append(arr[:n_test])
    return tuple(result)


def set_seed(seed: int) -> np.random.Generator:
    """Return a seeded numpy random Generator for reproducible experiments."""
    return np.random.default_rng(seed)
