"""Solving linear systems."""

from __future__ import annotations

import numpy as np
from scipy import linalg as _scipy_linalg


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the square linear system a @ x = b."""
    return np.linalg.solve(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def least_squares(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares solution to a @ x = b (works for over/under-determined systems)."""
    x, *_ = np.linalg.lstsq(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rcond=None)
    return x


def solve_triangular(a: np.ndarray, b: np.ndarray, lower: bool = True) -> np.ndarray:
    """Solve a triangular linear system a @ x = b."""
    return _scipy_linalg.solve_triangular(np.asarray(a, dtype=float), np.asarray(b, dtype=float), lower=lower)
