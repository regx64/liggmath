"""Matrix operations and constructors built on numpy arrays."""

from __future__ import annotations

import numpy as np


def identity(n: int) -> np.ndarray:
    return np.eye(n)


def zeros(rows: int, cols: int | None = None) -> np.ndarray:
    return np.zeros((rows, cols if cols is not None else rows))


def ones(rows: int, cols: int | None = None) -> np.ndarray:
    return np.ones((rows, cols if cols is not None else rows))


def random_matrix(
    rows: int, cols: int | None = None, low: float = 0.0, high: float = 1.0, seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(rows, cols if cols is not None else rows))


def transpose(m: np.ndarray) -> np.ndarray:
    return np.asarray(m).T


def trace(m: np.ndarray) -> float:
    return float(np.trace(np.asarray(m, dtype=float)))


def rank(m: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(np.asarray(m, dtype=float)))


def determinant(m: np.ndarray) -> float:
    return float(np.linalg.det(np.asarray(m, dtype=float)))


def inverse(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(m, dtype=float))


def pseudo_inverse(m: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.asarray(m, dtype=float))


def is_symmetric(m: np.ndarray, tol: float = 1e-8) -> bool:
    m = np.asarray(m, dtype=float)
    return m.shape[0] == m.shape[1] and np.allclose(m, m.T, atol=tol)


def is_orthogonal_matrix(m: np.ndarray, tol: float = 1e-8) -> bool:
    m = np.asarray(m, dtype=float)
    if m.shape[0] != m.shape[1]:
        return False
    return np.allclose(m @ m.T, np.eye(m.shape[0]), atol=tol)


def is_positive_definite(m: np.ndarray) -> bool:
    m = np.asarray(m, dtype=float)
    if not is_symmetric(m):
        return False
    try:
        np.linalg.cholesky(m)
        return True
    except np.linalg.LinAlgError:
        return False


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=float) @ np.asarray(b, dtype=float)


def hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise (Hadamard) product."""
    return np.asarray(a, dtype=float) * np.asarray(b, dtype=float)


def kronecker(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kronecker product of two matrices."""
    return np.kron(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
