"""Matrix decompositions: LU, QR, SVD, eigendecomposition, Cholesky."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import linalg as _scipy_linalg  # optional-ish, but declared as a dependency


class LUResult(NamedTuple):
    P: np.ndarray
    L: np.ndarray
    U: np.ndarray


class QRResult(NamedTuple):
    Q: np.ndarray
    R: np.ndarray


class SVDResult(NamedTuple):
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray


class EigResult(NamedTuple):
    values: np.ndarray
    vectors: np.ndarray


def lu(m: np.ndarray) -> LUResult:
    """PA = LU decomposition."""
    P, L, U = _scipy_linalg.lu(np.asarray(m, dtype=float))
    return LUResult(P=P, L=L, U=U)


def qr(m: np.ndarray) -> QRResult:
    """QR decomposition: m = Q @ R."""
    Q, R = np.linalg.qr(np.asarray(m, dtype=float))
    return QRResult(Q=Q, R=R)


def svd(m: np.ndarray, full_matrices: bool = False) -> SVDResult:
    """Singular value decomposition: m = U @ diag(S) @ Vt."""
    U, S, Vt = np.linalg.svd(np.asarray(m, dtype=float), full_matrices=full_matrices)
    return SVDResult(U=U, S=S, Vt=Vt)


def eig(m: np.ndarray) -> EigResult:
    """Eigendecomposition for general (possibly non-symmetric) matrices."""
    values, vectors = np.linalg.eig(np.asarray(m, dtype=float))
    return EigResult(values=values, vectors=vectors)


def eigh(m: np.ndarray) -> EigResult:
    """Eigendecomposition for symmetric/Hermitian matrices (real eigenvalues)."""
    values, vectors = np.linalg.eigh(np.asarray(m, dtype=float))
    return EigResult(values=values, vectors=vectors)


def cholesky(m: np.ndarray) -> np.ndarray:
    """Cholesky factor L such that m = L @ L.T (m must be symmetric positive-definite)."""
    return np.linalg.cholesky(np.asarray(m, dtype=float))
