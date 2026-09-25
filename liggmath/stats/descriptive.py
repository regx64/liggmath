"""Descriptive statistics utilities."""

from __future__ import annotations

import numpy as np


def mean(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    return np.mean(np.asarray(x, dtype=float), axis=axis)


def variance(x: np.ndarray, axis: int | None = None, ddof: int = 0) -> np.ndarray | float:
    return np.var(np.asarray(x, dtype=float), axis=axis, ddof=ddof)


def std(x: np.ndarray, axis: int | None = None, ddof: int = 0) -> np.ndarray | float:
    return np.std(np.asarray(x, dtype=float), axis=axis, ddof=ddof)


def covariance(x: np.ndarray, y: np.ndarray, ddof: int = 1) -> float:
    """Scalar covariance between two 1-D samples."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.cov(x, y, ddof=ddof)[0, 1])


def covariance_matrix(x: np.ndarray, rowvar: bool = False, ddof: int = 1) -> np.ndarray:
    """Covariance matrix of a dataset. rowvar=False means rows are observations, cols are variables."""
    return np.cov(np.asarray(x, dtype=float), rowvar=rowvar, ddof=ddof)


def correlation_matrix(x: np.ndarray, rowvar: bool = False) -> np.ndarray:
    return np.corrcoef(np.asarray(x, dtype=float), rowvar=rowvar)


def standardize(x: np.ndarray, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """Z-score standardization: (x - mean) / std along the given axis."""
    x = np.asarray(x, dtype=float)
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sigma + eps)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with the given window size (valid convolution)."""
    x = np.asarray(x, dtype=float)
    if window <= 0 or window > len(x):
        raise ValueError("window must be in [1, len(x)]")
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")
