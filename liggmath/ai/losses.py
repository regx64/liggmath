"""Loss functions commonly used in machine learning."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy loss. y_pred is expected to be probabilities in (0, 1)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Categorical cross-entropy loss for one-hot y_true and probability y_pred, shape (N, C)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=-1)))


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss: quadratic for small errors, linear for large ones."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return float(np.mean(0.5 * quadratic**2 + delta * linear))
