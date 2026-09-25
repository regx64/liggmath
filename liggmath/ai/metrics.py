"""Evaluation metrics for classification and regression."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Confusion matrix for integer class labels; rows = true, cols = predicted."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> float:
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    denom = p + r
    return float(2 * p * r / denom) if denom > 0 else 0.0


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination for regression."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
