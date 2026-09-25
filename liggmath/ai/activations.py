"""Activation functions and their derivatives, for use in neural networks."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid, evaluated at raw input x (not at sigmoid(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.asarray(x, dtype=float))


def relu_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x > 0).astype(float)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * x)


def leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1.0, alpha)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(x, dtype=float))


def tanh_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(np.asarray(x, dtype=float))
    return 1.0 - t**2


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the given axis."""
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (tanh approximation, as used in GPT/BERT)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def silu(x: np.ndarray) -> np.ndarray:
    """Sigmoid Linear Unit / Swish: x * sigmoid(x)."""
    x = np.asarray(x, dtype=float)
    return x * sigmoid(x)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
