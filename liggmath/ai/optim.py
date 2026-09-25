"""Simple gradient-based optimizers operating on numpy-array parameters.

Each optimizer takes a list/dict of parameter arrays and updates them in place
given matching gradient arrays, e.g.:

    opt = SGD(params, lr=0.01)
    ...
    opt.step(grads)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class SGD:
    """Vanilla stochastic gradient descent."""

    def __init__(self, params: Sequence[np.ndarray], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    def step(self, grads: Sequence[np.ndarray]) -> None:
        for p, g in zip(self.params, grads):
            p -= self.lr * g


class Momentum:
    """SGD with classical momentum."""

    def __init__(self, params: Sequence[np.ndarray], lr: float = 0.01, beta: float = 0.9):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.velocity = [np.zeros_like(p) for p in self.params]

    def step(self, grads: Sequence[np.ndarray]) -> None:
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.velocity[i] = self.beta * self.velocity[i] + (1 - self.beta) * g
            p -= self.lr * self.velocity[i]


class Adam:
    """Adam optimizer (Kingma & Ba, 2014)."""

    def __init__(
        self,
        params: Sequence[np.ndarray],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads: Sequence[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
