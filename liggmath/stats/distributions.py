"""Basic probability distribution functions."""

from __future__ import annotations

import math

import numpy as np


def normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Probability density function of the normal distribution."""
    x = np.asarray(x, dtype=float)
    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    return coeff * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def normal_cdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Cumulative distribution function of the normal distribution."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _erf((x - mu) / (sigma * math.sqrt(2.0))))


def _erf(x: np.ndarray) -> np.ndarray:
    # numpy has no built-in erf without scipy; use math.erf element-wise via vectorize
    vec_erf = np.vectorize(math.erf)
    return vec_erf(x)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for X ~ Binomial(n, p)."""
    if not (0 <= k <= n):
        return 0.0
    comb = math.comb(n, k)
    return comb * (p**k) * ((1 - p) ** (n - k))


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lambda)."""
    if k < 0:
        return 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)
