"""Statistics utilities: descriptive statistics and probability distributions."""

from .descriptive import (
    mean,
    variance,
    std,
    covariance,
    covariance_matrix,
    correlation_matrix,
    standardize,
    moving_average,
)
from .distributions import (
    normal_pdf,
    normal_cdf,
    binomial_pmf,
    poisson_pmf,
)

__all__ = [
    "mean",
    "variance",
    "std",
    "covariance",
    "covariance_matrix",
    "correlation_matrix",
    "standardize",
    "moving_average",
    "normal_pdf",
    "normal_cdf",
    "binomial_pmf",
    "poisson_pmf",
]
