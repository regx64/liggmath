import numpy as np
import pytest

from liggmath import stats


def test_descriptive_basics():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert stats.mean(x) == pytest.approx(2.5)
    assert stats.std(x) == pytest.approx(np.std(x))


def test_covariance_and_correlation():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    assert stats.covariance(x, y) == pytest.approx(np.cov(x, y)[0, 1])
    corr = stats.correlation_matrix(np.column_stack([x, y]))
    assert corr[0, 1] == pytest.approx(1.0)


def test_standardize():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = stats.standardize(x)
    assert np.mean(z) == pytest.approx(0.0, abs=1e-8)
    assert np.std(z) == pytest.approx(1.0, abs=1e-6)


def test_moving_average():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ma = stats.moving_average(x, 2)
    assert np.allclose(ma, [1.5, 2.5, 3.5, 4.5])


def test_normal_pdf_cdf():
    assert stats.normal_pdf(0.0) == pytest.approx(0.3989422804014327)
    assert stats.normal_cdf(0.0) == pytest.approx(0.5)


def test_binomial_poisson_pmf():
    assert stats.binomial_pmf(2, 4, 0.5) == pytest.approx(0.375)
    assert stats.poisson_pmf(0, 1.0) == pytest.approx(np.exp(-1.0))
