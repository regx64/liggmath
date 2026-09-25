import numpy as np
import pytest

from liggmath import linalg


def test_vector_dot_norm_normalize():
    a = np.array([3.0, 4.0])
    assert linalg.norm(a) == pytest.approx(5.0)
    assert linalg.dot(a, a) == pytest.approx(25.0)
    unit = linalg.normalize(a)
    assert linalg.norm(unit) == pytest.approx(1.0)


def test_angle_and_orthogonality():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert linalg.angle_between(a, b, degrees=True) == pytest.approx(90.0)
    assert linalg.is_orthogonal_vectors(a, b)


def test_cross_and_project():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert np.allclose(linalg.cross(a, b), [0.0, 0.0, 1.0])
    proj = linalg.project(np.array([2.0, 2.0]), np.array([1.0, 0.0]))
    assert np.allclose(proj, [2.0, 0.0])


def test_matrix_basics():
    m = np.array([[2.0, 0.0], [0.0, 3.0]])
    assert linalg.trace(m) == pytest.approx(5.0)
    assert linalg.determinant(m) == pytest.approx(6.0)
    assert linalg.rank(m) == 2
    inv = linalg.inverse(m)
    assert np.allclose(m @ inv, np.eye(2))


def test_symmetric_orthogonal_pd():
    sym = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert linalg.is_symmetric(sym)
    assert linalg.is_positive_definite(sym)
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert linalg.is_orthogonal_matrix(rot)


def test_decompositions_reconstruct():
    m = np.array([[4.0, 3.0], [6.0, 3.0]])

    P, L, U = linalg.lu(m)
    assert np.allclose(P @ m, L @ U)

    Q, R = linalg.qr(m)
    assert np.allclose(Q @ R, m)
    assert np.allclose(Q @ Q.T, np.eye(2))

    U_, S, Vt = linalg.svd(m)
    assert np.allclose(U_ @ np.diag(S) @ Vt, m)

    sym = np.array([[2.0, 1.0], [1.0, 2.0]])
    values, vectors = linalg.eigh(sym)
    for i in range(len(values)):
        assert np.allclose(sym @ vectors[:, i], values[i] * vectors[:, i])

    L = linalg.cholesky(sym)
    assert np.allclose(L @ L.T, sym)


def test_solve_and_least_squares():
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    x = linalg.solve(a, b)
    assert np.allclose(a @ x, b)

    a_over = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    b_over = np.array([2.0, 3.0, 5.0])
    x_ls = linalg.least_squares(a_over, b_over)
    residual = np.linalg.norm(a_over @ x_ls - b_over)
    assert residual < np.linalg.norm(a_over @ (x_ls + 1.0) - b_over)


def test_hadamard_kronecker():
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0, 4.0]])
    assert np.allclose(linalg.hadamard(a, b), [[3.0, 8.0]])
    kron = linalg.kronecker(np.eye(2), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert kron.shape == (4, 4)
