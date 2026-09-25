"""Vector operations built on numpy arrays."""

from __future__ import annotations

import numpy as np


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two 1-D vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.dot(a, b))


def norm(v: np.ndarray, ord: int | float | str | None = 2) -> float:
    """Vector norm (default: Euclidean / L2)."""
    return float(np.linalg.norm(np.asarray(v, dtype=float), ord=ord))


def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector in the direction of v."""
    v = np.asarray(v, dtype=float)
    n = norm(v)
    if n == 0:
        raise ValueError("cannot normalize the zero vector")
    return v / n


def angle_between(a: np.ndarray, b: np.ndarray, degrees: bool = False) -> float:
    """Angle between two vectors, in radians (or degrees)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cos_theta = dot(a, b) / (norm(a) * norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    return np.degrees(theta) if degrees else theta


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-D vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("cross product requires two 3-D vectors")
    return np.cross(a, b)


def project(a: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Orthogonal projection of vector a onto vector `onto`."""
    a, onto = np.asarray(a, dtype=float), np.asarray(onto, dtype=float)
    denom = dot(onto, onto)
    if denom == 0:
        raise ValueError("cannot project onto the zero vector")
    return (dot(a, onto) / denom) * onto


def is_orthogonal_vectors(a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
    """Whether two vectors are orthogonal within tolerance."""
    return abs(dot(a, b)) < tol
