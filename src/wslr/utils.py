from __future__ import annotations

from typing import Iterable

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def coalesce_errors(errors: Iterable[str | None]) -> str | None:
    values = [e for e in errors if e]
    if not values:
        return None
    return ";".join(values)


def safe_inverse(matrix: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None, "not_square_matrix"
    if not np.all(np.isfinite(matrix)):
        return None, "non_finite_matrix"
    try:
        inv = np.linalg.inv(matrix)
        if not np.all(np.isfinite(inv)):
            return None, "non_finite_inverse"
        return inv, None
    except np.linalg.LinAlgError:
        return None, "singular_matrix"


def safe_quadform(vector: np.ndarray, matrix_inv: np.ndarray) -> tuple[float, str | None]:
    if vector.ndim == 1:
        vector = vector.reshape(-1, 1)
    if vector.ndim != 2 or vector.shape[1] != 1:
        return np.nan, "invalid_vector_shape"
    if not np.all(np.isfinite(vector)):
        return np.nan, "non_finite_vector"
    value = float(vector.T @ matrix_inv @ vector)
    if not np.isfinite(value):
        return np.nan, "non_finite_stat"
    if value < 0:
        # Small negative values can happen numerically.
        if value > -1e-10:
            value = 0.0
        else:
            return np.nan, "negative_stat"
    return value, None
