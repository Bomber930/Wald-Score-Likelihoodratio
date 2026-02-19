from __future__ import annotations

from typing import Any

import numpy as np

from ..types import ModelFit, ScoreBundle
from ..utils import make_rng
from .base import Scenario
from .common import fit_ols, fit_ols_with_zero_constraints, ols_score_bundle


class LinearRegressionScenario(Scenario):
    default_n_list = [30, 60, 120, 300]
    default_effect_list = [0.0, 0.15, 0.3, 0.6]
    recommended_reps = 400

    def __init__(self, tested_indices: list[int], name_suffix: str) -> None:
        self.tested_indices = [int(i) for i in tested_indices]
        self.name = f"Regular: Linear regression ({name_suffix})"
        self.slug = f"linear_regression_{name_suffix.replace(' ', '_').lower()}"

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        x3 = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x1, x2, x3])
        beta = np.array([1.0, 0.25, -0.20, 0.30], dtype=float)
        for idx in self.tested_indices:
            beta[idx] = effect_size
        y = X @ beta + rng.normal(scale=1.0, size=n)
        return {"X": X, "y": y}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_ols(data["X"], data["y"])

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_ols_with_zero_constraints(data["X"], data["y"], self.tested_indices)

    def restriction(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.asarray(params[self.tested_indices], dtype=float)

    def restriction_jacobian(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        p = params.size
        R = np.zeros((len(self.tested_indices), p), dtype=float)
        for row, idx in enumerate(self.tested_indices):
            R[row, idx] = 1.0
        return R

    def score_components(
        self,
        data: dict[str, Any],
        null_fit: ModelFit,
        hypothesis_id: str = "main",
    ) -> ScoreBundle:
        if null_fit.params is None:
            return ScoreBundle(error_type="null_params_missing")
        sigma2 = null_fit.extra.get("sigma2", np.nan)
        return ols_score_bundle(
            y=data["y"],
            X_full=data["X"],
            tested_indices=self.tested_indices,
            null_params=null_fit.params,
            sigma2=float(sigma2),
        )

    def null_spec(self, hypothesis_id: str = "main") -> dict[str, Any]:
        idx_str = ", ".join(f"beta_{idx}" for idx in self.tested_indices)
        return {
            "hypothesis": f"H0: {idx_str} = 0",
            "target": "linear_coefficients",
            "df": len(self.tested_indices),
        }

    def notes(self) -> str:
        return "正則条件下の基礎例。単一・複数制約ともにn増加で3検定が漸近的一致を示す。"
