from __future__ import annotations

from typing import Any

import numpy as np

from ..types import ModelFit, ScoreBundle
from ..utils import make_rng
from .base import Scenario
from .common import fit_ols, fit_ols_with_zero_constraints, ols_score_bundle


class CollinearityScenario(Scenario):
    name = "Weak identification: severe collinearity"
    slug = "collinearity"
    default_n_list = [30, 60, 120, 250]
    default_effect_list = [0.0, 0.2, 0.4]
    recommended_reps = 350

    def __init__(self, collinearity_noise: float = 1e-8) -> None:
        self.collinearity_noise = float(collinearity_noise)

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        x1 = rng.normal(size=n)
        x2 = x1 + rng.normal(scale=self.collinearity_noise, size=n)
        x3 = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x1, x2, x3])
        beta = np.array([0.5, effect_size, 0.0, 0.3], dtype=float)
        y = X @ beta + rng.normal(scale=1.0, size=n)
        return {"X": X, "y": y}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_ols(data["X"], data["y"])

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_ols_with_zero_constraints(data["X"], data["y"], [1])

    def restriction(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.array([float(params[1])], dtype=float)

    def restriction_jacobian(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=float)

    def score_components(
        self,
        data: dict[str, Any],
        null_fit: ModelFit,
        hypothesis_id: str = "main",
    ) -> ScoreBundle:
        if null_fit.params is None:
            return ScoreBundle(error_type="null_params_missing")
        sigma2 = float(null_fit.extra.get("sigma2", np.nan))
        return ols_score_bundle(
            y=data["y"],
            X_full=data["X"],
            tested_indices=[1],
            null_params=null_fit.params,
            sigma2=sigma2,
        )

    def null_spec(self, hypothesis_id: str = "main") -> dict[str, Any]:
        return {"hypothesis": "H0: beta_1 = 0", "target": "collinear_slope", "df": 1}

    def notes(self) -> str:
        return "強共線性で情報行列がほぼ特異となり、分散推定不安定や失敗率上昇が起こる。"
