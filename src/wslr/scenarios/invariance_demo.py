from __future__ import annotations

from typing import Any

import numpy as np
import statsmodels.api as sm

from ..types import ModelFit, ScoreBundle
from ..utils import make_rng
from .base import Scenario
from .common import fit_glm, glm_score_bundle


class InvarianceDemoScenario(Scenario):
    name = "Invariance demo: beta=0 vs OR=1"
    slug = "invariance_demo"
    default_n_list = [25, 50, 100, 200]
    default_effect_list = [0.0, 0.2, 0.4, 0.7]
    recommended_reps = 400

    def hypothesis_ids(self) -> list[str]:
        return ["beta", "or"]

    def test_suite(self, hypothesis_id: str) -> list[str]:
        if hypothesis_id == "or":
            return ["wald"]
        return ["wald", "score", "lr"]

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        x = rng.normal(size=n)
        eta = -0.4 + effect_size * x
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p, size=n).astype(float)
        X_full = np.column_stack([np.ones(n), x])
        X_null = np.ones((n, 1), dtype=float)
        return {"X_full": X_full, "X_null": X_null, "x": x, "y": y}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_glm(data["X_full"], data["y"], sm.families.Binomial(), maxiter=200)

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_glm(data["X_null"], data["y"], sm.families.Binomial(), maxiter=200)

    def restriction(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        beta = float(params[1])
        if hypothesis_id == "or":
            return np.array([np.exp(beta) - 1.0], dtype=float)
        return np.array([beta], dtype=float)

    def restriction_jacobian(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        beta = float(params[1])
        if hypothesis_id == "or":
            return np.array([[0.0, np.exp(beta)]], dtype=float)
        return np.array([[0.0, 1.0]], dtype=float)

    def score_components(
        self,
        data: dict[str, Any],
        null_fit: ModelFit,
        hypothesis_id: str = "main",
    ) -> ScoreBundle:
        result = null_fit.extra.get("result")
        if result is None:
            return ScoreBundle(error_type="null_result_missing")
        mu = np.asarray(result.fittedvalues, dtype=float)
        var_w = mu * (1.0 - mu)
        X_test = data["x"].reshape(-1, 1)
        return glm_score_bundle(
            y=data["y"],
            X_null=data["X_null"],
            X_test=X_test,
            mu=mu,
            variance_weight=var_w,
        )

    def null_spec(self, hypothesis_id: str = "main") -> dict[str, Any]:
        if hypothesis_id == "or":
            return {"hypothesis": "H0: OR = 1", "target": "odds_ratio", "df": 1}
        return {"hypothesis": "H0: beta_1 = 0", "target": "logit_slope", "df": 1}

    def notes(self) -> str:
        return "同値仮説でもWaldはパラメータ化依存でズレ得る。LRは同じfull/null比較なので相対的に不変。"
