from __future__ import annotations

from typing import Any

import numpy as np
import statsmodels.api as sm

from ..types import ModelFit, ScoreBundle
from ..utils import make_rng
from .base import Scenario
from .common import fit_glm, glm_score_bundle


class LogisticLocalAlternativesScenario(Scenario):
    name = "Local alternatives: Logistic beta=c/sqrt(n)"
    slug = "logistic_local_alternatives"
    default_n_list = [40, 80, 150, 300]
    default_effect_list = [0.0, 0.5, 1.0, 1.5, 2.0]
    recommended_reps = 500

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        x = rng.normal(size=n)
        beta_local = float(effect_size) / np.sqrt(float(n))
        eta = -0.3 + beta_local * x
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p, size=n).astype(float)
        X_full = np.column_stack([np.ones(n), x])
        X_null = np.ones((n, 1), dtype=float)
        return {"X_full": X_full, "X_null": X_null, "x": x, "y": y, "beta_local": beta_local}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_glm(data["X_full"], data["y"], sm.families.Binomial(), maxiter=200)

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_glm(data["X_null"], data["y"], sm.families.Binomial(), maxiter=200)

    def restriction(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.array([float(params[1])], dtype=float)

    def restriction_jacobian(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
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
        return {"hypothesis": "H0: beta_1 = 0", "target": "local_logit_slope", "df": 1}

    def true_parameter_value(
        self,
        n: int,
        effect_size: float,
        hypothesis_id: str = "main",
    ) -> float:
        return float(effect_size) / np.sqrt(float(n))

    def notes(self) -> str:
        return (
            "This scenario sets beta_1=c/sqrt(n), so power grows more slowly than fixed-effect alternatives. "
            "Finite-sample differences among Wald/Score/LR can remain visible before asymptotic agreement dominates."
        )
