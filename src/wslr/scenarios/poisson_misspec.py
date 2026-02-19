from __future__ import annotations

from typing import Any

import numpy as np
import statsmodels.api as sm

from ..types import ModelFit, ScoreBundle
from ..utils import make_rng
from .base import Scenario
from .common import fit_glm, glm_score_bundle


class PoissonMisspecScenario(Scenario):
    name = "Misspecification: True NegBin, fitted Poisson"
    slug = "poisson_misspec"
    default_n_list = [40, 80, 150, 300]
    default_effect_list = [0.0, 0.2, 0.4, 0.7]
    recommended_reps = 350

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        x = rng.normal(size=n)
        eta = 0.3 + effect_size * x
        mu = np.exp(eta)
        r = 1.0 / self.alpha
        p = r / (r + mu)
        y = rng.negative_binomial(r, p, size=n).astype(float)
        X_full = np.column_stack([np.ones(n), x])
        X_null = np.ones((n, 1), dtype=float)
        return {"X_full": X_full, "X_null": X_null, "x": x, "y": y}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_glm(data["X_full"], data["y"], sm.families.Poisson(), maxiter=200)

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_glm(data["X_null"], data["y"], sm.families.Poisson(), maxiter=200)

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
        X_test = data["x"].reshape(-1, 1)
        var_w = np.clip(mu, 1e-12, np.inf)
        return glm_score_bundle(
            y=data["y"],
            X_null=data["X_null"],
            X_test=X_test,
            mu=mu,
            variance_weight=var_w,
        )

    def null_spec(self, hypothesis_id: str = "main") -> dict[str, Any]:
        return {"hypothesis": "H0: beta_1 = 0", "target": "poisson_slope", "df": 1}

    def notes(self) -> str:
        return "真の分布が過分散なのにPoisson仮定で推定。3検定が同時にサイズ崩壊する例。"
