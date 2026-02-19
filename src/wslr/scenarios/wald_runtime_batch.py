from __future__ import annotations

from typing import Any

import numpy as np

from ..tests_core import run_lr_test, run_score_test, run_wald_test
from ..types import ModelFit, ScoreBundle
from ..utils import make_rng
from .base import Scenario
from .common import fit_ols, fit_ols_with_zero_constraints, ols_score_bundle


class WaldRuntimeBatchScenario(Scenario):
    name = "Practical: Wald batch speed for many hypotheses"
    slug = "wald_runtime_batch"
    default_n_list = [80, 150, 300]
    default_effect_list = [0.0, 0.2, 0.4]
    recommended_reps = 250

    def __init__(self, p_features: int = 20, n_hypotheses: int = 10) -> None:
        self.p_features = int(p_features)
        self.n_hypotheses = int(n_hypotheses)
        self.tested_indices = [i + 1 for i in range(self.n_hypotheses)]

    def hypothesis_ids(self) -> list[str]:
        return [f"beta_{idx}" for idx in self.tested_indices]

    def _index_from_hypothesis(self, hypothesis_id: str) -> int:
        return int(hypothesis_id.split("_")[1])

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        X_raw = rng.normal(size=(n, self.p_features))
        X = np.column_stack([np.ones(n), X_raw])
        beta = np.zeros(self.p_features + 1, dtype=float)
        beta[0] = 0.5
        n_signal = min(5, self.p_features)
        for idx in range(1, n_signal + 1):
            beta[idx] = effect_size
        y = X @ beta + rng.normal(scale=1.0, size=n)
        return {"X": X, "y": y}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_ols(data["X"], data["y"])

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        idx = self._index_from_hypothesis(hypothesis_id)
        return fit_ols_with_zero_constraints(data["X"], data["y"], [idx])

    def restriction(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        idx = self._index_from_hypothesis(hypothesis_id)
        return np.array([float(params[idx])], dtype=float)

    def restriction_jacobian(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        idx = self._index_from_hypothesis(hypothesis_id)
        R = np.zeros((1, params.size), dtype=float)
        R[0, idx] = 1.0
        return R

    def score_components(
        self,
        data: dict[str, Any],
        null_fit: ModelFit,
        hypothesis_id: str = "main",
    ) -> ScoreBundle:
        if null_fit.params is None:
            return ScoreBundle(error_type="null_params_missing")
        idx = self._index_from_hypothesis(hypothesis_id)
        sigma2 = float(null_fit.extra.get("sigma2", np.nan))
        return ols_score_bundle(
            y=data["y"],
            X_full=data["X"],
            tested_indices=[idx],
            null_params=null_fit.params,
            sigma2=sigma2,
        )

    def null_spec(self, hypothesis_id: str = "main") -> dict[str, Any]:
        idx = self._index_from_hypothesis(hypothesis_id)
        return {"hypothesis": f"H0: beta_{idx} = 0", "target": "linear_coefficient", "df": 1}

    def true_parameter_value(
        self,
        n: int,
        effect_size: float,
        hypothesis_id: str = "main",
    ) -> float:
        idx = self._index_from_hypothesis(hypothesis_id)
        n_signal = min(5, self.p_features)
        if idx <= n_signal:
            return float(effect_size)
        return 0.0

    def notes(self) -> str:
        return (
            "When many hypotheses are tested from one shared full fit, Wald can be much faster than "
            "re-fitting null models for LR/Score."
        )

    def run_replication(
        self,
        seed: int,
        n: int,
        effect_size: float,
        rep: int,
        alpha: float = 0.05,
    ) -> list[dict[str, Any]]:
        data = self.generate(seed=seed, n=n, effect_size=effect_size)
        rows: list[dict[str, Any]] = []
        full_fit = self.fit_full(data, hypothesis_id="main")
        wald_full_share = full_fit.runtime_ms / max(1, len(self.hypothesis_ids()))
        null_stub = ModelFit(
            params=None,
            cov=None,
            loglike=None,
            converged=True,
            error_type=None,
            runtime_ms=0.0,
            extra={},
        )
        for hypothesis_id in self.hypothesis_ids():
            df = 1
            theta_true = self.true_parameter_value(n=n, effect_size=effect_size, hypothesis_id=hypothesis_id)
            if full_fit.params is None:
                wald = run_wald_test(full_fit, null_stub, None, None, df=df, alpha=alpha)
            else:
                restriction = self.restriction(full_fit.params, hypothesis_id=hypothesis_id)
                jacobian = self.restriction_jacobian(full_fit.params, hypothesis_id=hypothesis_id)
                wald = run_wald_test(full_fit, null_stub, restriction, jacobian, df=df, alpha=alpha)
            wald.runtime_ms = float(wald.runtime_ms + wald_full_share)
            rows.append(
                self._result_to_row(
                    result=wald,
                    hypothesis_id=hypothesis_id,
                    n=n,
                    effect_size=effect_size,
                    rep=rep,
                    seed=seed,
                    alpha=alpha,
                    theta_true=theta_true,
                )
            )
            null_fit = self.fit_null(data, hypothesis_id=hypothesis_id)
            score_bundle = self.score_components(data, null_fit, hypothesis_id=hypothesis_id)
            score = run_score_test(full_fit, null_fit, score_bundle, alpha=alpha)
            if np.isfinite(null_fit.runtime_ms):
                score.runtime_ms = float(score.runtime_ms + null_fit.runtime_ms)
            theta_hat = np.nan
            if full_fit.params is not None:
                theta_hat = float(self.restriction(full_fit.params, hypothesis_id=hypothesis_id)[0])
            lr = run_lr_test(full_fit, null_fit, df=df, theta_hat=theta_hat, alpha=alpha)
            if np.isfinite(null_fit.runtime_ms):
                lr.runtime_ms = float(lr.runtime_ms + null_fit.runtime_ms)
            lr.runtime_ms = float(lr.runtime_ms + wald_full_share)
            rows.append(
                self._result_to_row(
                    result=score,
                    hypothesis_id=hypothesis_id,
                    n=n,
                    effect_size=effect_size,
                    rep=rep,
                    seed=seed,
                    alpha=alpha,
                    theta_true=theta_true,
                )
            )
            rows.append(
                self._result_to_row(
                    result=lr,
                    hypothesis_id=hypothesis_id,
                    n=n,
                    effect_size=effect_size,
                    rep=rep,
                    seed=seed,
                    alpha=alpha,
                    theta_true=theta_true,
                )
            )
        return rows
