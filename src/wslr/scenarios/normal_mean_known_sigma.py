from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from ..types import ModelFit, ScoreBundle
from ..utils import make_rng
from .base import Scenario


class NormalMeanKnownSigmaScenario(Scenario):
    name = "Regular: Normal mean (sigma known)"
    slug = "normal_mean_known_sigma"
    default_n_list = [20, 50, 100, 300]
    default_effect_list = [0.0, 0.2, 0.4, 0.8]
    recommended_reps = 500

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = float(sigma)
        self.sigma2 = float(sigma * sigma)

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        x = rng.normal(loc=effect_size, scale=self.sigma, size=n)
        return {"x": x}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        start = perf_counter()
        try:
            x = np.asarray(data["x"], dtype=float).reshape(-1)
            n = x.size
            mu_hat = float(np.mean(x))
            cov = np.array([[self.sigma2 / n]], dtype=float)
            resid = x - mu_hat
            ll = -0.5 * n * np.log(2.0 * np.pi * self.sigma2) - 0.5 * np.sum((resid * resid) / self.sigma2)
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=np.array([mu_hat], dtype=float),
                cov=cov,
                loglike=float(ll),
                converged=True,
                error_type=None,
                runtime_ms=runtime_ms,
                extra={},
            )
        except Exception as exc:
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=None,
                cov=None,
                loglike=None,
                converged=False,
                error_type=f"normal_full_failed:{type(exc).__name__}",
                runtime_ms=runtime_ms,
                extra={},
            )

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        start = perf_counter()
        try:
            x = np.asarray(data["x"], dtype=float).reshape(-1)
            n = x.size
            mu0 = 0.0
            resid = x - mu0
            ll = -0.5 * n * np.log(2.0 * np.pi * self.sigma2) - 0.5 * np.sum((resid * resid) / self.sigma2)
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=np.array([mu0], dtype=float),
                cov=np.array([[self.sigma2 / n]], dtype=float),
                loglike=float(ll),
                converged=True,
                error_type=None,
                runtime_ms=runtime_ms,
                extra={},
            )
        except Exception as exc:
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=None,
                cov=None,
                loglike=None,
                converged=False,
                error_type=f"normal_null_failed:{type(exc).__name__}",
                runtime_ms=runtime_ms,
                extra={},
            )

    def restriction(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.array([float(params[0])], dtype=float)

    def restriction_jacobian(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.array([[1.0]], dtype=float)

    def score_components(
        self,
        data: dict[str, Any],
        null_fit: ModelFit,
        hypothesis_id: str = "main",
    ) -> ScoreBundle:
        try:
            x = np.asarray(data["x"], dtype=float).reshape(-1)
            n = x.size
            mu0 = 0.0
            score = np.array([np.sum((x - mu0) / self.sigma2)], dtype=float)
            info = np.array([[n / self.sigma2]], dtype=float)
            return ScoreBundle(score=score, information=info, df=1, error_type=None)
        except Exception as exc:
            return ScoreBundle(error_type=f"normal_score_failed:{type(exc).__name__}:{exc}")

    def null_spec(self, hypothesis_id: str = "main") -> dict[str, Any]:
        return {"hypothesis": "H0: mu = 0", "target": "mean", "df": 1}

    def notes(self) -> str:
        return "正則・解析的に扱える設定。nが増えるとWald/Score/LRはほぼ一致する。"
