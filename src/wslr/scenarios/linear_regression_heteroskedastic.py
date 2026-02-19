from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

from ..types import ModelFit, ScoreBundle, TestResult
from ..utils import make_rng
from .base import Scenario
from .common import fit_ols, fit_ols_with_zero_constraints, ols_score_bundle


def robust_wald_hc3_result(
    data: dict[str, Any],
    full_fit: ModelFit,
    null_fit: ModelFit,
    alpha: float = 0.05,
) -> TestResult:
    start = perf_counter()
    if full_fit.params is None:
        runtime_ms = (perf_counter() - start) * 1000.0
        return TestResult(
            test="wald_hc3",
            stat=np.nan,
            pvalue=np.nan,
            df=1,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=full_fit.error_type or "full_params_missing",
        )
    try:
        X = np.asarray(data["X"], dtype=float)
        y = np.asarray(data["y"], dtype=float).reshape(-1)
        result = sm.OLS(y, X).fit(cov_type="HC3")
        params = np.asarray(result.params, dtype=float)
        cov = np.asarray(result.cov_params(), dtype=float)
        if params.size < 2 or cov.shape[0] < 2 or cov.shape[1] < 2:
            runtime_ms = (perf_counter() - start) * 1000.0
            return TestResult(
                test="wald_hc3",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="invalid_hc3_dimension",
            )
        beta_1 = float(params[1])
        var_11 = float(cov[1, 1])
        if (not np.isfinite(var_11)) or var_11 <= 0:
            runtime_ms = (perf_counter() - start) * 1000.0
            return TestResult(
                test="wald_hc3",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="invalid_hc3_variance",
            )
        stat = float((beta_1 * beta_1) / var_11)
        if (not np.isfinite(stat)) or stat < 0:
            runtime_ms = (perf_counter() - start) * 1000.0
            return TestResult(
                test="wald_hc3",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="invalid_hc3_stat",
            )
        pvalue = float(chi2.sf(stat, 1))
        if not np.isfinite(pvalue):
            runtime_ms = (perf_counter() - start) * 1000.0
            return TestResult(
                test="wald_hc3",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="invalid_hc3_pvalue",
            )
        crit = float(np.sqrt(chi2.ppf(1.0 - alpha, 1)))
        half = crit * float(np.sqrt(var_11))
        ci_lower = float(beta_1 - half)
        ci_upper = float(beta_1 + half)
        runtime_ms = (perf_counter() - start) * 1000.0
        return TestResult(
            test="wald_hc3",
            stat=stat,
            pvalue=pvalue,
            df=1,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=None,
            theta_hat=beta_1,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_method="inverted_wald_hc3_chi2",
        )
    except Exception as exc:
        runtime_ms = (perf_counter() - start) * 1000.0
        return TestResult(
            test="wald_hc3",
            stat=np.nan,
            pvalue=np.nan,
            df=1,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=f"wald_hc3_failed:{type(exc).__name__}",
        )


class LinearRegressionHeteroskedasticScenario(Scenario):
    name = "Misspecification: Linear regression heteroskedastic"
    slug = "linear_regression_heteroskedastic"
    default_n_list = [40, 80, 150, 300]
    default_effect_list = [0.0, 0.2, 0.4, 0.7]
    recommended_reps = 350

    def __init__(self, hetero_gamma: float = 2.0) -> None:
        self.hetero_gamma = float(hetero_gamma)

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x1, x2])
        beta = np.array([0.6, effect_size, 0.3], dtype=float)
        sigma2 = 1.0 + self.hetero_gamma * (x1 * x1)
        eps = rng.normal(scale=np.sqrt(np.clip(sigma2, 1e-12, np.inf)), size=n)
        y = X @ beta + eps
        return {"X": X, "y": y, "x1": x1, "sigma2_true": sigma2}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_ols(data["X"], data["y"])

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        return fit_ols_with_zero_constraints(data["X"], data["y"], [1])

    def restriction(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.array([float(params[1])], dtype=float)

    def restriction_jacobian(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0]], dtype=float)

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
        return {"hypothesis": "H0: beta_1 = 0", "target": "heteroskedastic_slope", "df": 1}

    def notes(self) -> str:
        return (
            "Heteroskedastic errors violate the constant-variance likelihood assumption, "
            "so nominal size can drift for Wald/Score/LR under the homoskedastic model. "
            "HC3-robust Wald can partially recover calibration."
        )

    def extra_tests(
        self,
        data: dict[str, Any],
        full_fit: ModelFit,
        null_fit: ModelFit,
        hypothesis_id: str = "main",
        alpha: float = 0.05,
    ) -> list[TestResult]:
        robust = robust_wald_hc3_result(data=data, full_fit=full_fit, null_fit=null_fit, alpha=alpha)
        return [robust]
