from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

from ..types import ModelFit, ScoreBundle, TestResult
from ..utils import make_rng
from .base import Scenario
from .common import fit_glm, glm_score_bundle


def cluster_wald_result(
    data: dict[str, Any],
    full_fit: ModelFit,
    null_fit: ModelFit,
    alpha: float = 0.05,
) -> TestResult:
    start = perf_counter()
    if full_fit.params is None:
        runtime_ms = (perf_counter() - start) * 1000.0
        return TestResult(
            test="wald_cluster",
            stat=np.nan,
            pvalue=np.nan,
            df=1,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=full_fit.error_type or "full_params_missing",
        )
    try:
        X = np.asarray(data["X_full"], dtype=float)
        y = np.asarray(data["y"], dtype=float).reshape(-1)
        groups = np.asarray(data["cluster_id"], dtype=int).reshape(-1)
        if groups.size != y.size:
            runtime_ms = (perf_counter() - start) * 1000.0
            return TestResult(
                test="wald_cluster",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="group_length_mismatch",
            )
        result = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            maxiter=200,
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
        params = np.asarray(result.params, dtype=float)
        cov = np.asarray(result.cov_params(), dtype=float)
        if params.size < 2 or cov.shape[0] < 2 or cov.shape[1] < 2:
            runtime_ms = (perf_counter() - start) * 1000.0
            return TestResult(
                test="wald_cluster",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="invalid_cluster_cov_dimension",
            )
        beta_1 = float(params[1])
        var_11 = float(cov[1, 1])
        if (not np.isfinite(var_11)) or var_11 <= 0:
            runtime_ms = (perf_counter() - start) * 1000.0
            return TestResult(
                test="wald_cluster",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="invalid_cluster_variance",
            )
        stat = float((beta_1 * beta_1) / var_11)
        if (not np.isfinite(stat)) or stat < 0:
            runtime_ms = (perf_counter() - start) * 1000.0
            return TestResult(
                test="wald_cluster",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="invalid_cluster_stat",
            )
        pvalue = float(chi2.sf(stat, 1))
        if not np.isfinite(pvalue):
            runtime_ms = (perf_counter() - start) * 1000.0
            return TestResult(
                test="wald_cluster",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="invalid_cluster_pvalue",
            )
        crit = float(np.sqrt(chi2.ppf(1.0 - alpha, 1)))
        half = crit * float(np.sqrt(var_11))
        ci_lower = float(beta_1 - half)
        ci_upper = float(beta_1 + half)
        runtime_ms = (perf_counter() - start) * 1000.0
        return TestResult(
            test="wald_cluster",
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
            ci_method="inverted_wald_cluster_chi2",
        )
    except Exception as exc:
        runtime_ms = (perf_counter() - start) * 1000.0
        return TestResult(
            test="wald_cluster",
            stat=np.nan,
            pvalue=np.nan,
            df=1,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=f"wald_cluster_failed:{type(exc).__name__}",
        )


def make_cluster_ids(n: int, cluster_size: int) -> np.ndarray:
    idx = np.arange(n, dtype=int)
    return idx // cluster_size


class LogisticClusteredScenario(Scenario):
    name = "Dependence: Logistic with clustered outcomes"
    slug = "logistic_clustered"
    default_n_list = [40, 80, 150, 300]
    default_effect_list = [0.0, 0.3, 0.6, 0.9]
    recommended_reps = 350

    def __init__(self, cluster_size: int = 8, cluster_sd: float = 1.0) -> None:
        self.cluster_size = int(cluster_size)
        self.cluster_sd = float(cluster_sd)

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        cluster_id = make_cluster_ids(n, cluster_size=max(2, self.cluster_size))
        n_clusters = int(cluster_id.max()) + 1
        x = rng.normal(size=n)
        b_cluster = rng.normal(loc=0.0, scale=self.cluster_sd, size=n_clusters)
        eta = -0.4 + effect_size * x + b_cluster[cluster_id]
        p = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, p, size=n).astype(float)
        X_full = np.column_stack([np.ones(n), x])
        X_null = np.ones((n, 1), dtype=float)
        return {
            "X_full": X_full,
            "X_null": X_null,
            "x": x,
            "y": y,
            "cluster_id": cluster_id.astype(int),
        }

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
        return {"hypothesis": "H0: beta_1 = 0", "target": "clustered_logit_slope", "df": 1}

    def notes(self) -> str:
        return (
            "Clustered dependence violates the independent-observation information matrix, "
            "so naive Wald/Score/LR under an iid logit can over-reject. "
            "Cluster-robust Wald can improve size calibration."
        )

    def extra_tests(
        self,
        data: dict[str, Any],
        full_fit: ModelFit,
        null_fit: ModelFit,
        hypothesis_id: str = "main",
        alpha: float = 0.05,
    ) -> list[TestResult]:
        cluster_wald = cluster_wald_result(data=data, full_fit=full_fit, null_fit=null_fit, alpha=alpha)
        return [cluster_wald]
