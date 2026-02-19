from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.stats import chi2

from .types import ModelFit, ScoreBundle, TestResult
from .utils import coalesce_errors, safe_inverse, safe_quadform


def build_failed_result(
    test: str,
    df: int,
    runtime_ms: float,
    converged_full: bool,
    converged_null: bool,
    error_type: str,
) -> TestResult:
    return TestResult(
        test=test,
        stat=np.nan,
        pvalue=np.nan,
        df=df,
        runtime_ms=runtime_ms,
        converged_full=converged_full,
        converged_null=converged_null,
        error_type=error_type,
    )


def run_wald_test(
    full_fit: ModelFit,
    null_fit: ModelFit,
    restriction: np.ndarray | None,
    jacobian: np.ndarray | None,
    df: int,
) -> TestResult:
    start = perf_counter()
    base_error = coalesce_errors([full_fit.error_type, null_fit.error_type])
    if base_error:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="wald",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=base_error,
        )
    if full_fit.params is None:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="wald",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="full_params_missing",
        )
    if full_fit.cov is None:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="wald",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="cov_missing",
        )
    if restriction is None or jacobian is None:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="wald",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="restriction_missing",
        )

    try:
        r = np.asarray(restriction, dtype=float).reshape(-1, 1)
        R = np.asarray(jacobian, dtype=float)
        V = np.asarray(full_fit.cov, dtype=float)
    except Exception:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="wald",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="restriction_cast_failed",
        )

    if R.ndim != 2:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="wald",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="invalid_jacobian",
        )
    middle = R @ V @ R.T
    middle_inv, inv_error = safe_inverse(middle)
    if inv_error:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="wald",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=inv_error,
        )
    stat, stat_error = safe_quadform(r, middle_inv)
    if stat_error:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="wald",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=stat_error,
        )
    pvalue = float(chi2.sf(stat, df))
    runtime_ms = (perf_counter() - start) * 1000.0
    if not np.isfinite(pvalue):
        return build_failed_result(
            test="wald",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="non_finite_pvalue",
        )
    return TestResult(
        test="wald",
        stat=stat,
        pvalue=pvalue,
        df=df,
        runtime_ms=runtime_ms,
        converged_full=full_fit.converged,
        converged_null=null_fit.converged,
        error_type=None,
    )


def run_score_test(full_fit: ModelFit, null_fit: ModelFit, bundle: ScoreBundle) -> TestResult:
    start = perf_counter()
    df = int(bundle.df)
    base_error = coalesce_errors([null_fit.error_type, bundle.error_type])
    if base_error:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="score",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=base_error,
        )
    if bundle.score is None or bundle.information is None:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="score",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="score_components_missing",
        )
    score = np.asarray(bundle.score, dtype=float).reshape(-1, 1)
    info = np.asarray(bundle.information, dtype=float)
    info_inv, inv_error = safe_inverse(info)
    if inv_error:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="score",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=inv_error,
        )
    stat, stat_error = safe_quadform(score, info_inv)
    if stat_error:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="score",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=stat_error,
        )
    pvalue = float(chi2.sf(stat, df))
    runtime_ms = (perf_counter() - start) * 1000.0
    if not np.isfinite(pvalue):
        return build_failed_result(
            test="score",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="non_finite_pvalue",
        )
    return TestResult(
        test="score",
        stat=stat,
        pvalue=pvalue,
        df=df,
        runtime_ms=runtime_ms,
        converged_full=full_fit.converged,
        converged_null=null_fit.converged,
        error_type=None,
    )


def run_lr_test(full_fit: ModelFit, null_fit: ModelFit, df: int) -> TestResult:
    start = perf_counter()
    base_error = coalesce_errors([full_fit.error_type, null_fit.error_type])
    if base_error:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="lr",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=base_error,
        )
    if full_fit.loglike is None or null_fit.loglike is None:
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="lr",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="loglike_missing",
        )
    stat = float(2.0 * (full_fit.loglike - null_fit.loglike))
    if not np.isfinite(stat):
        runtime_ms = (perf_counter() - start) * 1000.0
        return build_failed_result(
            test="lr",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="non_finite_stat",
        )
    if stat < 0:
        if stat > -1e-10:
            stat = 0.0
        else:
            runtime_ms = (perf_counter() - start) * 1000.0
            return build_failed_result(
                test="lr",
                df=df,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="negative_lr_stat",
            )
    pvalue = float(chi2.sf(stat, df))
    runtime_ms = (perf_counter() - start) * 1000.0
    if not np.isfinite(pvalue):
        return build_failed_result(
            test="lr",
            df=df,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type="non_finite_pvalue",
        )
    return TestResult(
        test="lr",
        stat=stat,
        pvalue=pvalue,
        df=df,
        runtime_ms=runtime_ms,
        converged_full=full_fit.converged,
        converged_null=null_fit.converged,
        error_type=None,
    )
