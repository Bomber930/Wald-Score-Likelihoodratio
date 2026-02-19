from __future__ import annotations

from time import perf_counter
import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

from ..types import ModelFit, ScoreBundle
from ..utils import safe_inverse


def fit_ols(X: np.ndarray, y: np.ndarray) -> ModelFit:
    start = perf_counter()
    try:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        n_obs, n_params = X_arr.shape
        beta, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        residual = y_arr - X_arr @ beta
        sse = float(np.sum(residual * residual))
        sigma2 = sse / n_obs
        xtx = X_arr.T @ X_arr
        cond_xtx = float(np.linalg.cond(xtx))
        if not np.isfinite(cond_xtx) or cond_xtx > 1e12:
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=beta,
                cov=None,
                loglike=None,
                converged=False,
                error_type="ill_conditioned_xtx",
                runtime_ms=runtime_ms,
                extra={"sse": sse, "sigma2": sigma2, "X": X_arr, "y": y_arr, "cond_xtx": cond_xtx},
            )
        xtx_inv, inv_error = safe_inverse(xtx)
        if inv_error:
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=beta,
                cov=None,
                loglike=None,
                converged=False,
                error_type=inv_error,
                runtime_ms=runtime_ms,
                extra={"sse": sse, "sigma2": sigma2, "X": X_arr, "y": y_arr},
            )
        cov = sigma2 * xtx_inv
        if sigma2 <= 0:
            sigma2 = 1e-12
        loglike = -0.5 * n_obs * (np.log(2.0 * np.pi * sigma2) + 1.0)
        runtime_ms = (perf_counter() - start) * 1000.0
        return ModelFit(
            params=beta,
            cov=cov,
            loglike=float(loglike),
            converged=True,
            error_type=None,
            runtime_ms=runtime_ms,
            extra={"sse": sse, "sigma2": sigma2, "X": X_arr, "y": y_arr, "rank": n_params},
        )
    except Exception as exc:
        runtime_ms = (perf_counter() - start) * 1000.0
        return ModelFit(
            params=None,
            cov=None,
            loglike=None,
            converged=False,
            error_type=f"ols_fit_failed:{type(exc).__name__}",
            runtime_ms=runtime_ms,
            extra={"exception": str(exc)},
        )


def fit_ols_with_zero_constraints(
    X: np.ndarray,
    y: np.ndarray,
    zero_indices: list[int],
) -> ModelFit:
    start = perf_counter()
    try:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        n_obs, n_params = X_arr.shape
        zero_set = set(int(i) for i in zero_indices)
        free_indices = [i for i in range(n_params) if i not in zero_set]
        X_free = X_arr[:, free_indices]
        sub_fit = fit_ols(X_free, y_arr)
        if sub_fit.params is None:
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=None,
                cov=None,
                loglike=sub_fit.loglike,
                converged=False,
                error_type=sub_fit.error_type or "restricted_ols_failed",
                runtime_ms=runtime_ms,
                extra={"free_indices": free_indices, "zero_indices": zero_indices},
            )
        params = np.zeros(n_params, dtype=float)
        params[free_indices] = sub_fit.params
        cov = None
        if sub_fit.cov is not None:
            cov = np.zeros((n_params, n_params), dtype=float)
            cov[np.ix_(free_indices, free_indices)] = sub_fit.cov
        runtime_ms = (perf_counter() - start) * 1000.0
        return ModelFit(
            params=params,
            cov=cov,
            loglike=sub_fit.loglike,
            converged=sub_fit.converged,
            error_type=sub_fit.error_type,
            runtime_ms=runtime_ms,
            extra={
                "free_indices": free_indices,
                "zero_indices": zero_indices,
                "sse": sub_fit.extra.get("sse"),
                "sigma2": sub_fit.extra.get("sigma2"),
                "X": X_arr,
                "y": y_arr,
            },
        )
    except Exception as exc:
        runtime_ms = (perf_counter() - start) * 1000.0
        return ModelFit(
            params=None,
            cov=None,
            loglike=None,
            converged=False,
            error_type=f"restricted_ols_failed:{type(exc).__name__}",
            runtime_ms=runtime_ms,
            extra={"exception": str(exc)},
        )


def fit_glm(
    X: np.ndarray,
    y: np.ndarray,
    family: sm.families.Family,
    maxiter: int = 200,
) -> ModelFit:
    start = perf_counter()
    try:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        model = sm.GLM(y_arr, X_arr, family=family)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            result = model.fit(maxiter=maxiter, disp=False)
        converged = bool(getattr(result, "converged", True))
        error_type = None if converged else "not_converged"
        cov = None
        try:
            cov_params = result.cov_params()
            cov = np.asarray(cov_params, dtype=float)
        except Exception:
            error_type = "cov_failed" if error_type is None else f"{error_type};cov_failed"
        runtime_ms = (perf_counter() - start) * 1000.0
        return ModelFit(
            params=np.asarray(result.params, dtype=float),
            cov=cov,
            loglike=float(result.llf),
            converged=converged,
            error_type=error_type,
            runtime_ms=runtime_ms,
            extra={"result": result, "X": X_arr, "y": y_arr, "model": model},
        )
    except PerfectSeparationWarning:
        runtime_ms = (perf_counter() - start) * 1000.0
        return ModelFit(
            params=None,
            cov=None,
            loglike=None,
            converged=False,
            error_type="perfect_separation_warning",
            runtime_ms=runtime_ms,
            extra={},
        )
    except PerfectSeparationError:
        runtime_ms = (perf_counter() - start) * 1000.0
        return ModelFit(
            params=None,
            cov=None,
            loglike=None,
            converged=False,
            error_type="perfect_separation",
            runtime_ms=runtime_ms,
            extra={},
        )
    except np.linalg.LinAlgError:
        runtime_ms = (perf_counter() - start) * 1000.0
        return ModelFit(
            params=None,
            cov=None,
            loglike=None,
            converged=False,
            error_type="linear_algebra_error",
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
            error_type=f"glm_fit_failed:{type(exc).__name__}",
            runtime_ms=runtime_ms,
            extra={"exception": str(exc)},
        )


def glm_score_bundle(
    y: np.ndarray,
    X_null: np.ndarray,
    X_test: np.ndarray,
    mu: np.ndarray,
    variance_weight: np.ndarray,
) -> ScoreBundle:
    try:
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        X0 = np.asarray(X_null, dtype=float)
        X1 = np.asarray(X_test, dtype=float)
        mu_arr = np.asarray(mu, dtype=float).reshape(-1)
        w = np.asarray(variance_weight, dtype=float).reshape(-1)
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            return ScoreBundle(error_type="invalid_weights")
        resid = y_arr - mu_arr
        u_test = X1.T @ resid
        wx0 = X0 * w[:, None]
        wx1 = X1 * w[:, None]
        i_tt = X1.T @ wx1
        i_tn = X1.T @ wx0
        i_nn = X0.T @ wx0
        i_nn_inv, inv_error = safe_inverse(i_nn)
        if inv_error:
            return ScoreBundle(error_type=f"score_nuisance_info:{inv_error}")
        info_eff = i_tt - i_tn @ i_nn_inv @ i_tn.T
        if not np.all(np.isfinite(info_eff)):
            return ScoreBundle(error_type="score_info_non_finite")
        return ScoreBundle(score=u_test.reshape(-1), information=info_eff, df=X1.shape[1], error_type=None)
    except Exception as exc:
        return ScoreBundle(error_type=f"score_bundle_failed:{type(exc).__name__}:{exc}")


def ols_score_bundle(
    y: np.ndarray,
    X_full: np.ndarray,
    tested_indices: list[int],
    null_params: np.ndarray,
    sigma2: float,
) -> ScoreBundle:
    try:
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        X = np.asarray(X_full, dtype=float)
        p = X.shape[1]
        tested = [int(i) for i in tested_indices]
        nuisance = [i for i in range(p) if i not in tested]
        X0 = X[:, nuisance]
        X1 = X[:, tested]
        beta0 = np.asarray(null_params, dtype=float).reshape(-1)
        fitted = X @ beta0
        resid = y_arr - fitted
        if not np.isfinite(sigma2) or sigma2 <= 0:
            return ScoreBundle(error_type="invalid_sigma2")
        score = (X1.T @ resid) / sigma2
        xtx0 = X0.T @ X0
        xtx0_inv, inv_error = safe_inverse(xtx0)
        if inv_error:
            return ScoreBundle(error_type=f"nuisance_xtx:{inv_error}")
        projection = X0 @ xtx0_inv @ X0.T
        m0 = np.eye(X.shape[0]) - projection
        info = (X1.T @ m0 @ X1) / sigma2
        return ScoreBundle(score=score, information=info, df=len(tested), error_type=None)
    except Exception as exc:
        return ScoreBundle(error_type=f"ols_score_bundle_failed:{type(exc).__name__}:{exc}")
