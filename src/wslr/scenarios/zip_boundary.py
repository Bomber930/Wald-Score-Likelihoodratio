from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ..types import ModelFit, ScoreBundle, TestResult
from ..utils import make_rng, safe_inverse
from .base import Scenario


def zip_loglike(y: np.ndarray, pi: float, lam: float) -> float:
    if not np.isfinite(pi) or not np.isfinite(lam):
        return -np.inf
    if pi < 0.0 or pi >= 1.0 or lam <= 0.0:
        return -np.inf
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    is_zero = y_arr == 0.0
    ll = np.zeros_like(y_arr, dtype=float)
    p0 = pi + (1.0 - pi) * np.exp(-lam)
    if p0 <= 0.0 or p0 >= 1.0 + 1e-12:
        return -np.inf
    ll[is_zero] = np.log(p0)
    y_pos = y_arr[~is_zero]
    if y_pos.size > 0:
        ll_pos = np.log1p(-pi) - lam + y_pos * np.log(lam) - gammaln(y_pos + 1.0)
        ll[~is_zero] = ll_pos
    if np.any(~np.isfinite(ll)):
        return -np.inf
    return float(np.sum(ll))


def zip_negloglike(theta: np.ndarray, y: np.ndarray) -> float:
    pi = float(theta[0])
    eta = float(theta[1])
    lam = float(np.exp(eta))
    ll = zip_loglike(y, pi=pi, lam=lam)
    if not np.isfinite(ll):
        return 1e12
    return float(-ll)


def numerical_hessian(theta: np.ndarray, y: np.ndarray, step: float = 1e-5) -> np.ndarray:
    p = theta.size
    hess = np.zeros((p, p), dtype=float)
    f0 = zip_negloglike(theta, y)
    for i in range(p):
        e_i = np.zeros(p, dtype=float)
        e_i[i] = step
        f_plus = zip_negloglike(theta + e_i, y)
        f_minus = zip_negloglike(theta - e_i, y)
        hess[i, i] = (f_plus - 2.0 * f0 + f_minus) / (step * step)
        for j in range(i + 1, p):
            e_j = np.zeros(p, dtype=float)
            e_j[j] = step
            f_pp = zip_negloglike(theta + e_i + e_j, y)
            f_pm = zip_negloglike(theta + e_i - e_j, y)
            f_mp = zip_negloglike(theta - e_i + e_j, y)
            f_mm = zip_negloglike(theta - e_i - e_j, y)
            val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step * step)
            hess[i, j] = val
            hess[j, i] = val
    return hess


class ZIPBoundaryScenario(Scenario):
    name = "Non-regular: ZIP boundary H0: pi=0"
    slug = "zip_boundary"
    default_n_list = [60, 120]
    default_effect_list = [0.0, 0.1, 0.2]
    recommended_reps = 80

    def __init__(self, bootstrap_B: int = 60) -> None:
        self.bootstrap_B = int(bootstrap_B)

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        lam = 1.4
        pi = float(np.clip(effect_size, 0.0, 0.4))
        z = rng.binomial(1, pi, size=n)
        y_pois = rng.poisson(lam=lam, size=n)
        y = np.where(z == 1, 0, y_pois).astype(float)
        return {"y": y, "seed": seed}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        start = perf_counter()
        y = np.asarray(data["y"], dtype=float).reshape(-1)
        lam_start = max(float(np.mean(y)), 1e-6)
        pi_start = float(np.clip(np.mean(y == 0.0) - np.exp(-lam_start), 1e-4, 0.8))
        x0 = np.array([pi_start, np.log(lam_start)], dtype=float)
        bounds = [(0.0, 1.0 - 1e-8), (-10.0, 10.0)]
        try:
            opt = minimize(zip_negloglike, x0=x0, args=(y,), method="L-BFGS-B", bounds=bounds)
            converged = bool(opt.success)
            if not converged:
                runtime_ms = (perf_counter() - start) * 1000.0
                return ModelFit(
                    params=None,
                    cov=None,
                    loglike=None,
                    converged=False,
                    error_type=f"zip_full_not_converged:{opt.message}",
                    runtime_ms=runtime_ms,
                    extra={},
                )
            theta_hat = np.asarray(opt.x, dtype=float)
            pi_hat = float(theta_hat[0])
            lam_hat = float(np.exp(theta_hat[1]))
            ll = zip_loglike(y, pi=pi_hat, lam=lam_hat)
            hess = numerical_hessian(theta_hat, y)
            cov, inv_error = safe_inverse(hess)
            err = inv_error
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=theta_hat,
                cov=cov,
                loglike=ll if np.isfinite(ll) else None,
                converged=converged,
                error_type=err,
                runtime_ms=runtime_ms,
                extra={"pi_hat": pi_hat, "lam_hat": lam_hat},
            )
        except Exception as exc:
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=None,
                cov=None,
                loglike=None,
                converged=False,
                error_type=f"zip_full_failed:{type(exc).__name__}",
                runtime_ms=runtime_ms,
                extra={"exception": str(exc)},
            )

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        start = perf_counter()
        try:
            y = np.asarray(data["y"], dtype=float).reshape(-1)
            n = y.size
            lam_hat = max(float(np.mean(y)), 1e-8)
            eta_hat = float(np.log(lam_hat))
            ll = float(np.sum(y * eta_hat - lam_hat - gammaln(y + 1.0)))
            cov = np.array([[np.nan, np.nan], [np.nan, 1.0 / (n * lam_hat)]], dtype=float)
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=np.array([0.0, eta_hat], dtype=float),
                cov=cov,
                loglike=ll,
                converged=True,
                error_type=None,
                runtime_ms=runtime_ms,
                extra={"lam_hat": lam_hat},
            )
        except Exception as exc:
            runtime_ms = (perf_counter() - start) * 1000.0
            return ModelFit(
                params=None,
                cov=None,
                loglike=None,
                converged=False,
                error_type=f"zip_null_failed:{type(exc).__name__}",
                runtime_ms=runtime_ms,
                extra={"exception": str(exc)},
            )

    def restriction(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.array([float(params[0])], dtype=float)

    def restriction_jacobian(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        return np.array([[1.0, 0.0]], dtype=float)

    def score_components(
        self,
        data: dict[str, Any],
        null_fit: ModelFit,
        hypothesis_id: str = "main",
    ) -> ScoreBundle:
        if null_fit.params is None:
            return ScoreBundle(error_type="null_params_missing")
        try:
            y = np.asarray(data["y"], dtype=float).reshape(-1)
            n = y.size
            lam = float(np.exp(null_fit.params[1]))
            is_zero = y == 0.0
            score_i = np.where(is_zero, np.exp(lam) - 1.0, -1.0)
            score = float(np.sum(score_i))
            i_pp = n * (np.exp(lam) - 1.0)
            i_peta = -n * lam
            i_etaeta = n * lam
            i_eff = i_pp - (i_peta * i_peta) / i_etaeta
            if not np.isfinite(i_eff) or i_eff <= 0:
                return ScoreBundle(error_type="invalid_effective_information")
            return ScoreBundle(
                score=np.array([score], dtype=float),
                information=np.array([[i_eff]], dtype=float),
                df=1,
                error_type=None,
            )
        except Exception as exc:
            return ScoreBundle(error_type=f"zip_score_failed:{type(exc).__name__}:{exc}")

    def null_spec(self, hypothesis_id: str = "main") -> dict[str, Any]:
        return {"hypothesis": "H0: pi = 0 (boundary)", "target": "zero_inflation_prob", "df": 1}

    def notes(self) -> str:
        return "境界仮説でχ²近似が壊れ、Wald/Score/LRすべてサイズが歪み得る。LRのparametric bootstrapで校正可能。"

    def extra_tests(
        self,
        data: dict[str, Any],
        full_fit: ModelFit,
        null_fit: ModelFit,
        hypothesis_id: str = "main",
        alpha: float = 0.05,
    ) -> list[TestResult]:
        start = perf_counter()
        if full_fit.loglike is None or null_fit.loglike is None:
            runtime_ms = (perf_counter() - start) * 1000.0
            failed = TestResult(
                test="lr_bootstrap",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="observed_lr_missing",
            )
            return [failed]
        lr_obs = 2.0 * (full_fit.loglike - null_fit.loglike)
        if lr_obs < 0 and lr_obs > -1e-10:
            lr_obs = 0.0
        if lr_obs < 0:
            runtime_ms = (perf_counter() - start) * 1000.0
            failed = TestResult(
                test="lr_bootstrap",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="negative_observed_lr",
            )
            return [failed]
        if null_fit.params is None:
            runtime_ms = (perf_counter() - start) * 1000.0
            failed = TestResult(
                test="lr_bootstrap",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="null_params_missing",
            )
            return [failed]
        lam0 = float(np.exp(null_fit.params[1]))
        y = np.asarray(data["y"], dtype=float).reshape(-1)
        n = y.size
        seed = int(data.get("seed", 0))
        rng = make_rng(seed + 333_333)
        lr_boot: list[float] = []
        for _ in range(self.bootstrap_B):
            y_b = rng.poisson(lam0, size=n).astype(float)
            data_b = {"y": y_b, "seed": seed}
            full_b = self.fit_full(data_b)
            null_b = self.fit_null(data_b)
            if full_b.loglike is None or null_b.loglike is None:
                continue
            lr_b = 2.0 * (full_b.loglike - null_b.loglike)
            if np.isfinite(lr_b) and lr_b >= 0.0:
                lr_boot.append(float(lr_b))
        runtime_ms = (perf_counter() - start) * 1000.0
        if len(lr_boot) < max(10, self.bootstrap_B // 5):
            failed = TestResult(
                test="lr_bootstrap",
                stat=np.nan,
                pvalue=np.nan,
                df=1,
                runtime_ms=runtime_ms,
                converged_full=full_fit.converged,
                converged_null=null_fit.converged,
                error_type="bootstrap_insufficient_valid_draws",
            )
            return [failed]
        lr_arr = np.asarray(lr_boot, dtype=float)
        p_boot = float((1.0 + np.sum(lr_arr >= lr_obs)) / (lr_arr.size + 1.0))
        result = TestResult(
            test="lr_bootstrap",
            stat=float(lr_obs),
            pvalue=p_boot,
            df=1,
            runtime_ms=runtime_ms,
            converged_full=full_fit.converged,
            converged_null=null_fit.converged,
            error_type=None,
        )
        return [result]
