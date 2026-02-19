from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ModelFit:
    params: np.ndarray | None = None
    cov: np.ndarray | None = None
    loglike: float | None = None
    converged: bool = False
    error_type: str | None = None
    runtime_ms: float = np.nan
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreBundle:
    score: np.ndarray | None = None
    information: np.ndarray | None = None
    df: int = 1
    error_type: str | None = None


@dataclass
class TestResult:
    test: str
    stat: float = np.nan
    pvalue: float = np.nan
    df: int = 1
    runtime_ms: float = np.nan
    converged_full: bool = False
    converged_null: bool = False
    error_type: str | None = None
    theta_hat: float = np.nan
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    ci_method: str | None = None

    def failed(self) -> bool:
        if self.error_type is not None:
            return True
        if not np.isfinite(self.stat):
            return True
        if not np.isfinite(self.pvalue):
            return True
        return False

    def to_record(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "stat": float(self.stat) if np.isfinite(self.stat) else np.nan,
            "pvalue": float(self.pvalue) if np.isfinite(self.pvalue) else np.nan,
            "df": int(self.df),
            "runtime_ms": float(self.runtime_ms) if np.isfinite(self.runtime_ms) else np.nan,
            "converged_full": bool(self.converged_full),
            "converged_null": bool(self.converged_null),
            "error_type": self.error_type,
            "theta_hat": float(self.theta_hat) if np.isfinite(self.theta_hat) else np.nan,
            "ci_lower": float(self.ci_lower) if np.isfinite(self.ci_lower) else np.nan,
            "ci_upper": float(self.ci_upper) if np.isfinite(self.ci_upper) else np.nan,
            "ci_method": self.ci_method,
            "failed": bool(self.failed()),
        }
