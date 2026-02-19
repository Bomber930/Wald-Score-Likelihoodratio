from __future__ import annotations

from typing import Any

import numpy as np

from ..tests_core import run_lr_test, run_score_test, run_wald_test
from ..types import ModelFit, ScoreBundle
from ..utils import make_rng
from .base import Scenario
from .common import fit_ols, fit_ols_with_zero_constraints, ols_score_bundle


class PHackingScenario(Scenario):
    name = "Overfitting/p-hacking: stepwise then test"
    slug = "phacking_stepwise"
    default_n_list = [80, 160, 320]
    default_effect_list = [0.0]
    recommended_reps = 500

    def __init__(self, n_features: int = 20) -> None:
        self.n_features = int(n_features)

    def hypothesis_ids(self) -> list[str]:
        return ["same_data", "split_data"]

    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        rng = make_rng(seed)
        X = rng.normal(size=(n, self.n_features))
        y = rng.normal(size=n) + effect_size * X[:, 0]
        return {"X": X, "y": y, "seed": seed}

    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        raise NotImplementedError("PHackingScenario uses custom run_replication.")

    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        raise NotImplementedError("PHackingScenario uses custom run_replication.")

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
        raise NotImplementedError("PHackingScenario uses custom run_replication.")

    def null_spec(self, hypothesis_id: str = "main") -> dict[str, Any]:
        if hypothesis_id == "split_data":
            text = "H0: selected beta = 0 (selection on train, testing on test split)"
        else:
            text = "H0: selected beta = 0 (selection and testing on same data)"
        return {"hypothesis": text, "target": "selected_linear_slope", "df": 1}

    def notes(self) -> str:
        return "変数選択後に同データで検定すると偽陽性率が増える。探索/検定分離でサイズが回復する。"

    def select_feature_index(self, X: np.ndarray, y: np.ndarray) -> int:
        y_centered = y - np.mean(y)
        x_centered = X - np.mean(X, axis=0, keepdims=True)
        numerator = np.sum(x_centered * y_centered[:, None], axis=0)
        denom = np.sqrt(np.sum(x_centered * x_centered, axis=0) * np.sum(y_centered * y_centered))
        corr = np.divide(
            numerator,
            np.where(denom <= 1e-12, np.nan, denom),
            out=np.full_like(numerator, np.nan),
            where=denom > 1e-12,
        )
        abs_corr = np.abs(corr)
        if np.all(~np.isfinite(abs_corr)):
            return 0
        return int(np.nanargmax(abs_corr))

    def build_design(self, X: np.ndarray, feature_index: int) -> np.ndarray:
        x = X[:, feature_index]
        n = X.shape[0]
        return np.column_stack([np.ones(n), x])

    def run_single_dataset(self, X: np.ndarray, y: np.ndarray) -> tuple[ModelFit, ModelFit, ScoreBundle]:
        full_fit = fit_ols(X, y)
        null_fit = fit_ols_with_zero_constraints(X, y, [1])
        if null_fit.params is None:
            score_bundle = ScoreBundle(error_type="null_params_missing")
        else:
            sigma2 = float(null_fit.extra.get("sigma2", np.nan))
            score_bundle = ols_score_bundle(
                y=y,
                X_full=X,
                tested_indices=[1],
                null_params=null_fit.params,
                sigma2=sigma2,
            )
        return full_fit, null_fit, score_bundle

    def run_replication(
        self,
        seed: int,
        n: int,
        effect_size: float,
        rep: int,
        alpha: float = 0.05,
    ) -> list[dict[str, Any]]:
        data = self.generate(seed=seed, n=n, effect_size=effect_size)
        X = np.asarray(data["X"], dtype=float)
        y = np.asarray(data["y"], dtype=float).reshape(-1)
        rows: list[dict[str, Any]] = []

        idx_same = self.select_feature_index(X, y)
        X_same = self.build_design(X, idx_same)
        full_same, null_same, score_same = self.run_single_dataset(X_same, y)
        wald_same = run_wald_test(
            full_same,
            null_same,
            self.restriction(full_same.params if full_same.params is not None else np.array([0.0, 0.0])),
            self.restriction_jacobian(np.array([0.0, 0.0] if full_same.params is None else full_same.params)),
            df=1,
        )
        if np.isfinite(full_same.runtime_ms):
            wald_same.runtime_ms = float(wald_same.runtime_ms + full_same.runtime_ms)
        score_res_same = run_score_test(full_same, null_same, score_same)
        if np.isfinite(null_same.runtime_ms):
            score_res_same.runtime_ms = float(score_res_same.runtime_ms + null_same.runtime_ms)
        lr_same = run_lr_test(full_same, null_same, df=1)
        if np.isfinite(full_same.runtime_ms):
            lr_same.runtime_ms = float(lr_same.runtime_ms + full_same.runtime_ms)
        if np.isfinite(null_same.runtime_ms):
            lr_same.runtime_ms = float(lr_same.runtime_ms + null_same.runtime_ms)
        rows.append(self._result_to_row(wald_same, "same_data", n, effect_size, rep, seed, alpha))
        rows.append(self._result_to_row(score_res_same, "same_data", n, effect_size, rep, seed, alpha))
        rows.append(self._result_to_row(lr_same, "same_data", n, effect_size, rep, seed, alpha))

        rng = make_rng(seed + 777_777)
        perm = rng.permutation(n)
        half = n // 2
        idx_train = perm[:half]
        idx_test = perm[half:]
        X_train = X[idx_train, :]
        y_train = y[idx_train]
        X_test = X[idx_test, :]
        y_test = y[idx_test]
        idx_split = self.select_feature_index(X_train, y_train)
        X_split = self.build_design(X_test, idx_split)
        full_split, null_split, score_split = self.run_single_dataset(X_split, y_test)
        wald_split = run_wald_test(
            full_split,
            null_split,
            self.restriction(full_split.params if full_split.params is not None else np.array([0.0, 0.0])),
            self.restriction_jacobian(np.array([0.0, 0.0] if full_split.params is None else full_split.params)),
            df=1,
        )
        if np.isfinite(full_split.runtime_ms):
            wald_split.runtime_ms = float(wald_split.runtime_ms + full_split.runtime_ms)
        score_res_split = run_score_test(full_split, null_split, score_split)
        if np.isfinite(null_split.runtime_ms):
            score_res_split.runtime_ms = float(score_res_split.runtime_ms + null_split.runtime_ms)
        lr_split = run_lr_test(full_split, null_split, df=1)
        if np.isfinite(full_split.runtime_ms):
            lr_split.runtime_ms = float(lr_split.runtime_ms + full_split.runtime_ms)
        if np.isfinite(null_split.runtime_ms):
            lr_split.runtime_ms = float(lr_split.runtime_ms + null_split.runtime_ms)
        rows.append(self._result_to_row(wald_split, "split_data", n, effect_size, rep, seed, alpha))
        rows.append(self._result_to_row(score_res_split, "split_data", n, effect_size, rep, seed, alpha))
        rows.append(self._result_to_row(lr_split, "split_data", n, effect_size, rep, seed, alpha))
        return rows
