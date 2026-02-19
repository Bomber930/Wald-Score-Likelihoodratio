from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..tests_core import run_lr_test, run_score_test, run_wald_test
from ..types import ModelFit, ScoreBundle, TestResult


class Scenario(ABC):
    name: str = "base"
    slug: str = "base"
    default_n_list: list[int] = [50, 100, 200]
    default_effect_list: list[float] = [0.0, 0.2, 0.5]
    recommended_reps: int = 200

    def hypothesis_ids(self) -> list[str]:
        return ["main"]

    def test_suite(self, hypothesis_id: str) -> list[str]:
        return ["wald", "score", "lr"]

    @abstractmethod
    def generate(self, seed: int, n: int, effect_size: float) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def fit_full(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        raise NotImplementedError

    @abstractmethod
    def fit_null(self, data: dict[str, Any], hypothesis_id: str = "main") -> ModelFit:
        raise NotImplementedError

    @abstractmethod
    def restriction(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def restriction_jacobian(self, params: np.ndarray, hypothesis_id: str = "main") -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def score_components(
        self,
        data: dict[str, Any],
        null_fit: ModelFit,
        hypothesis_id: str = "main",
    ) -> ScoreBundle:
        raise NotImplementedError

    @abstractmethod
    def null_spec(self, hypothesis_id: str = "main") -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def notes(self) -> str:
        raise NotImplementedError

    def extra_tests(
        self,
        data: dict[str, Any],
        full_fit: ModelFit,
        null_fit: ModelFit,
        hypothesis_id: str = "main",
    ) -> list[TestResult]:
        return []

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
        for hypothesis_id in self.hypothesis_ids():
            full_fit = self.fit_full(data, hypothesis_id=hypothesis_id)
            null_fit = self.fit_null(data, hypothesis_id=hypothesis_id)
            df = int(self.null_spec(hypothesis_id).get("df", 1))
            test_names = self.test_suite(hypothesis_id)
            if "wald" in test_names:
                restriction = None
                jacobian = None
                if full_fit.params is not None:
                    restriction = self.restriction(full_fit.params, hypothesis_id=hypothesis_id)
                    jacobian = self.restriction_jacobian(full_fit.params, hypothesis_id=hypothesis_id)
                wald = run_wald_test(full_fit, null_fit, restriction, jacobian, df=df)
                if np.isfinite(full_fit.runtime_ms):
                    wald.runtime_ms = float(wald.runtime_ms + full_fit.runtime_ms)
                rows.append(
                    self._result_to_row(
                        result=wald,
                        hypothesis_id=hypothesis_id,
                        n=n,
                        effect_size=effect_size,
                        rep=rep,
                        seed=seed,
                        alpha=alpha,
                    )
                )
            if "score" in test_names:
                bundle = self.score_components(data, null_fit, hypothesis_id=hypothesis_id)
                score = run_score_test(full_fit, null_fit, bundle)
                if np.isfinite(null_fit.runtime_ms):
                    score.runtime_ms = float(score.runtime_ms + null_fit.runtime_ms)
                rows.append(
                    self._result_to_row(
                        result=score,
                        hypothesis_id=hypothesis_id,
                        n=n,
                        effect_size=effect_size,
                        rep=rep,
                        seed=seed,
                        alpha=alpha,
                    )
                )
            if "lr" in test_names:
                lr = run_lr_test(full_fit, null_fit, df=df)
                if np.isfinite(full_fit.runtime_ms):
                    lr.runtime_ms = float(lr.runtime_ms + full_fit.runtime_ms)
                if np.isfinite(null_fit.runtime_ms):
                    lr.runtime_ms = float(lr.runtime_ms + null_fit.runtime_ms)
                rows.append(
                    self._result_to_row(
                        result=lr,
                        hypothesis_id=hypothesis_id,
                        n=n,
                        effect_size=effect_size,
                        rep=rep,
                        seed=seed,
                        alpha=alpha,
                    )
                )
            extra = self.extra_tests(data, full_fit, null_fit, hypothesis_id=hypothesis_id)
            for item in extra:
                rows.append(
                    self._result_to_row(
                        result=item,
                        hypothesis_id=hypothesis_id,
                        n=n,
                        effect_size=effect_size,
                        rep=rep,
                        seed=seed,
                        alpha=alpha,
                    )
                )
        return rows

    def _result_to_row(
        self,
        result: TestResult,
        hypothesis_id: str,
        n: int,
        effect_size: float,
        rep: int,
        seed: int,
        alpha: float,
    ) -> dict[str, Any]:
        record = result.to_record()
        pvalue = record["pvalue"]
        reject = bool(np.isfinite(pvalue) and pvalue < alpha)
        record["reject"] = reject
        record["scenario"] = self.name
        record["scenario_slug"] = self.slug
        record["hypothesis_id"] = hypothesis_id
        record["n"] = int(n)
        record["effect_size"] = float(effect_size)
        record["rep"] = int(rep)
        record["seed"] = int(seed)
        record["alpha"] = float(alpha)
        record["null_spec"] = self.null_spec(hypothesis_id).get("hypothesis", "")
        record["null_df"] = int(self.null_spec(hypothesis_id).get("df", 1))
        record["notes"] = self.notes()
        return record
