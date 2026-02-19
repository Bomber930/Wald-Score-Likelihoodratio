from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..scenarios import Scenario, get_all_scenarios
from ..utils import make_rng


@dataclass
class RunnerConfig:
    output_dir: Path
    alpha: float = 0.05
    reps: int | None = None
    base_seed: int = 20260219
    scenario_slugs: list[str] | None = None
    n_list: list[int] | None = None
    effect_list: list[float] | None = None


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def select_scenarios(all_scenarios: Iterable[Scenario], slugs: list[str] | None) -> list[Scenario]:
    scenarios = list(all_scenarios)
    if slugs is None or len(slugs) == 0:
        return scenarios
    slug_set = {s.strip() for s in slugs}
    selected = [s for s in scenarios if s.slug in slug_set]
    return selected


def scenario_reps(config: RunnerConfig, scenario: Scenario) -> int:
    if config.reps is not None:
        return int(config.reps)
    return int(scenario.recommended_reps)


def scenario_n_list(config: RunnerConfig, scenario: Scenario) -> list[int]:
    if config.n_list is not None and len(config.n_list) > 0:
        return [int(x) for x in config.n_list]
    return [int(x) for x in scenario.default_n_list]


def scenario_effect_list(config: RunnerConfig, scenario: Scenario) -> list[float]:
    if config.effect_list is not None and len(config.effect_list) > 0:
        return [float(x) for x in config.effect_list]
    return [float(x) for x in scenario.default_effect_list]


def make_runner_failure_row(
    scenario: Scenario,
    hypothesis_id: str,
    n: int,
    effect_size: float,
    rep: int,
    seed: int,
    alpha: float,
    exc: Exception,
) -> dict[str, object]:
    return {
        "test": "runner_failure",
        "stat": np.nan,
        "pvalue": np.nan,
        "df": np.nan,
        "runtime_ms": np.nan,
        "converged_full": False,
        "converged_null": False,
        "error_type": f"runner_failed:{type(exc).__name__}:{exc}",
        "theta_hat": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "ci_method": None,
        "failed": True,
        "reject": False,
        "scenario": scenario.name,
        "scenario_slug": scenario.slug,
        "hypothesis_id": hypothesis_id,
        "n": int(n),
        "effect_size": float(effect_size),
        "rep": int(rep),
        "seed": int(seed),
        "alpha": float(alpha),
        "null_spec": "",
        "null_df": np.nan,
        "theta_true": np.nan,
        "notes": scenario.notes(),
    }


def run_simulation(config: RunnerConfig) -> pd.DataFrame:
    ensure_output_dir(config.output_dir)
    scenario_list = select_scenarios(get_all_scenarios(), config.scenario_slugs)
    seed_rng = make_rng(config.base_seed)
    rows: list[dict[str, object]] = []
    for scenario in scenario_list:
        reps = scenario_reps(config, scenario)
        n_list = scenario_n_list(config, scenario)
        effect_list = scenario_effect_list(config, scenario)
        for n in n_list:
            for effect_size in effect_list:
                for rep in range(reps):
                    seed = int(seed_rng.integers(0, 2**63 - 1))
                    try:
                        out_rows = scenario.run_replication(
                            seed=seed,
                            n=n,
                            effect_size=effect_size,
                            rep=rep,
                            alpha=config.alpha,
                        )
                        rows.extend(out_rows)
                    except Exception as exc:
                        for hypothesis_id in scenario.hypothesis_ids():
                            rows.append(
                                make_runner_failure_row(
                                    scenario=scenario,
                                    hypothesis_id=hypothesis_id,
                                    n=n,
                                    effect_size=effect_size,
                                    rep=rep,
                                    seed=seed,
                                    alpha=config.alpha,
                                    exc=exc,
                                )
                            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["is_null"] = np.isclose(df["effect_size"], 0.0)
    csv_path = config.output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    return df
