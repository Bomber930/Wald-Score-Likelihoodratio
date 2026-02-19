from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wslr.runner.harness import RunnerConfig, run_simulation, select_scenarios
from wslr.runner.report import write_report
from wslr.scenarios import get_all_scenarios


def parse_int_list(text: str | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return None
    return [int(v.strip()) for v in text.split(",") if v.strip() != ""]


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None or text.strip() == "":
        return None
    return [float(v.strip()) for v in text.split(",") if v.strip() != ""]


def parse_slug_list(text: str | None) -> list[str] | None:
    if text is None or text.strip() == "":
        return None
    return [v.strip() for v in text.split(",") if v.strip() != ""]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Wald/Score/LR educational simulations.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to store csv/png/report.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--reps", type=int, default=None, help="Replications per grid cell (override scenario default).")
    parser.add_argument("--seed", type=int, default=20260219, help="Base seed for reproducibility.")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenario slugs.")
    parser.add_argument("--n-list", type=str, default=None, help="Override n grid. Example: 40,80,160")
    parser.add_argument("--effect-list", type=str, default=None, help="Override effect grid. Example: 0,0.2,0.5")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios and exit.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    all_scenarios = get_all_scenarios()
    if args.list_scenarios:
        for scenario in all_scenarios:
            print(f"{scenario.slug}\t{scenario.name}")
        return 0
    output_dir = Path(args.output_dir).resolve()
    scenario_slugs = parse_slug_list(args.scenarios)
    n_list = parse_int_list(args.n_list)
    effect_list = parse_float_list(args.effect_list)
    config = RunnerConfig(
        output_dir=output_dir,
        alpha=float(args.alpha),
        reps=args.reps,
        base_seed=int(args.seed),
        scenario_slugs=scenario_slugs,
        n_list=n_list,
        effect_list=effect_list,
    )
    df = run_simulation(config)
    selected_scenarios = select_scenarios(all_scenarios, scenario_slugs)
    report_path = write_report(df, selected_scenarios, output_dir=output_dir, alpha=float(args.alpha))
    print(f"Saved: {output_dir / 'results.csv'}")
    print(f"Saved: {report_path}")
    print(f"Rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
