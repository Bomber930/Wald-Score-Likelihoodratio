from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..scenarios import Scenario
from .plots import add_test_label, generate_scenario_plots


def ensure_column(df: pd.DataFrame, name: str, default_value: float | str | None = np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default_value] * len(df), index=df.index)


def compute_test_summary(df: pd.DataFrame) -> pd.DataFrame:
    labeled = add_test_label(df)
    records: list[dict[str, object]] = []
    max_effect = float(labeled["effect_size"].max()) if not labeled.empty else 0.0
    grouped = labeled.groupby("test_label", dropna=False)
    for test_label, grp in grouped:
        null_grp = grp[grp["is_null"]]
        alt_grp = grp[grp["effect_size"] == max_effect]
        null_valid = null_grp[(~null_grp["failed"]) & np.isfinite(null_grp["pvalue"])]
        alt_valid = alt_grp[(~alt_grp["failed"]) & np.isfinite(alt_grp["pvalue"])]
        size_rate = float(null_valid["reject"].mean()) if not null_valid.empty else np.nan
        power_rate = float(alt_valid["reject"].mean()) if not alt_valid.empty else np.nan
        fail_rate = float(grp["failed"].mean()) if not grp.empty else np.nan
        runtime_vals = grp["runtime_ms"].to_numpy(dtype=float)
        runtime_vals = runtime_vals[np.isfinite(runtime_vals)]
        runtime_mean = float(np.mean(runtime_vals)) if runtime_vals.size > 0 else np.nan
        records.append(
            {
                "test_label": str(test_label),
                "size": size_rate,
                "power_at_max_effect": power_rate,
                "failure_rate": fail_rate,
                "mean_runtime_ms": runtime_mean,
            }
        )
    if len(records) == 0:
        return pd.DataFrame(columns=["test_label", "size", "power_at_max_effect", "failure_rate", "mean_runtime_ms"])
    return pd.DataFrame(records).sort_values("test_label")


def compute_ci_summary(df: pd.DataFrame) -> pd.DataFrame:
    labeled = add_test_label(df.copy())
    labeled["theta_true"] = ensure_column(labeled, "theta_true", np.nan)
    labeled["ci_lower"] = ensure_column(labeled, "ci_lower", np.nan)
    labeled["ci_upper"] = ensure_column(labeled, "ci_upper", np.nan)
    labeled["ci_method"] = ensure_column(labeled, "ci_method", None)

    records: list[dict[str, object]] = []
    grouped = labeled.groupby("test_label", dropna=False)
    for test_label, grp in grouped:
        df_vals = grp["df"].to_numpy(dtype=float)
        eligible = np.isfinite(df_vals) & (df_vals == 1.0)
        ci_low = grp["ci_lower"].to_numpy(dtype=float)
        ci_up = grp["ci_upper"].to_numpy(dtype=float)
        theta_true = grp["theta_true"].to_numpy(dtype=float)

        valid_ci = eligible & np.isfinite(ci_low) & np.isfinite(ci_up) & (ci_low <= ci_up)
        n_eligible = int(np.sum(eligible))
        n_valid = int(np.sum(valid_ci))
        ci_valid_rate = float(n_valid / n_eligible) if n_eligible > 0 else np.nan
        ci_invalid_rate = float(1.0 - ci_valid_rate) if np.isfinite(ci_valid_rate) else np.nan

        widths = ci_up[valid_ci] - ci_low[valid_ci]
        mean_width = float(np.mean(widths)) if widths.size > 0 else np.nan
        median_width = float(np.median(widths)) if widths.size > 0 else np.nan

        valid_with_truth = valid_ci & np.isfinite(theta_true)
        if np.any(valid_with_truth):
            covered = (ci_low <= theta_true) & (theta_true <= ci_up)
            left_miss = ci_up < theta_true
            right_miss = ci_low > theta_true
            coverage = float(np.mean(covered[valid_with_truth]))
            left_rate = float(np.mean(left_miss[valid_with_truth]))
            right_rate = float(np.mean(right_miss[valid_with_truth]))
            asym = float(right_rate - left_rate)
        else:
            coverage = np.nan
            left_rate = np.nan
            right_rate = np.nan
            asym = np.nan

        ci_method_series = grp["ci_method"].dropna()
        if n_eligible == 0:
            method_label = "unsupported_df!=1"
        elif ci_method_series.empty:
            method_label = "NA"
        else:
            method_label = str(ci_method_series.mode().iloc[0])

        records.append(
            {
                "test_label": str(test_label),
                "ci_method": method_label,
                "ci_valid_rate": ci_valid_rate,
                "ci_invalid_rate": ci_invalid_rate,
                "coverage": coverage,
                "left_miss_rate": left_rate,
                "right_miss_rate": right_rate,
                "asymmetry_right_minus_left": asym,
                "median_ci_width": median_width,
                "mean_ci_width": mean_width,
            }
        )

    if len(records) == 0:
        return pd.DataFrame(
            columns=[
                "test_label",
                "ci_method",
                "ci_valid_rate",
                "ci_invalid_rate",
                "coverage",
                "left_miss_rate",
                "right_miss_rate",
                "asymmetry_right_minus_left",
                "median_ci_width",
                "mean_ci_width",
            ]
        )
    return pd.DataFrame(records).sort_values("test_label")


def fmt_num(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def markdown_table_from_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "No summary rows."
    headers = df.columns.tolist()
    line_header = "| " + " | ".join(headers) + " |"
    line_sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = [str(row[h]) for h in headers]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([line_header, line_sep] + rows)


def scenario_interpretation(slug: str, summary_df: pd.DataFrame, alpha: float) -> str:
    if summary_df.empty:
        return "有効な検定結果が不足しているため、解釈を保留します。"
    if slug == "logistic_separation":
        return "分離により full MLE が不安定になり、Wald/LR は失敗率や歪みが増え、Score が相対的に残りやすい設計です。"
    if slug == "zip_boundary":
        return "境界帰無では通常のχ²近似が崩れるため、naive な p 値はサイズを外しやすく、bootstrap 補正の有効性を見るシナリオです。"
    if slug == "phacking_stepwise":
        return "同一データで変数選択と検定を行うと size が膨らみ、サンプルスプリットで緩和されることを確認するシナリオです。"
    if slug == "wald_runtime_batch":
        return "複数仮説を同一 full fit で処理できるため、Wald の実務上の速度優位を確認するシナリオです。"
    return "n の増加に伴い 3 検定が近づくか、どの条件で差が拡大するかを比較するシナリオです。"


def scenario_ci_expectation(slug: str, alpha: float) -> str:
    target = 1.0 - alpha
    if slug in {"normal_mean_known_sigma", "linear_regression_single_constraint"}:
        return (
            f"想定: 正則条件下では被覆率は概ね {target:.2f} に近づき、左右ミス率は近い値になります。"
            "不適切行動は少ない想定です。"
        )
    if slug == "linear_regression_multiple_constraints":
        return "想定: この章は df=2 の同時検定なので、単一パラメータの二側CIは対象外 (unsupported_df!=1) です。"
    if slug == "linear_regression_heteroskedastic":
        return (
            "想定: 等分散仮定の Wald/Score/LR は被覆率が崩れやすく、左右ミス率が偏る可能性があります。"
            "`wald_hc3` は被覆率改善が期待されます。"
        )
    if slug == "logistic_clustered":
        return (
            "想定: 独立仮定の推論は過小SEにより被覆率が低下しやすく、過剰棄却を起こしやすいです。"
            "`wald_cluster` は補正で改善が期待されます。"
        )
    if slug == "logistic_separation":
        return (
            "想定: 分離で MLE が発散し、CI の計算不能率が上昇します。"
            "有限区間が得られても片側に偏った失敗が起きやすいです。"
        )
    if slug == "logistic_small_sample":
        return (
            "想定: 小標本近似誤差で被覆率が名目からずれ、左右非対称なミス率が出る可能性があります。"
        )
    if slug == "logistic_local_alternatives":
        return (
            "想定: 局所対立では固定効果より情報増加が遅く、区間幅の収束が遅いです。"
            "有限標本では検定間の被覆差が残る可能性があります。"
        )
    if slug == "collinearity":
        return (
            "想定: 情報行列のほぼ特異性により区間幅の不安定化・失敗率上昇が起き、"
            "過大/過小被覆の混在が起こり得ます。"
        )
    if slug == "poisson_misspec":
        return (
            "想定: 分布ミススペックでモデルベースSEが不適切になり、被覆率低下や片側ミス偏りが起きやすいです。"
        )
    if slug == "zip_boundary":
        return (
            "想定: 境界問題で二側対称CIの仮定が壊れ、被覆率悪化や下限の不自然な挙動が起きます。"
            "bootstrap 系の補正が必要です。"
        )
    if slug == "invariance_demo":
        return (
            "想定: Wald はパラメータ化依存で被覆や左右ミス率が変化し得ます。"
            "同値仮説でも区間形状が一致しない点に注意が必要です。"
        )
    if slug == "phacking_stepwise":
        return (
            "想定: 同一データ選択では区間が過度に楽観的になり被覆率が低下します。"
            "split では改善する一方で幅は広がりやすいです。"
        )
    return "想定: 条件が悪化するほど被覆率低下・左右非対称化・計算不能増加が起きやすくなります。"


def scenario_ci_empirical(ci_summary: pd.DataFrame, alpha: float) -> str:
    if ci_summary.empty:
        return "実測: CI の集計対象がありません。"
    target = 1.0 - alpha
    valid_cov = ci_summary[np.isfinite(ci_summary["coverage"])].copy()
    valid_asym = ci_summary[np.isfinite(ci_summary["asymmetry_right_minus_left"])].copy()

    lines: list[str] = []
    lines.append(f"実測: 目標被覆率は {target:.3f} です。")

    if not valid_cov.empty:
        valid_cov["coverage_gap"] = valid_cov["coverage"] - target
        worst_cov = valid_cov.iloc[int(np.argmax(np.abs(valid_cov["coverage_gap"].to_numpy(dtype=float))))]
        lines.append(
            f"被覆率乖離最大: `{worst_cov['test_label']}` が {fmt_num(float(worst_cov['coverage']))} "
            f"(差 {fmt_num(float(worst_cov['coverage_gap']))})。"
        )

    if not valid_asym.empty:
        abs_asym = np.abs(valid_asym["asymmetry_right_minus_left"].to_numpy(dtype=float))
        idx = int(np.argmax(abs_asym))
        worst_asym = valid_asym.iloc[idx]
        lines.append(
            f"左右非対称最大: `{worst_asym['test_label']}` で right-left="
            f"{fmt_num(float(worst_asym['asymmetry_right_minus_left']))}。"
        )

    invalid = ci_summary[np.isfinite(ci_summary["ci_invalid_rate"])].copy()
    if not invalid.empty:
        worst_invalid = invalid.iloc[int(np.argmax(invalid["ci_invalid_rate"].to_numpy(dtype=float)))]
        lines.append(
            f"不適切行動(計算不能/無効CI)最大: `{worst_invalid['test_label']}` の無効率 "
            f"{fmt_num(float(worst_invalid['ci_invalid_rate']))}。"
        )

    return " ".join(lines)


def write_report(
    df: pd.DataFrame,
    scenarios: list[Scenario],
    output_dir: Path,
    alpha: float,
) -> Path:
    report_path = output_dir / "report.md"
    lines: list[str] = []
    lines.append("# Wald / Score / LR シミュレーションレポート")
    lines.append("")
    lines.append(f"- alpha: {alpha}")
    lines.append(f"- total rows: {len(df)}")
    lines.append("- CI方針: df=1 のとき、各検定のχ²近似を反転した区間を表示 (Score/LR は局所二次近似)。")
    lines.append("")

    for scenario in scenarios:
        sdf = df[df["scenario_slug"] == scenario.slug].copy()
        scenario_dir = output_dir / scenario.slug
        files = generate_scenario_plots(sdf, scenario_dir, alpha=alpha)

        lines.append(f"## {scenario.name}")
        lines.append("")
        lines.append(f"- slug: `{scenario.slug}`")
        lines.append(f"- notes: {scenario.notes()}")
        for hypothesis_id in scenario.hypothesis_ids():
            spec = scenario.null_spec(hypothesis_id)
            lines.append(f"- {hypothesis_id}: {spec.get('hypothesis', '')} (df={spec.get('df', 1)})")
        lines.append("")

        summary = compute_test_summary(sdf)
        if not summary.empty:
            summary_fmt = summary.copy()
            summary_fmt["size"] = summary_fmt["size"].map(fmt_num)
            summary_fmt["power_at_max_effect"] = summary_fmt["power_at_max_effect"].map(fmt_num)
            summary_fmt["failure_rate"] = summary_fmt["failure_rate"].map(fmt_num)
            summary_fmt["mean_runtime_ms"] = summary_fmt["mean_runtime_ms"].map(fmt_num)
            lines.append("### Summary")
            lines.append("")
            lines.append(markdown_table_from_df(summary_fmt))
            lines.append("")

        ci_summary = compute_ci_summary(sdf)
        ci_fmt = ci_summary.copy()
        if not ci_fmt.empty:
            for col in [
                "ci_valid_rate",
                "ci_invalid_rate",
                "coverage",
                "left_miss_rate",
                "right_miss_rate",
                "asymmetry_right_minus_left",
                "median_ci_width",
                "mean_ci_width",
            ]:
                ci_fmt[col] = ci_fmt[col].map(fmt_num)
            lines.append("### Inverted CI Summary")
            lines.append("")
            lines.append(markdown_table_from_df(ci_fmt))
            lines.append("")

        lines.append("### CI Interpretation")
        lines.append("")
        lines.append(scenario_ci_expectation(scenario.slug, alpha=alpha))
        lines.append("")
        lines.append(scenario_ci_empirical(ci_summary, alpha=alpha))
        lines.append("")

        lines.append("### Interpretation")
        lines.append("")
        lines.append(scenario_interpretation(scenario.slug, summary, alpha=alpha))
        lines.append("")

        lines.append("### Figures")
        lines.append("")
        for file_name in files:
            lines.append(f"- ![{file_name}]({scenario.slug}/{file_name})")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
