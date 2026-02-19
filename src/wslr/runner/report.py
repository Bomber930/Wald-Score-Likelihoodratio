from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..scenarios import Scenario
from .plots import add_test_label, generate_scenario_plots


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
        return "有効な推定結果が少なく、解釈に十分な統計量が得られませんでした。"
    top_fail = summary_df.sort_values("failure_rate", ascending=False).iloc[0]
    low_fail = summary_df.sort_values("failure_rate", ascending=True).iloc[0]
    if slug == "logistic_separation":
        return (
            f"分離設定では失敗率の高い検定が目立ち、特に `{top_fail['test_label']}` が不安定です。"
            f" 一方で `{low_fail['test_label']}` は比較的計算が成立しやすく、帰無点ベース検定の利点が出ます。"
        )
    if slug == "zip_boundary":
        return (
            "境界仮説では通常のχ²近似が当てはまりにくく、サイズがalphaからずれます。"
            " `lr_bootstrap` のサイズが他よりalphaに近ければ、校正の有効性が確認できます。"
        )
    if slug == "phacking_stepwise":
        return (
            "同一データ選択の系列（`*:same_data`）は、split系列（`*:split_data`）よりサイズが大きくなりやすく、"
            "選択バイアスによる偽陽性増加が見えます。"
        )
    if slug == "invariance_demo":
        return (
            "同値仮説 `beta=0` と `OR=1` でWaldの値がずれ得る一方、LRは同じfull/null比較のため相対的に不変です。"
        )
    if slug == "wald_runtime_batch":
        return (
            "多数仮説の同時処理では、Waldがfull推定再利用できるため平均runtimeで有利になりやすい設定です。"
        )
    if slug == "poisson_misspec":
        return (
            f"モデルミススペック下ではサイズがalpha={alpha:.2f}から体系的に外れ、3検定が同時に崩れることがあります。"
        )
    if slug == "collinearity":
        return "共線性で情報行列が不安定になり、失敗率上昇や推定分散の増大が確認できます。"
    return "n増加で3検定が近づくか、または設定由来の不安定性がどの検定に出るかを確認できます。"


def write_report(
    df: pd.DataFrame,
    scenarios: list[Scenario],
    output_dir: Path,
    alpha: float,
) -> Path:
    report_path = output_dir / "report.md"
    lines: list[str] = []
    lines.append("# Wald / Score / LR シミュレーション教材レポート")
    lines.append("")
    lines.append(f"- alpha: {alpha}")
    lines.append(f"- total rows: {len(df)}")
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
