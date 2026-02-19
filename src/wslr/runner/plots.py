from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_test_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n_hyp = out["hypothesis_id"].nunique(dropna=True)
    if n_hyp <= 1:
        out["test_label"] = out["test"]
        return out
    if n_hyp <= 3:
        out["test_label"] = out["test"] + ":" + out["hypothesis_id"]
        return out
    out["test_label"] = out["test"]
    return out


def save_no_data_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pvalue_histogram(df: pd.DataFrame, out_path: Path) -> None:
    title = "Null p-value histograms by test"
    null_df = df[df["is_null"]].copy()
    mask_valid = (~null_df["failed"]) & np.isfinite(null_df["pvalue"])
    null_df = null_df[mask_valid]
    if null_df.empty:
        save_no_data_figure(out_path, title, "No valid null p-values.")
        return
    test_labels = sorted(null_df["test_label"].unique().tolist())
    n_tests = len(test_labels)
    ncols = min(3, n_tests)
    nrows = int(np.ceil(n_tests / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(-1)
    bins = np.linspace(0.0, 1.0, 21)
    for idx, test_label in enumerate(test_labels):
        ax = axes_arr[idx]
        vals = null_df.loc[null_df["test_label"] == test_label, "pvalue"].to_numpy(dtype=float)
        ax.hist(vals, bins=bins, alpha=0.8, color="#4472C4", edgecolor="black")
        ax.set_title(f"{test_label} (n={vals.size})")
        ax.set_xlabel("p-value")
        ax.set_ylabel("count")
        ax.set_xlim(0.0, 1.0)
    for idx in range(n_tests, axes_arr.size):
        axes_arr[idx].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def aggregate_size_or_power(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str | int]] = []
    grouped = df.groupby(["test_label", "n", "effect_size"], dropna=False)
    for (test_label, n, effect_size), grp in grouped:
        valid = (~grp["failed"]) & np.isfinite(grp["pvalue"])
        n_valid = int(valid.sum())
        if n_valid > 0:
            reject_rate = float(grp.loc[valid, "reject"].mean())
        else:
            reject_rate = np.nan
        records.append(
            {
                "test_label": str(test_label),
                "n": int(n),
                "effect_size": float(effect_size),
                "n_valid": n_valid,
                "reject_rate": reject_rate,
            }
        )
    return pd.DataFrame(records)


def plot_size_vs_n(df: pd.DataFrame, out_path: Path, alpha: float) -> None:
    title = "Empirical size (Type I error) vs n"
    null_df = df[df["is_null"]].copy()
    if null_df.empty:
        save_no_data_figure(out_path, title, "No null samples.")
        return
    agg = aggregate_size_or_power(null_df)
    if agg.empty:
        save_no_data_figure(out_path, title, "No valid aggregates.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for test_label in sorted(agg["test_label"].unique().tolist()):
        sub = agg[agg["test_label"] == test_label].sort_values("n")
        ax.plot(sub["n"], sub["reject_rate"], marker="o", label=test_label)
    ax.axhline(alpha, color="black", linestyle="--", linewidth=1.0, label=f"alpha={alpha:.2f}")
    ax.set_xlabel("n")
    ax.set_ylabel("reject rate under H0")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_power_curve(df: pd.DataFrame, out_path: Path) -> None:
    title = "Power curves by effect size"
    alt_df = df[~df["is_null"]].copy()
    if alt_df.empty:
        save_no_data_figure(out_path, title, "No non-null samples.")
        return
    agg = aggregate_size_or_power(alt_df)
    if agg.empty:
        save_no_data_figure(out_path, title, "No valid aggregates.")
        return
    n_values = sorted(agg["n"].unique().tolist())
    n_panels = len(n_values)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.7 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(-1)
    test_labels = sorted(agg["test_label"].unique().tolist())
    for idx, n in enumerate(n_values):
        ax = axes_arr[idx]
        sub_n = agg[agg["n"] == n]
        for test_label in test_labels:
            sub = sub_n[sub_n["test_label"] == test_label].sort_values("effect_size")
            ax.plot(sub["effect_size"], sub["reject_rate"], marker="o", label=test_label)
        ax.set_title(f"n={n}")
        ax.set_xlabel("effect size")
        ax.set_ylabel("reject rate")
        ax.set_ylim(0.0, 1.0)
    for idx in range(n_panels, axes_arr.size):
        axes_arr[idx].axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_failure_rate(df: pd.DataFrame, out_path: Path) -> None:
    title = "Failure rate vs n"
    records: list[dict[str, float | str | int]] = []
    grouped = df.groupby(["test_label", "n"], dropna=False)
    for (test_label, n), grp in grouped:
        failure_rate = float(grp["failed"].mean())
        records.append({"test_label": str(test_label), "n": int(n), "failure_rate": failure_rate})
    agg = pd.DataFrame(records)
    if agg.empty:
        save_no_data_figure(out_path, title, "No data.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for test_label in sorted(agg["test_label"].unique().tolist()):
        sub = agg[agg["test_label"] == test_label].sort_values("n")
        ax.plot(sub["n"], sub["failure_rate"], marker="o", label=test_label)
    ax.set_xlabel("n")
    ax.set_ylabel("failure rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_runtime(df: pd.DataFrame, out_path: Path) -> None:
    title = "Runtime comparison vs n"
    records: list[dict[str, float | str | int]] = []
    grouped = df.groupby(["test_label", "n"], dropna=False)
    for (test_label, n), grp in grouped:
        vals = grp["runtime_ms"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean_ms = np.nan
        else:
            mean_ms = float(np.mean(vals))
        records.append({"test_label": str(test_label), "n": int(n), "runtime_ms": mean_ms})
    agg = pd.DataFrame(records)
    if agg.empty:
        save_no_data_figure(out_path, title, "No runtime data.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for test_label in sorted(agg["test_label"].unique().tolist()):
        sub = agg[agg["test_label"] == test_label].sort_values("n")
        ax.plot(sub["n"], sub["runtime_ms"], marker="o", label=test_label)
    ax.set_xlabel("n")
    ax.set_ylabel("mean runtime (ms)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_scenario_plots(df: pd.DataFrame, scenario_dir: Path, alpha: float) -> list[str]:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    labeled_df = add_test_label(df)
    files: list[str] = []
    p_hist = scenario_dir / "pvalue_hist.png"
    plot_pvalue_histogram(labeled_df, p_hist)
    files.append(p_hist.name)
    p_size = scenario_dir / "size_vs_n.png"
    plot_size_vs_n(labeled_df, p_size, alpha=alpha)
    files.append(p_size.name)
    p_power = scenario_dir / "power_curves.png"
    plot_power_curve(labeled_df, p_power)
    files.append(p_power.name)
    p_fail = scenario_dir / "failure_rate_vs_n.png"
    plot_failure_rate(labeled_df, p_fail)
    files.append(p_fail.name)
    p_runtime = scenario_dir / "runtime_vs_n.png"
    plot_runtime(labeled_df, p_runtime)
    files.append(p_runtime.name)
    return files
