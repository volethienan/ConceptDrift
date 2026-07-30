from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_METHODS = [
    "static",
    "experience_replay",
    "ssf_budget",
    "dasp_v16_no_detector",
    "dasp_v16",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the bounded natural chronological-prefix track."
    )
    parser.add_argument("output_dir")
    args = parser.parse_args()
    root = Path(args.output_dir)
    summaries = pd.read_csv(root / "summary_by_run.csv")
    destination = root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)

    window_frames: list[pd.DataFrame] = []
    for _, row in summaries.iterrows():
        path = Path(row["run_dir"]) / "windows.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["dataset"] = row["dataset"]
        frame["method"] = row["method"]
        frame["model_seed"] = row["model_seed"]
        window_frames.append(frame)
    windows = pd.concat(window_frames, ignore_index=True)
    windows.to_csv(destination / "all_windows.csv", index=False)
    evaluated = windows[~windows["warmup"].astype(bool)]
    phases = (
        evaluated.groupby(
            ["dataset", "method", "model_seed", "phase"], as_index=False
        )
        .agg(
            macro_f1=("macro_f1", "mean"),
            attack_recall=("attack_recall", "mean"),
            attack_rate=("attack_rate", "mean"),
        )
    )
    phases.to_csv(destination / "phase_metrics.csv", index=False)
    aggregate = (
        summaries.groupby(["dataset", "method"], as_index=False)
        .agg(
            macro_f1=("mean_macro_f1", "mean"),
            attack_recall=("mean_attack_recall", "mean"),
            p95_prediction_seconds=(
                "p95_prediction_decision_seconds",
                "median",
            ),
            throughput=(
                "prediction_decision_throughput_flows_per_second",
                "median",
            ),
        )
    )
    aggregate.to_csv(destination / "aggregate.csv", index=False)

    methods = [
        method for method in PLOT_METHODS if method in set(windows["method"])
    ]
    datasets = list(dict.fromkeys(windows["dataset"].tolist()))
    fig, axes = plt.subplots(
        len(datasets), 1, figsize=(8.2, 2.25 * len(datasets)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, dataset in zip(axes, datasets):
        subset = evaluated[
            (evaluated["dataset"] == dataset)
            & (evaluated["method"].isin(methods))
        ]
        for method in methods:
            mean = (
                subset[subset["method"] == method]
                .groupby("window")["macro_f1"]
                .mean()
            )
            axis.plot(mean.index, mean.values, label=method, linewidth=1.3)
        axis.set_ylabel(f"{dataset}\nMacro-F1", fontsize=8)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Exact chronological-prefix window")
    axes[0].legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(
            destination / f"natural_macro_f1_curves.{suffix}", dpi=220
        )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    grouped = (
        summaries.groupby("method", as_index=False)
        .agg(
            macro_f1=("mean_macro_f1", "mean"),
            latency=("p95_prediction_decision_seconds", "median"),
        )
    )
    axis.scatter(grouped["latency"] * 1000, grouped["macro_f1"], s=42)
    for _, row in grouped.iterrows():
        axis.annotate(
            row["method"],
            (row["latency"] * 1000, row["macro_f1"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Median p95 prediction+decision latency (ms/window)")
    axis.set_ylabel("Mean Macro-F1 across five chronological prefixes")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(
            destination / f"natural_performance_latency.{suffix}", dpi=220
        )
    plt.close(fig)

    lines = [
        "% Generated automatically from the natural-prefix outputs.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Dataset & Method & Macro-F1 & Attack Recall & p95 ms \\\\",
        "\\midrule",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"{str(row['dataset']).replace('_', '\\_')} & "
            f"{str(row['method']).replace('_', '\\_')} & "
            f"{row['macro_f1']:.4f} & {row['attack_recall']:.4f} & "
            f"{1000 * row['p95_prediction_seconds']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (destination / "natural_results_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        "Natural-track analysis complete. Model seeds quantify initialization "
        "sensitivity only; they are not independent temporal replicates.",
        flush=True,
    )


if __name__ == "__main__":
    main()
