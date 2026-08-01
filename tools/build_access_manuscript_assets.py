from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs_confirmation_v6_10scenario"
FIGURES = ROOT / "ACCESS_latex_template_20260513" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def _save_dataset_plot(df: pd.DataFrame) -> None:
    method_order = [
        "online_mlp",
        "experience_replay",
        "online_ewc",
        "agem",
        "ssf",
        "repshield",
        "cdda_md",
        "modular_dasp",
    ]
    pivot = df.pivot_table(
        index="dataset", columns="method", values="mean_macro_f1", aggfunc="mean"
    ).reindex(columns=method_order)
    labels = [
        "CIC-IDS2017",
        "NF-BoT-IoT-v3",
        "NF-CSE-CIC-IDS2018-v3",
        "NF-ToN-IoT-v3",
        "NF-UNSW-NB15-v3",
    ]
    x = np.arange(len(pivot.index))
    width = 0.095
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = ["#9aa7b8"] * 7 + ["#c23b3b"]
    for index, method in enumerate(method_order):
        offset = (index - (len(method_order) - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[method].values,
            width,
            label=method.replace("_", " "),
            color=colors[index],
            edgecolor="white",
            linewidth=0.25,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Mean Macro-F1")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_macro_f1.pdf", bbox_inches="tight")
    plt.close(fig)


def _save_scenario_heatmap(df: pd.DataFrame) -> None:
    paired = df[df["method"].isin(["modular_dasp", "ssf"])]
    pivot = paired.pivot_table(
        index=["dataset", "scenario", "stream_seed", "model_seed"],
        columns="method",
        values="mean_macro_f1",
    )
    pivot["gap"] = pivot["modular_dasp"] - pivot["ssf"]
    table = pivot.reset_index().pivot_table(
        index="dataset", columns="scenario", values="gap", aggfunc="mean"
    )
    scenario_order = [
        "S0",
        "S1-Sudden",
        "S2-Gradual",
        "S3-Incremental",
        "S4-Recurrent",
        "S5-Blip",
        "S6-Combined",
        "S7-Local-Low",
        "S7-Local-Medium",
        "S7-Local-High",
    ]
    table = table.reindex(columns=scenario_order)
    labels = [
        "CIC",
        "BoT-IoT",
        "CSE-CIC",
        "ToN-IoT",
        "UNSW",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    vmax = max(abs(float(table.min().min())), abs(float(table.max().max())))
    image = ax.imshow(table.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(scenario_order)))
    ax.set_xticklabels(["S0", "S1", "S2", "S3", "S4", "S5", "S6", "L", "M", "H"], fontsize=8)
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table.iloc[row, col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=6.4)
    ax.set_xlabel("Scenario; L/M/H = local low/medium/high")
    ax.set_title("Macro-F1 gap: modular DASP minus SSF")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Macro-F1 gap")
    fig.tight_layout()
    fig.savefig(FIGURES / "scenario_gap_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def _save_frontier(df: pd.DataFrame) -> None:
    grouped = df.groupby("method").agg(
        macro_f1=("mean_macro_f1", "mean"),
        update_seconds=("total_update_seconds", "mean"),
        gradient_evaluations=("total_gradient_evaluations", "mean"),
    )
    names = list(grouped.index)
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    sizes = 40 + 0.15 * grouped["gradient_evaluations"].values
    colors = ["#c23b3b" if name == "modular_dasp" else "#68788d" for name in names]
    ax.scatter(
        grouped["update_seconds"],
        grouped["macro_f1"],
        s=sizes,
        c=colors,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.7,
    )
    for name, row in grouped.iterrows():
        ax.annotate(name.replace("_", " "), (row.update_seconds, row.macro_f1), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Mean update time per stream (s)")
    ax.set_ylabel("Mean Macro-F1")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_update_frontier.pdf", bbox_inches="tight")
    plt.close(fig)



def _resolve_windows_path(row: pd.Series) -> Path:
    source = str(row["summary_source"])
    if source != "computed":
        summary_path = Path(source)
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
    else:
        summary_path = (
            RESULTS
            / str(row["dataset"])
            / str(row["scenario"])
            / f"{row['method']}_stream{int(row['stream_seed'])}_model{int(row['model_seed'])}"
            / "summary.json"
        )
    windows_path = summary_path.parent / "windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(windows_path)
    return windows_path


def _save_window_trajectory_plots(df: pd.DataFrame) -> None:
    datasets = [
        "cic-ids2017",
        "nf-bot-iot-v3",
        "nf-cse-cic-ids2018-v3",
        "nf-ton-iot-v3",
        "nf-unsw-nb15-v3",
    ]
    dataset_labels = {
        "cic-ids2017": "CIC-IDS2017",
        "nf-bot-iot-v3": "NF-BoT-IoT-v3",
        "nf-cse-cic-ids2018-v3": "NF-CSE-CIC-IDS2018-v3",
        "nf-ton-iot-v3": "NF-ToN-IoT-v3",
        "nf-unsw-nb15-v3": "NF-UNSW-NB15-v3",
    }
    dataset_stems = {
        "cic-ids2017": "cic_ids2017",
        "nf-bot-iot-v3": "nf_bot_iot_v3",
        "nf-cse-cic-ids2018-v3": "nf_cse_cic_ids2018_v3",
        "nf-ton-iot-v3": "nf_ton_iot_v3",
        "nf-unsw-nb15-v3": "nf_unsw_nb15_v3",
    }
    scenarios = [
        "S0",
        "S1-Sudden",
        "S2-Gradual",
        "S3-Incremental",
        "S4-Recurrent",
        "S5-Blip",
        "S6-Combined",
        "S7-Local-Low",
        "S7-Local-Medium",
        "S7-Local-High",
    ]
    scenario_labels = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "L", "M", "H"]
    methods = [
        "modular_dasp",
        "ssf",
        "experience_replay",
        "repshield",
        "agem",
        "online_ewc",
        "online_mlp",
        "cdda_md",
    ]
    styles = {
        "modular_dasp": ("DASP", "#c62828", "-", 2.15),
        "ssf": ("SSF", "#1565c0", "--", 1.45),
        "experience_replay": ("Experience Replay", "#2e7d32", "-.", 1.15),
        "repshield": ("RepShield", "#ef6c00", (0, (4, 1, 1, 1)), 1.05),
        "agem": ("A-GEM", "#00838f", (0, (5, 2)), 1.05),
        "online_ewc": ("Online EWC", "#6a1b9a", ":", 1.05),
        "online_mlp": ("Online MLP", "#8d6e63", (0, (2, 1)), 1.00),
        "cdda_md": ("CDDA-MD", "#616161", (0, (1, 1)), 1.00),
    }

    selected = df[df["dataset"].isin(datasets) & df["method"].isin(methods)]
    frames: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        windows = pd.read_csv(
            _resolve_windows_path(row),
            usecols=["window", "macro_f1"],
        )
        windows = windows[windows["window"] >= 4].copy()
        windows["dataset"] = row["dataset"]
        windows["scenario"] = row["scenario"]
        windows["method"] = row["method"]
        windows["stream_seed"] = int(row["stream_seed"])
        windows["model_seed"] = int(row["model_seed"])
        frames.append(windows)
    window_data = pd.concat(frames, ignore_index=True)
    summary = (
        window_data.groupby(["dataset", "scenario", "method", "window"], as_index=False)
        .agg(mean=("macro_f1", "mean"), std=("macro_f1", "std"), n=("macro_f1", "size"))
    )
    summary["ci95"] = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["n"].clip(lower=1))

    window_count = 32
    gap = 2
    segment = window_count + gap
    centers = [index * segment + (window_count - 1) / 2 for index in range(len(scenarios))]
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(11.2, 4.65))
        legend_handles = []
        legend_labels = []
        for scenario_index, scenario in enumerate(scenarios):
            base = scenario_index * segment
            if scenario_index:
                ax.axvline(base - gap / 2, color="#b0b0b0", linewidth=0.55, alpha=0.65)
            if scenario != "S0":
                ax.axvline(base + 8, color="#8c8c8c", linewidth=0.55, linestyle=":", alpha=0.75)
            for method in methods:
                values = summary[
                    (summary["dataset"] == dataset)
                    & (summary["scenario"] == scenario)
                    & (summary["method"] == method)
                ].sort_values("window")
                x = base + values["window"].to_numpy(dtype=float) - 4.0
                y = values["mean"].to_numpy(dtype=float)
                label, color, linestyle, linewidth = styles[method]
                line, = ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label if scenario_index == 0 else None,
                    alpha=1.0 if method == "modular_dasp" else 0.88,
                    zorder=8 if method == "modular_dasp" else (6 if method == "ssf" else 4),
                )
                if scenario_index == 0:
                    legend_handles.append(line)
                    legend_labels.append(label)
                if method == "modular_dasp":
                    ci = values["ci95"].to_numpy(dtype=float)
                    ax.fill_between(
                        x,
                        np.clip(y - ci, 0.0, 1.0),
                        np.clip(y + ci, 0.0, 1.0),
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                        zorder=2,
                    )
        dataset_min = float(summary.loc[summary["dataset"] == dataset, "mean"].min())
        y_min = max(0.0, np.floor((dataset_min - 0.025) * 20.0) / 20.0)
        ax.set_title(
            f"Window-level Macro-F1 comparison on {dataset_labels[dataset]}",
            fontsize=11.5,
            fontweight="bold",
            pad=6,
        )
        ax.set_ylabel("Window Macro-F1", fontsize=10.0)
        ax.set_ylim(y_min, 1.015)
        ax.grid(axis="y", color="#bdbdbd", alpha=0.42, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8.9)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#777777")
        ax.set_xticks(centers)
        ax.set_xticklabels(scenario_labels, fontsize=9.0)
        ax.set_xlabel(
            "Independent scenario segments (32 post-warm-up windows per segment)",
            fontsize=9.8,
        )
        ax.text(
            0.995,
            0.025,
            "Dotted markers: first nonstationary window",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.8,
            color="#666666",
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            fontsize=9.1,
            bbox_to_anchor=(0.5, 1.025),
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        stem = f"window_macro_f1_{dataset_stems[dataset]}"
        for suffix, kwargs in {
            "pdf": {},
            "svg": {},
            "png": {"dpi": 600},
        }.items():
            fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
        plt.close(fig)

def _save_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 2.5))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    boxes = [
        (0.1, 1.25, 1.35, 0.55, "Current flow\nwindow $W_t$", "#e8f1fb"),
        (1.75, 1.25, 1.35, 0.55, "Prequential\nprediction", "#e8f1fb"),
        (3.4, 1.25, 1.35, 0.55, "Module\nrepresentations", "#e8f1fb"),
        (5.05, 1.25, 1.35, 0.55, "Drift\nattribution", "#fff0d8"),
        (6.7, 1.25, 1.35, 0.55, "Top-$k$\nrouter", "#fff0d8"),
        (8.35, 1.25, 1.35, 0.55, "Localized\nrepair", "#e8f5e9"),
        (10.0, 1.25, 1.35, 0.55, "Guard and\ncommit", "#e8f5e9"),
    ]
    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06", linewidth=1.0, edgecolor="#334e68", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)
    for left, right in zip(boxes[:-1], boxes[1:]):
        x1 = left[0] + left[2]
        x2 = right[0]
        ax.add_patch(FancyArrowPatch((x1 + 0.03, left[1] + left[3] / 2), (x2 - 0.03, right[1] + right[3] / 2), arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#334e68"))
    ax.add_patch(FancyBboxPatch((3.55, 0.25), 1.05, 0.45, boxstyle="round,pad=0.03,rounding_size=0.05", linewidth=1.0, edgecolor="#4c6b50", facecolor="#edf7ed"))
    ax.text(4.075, 0.475, "Replay memory\n512 balanced", ha="center", va="center", fontsize=7)
    ax.add_patch(FancyArrowPatch((4.075, 0.72), (7.35, 1.22), arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color="#4c6b50", connectionstyle="arc3,rad=0.15"))
    ax.text(0.1, 0.1, "Labels arrive after $\delta=1$ window; inactive modules remain frozen.", fontsize=7.5, color="#334e68")
    fig.tight_layout(pad=0.2)
    fig.savefig(FIGURES / "dasp_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_tables(df: pd.DataFrame) -> None:
    output = ROOT / "ACCESS_latex_template_20260513" / "generated_tables.tex"
    method = df.groupby("method").agg(
        macro=("mean_macro_f1", "mean"),
        update=("total_update_seconds", "mean"),
        gradients=("total_gradient_evaluations", "mean"),
        throughput=("prediction_decision_throughput_flows_per_second", "mean"),
    ).sort_values("macro", ascending=False)
    dataset = df.groupby("dataset").agg(
        macro=("mean_macro_f1", "mean"),
        update=("total_update_seconds", "mean"),
    )
    rows = []
    rows.append(r"\begin{table*}[t]")
    rows.append(r"\caption{Confirmation results over 4,000 runs. Values are means across five stream seeds and two model seeds for each dataset--scenario cell.}")
    rows.append(r"\label{tab:overall}")
    rows.append(r"\centering")
    rows.append(r"\begin{tabular}{lrrrr}")
    rows.append(r"\toprule")
    rows.append(r"Method & Macro-F1 & Update (s) & Grad. eval. & Throughput (flows/s)\\")
    rows.append(r"\midrule")
    for name, values in method.iterrows():
        label = name.replace("_", " ")
        rows.append(f"{label} & {values['macro']:.4f} & {values['update']:.3f} & {values['gradients']:.1f} & {values['throughput']:,.0f}\\\\")
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    rows.append(r"\begin{table}[t]")
    rows.append(r"\caption{Dataset-level mean Macro-F1 and update cost.}")
    rows.append(r"\label{tab:dataset}")
    rows.append(r"\centering")
    rows.append(r"\begin{tabular}{lrr}")
    rows.append(r"\toprule")
    rows.append(r"Dataset & Macro-F1 & Update (s)\\")
    rows.append(r"\midrule")
    for name, values in dataset.iterrows():
        rows.append(f"{name.replace('_', r'\_')} & {values['macro']:.4f} & {values['update']:.3f}\\\\")
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(RESULTS / "full_summary.csv")
    _save_dataset_plot(df)
    _save_scenario_heatmap(df)
    _save_frontier(df)
    _save_window_trajectory_plots(df)
    _save_architecture()
    _write_tables(df)
    print(FIGURES)


if __name__ == "__main__":
    main()
