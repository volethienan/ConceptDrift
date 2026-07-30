from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs_dasp_mock_benchmark")
METHODS = [
    "static",
    "periodic_ft",
    "online_mlp",
    "experience_replay",
    "online_ewc",
    "agem",
    "ssf_budget",
    "repshield",
    "cdda_md",
    "dasp",
]
DISPLAY = {
    "static": "Frozen",
    "periodic_ft": "Periodic FT",
    "online_mlp": "Online MLP",
    "experience_replay": "Experience Replay",
    "online_ewc": "Online EWC",
    "agem": "A-GEM",
    "ssf_budget": "SSF-budget",
    "repshield": "RepShield",
    "cdda_md": "CDDA-MD",
    "dasp": "DASP",
}


def transition_starts(frame: pd.DataFrame) -> list[int]:
    phases = frame.sort_values("window")[["window", "phase"]]
    previous = phases["phase"].shift(1)
    return phases.loc[phases["phase"] != previous, "window"].astype(int).tolist()[1:]


def bootstrap_ci(values: np.ndarray, seed: int = 18001) -> tuple[float, float]:
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    summary = pd.read_csv(ROOT / "summary_by_run.csv")
    episode_rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        windows = pd.read_csv(Path(row.run_dir) / "windows.csv")
        starts = transition_starts(windows)
        if row.scenario == "S0":
            starts = []
        for episode, start in enumerate(starts, start=1):
            early = windows[
                (windows["window"] >= start) & (windows["window"] < start + 4)
            ]
            episode_rows.append(
                {
                    "scenario": row.scenario,
                    "stream_seed": row.stream_seed,
                    "model_seed": row.model_seed,
                    "method": row.method,
                    "episode": episode,
                    "transition_window": start,
                    "early_macro_f1": early["macro_f1"].mean(),
                    "early_attack_recall": early["attack_recall"].mean(),
                }
            )
    episodes = pd.DataFrame(episode_rows)
    episodes.to_csv(ROOT / "early_episodes.csv", index=False)

    first = episodes[episodes["episode"] == 1].copy()
    cell_key = ["scenario", "stream_seed", "model_seed"]
    macro_pivot = first.pivot(index=cell_key, columns="method", values="early_macro_f1")
    recall_pivot = first.pivot(
        index=cell_key, columns="method", values="early_attack_recall"
    )
    ranks = macro_pivot[METHODS].rank(axis=1, ascending=False, method="average")
    method_summary = pd.DataFrame(
        {
            "method": METHODS,
            "display": [DISPLAY[value] for value in METHODS],
            "mean_early_macro_f1": [
                macro_pivot[value].mean() for value in METHODS
            ],
            "mean_early_attack_recall": [
                recall_pivot[value].mean() for value in METHODS
            ],
            "mean_rank": [ranks[value].mean() for value in METHODS],
            "mean_full_macro_f1": [
                summary.loc[summary["method"] == value, "mean_macro_f1"].mean()
                for value in METHODS
            ],
            "median_prediction_decision_ms": [
                1000
                * summary.loc[
                    summary["method"] == value,
                    "median_prediction_decision_seconds",
                ].mean()
                for value in METHODS
            ],
            "mean_peak_memory_mb": [
                summary.loc[summary["method"] == value, "peak_rss_bytes"].mean()
                / (1024**2)
                for value in METHODS
            ],
        }
    ).sort_values("mean_rank")
    method_summary.to_csv(ROOT / "method_summary.csv", index=False)

    paired_rows: list[dict[str, object]] = []
    for comparator in METHODS:
        if comparator == "dasp":
            continue
        macro_delta = (macro_pivot["dasp"] - macro_pivot[comparator]).to_numpy()
        recall_delta = (recall_pivot["dasp"] - recall_pivot[comparator]).to_numpy()
        macro_ci = bootstrap_ci(macro_delta)
        recall_ci = bootstrap_ci(recall_delta, seed=18002)
        paired_rows.append(
            {
                "comparator": comparator,
                "display": DISPLAY[comparator],
                "n_cells": len(macro_delta),
                "mean_delta_early_macro_f1": macro_delta.mean(),
                "macro_ci_low": macro_ci[0],
                "macro_ci_high": macro_ci[1],
                "macro_wins": int((macro_delta > 0).sum()),
                "mean_delta_early_attack_recall": recall_delta.mean(),
                "recall_ci_low": recall_ci[0],
                "recall_ci_high": recall_ci[1],
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(ROOT / "paired_comparisons.csv", index=False)

    ssf = paired[paired["comparator"] == "ssf_budget"].iloc[0]
    dasp_summary = summary[summary["method"] == "dasp"]
    criteria = {
        "first_drift_cells": int(len(macro_pivot)),
        "mean_delta_early_macro_f1_vs_ssf": float(
            ssf["mean_delta_early_macro_f1"]
        ),
        "mean_delta_early_attack_recall_vs_ssf": float(
            ssf["mean_delta_early_attack_recall"]
        ),
        "macro_wins_vs_ssf": int(ssf["macro_wins"]),
        "mean_early_macro_f1_rank": float(ranks["dasp"].mean()),
        "global_update_count": int(dasp_summary["global_update_count"].sum()),
        "maximum_labels_per_window": int(
            max(
                pd.read_csv(Path(row.run_dir) / "windows.csv")[
                    "ground_truth_update_labels"
                ].max()
                for row in dasp_summary.itertuples(index=False)
            )
        ),
    }
    criteria["advances"] = bool(
        criteria["mean_delta_early_macro_f1_vs_ssf"] > 0
        and criteria["mean_delta_early_attack_recall_vs_ssf"] >= -0.01
        and criteria["macro_wins_vs_ssf"] >= 6
        and criteria["mean_early_macro_f1_rank"] <= 3.0
        and criteria["global_update_count"] == 0
        and criteria["maximum_labels_per_window"] <= 64
    )
    (ROOT / "advancement_criteria.json").write_text(
        json.dumps(criteria, indent=2), encoding="utf-8"
    )

    plot = method_summary.sort_values("mean_early_macro_f1")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#1479B8" if value == "DASP" else "#9AA4AD" for value in plot["display"]]
    ax.barh(plot["display"], plot["mean_early_macro_f1"], color=colors)
    ax.set_xlabel("Mean early Macro-F1 (first four post-transition windows)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "mock_early_macro_f1.png", dpi=220)
    plt.close(fig)
    print(json.dumps(criteria, indent=2))


if __name__ == "__main__":
    main()
