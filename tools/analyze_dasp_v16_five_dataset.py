from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPISODES = {
    "S1": [("first", 12, 35)],
    "S2": [("gradual", 12, 35)],
    "S3": [("first", 12, 23), ("recurrence", 24, 35)],
    "S4": [("local", 12, 35)],
    "S5": [("severe", 12, 35)],
    "S1-Sudden": [("sudden", 12, 35)],
    "S2-Gradual": [("gradual", 12, 35)],
    "S3-Incremental": [("incremental", 12, 35)],
    "S4-Recurrent": [
        ("departure", 12, 23),
        ("recurrence", 24, 35),
    ],
    "S5-Blip": [("blip_and_return", 12, 16)],
    "S6-Combined": [
        ("sudden_and_incremental", 12, 23),
        ("C_and_recurrence", 24, 35),
    ],
    "S7-Local-Low": [("local_low", 12, 35)],
    "S7-Local-Medium": [("local_medium", 12, 35)],
    "S7-Local-High": [("local_high", 12, 35)],
    "S8-Stress": [("severe", 12, 35)],
}
DASP_METHOD = "dasp"
PLOT_METHODS = [
    "online_mlp",
    "experience_replay",
    "online_ewc",
    "agem",
    "ssf_budget",
    "repshield",
    "cdda_md",
    DASP_METHOD,
]


def _run_dir(root: Path, row: pd.Series) -> Path:
    if "run_dir" in row and isinstance(row["run_dir"], str):
        candidate = Path(row["run_dir"])
        if candidate.exists():
            return candidate
    return (
        root
        / str(row["dataset"])
        / str(row["scenario"])
        / (
            f"{row['method']}_stream{int(row['stream_seed'])}"
            f"_model{int(row['model_seed'])}"
        )
    )


def _episode_metrics(
    windows: pd.DataFrame,
    onset: int,
    end: int,
) -> dict[str, float | int | bool]:
    early = windows[
        windows["window"].between(onset, min(onset + 4, end))
    ]
    if len(early) != 5:
        raise ValueError(f"Expected five early windows at onset={onset}")
    reference = windows[windows["window"].between(7, 11)]["macro_f1"]
    if len(reference) != 5:
        raise ValueError("Expected reference windows W7-W11")
    reference_f1 = float(reference.median())
    recovery_floor = 0.90 * reference_f1
    post = windows[windows["window"].between(onset, end)].sort_values("window")
    recovered_at: int | None = None
    values = post["macro_f1"].to_numpy(dtype=float)
    indexes = post["window"].to_numpy(dtype=int)
    for index in range(max(0, len(values) - 1)):
        if values[index] >= recovery_floor and values[index + 1] >= recovery_floor:
            recovered_at = int(indexes[index])
            break
    return {
        "early_macro_f1": float(early["macro_f1"].mean()),
        "early_attack_recall": float(early["attack_recall"].mean()),
        "minimum_post_drift_macro_f1": float(post["macro_f1"].min()),
        "reference_macro_f1": reference_f1,
        "early_performance_drop_macro_f1": float(
            reference_f1 - early["macro_f1"].mean()
        ),
        "maximum_performance_drop_macro_f1": float(
            reference_f1 - post["macro_f1"].min()
        ),
        "onset_performance_drop_macro_f1": float(
            reference_f1 - float(post.iloc[0]["macro_f1"])
        ),
        "recovery_floor": recovery_floor,
        "recovery_time": (
            int(recovered_at - onset) if recovered_at is not None else np.nan
        ),
        "recovery_censored": recovered_at is None,
        "recovery_deficit_at_5": float(
            np.maximum(
                0.0, recovery_floor - early["macro_f1"].to_numpy(dtype=float)
            ).sum()
        ),
    }


def _matched_detector_metrics(
    windows: pd.DataFrame, scenario: str
) -> dict[str, float | int]:
    def flag(column: str) -> pd.Series:
        if column not in windows:
            return pd.Series(False, index=windows.index, dtype=bool)
        return (
            windows[column]
            .astype("string")
            .str.lower()
            .eq("true")
            .fillna(False)
            .astype(bool)
        )

    if scenario == "S0":
        true_events: list[tuple[int, int]] = []
    elif scenario in {"S2", "S2-Gradual", "S3-Incremental"}:
        true_events = [(12, 25)]
    elif scenario in {"S3", "S4-Recurrent"}:
        true_events = [(12, 17), (24, 29)]
    elif scenario == "S5-Blip":
        true_events = [(12, 13)]
    elif scenario == "S6-Combined":
        true_events = [(12, 15), (16, 23), (28, 35)]
    else:
        true_events = [(12, 17)]
    learned = flag("triggered") | flag("fast_update_warning")
    supervised = flag("v16_performance_triggered") | flag(
        "v17_performance_triggered"
    )
    alarms = sorted(
        windows.loc[learned | supervised, "window"]
        .astype(int)
        .unique()
        .tolist()
    )
    unused = set(alarms)
    delays: list[int] = []
    for onset, tolerance_end in true_events:
        candidates = [
            alarm
            for alarm in sorted(unused)
            if onset <= alarm <= tolerance_end
        ]
        if candidates:
            alarm = candidates[0]
            unused.remove(alarm)
            delays.append(alarm - onset)
    matched = len(delays)
    return {
        "detector_true_events": len(true_events),
        "detector_alarms": len(alarms),
        "detector_matched": matched,
        "detector_missed": len(true_events) - matched,
        "detector_false_alarms": len(unused),
        "detector_precision": (
            float(matched / len(alarms)) if alarms else (1.0 if not true_events else 0.0)
        ),
        "detector_recall": (
            float(matched / len(true_events)) if true_events else np.nan
        ),
        "mean_detection_delay": float(np.mean(delays)) if delays else np.nan,
    }


def _load_windows(root: Path, summaries: pd.DataFrame) -> pd.DataFrame:
    all_windows: list[pd.DataFrame] = []
    for _, row in summaries.iterrows():
        path = _run_dir(root, row) / "windows.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        for key in ["dataset", "scenario", "method", "stream_seed", "model_seed"]:
            frame[key] = row[key]
        all_windows.append(frame)
    return pd.concat(all_windows, ignore_index=True)


def _build_episode_table(
    summaries: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["dataset", "scenario", "method", "stream_seed", "model_seed"]
    for key_values, group in windows.groupby(keys, sort=False):
        dataset, scenario, method, stream_seed, model_seed = key_values
        detector = _matched_detector_metrics(group, str(scenario))
        for episode, onset, end in EPISODES.get(str(scenario), []):
            values = _episode_metrics(group, onset, end)
            rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "episode": episode,
                    "method": method,
                    "stream_seed": int(stream_seed),
                    "model_seed": int(model_seed),
                    **values,
                    **detector,
                    "dasp_commits": int(
                        (group.get("accepted", "") == "dasp").sum()
                    ),
                    "global_update_count": int(
                        group.get(
                            "global_update",
                            pd.Series(False, index=group.index),
                        )
                        .fillna(False)
                        .sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired(
    episodes: pd.DataFrame,
    method: str,
    reference: str,
    metric: str,
) -> pd.DataFrame:
    keys = [
        "dataset",
        "scenario",
        "episode",
        "stream_seed",
        "model_seed",
    ]
    left = (
        episodes[episodes["method"] == method]
        .set_index(keys)[metric]
        .rename("method_value")
    )
    right = (
        episodes[episodes["method"] == reference]
        .set_index(keys)[metric]
        .rename("reference_value")
    )
    paired = pd.concat([left, right], axis=1).dropna().reset_index()
    paired["difference"] = paired["method_value"] - paired["reference_value"]
    return paired


def _hierarchical_bootstrap(
    paired: pd.DataFrame,
    iterations: int = 10_000,
    seed: int = 20260726,
) -> dict[str, float | int]:
    if paired.empty:
        return {
            "n_pairs": 0,
            "mean_difference": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    model_averaged = (
        paired.groupby(
            ["dataset", "scenario", "episode", "stream_seed"],
            as_index=False,
        )["difference"]
        .mean()
    )
    datasets = model_averaged["dataset"].unique()
    rng = np.random.default_rng(seed)
    stream_effects = [
        (
            model_averaged[model_averaged["dataset"] == dataset]
            .groupby("stream_seed")["difference"]
            .mean()
            .to_numpy(dtype=float)
        )
        for dataset in datasets
    ]
    dataset_count = len(stream_effects)
    components = np.empty(
        (iterations, dataset_count, dataset_count), dtype=float
    )
    for dataset_index, effects in enumerate(stream_effects):
        sampled_indices = rng.integers(
            0,
            len(effects),
            size=(iterations, dataset_count, len(effects)),
        )
        components[:, dataset_index, :] = effects[sampled_indices].mean(
            axis=2
        )
    sampled_dataset_indices = rng.integers(
        0, dataset_count, size=(iterations, dataset_count)
    )
    bootstrap_rows = np.arange(iterations)[:, None]
    bootstrap_slots = np.arange(dataset_count)[None, :]
    boot = components[
        bootstrap_rows, sampled_dataset_indices, bootstrap_slots
    ].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "n_pairs": int(len(paired)),
        "n_datasets": int(len(datasets)),
        "n_streams": int(model_averaged["stream_seed"].nunique()),
        "mean_difference": float(paired["difference"].mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "win_rate": float((paired["difference"] > 0).mean()),
    }


def _cluster_sign_flip_test(
    paired: pd.DataFrame,
    *,
    seed: int = 20260729,
    monte_carlo_draws: int = 100_000,
) -> dict[str, float | int | str]:
    """Two-sided paired randomization test at the independent stream level."""
    if paired.empty:
        return {
            "cluster_sign_flip_p_two_sided": np.nan,
            "independent_stream_clusters": 0,
            "sign_flip_mode": "unavailable",
            "sign_flip_draws": 0,
        }
    clusters = (
        paired.groupby(["dataset", "stream_seed"], as_index=False)[
            "difference"
        ]
        .mean()["difference"]
        .to_numpy(dtype=float)
    )
    observed = abs(float(clusters.mean()))
    n_clusters = len(clusters)
    tolerance = 1e-15
    if n_clusters <= 16:
        sign_matrix = np.asarray(
            list(itertools.product([-1.0, 1.0], repeat=n_clusters)),
            dtype=float,
        )
        statistics = np.abs((sign_matrix * clusters).mean(axis=1))
        p_value = float(np.mean(statistics >= observed - tolerance))
        mode = "exact"
        draws = len(statistics)
    else:
        rng = np.random.default_rng(seed)
        extreme = 0
        remaining = int(monte_carlo_draws)
        while remaining:
            batch = min(10_000, remaining)
            signs = rng.choice(
                np.asarray([-1.0, 1.0]),
                size=(batch, n_clusters),
                replace=True,
            )
            statistics = np.abs((signs * clusters).mean(axis=1))
            extreme += int(np.sum(statistics >= observed - tolerance))
            remaining -= batch
        p_value = float((extreme + 1) / (monte_carlo_draws + 1))
        mode = "monte_carlo"
        draws = int(monte_carlo_draws)
    return {
        "cluster_sign_flip_p_two_sided": p_value,
        "independent_stream_clusters": int(n_clusters),
        "sign_flip_mode": mode,
        "sign_flip_draws": int(draws),
    }


def _holm_adjust(frame: pd.DataFrame, p_column: str) -> pd.DataFrame:
    adjusted = frame.copy()
    adjusted["holm_adjusted_p"] = np.nan
    finite = adjusted[p_column].dropna().sort_values()
    running = 0.0
    family_size = len(finite)
    for rank, (index, raw) in enumerate(finite.items()):
        value = min(1.0, (family_size - rank) * float(raw))
        running = max(running, value)
        adjusted.loc[index, "holm_adjusted_p"] = running
    adjusted["holm_reject_0_05"] = (
        adjusted["holm_adjusted_p"].fillna(1.0) < 0.05
    )
    return adjusted


def _all_baseline_statistics(
    episodes: pd.DataFrame,
    metric: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    baselines = sorted(
        set(episodes["method"].astype(str)).difference({DASP_METHOD})
    )
    for index, baseline in enumerate(baselines):
        paired = _paired(
            episodes,
            DASP_METHOD,
            baseline,
            metric,
        )
        statistics = _hierarchical_bootstrap(
            paired,
            iterations,
            seed=seed + index,
        )
        randomization = _cluster_sign_flip_test(
            paired,
            seed=seed + 1000 + index,
        )
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                **statistics,
                **randomization,
                "ci_supports_dasp_superiority": bool(
                    np.isfinite(statistics.get("ci95_low", np.nan))
                    and float(statistics["ci95_low"]) > 0
                ),
            }
        )
    return _holm_adjust(
        pd.DataFrame(rows), "cluster_sign_flip_p_two_sided"
    )


def _causal_ablation_statistics(
    episodes: pd.DataFrame,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    required = {
        DASP_METHOD,
        "dasp_v16_no_detector",
        "dasp_v16_oracle",
    }
    if not required.issubset(set(episodes["method"].astype(str))):
        return {
            "available": False,
            "missing_methods": sorted(
                required.difference(set(episodes["method"].astype(str)))
            ),
        }
    contrasts: dict[str, dict[str, object]] = {}
    specifications = {
        "proposed_minus_detector_off": (
            DASP_METHOD,
            "dasp_v16_no_detector",
        ),
        "proposed_minus_oracle": (
            DASP_METHOD,
            "dasp_v16_oracle",
        ),
        "oracle_minus_detector_off": (
            "dasp_v16_oracle",
            "dasp_v16_no_detector",
        ),
    }
    paired_frames: dict[str, pd.DataFrame] = {}
    for index, (name, (method, reference)) in enumerate(
        specifications.items()
    ):
        paired = _paired(episodes, method, reference, metric)
        paired_frames[name] = paired
        contrasts[name] = {
            **_hierarchical_bootstrap(
                paired, iterations, seed=seed + index
            ),
            **_cluster_sign_flip_test(
                paired, seed=seed + 100 + index
            ),
        }
    proposed_gain = float(
        paired_frames["proposed_minus_detector_off"][
            "difference"
        ].mean()
    )
    oracle_gain = float(
        paired_frames["oracle_minus_detector_off"]["difference"].mean()
    )
    gap_closed = (
        proposed_gain / oracle_gain
        if np.isfinite(oracle_gain) and oracle_gain > 0
        else np.nan
    )
    return {
        "available": True,
        "metric": metric,
        "contrasts": contrasts,
        "proposed_fraction_of_oracle_gain": float(gap_closed),
        "interpretation": (
            "A positive proposed-minus-detector-off effect attributes value "
            "to triggering; proposed-minus-oracle quantifies the remaining "
            "oracle gap. The gain fraction is descriptive and is undefined "
            "when oracle does not improve over detector-off."
        ),
    }


def _audit(
    root: Path,
    summaries: pd.DataFrame,
    windows: pd.DataFrame,
) -> dict[str, object]:
    keys = ["dataset", "scenario", "stream_seed", "model_seed", "method"]
    duplicate_runs = int(summaries.duplicated(keys).sum())
    dasp = summaries[summaries["method"] == DASP_METHOD]
    global_violations = int((dasp["global_update_count"] != 0).sum())
    ssf = windows[windows["method"] == "ssf_budget"]
    max_ssf_labels = (
        float(ssf["ssf_ground_truth_inserted"].fillna(0).max())
        if "ssf_ground_truth_inserted" in ssf
        else np.nan
    )
    max_ssf_pseudo = (
        float(ssf["ssf_pseudo_inserted"].fillna(0).max())
        if "ssf_pseudo_inserted" in ssf
        else np.nan
    )
    return {
        "run_count": int(len(summaries)),
        "duplicate_run_cells": duplicate_runs,
        "missing_window_files": 0,
        "dasp_global_update_violations": global_violations,
        "ssf_budget_max_ground_truth_insertions_per_window": max_ssf_labels,
        "ssf_budget_max_pseudo_insertions_per_window": max_ssf_pseudo,
        "all_finite_primary_metrics": bool(
            np.isfinite(summaries["mean_macro_f1"]).all()
        ),
        "audit_passed": bool(
            duplicate_runs == 0
            and global_violations == 0
            and (
                not np.isfinite(max_ssf_labels)
                or max_ssf_labels <= 256
            )
            and (
                not np.isfinite(max_ssf_pseudo)
                or max_ssf_pseudo == 0
            )
        ),
    }


def _development_gate(
    episodes: pd.DataFrame,
    summaries: pd.DataFrame,
) -> dict[str, object]:
    pair = _paired(
        episodes,
        DASP_METHOD,
        "dasp_v16_no_detector",
        "early_macro_f1",
    )
    proposed = summaries[summaries["method"] == DASP_METHOD]
    no_detector = summaries[
        summaries["method"] == "dasp_v16_no_detector"
    ]
    s0_keys = ["dataset", "scenario", "stream_seed", "model_seed"]
    s0 = pd.concat(
        [
            proposed[proposed["scenario"] == "S0"]
            .set_index(s0_keys)["mean_macro_f1"]
            .rename("proposed"),
            no_detector[no_detector["scenario"] == "S0"]
            .set_index(s0_keys)["mean_macro_f1"]
            .rename("no_detector"),
        ],
        axis=1,
    ).dropna()
    conditions = {
        "early_macro_f1_above_no_detector": bool(
            len(pair) and pair["difference"].mean() > 0
        ),
        "nonzero_adaptation_events": bool(
            proposed["adaptation_triggers"].fillna(0).sum()
            + proposed.get(
                "v16_performance_triggers", pd.Series(dtype=float)
            ).fillna(0).sum()
            + proposed.get("dasp_commits", pd.Series(dtype=float)).fillna(0).sum()
            > 0
        ),
        "nonzero_strong_commits": bool(
            proposed.get("dasp_commits", pd.Series(dtype=float)).fillna(0).sum()
            > 0
        ),
        "zero_global_updates": bool(
            (proposed["global_update_count"] == 0).all()
        ),
        "s0_within_0_005": bool(
            len(s0)
            and (s0["proposed"] - s0["no_detector"]).mean() >= -0.005
        ),
    }
    return {
        **conditions,
        "mean_early_macro_f1_difference": (
            float(pair["difference"].mean()) if len(pair) else np.nan
        ),
        "mean_s0_difference": (
            float((s0["proposed"] - s0["no_detector"]).mean())
            if len(s0)
            else np.nan
        ),
        "advancement_passed": all(conditions.values()),
    }


def _plot_curves(
    destination: Path,
    windows: pd.DataFrame,
) -> None:
    methods = [method for method in PLOT_METHODS if method in set(windows["method"])]
    scenarios = [
        scenario
        for scenario in [
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
            "S8-Stress",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
        ]
        if scenario in set(windows["scenario"])
    ]
    fig, axes = plt.subplots(
        len(scenarios), 1, figsize=(8.2, 2.35 * len(scenarios)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, scenario in zip(axes, scenarios):
        subset = windows[
            (windows["scenario"] == scenario)
            & (windows["method"].isin(methods))
            & (~windows["warmup"].astype(bool))
        ]
        for method in methods:
            values = subset[subset["method"] == method]
            grouped = values.groupby("window")["macro_f1"]
            mean = grouped.mean()
            sem = grouped.sem().fillna(0.0)
            x = mean.index.to_numpy(dtype=int)
            y = mean.to_numpy(dtype=float)
            band = 1.96 * sem.to_numpy(dtype=float)
            axis.plot(x, y, label=method, linewidth=1.5)
            axis.fill_between(x, y - band, y + band, alpha=0.13)
        if scenario != "S0":
            axis.axvline(12, color="black", linestyle="--", linewidth=0.8)
        if scenario in {"S3", "S4-Recurrent"}:
            axis.axvline(24, color="black", linestyle=":", linewidth=0.8)
        if scenario == "S6-Combined":
            for onset in [16, 28]:
                axis.axvline(
                    onset,
                    color="black",
                    linestyle=":",
                    linewidth=0.8,
                )
        axis.set_ylabel(f"{scenario}\nMacro-F1")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Stream window (warm-up W0-W3 excluded)")
    axes[0].legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(destination / f"macro_f1_stream_curves.{suffix}", dpi=220)
    plt.close(fig)


def _plot_forest(destination: Path, paired: pd.DataFrame) -> None:
    if paired.empty:
        return
    grouped = (
        paired.groupby(["dataset", "scenario", "episode"])["difference"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    grouped["se"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"])
    grouped["low"] = grouped["mean"] - 1.96 * grouped["se"]
    grouped["high"] = grouped["mean"] + 1.96 * grouped["se"]
    grouped["label"] = (
        grouped["dataset"]
        + " / "
        + grouped["scenario"]
        + "-"
        + grouped["episode"]
    )
    height = max(4.0, 0.34 * len(grouped) + 1.3)
    fig, axis = plt.subplots(figsize=(8.0, height))
    y = np.arange(len(grouped))
    axis.errorbar(
        grouped["mean"],
        y,
        xerr=[
            grouped["mean"] - grouped["low"],
            grouped["high"] - grouped["mean"],
        ],
        fmt="o",
        color="#1f4e79",
        ecolor="#6f8fae",
        capsize=2,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, grouped["label"], fontsize=7)
    axis.set_xlabel("Paired EarlyMacroF1@5 difference: DASP − SSF-budget")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(destination / f"early_macro_f1_forest.{suffix}", dpi=220)
    plt.close(fig)


def _plot_all_baseline_forest(
    destination: Path,
    statistics: pd.DataFrame,
    metric: str,
) -> None:
    plotted = statistics[statistics["metric"] == metric].copy()
    # CDDA-MD is a documented degenerate configuration on this binary stream
    # (Macro-F1 ~= 0.486, attack recall = 0) and would collapse the useful
    # near-zero effect scale for every viable comparator.
    plotted = plotted[plotted["baseline"] != "cdda_md"]
    plotted = plotted.sort_values("mean_difference")
    if plotted.empty:
        return
    fig, axis = plt.subplots(
        figsize=(8.4, max(4.8, 0.38 * len(plotted) + 1.5))
    )
    y = np.arange(len(plotted))
    means = plotted["mean_difference"].to_numpy(dtype=float)
    lows = plotted["ci95_low"].to_numpy(dtype=float)
    highs = plotted["ci95_high"].to_numpy(dtype=float)
    colors = [
        "#2a9d8f" if low > 0 else "#d1495b" if high < 0 else "#6c757d"
        for low, high in zip(lows, highs)
    ]
    for position, mean, low, high, color in zip(
        y, means, lows, highs, colors
    ):
        axis.errorbar(
            mean,
            position,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
        )
    axis.axvline(0, color="black", linewidth=0.9)
    axis.set_yticks(y, plotted["baseline"].str.replace("_", " "))
    axis.set_xlabel(f"Paired difference: DASP − baseline ({metric})")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    stem = f"dasp_all_baselines_{metric}_forest"
    for suffix in ["png", "pdf"]:
        fig.savefig(destination / f"{stem}.{suffix}", dpi=220)
    plt.close(fig)


def _run_performance(
    summaries: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    summaries = summaries.copy()
    if "peak_rss_increment_bytes" not in summaries:
        summaries["peak_rss_increment_bytes"] = summaries["peak_rss_bytes"]
    if "peak_cuda_reserved_bytes" not in summaries:
        summaries["peak_cuda_reserved_bytes"] = summaries[
            "peak_cuda_allocated_bytes"
        ]
    keys = ["dataset", "scenario", "method", "stream_seed", "model_seed"]
    evaluated = windows[~windows["warmup"].astype(bool)].copy()
    # prediction_decision_seconds already includes preprocessing, forward
    # inference, and drift scoring. Add metric computation and update exactly
    # once to obtain the complete test-then-train window latency.
    components = [
        "prediction_decision_seconds",
        "metric_seconds",
        "update_seconds",
    ]
    missing = [column for column in components if column not in evaluated]
    if missing:
        raise ValueError(f"Window timing columns are missing: {missing}")
    evaluated["end_to_end_seconds"] = evaluated[components].sum(axis=1)
    gradient_event = (
        pd.to_numeric(evaluated["gradient_evaluations"], errors="coerce")
        .fillna(0)
        .gt(0)
    )
    sample_event = (
        pd.to_numeric(evaluated["sample_updates"], errors="coerce")
        .fillna(0)
        .gt(0)
    )
    evaluated["adaptation_event"] = gradient_event | sample_event
    per_run = (
        evaluated.groupby(keys, as_index=False)
        .agg(
            evaluated_windows=("window", "size"),
            total_evaluated_flows=("n_samples", "sum"),
            adaptation_event_count=("adaptation_event", "sum"),
            mean_end_to_end_seconds=("end_to_end_seconds", "mean"),
            median_end_to_end_seconds=("end_to_end_seconds", "median"),
            p95_end_to_end_seconds=(
                "end_to_end_seconds",
                lambda values: float(values.quantile(0.95)),
            ),
            total_end_to_end_seconds=("end_to_end_seconds", "sum"),
            total_update_seconds=("update_seconds", "sum"),
            total_gradient_evaluations=("gradient_evaluations", "sum"),
            total_ground_truth_update_labels=(
                "ground_truth_update_labels",
                "sum",
            ),
        )
    )
    per_run["adaptation_event_rate"] = (
        per_run["adaptation_event_count"] / per_run["evaluated_windows"]
    )
    per_run["end_to_end_throughput_flows_per_second"] = (
        per_run["total_evaluated_flows"]
        / per_run["total_end_to_end_seconds"].replace(0.0, np.nan)
    )
    event_rows = evaluated[evaluated["adaptation_event"]]
    if event_rows.empty:
        event_latency = pd.DataFrame(
            columns=keys
            + [
                "median_adaptation_event_seconds",
                "p95_adaptation_event_seconds",
            ]
        )
    else:
        event_latency = (
            event_rows.groupby(keys, as_index=False)
            .agg(
                median_adaptation_event_seconds=("update_seconds", "median"),
                p95_adaptation_event_seconds=(
                    "update_seconds",
                    lambda values: float(values.quantile(0.95)),
                ),
            )
        )
    per_run = per_run.merge(
        event_latency,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    summary_columns = keys + [
        "mean_macro_f1",
        "mean_attack_recall",
        "p95_forward_seconds",
        "p95_prediction_decision_seconds",
        "p95_update_seconds",
        "prediction_decision_throughput_flows_per_second",
        "peak_rss_bytes",
        "peak_rss_increment_bytes",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "memory_payload_bytes",
        "model_parameter_bytes",
    ]
    missing = [column for column in summary_columns if column not in summaries]
    if missing:
        raise ValueError(f"Run performance columns are missing: {missing}")
    return summaries[summary_columns].merge(
        per_run, on=keys, how="inner", validate="one_to_one"
    )


def _plot_latency(destination: Path, performance: pd.DataFrame) -> None:
    plotted = performance[performance["method"].isin(PLOT_METHODS)].copy()
    panels = [
        ("inference_p95_ms", "p95 forward inference (ms/window)"),
        ("window_update_p95_ms", "p95 update over all windows (ms/window)"),
        (
            "adaptation_event_p95_ms",
            "p95 update conditional on adaptation (ms/event)",
        ),
        ("end_to_end_p95_ms", "p95 end-to-end processing (ms/window)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharey=True)
    axes = axes.ravel()
    colors = dict(
        zip(
            PLOT_METHODS,
            plt.get_cmap("tab20").colors[: len(PLOT_METHODS)],
        )
    )
    for axis, (column, label) in zip(axes, panels):
        for _, row in plotted.iterrows():
            method = str(row["method"])
            axis.scatter(
                row[column],
                row["macro_f1"],
                s=90 if method == DASP_METHOD else 48,
                marker="*" if method == DASP_METHOD else "o",
                color=colors[method],
                edgecolor="black" if method == DASP_METHOD else "none",
                linewidth=0.6,
                label=method,
            )
        axis.set_xlabel(label)
        axis.grid(alpha=0.25)
    lower = float(plotted["macro_f1"].min())
    upper = float(plotted["macro_f1"].max())
    margin = max(0.0005, 0.12 * (upper - lower))
    axes[0].set_ylim(lower - margin, upper + margin)
    axes[0].set_ylabel("Mean post-warm-up Macro-F1")
    axes[2].set_ylabel("Mean post-warm-up Macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(
            destination / f"accuracy_performance_tradeoff.{suffix}", dpi=220
        )
    plt.close(fig)


def _write_performance_outputs(
    destination: Path,
    summaries: pd.DataFrame,
    windows: pd.DataFrame,
) -> None:
    runs = _run_performance(summaries, windows)
    performance = (
        runs.groupby("method", as_index=False)
        .agg(
            n_runs=("method", "size"),
            macro_f1=("mean_macro_f1", "mean"),
            attack_recall=("mean_attack_recall", "mean"),
            inference_p95_ms=("p95_forward_seconds", lambda x: 1000 * x.median()),
            prediction_decision_p95_ms=(
                "p95_prediction_decision_seconds",
                lambda x: 1000 * x.median(),
            ),
            window_update_p95_ms=(
                "p95_update_seconds",
                lambda x: 1000 * x.median(),
            ),
            adaptation_event_median_ms=(
                "median_adaptation_event_seconds",
                lambda x: 1000 * x.median(),
            ),
            adaptation_event_p95_ms=(
                "p95_adaptation_event_seconds",
                lambda x: 1000 * x.median(),
            ),
            adaptation_event_rate=("adaptation_event_rate", "mean"),
            mean_end_to_end_ms=(
                "mean_end_to_end_seconds",
                lambda x: 1000 * x.median(),
            ),
            end_to_end_p95_ms=(
                "p95_end_to_end_seconds", lambda x: 1000 * x.median()
            ),
            total_end_to_end_seconds=(
                "total_end_to_end_seconds",
                "median",
            ),
            total_update_seconds=("total_update_seconds", "median"),
            total_gradient_evaluations=(
                "total_gradient_evaluations",
                "median",
            ),
            total_ground_truth_update_labels=(
                "total_ground_truth_update_labels",
                "median",
            ),
            throughput_flows_per_second=(
                "prediction_decision_throughput_flows_per_second",
                "median",
            ),
            end_to_end_throughput_flows_per_second=(
                "end_to_end_throughput_flows_per_second",
                "median",
            ),
            peak_rss_mib=("peak_rss_bytes", lambda x: x.median() / 2**20),
            peak_rss_increment_mib=(
                "peak_rss_increment_bytes",
                lambda x: x.median() / 2**20,
            ),
            peak_cuda_allocated_mib=(
                "peak_cuda_allocated_bytes",
                lambda x: x.median() / 2**20,
            ),
            peak_cuda_reserved_mib=(
                "peak_cuda_reserved_bytes",
                lambda x: x.median() / 2**20,
            ),
            memory_payload_mib=(
                "memory_payload_bytes", lambda x: x.median() / 2**20
            ),
            model_mib=("model_parameter_bytes", lambda x: x.median() / 2**20),
        )
        .sort_values("macro_f1", ascending=False)
    )
    performance.to_csv(destination / "performance_by_method.csv", index=False)

    keys = ["dataset", "scenario", "stream_seed", "model_seed"]
    metric_map = {
        "macro_f1": "mean_macro_f1",
        "attack_recall": "mean_attack_recall",
        "inference": "p95_forward_seconds",
        "adaptation": "p95_update_seconds",
        "event_adaptation": "p95_adaptation_event_seconds",
        "end_to_end": "p95_end_to_end_seconds",
        "throughput": "prediction_decision_throughput_flows_per_second",
        "peak_rss": "peak_rss_bytes",
        "peak_cuda": "peak_cuda_allocated_bytes",
    }
    proposed = runs[runs["method"] == DASP_METHOD].set_index(keys)
    paired_frames: list[pd.DataFrame] = []
    for baseline in sorted(set(runs["method"]).difference({DASP_METHOD})):
        reference = runs[runs["method"] == baseline].set_index(keys)
        common = proposed.index.intersection(reference.index)
        if common.empty:
            continue
        pair = pd.DataFrame(index=common).reset_index()
        pair["baseline"] = baseline
        for label, column in metric_map.items():
            pair[f"dasp_{label}"] = proposed.loc[common, column].to_numpy()
            pair[f"baseline_{label}"] = reference.loc[common, column].to_numpy()
        pair["delta_macro_f1"] = (
            pair["dasp_macro_f1"] - pair["baseline_macro_f1"]
        )
        pair["delta_attack_recall"] = (
            pair["dasp_attack_recall"] - pair["baseline_attack_recall"]
        )
        for label in [
            "inference",
            "adaptation",
            "event_adaptation",
            "end_to_end",
        ]:
            denominator = pair[f"dasp_{label}"].replace(0.0, np.nan)
            pair[f"{label}_speedup"] = pair[f"baseline_{label}"] / denominator
        pair["throughput_ratio"] = (
            pair["dasp_throughput"]
            / pair["baseline_throughput"].replace(0.0, np.nan)
        )
        pair["peak_rss_ratio"] = (
            pair["dasp_peak_rss"]
            / pair["baseline_peak_rss"].replace(0.0, np.nan)
        )
        pair["peak_cuda_ratio"] = (
            pair["dasp_peak_cuda"]
            / pair["baseline_peak_cuda"].replace(0.0, np.nan)
        )
        paired_frames.append(pair)
    paired = (
        pd.concat(paired_frames, ignore_index=True)
        if paired_frames
        else pd.DataFrame()
    )
    paired.to_csv(destination / "paired_dasp_performance_cells.csv", index=False)
    if not paired.empty:
        comparison = (
            paired.groupby("baseline", as_index=False)
            .agg(
                n_pairs=("delta_macro_f1", "size"),
                delta_macro_f1=("delta_macro_f1", "mean"),
                macro_f1_win_rate=("delta_macro_f1", lambda x: (x > 0).mean()),
                delta_attack_recall=("delta_attack_recall", "mean"),
                attack_recall_win_rate=(
                    "delta_attack_recall", lambda x: (x > 0).mean()
                ),
                inference_speedup=("inference_speedup", "median"),
                inference_faster_rate=(
                    "inference_speedup", lambda x: (x > 1).mean()
                ),
                adaptation_speedup=("adaptation_speedup", "median"),
                adaptation_faster_rate=(
                    "adaptation_speedup", lambda x: (x > 1).mean()
                ),
                event_adaptation_speedup=(
                    "event_adaptation_speedup",
                    "median",
                ),
                event_adaptation_faster_rate=(
                    "event_adaptation_speedup",
                    lambda x: (x > 1).mean(),
                ),
                end_to_end_speedup=("end_to_end_speedup", "median"),
                end_to_end_faster_rate=(
                    "end_to_end_speedup", lambda x: (x > 1).mean()
                ),
                throughput_ratio=("throughput_ratio", "median"),
                peak_rss_ratio=("peak_rss_ratio", "median"),
                peak_cuda_ratio=("peak_cuda_ratio", "median"),
            )
        )
        comparison.to_csv(
            destination / "paired_dasp_performance_summary.csv", index=False
        )
    _plot_latency(destination, performance)


def _write_latex_table(destination: Path, episodes: pd.DataFrame) -> None:
    table = (
        episodes.groupby(["scenario", "episode", "method"], as_index=False)
        .agg(
            early_macro_f1=("early_macro_f1", "mean"),
            early_attack_recall=("early_attack_recall", "mean"),
            recovery_deficit=("recovery_deficit_at_5", "mean"),
        )
    )
    lines = [
        "% Generated automatically; do not edit numeric values by hand.",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Scenario & Episode & Method & Early F1 & Attack Recall & Deficit \\\\",
        "\\midrule",
    ]
    for _, row in table.iterrows():
        method = str(row["method"]).replace("_", "\\_")
        lines.append(
            f"{row['scenario']} & {row['episode']} & {method} & "
            f"{row['early_macro_f1']:.4f} & "
            f"{row['early_attack_recall']:.4f} & "
            f"{row['recovery_deficit']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (destination / "primary_results_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    global DASP_METHOD
    parser = argparse.ArgumentParser(
        description="Analyze the locked five-dataset Selective-DASP matrix."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()

    root = Path(args.output_dir)
    summary_path = root / "summary_by_run.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summaries = pd.read_csv(summary_path)
    required = {
        "dataset",
        "scenario",
        "method",
        "stream_seed",
        "model_seed",
        "mean_macro_f1",
    }
    missing = required.difference(summaries.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    present_methods = set(summaries["method"].astype(str))
    if "dasp_v16" in present_methods:
        DASP_METHOD = "dasp_v16"
    elif "dasp" not in present_methods:
        raise ValueError(
            "Summary contains neither proposed method 'dasp_v16' nor 'dasp'"
        )

    destination = root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    windows = _load_windows(root, summaries)
    episodes = _build_episode_table(summaries, windows)
    episodes.to_csv(destination / "episode_metrics.csv", index=False)
    windows.to_csv(destination / "all_windows.csv", index=False)

    primary_pair = _paired(
        episodes, DASP_METHOD, "ssf_budget", "early_macro_f1"
    )
    recall_pair = _paired(
        episodes, DASP_METHOD, "ssf_budget", "early_attack_recall"
    )
    primary_pair.to_csv(
        destination / "paired_early_macro_f1.csv", index=False
    )
    recall_pair.to_csv(
        destination / "paired_early_attack_recall.csv", index=False
    )
    statistics = {
        "early_macro_f1": {
            **_hierarchical_bootstrap(
                primary_pair, args.bootstrap_iterations
            ),
            **_cluster_sign_flip_test(primary_pair),
            "multiplicity_role": (
                "single prespecified primary contrast; no adjustment"
            ),
        },
        "early_attack_recall": {
            **_hierarchical_bootstrap(
                recall_pair,
                args.bootstrap_iterations,
                seed=20260727,
            ),
            **_cluster_sign_flip_test(
                recall_pair, seed=20260730
            ),
            "multiplicity_role": (
                "prespecified safety/non-inferiority endpoint"
            ),
        },
    }
    (destination / "primary_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    causal_ablations = {
        metric: _causal_ablation_statistics(
            episodes,
            metric,
            args.bootstrap_iterations,
            seed=20260731 + index * 100,
        )
        for index, metric in enumerate(
            ["early_macro_f1", "early_attack_recall"]
        )
    }
    (destination / "causal_ablation_statistics.json").write_text(
        json.dumps(causal_ablations, indent=2), encoding="utf-8"
    )
    all_baseline_statistics = pd.concat(
        [
            _all_baseline_statistics(
                episodes,
                "early_macro_f1",
                args.bootstrap_iterations,
                seed=20260728,
            ),
            _all_baseline_statistics(
                episodes,
                "early_attack_recall",
                args.bootstrap_iterations,
                seed=20261728,
            ),
        ],
        ignore_index=True,
    )
    all_baseline_statistics.to_csv(
        destination / "paired_dasp_all_baselines_statistics.csv",
        index=False,
    )
    _plot_all_baseline_forest(
        destination,
        all_baseline_statistics,
        "early_macro_f1",
    )
    _plot_all_baseline_forest(
        destination,
        all_baseline_statistics,
        "early_attack_recall",
    )
    audit = _audit(root, summaries, windows)
    (destination / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if "stage" in summaries and set(summaries["stage"]) == {"development"}:
        gate = _development_gate(episodes, summaries)
        gate["integrity_audit_passed"] = audit["audit_passed"]
        gate["advancement_passed"] = bool(
            gate["advancement_passed"] and audit["audit_passed"]
        )
        (destination / "development_advancement.json").write_text(
            json.dumps(gate, indent=2), encoding="utf-8"
        )

    _plot_curves(destination, windows)
    _plot_forest(destination, primary_pair)
    _write_performance_outputs(destination, summaries, windows)
    _write_latex_table(destination, episodes)
    print(f"analysis complete: {destination}", flush=True)


if __name__ == "__main__":
    main()
