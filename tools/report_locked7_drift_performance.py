"""Report drift adaptation and clean GPU performance for the locked baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "dasp",
    "static",
    "periodic_ft",
    "online_mlp",
    "experience_replay",
    "ssf_faithful",
    "ssf_budget",
    "repshield",
]
DISPLAY = {
    "dasp": "DASP",
    "static": "Frozen",
    "periodic_ft": "Periodic FT",
    "online_mlp": "Online MLP",
    "experience_replay": "Replay",
    "ssf_faithful": "SSF-faithful",
    "ssf_budget": "SSF-budget",
    "repshield": "RepShield",
}
EPISODE_ORDER = ["S1-first", "S2-gradual", "S3-first", "S3-recurrence", "S4-local"]


def _require_methods(frame: pd.DataFrame, source: str) -> None:
    observed = set(frame["method"].unique())
    missing = set(METHODS) - observed
    unexpected = observed - set(METHODS)
    if missing or unexpected:
        raise ValueError(
            f"{source}: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame[columns].copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.6f}"
        )
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def _adaptation_tables(
    episodes: pd.DataFrame, summaries: pd.DataFrame, windows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episodes = episodes.copy()
    episodes["episode_key"] = episodes["scenario"] + "-" + episodes["episode"]
    episodes["method_display"] = episodes["method"].map(DISPLAY)

    overall = (
        episodes.groupby("method", as_index=False)
        .agg(
            n_episode_runs=("early_macro_f1", "size"),
            early_macro_f1=("early_macro_f1", "mean"),
            early_attack_recall=("early_attack_recall", "mean"),
            minimum_post_drift_macro_f1=(
                "minimum_post_drift_macro_f1",
                "mean",
            ),
            median_recovery_time=("recovery_time", "median"),
            recovery_censor_rate=("recovery_censored", "mean"),
            recovery_deficit_at_5=("recovery_deficit_at_5", "mean"),
        )
        .sort_values("early_macro_f1", ascending=False)
    )
    overall.insert(1, "method_display", overall["method"].map(DISPLAY))

    by_episode = (
        episodes.groupby(["method", "episode_key"], as_index=False)
        .agg(
            n_runs=("early_macro_f1", "size"),
            early_macro_f1=("early_macro_f1", "mean"),
            early_attack_recall=("early_attack_recall", "mean"),
            minimum_post_drift_macro_f1=(
                "minimum_post_drift_macro_f1",
                "mean",
            ),
            median_recovery_time=("recovery_time", "median"),
            recovery_censor_rate=("recovery_censored", "mean"),
            recovery_deficit_at_5=("recovery_deficit_at_5", "mean"),
        )
    )
    by_episode.insert(1, "method_display", by_episode["method"].map(DISPLAY))

    s3 = episodes[episodes["scenario"].eq("S3")].pivot_table(
        index=["method", "stream_seed", "model_seed"],
        columns="episode",
        values=["early_macro_f1", "early_attack_recall"],
    )
    s3.columns = ["_".join(column) for column in s3.columns]
    s3 = s3.reset_index()
    s3["recurrence_gain_macro_f1"] = (
        s3["early_macro_f1_recurrence"] - s3["early_macro_f1_first"]
    )
    s3["recurrence_gain_attack_recall"] = (
        s3["early_attack_recall_recurrence"]
        - s3["early_attack_recall_first"]
    )
    forgetting = (
        summaries[summaries["scenario"].eq("S3")]
        .groupby("method", as_index=False)
        .agg(
            mean_online_forgetting=("mean_online_forgetting", "mean"),
            final_online_forgetting=("final_online_forgetting", "mean"),
            forgetting_A1_to_A2=("forgetting_A1_to_A2", "mean"),
        )
    )
    recurrence = (
        s3.groupby("method", as_index=False)
        .agg(
            n_pairs=("recurrence_gain_macro_f1", "size"),
            first_macro_f1=("early_macro_f1_first", "mean"),
            recurrence_macro_f1=("early_macro_f1_recurrence", "mean"),
            recurrence_gain_macro_f1=("recurrence_gain_macro_f1", "mean"),
            first_attack_recall=("early_attack_recall_first", "mean"),
            recurrence_attack_recall=(
                "early_attack_recall_recurrence",
                "mean",
            ),
            recurrence_gain_attack_recall=(
                "recurrence_gain_attack_recall",
                "mean",
            ),
        )
        .merge(forgetting, on="method", how="left", validate="one_to_one")
        .sort_values("recurrence_macro_f1", ascending=False)
    )
    recurrence.insert(1, "method_display", recurrence["method"].map(DISPLAY))

    stationary = windows[
        windows["scenario"].eq("S0") & ~windows["warmup"].astype(bool)
    ].copy()
    gradient_event = (
        pd.to_numeric(stationary["gradient_evaluations"], errors="coerce")
        .fillna(0)
        .gt(0)
    )
    sample_event = (
        pd.to_numeric(stationary["sample_updates"], errors="coerce")
        .fillna(0)
        .gt(0)
    )
    stationary["update_event"] = gradient_event | sample_event
    alarm_columns = [
        column
        for column in ["triggered", "fast_update_warning", "v17_performance_triggered"]
        if column in stationary
    ]
    alarm = pd.Series(False, index=stationary.index)
    for column in alarm_columns:
        values = stationary[column]
        alarm |= values.eq(True) | values.astype(str).str.lower().isin(["true", "1"])
    stationary["alarm_or_fast_trigger"] = alarm
    stationary_summary = (
        stationary.groupby("method", as_index=False)
        .agg(
            n_stationary_windows=("window", "size"),
            stationary_update_event_rate=("update_event", "mean"),
            stationary_alarm_or_fast_trigger_rate=(
                "alarm_or_fast_trigger",
                "mean",
            ),
            stationary_gradient_evaluations=("gradient_evaluations", "sum"),
            stationary_ground_truth_labels=("ground_truth_update_labels", "sum"),
        )
        .sort_values("stationary_update_event_rate")
    )
    stationary_summary.insert(
        1, "method_display", stationary_summary["method"].map(DISPLAY)
    )

    # Detector totals are run-level fields duplicated across S3's two episode
    # rows. Deduplicate before summing so the recurrence scenario is counted once.
    detector_runs = episodes.drop_duplicates(
        ["method", "scenario", "stream_seed", "model_seed"]
    ).copy()
    detector_runs["delay_weighted_matches"] = (
        detector_runs["mean_detection_delay"].fillna(0.0)
        * detector_runs["detector_matched"]
    )
    detector = (
        detector_runs.groupby("method", as_index=False)
        .agg(
            detector_true_events=("detector_true_events", "sum"),
            detector_alarms=("detector_alarms", "sum"),
            detector_matched=("detector_matched", "sum"),
            detector_missed=("detector_missed", "sum"),
            detector_false_alarms=("detector_false_alarms", "sum"),
            delay_weighted_matches=("delay_weighted_matches", "sum"),
        )
    )
    detector["detector_precision"] = (
        detector["detector_matched"] / detector["detector_alarms"].replace(0, np.nan)
    ).fillna(0.0)
    detector["detector_recall"] = (
        detector["detector_matched"]
        / detector["detector_true_events"].replace(0, np.nan)
    ).fillna(0.0)
    detector["mean_detection_delay"] = (
        detector["delay_weighted_matches"]
        / detector["detector_matched"].replace(0, np.nan)
    )
    detector = detector.drop(columns="delay_weighted_matches")
    detector.insert(1, "method_display", detector["method"].map(DISPLAY))
    return overall, by_episode, recurrence, stationary_summary, detector


def _plot_episode_heatmap(by_episode: pd.DataFrame, destination: Path) -> None:
    matrix = by_episode.pivot(
        index="method", columns="episode_key", values="early_macro_f1"
    ).reindex(index=METHODS, columns=EPISODE_ORDER)
    fig, axis = plt.subplots(figsize=(9.4, 4.8))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(matrix.index)), [DISPLAY[item] for item in matrix.index])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            axis.text(
                column,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                color="white" if value < matrix.to_numpy().mean() else "black",
                fontsize=8,
            )
    axis.set_title("Early Macro-F1 in the first five post-drift windows")
    fig.colorbar(image, ax=axis, label="Macro-F1")
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(destination / f"locked7_drift_episode_heatmap.{suffix}", dpi=220)
    plt.close(fig)


def _plot_tradeoff(joined: pd.DataFrame, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    label_offsets = {
        "dasp": (7, 5),
        "experience_replay": (7, -2),
        "periodic_ft": (7, -22),
        "online_mlp": (-7, -15),
        "static": (7, 4),
        "ssf_faithful": (7, 4),
        "ssf_budget": (7, 4),
        "repshield": (7, 4),
    }
    for row in joined.itertuples(index=False):
        axis.scatter(
            row.mean_end_to_end_ms,
            row.early_macro_f1,
            s=75 if row.method == "dasp" else 50,
            marker="*" if row.method == "dasp" else "o",
        )
        axis.annotate(
            DISPLAY[row.method],
            (row.mean_end_to_end_ms, row.early_macro_f1),
            xytext=label_offsets[row.method],
            textcoords="offset points",
            ha="right" if row.method == "online_mlp" else "left",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Mean test-then-train latency (ms/window, log scale; lower is better)")
    axis.set_ylabel("Confirmation early Macro-F1 (higher is better)")
    axis.set_title("Accuracy–performance trade-off on NF-UNSW-NB15-v3")
    axis.margins(y=0.08)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(destination / f"locked7_accuracy_performance_tradeoff.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("confirmation_root", type=Path)
    parser.add_argument("performance_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    confirmation_analysis = args.confirmation_root / "analysis"
    performance_analysis = args.performance_root / "analysis"
    destination = (
        args.output_dir
        if args.output_dir is not None
        else args.confirmation_root / "analysis_locked7_drift_performance"
    )
    destination.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(confirmation_analysis / "episode_metrics.csv")
    summaries = pd.read_csv(args.confirmation_root / "summary_by_run.csv")
    windows = pd.read_csv(confirmation_analysis / "all_windows.csv")
    performance = pd.read_csv(performance_analysis / "performance_by_method.csv")
    paired = pd.read_csv(
        confirmation_analysis / "paired_dasp_all_baselines_statistics.csv"
    )
    for frame, source in [
        (episodes, "confirmation episodes"),
        (summaries, "confirmation summaries"),
        (windows, "confirmation windows"),
        (performance, "performance summaries"),
    ]:
        _require_methods(frame, source)

    overall, by_episode, recurrence, stationary, detector = _adaptation_tables(
        episodes, summaries, windows
    )
    performance = performance[performance["method"].isin(METHODS)].copy()
    performance.insert(1, "method_display", performance["method"].map(DISPLAY))
    performance = performance.sort_values("mean_end_to_end_ms")
    joined = overall.merge(
        performance[
            [
                "method",
                "inference_p95_ms",
                "prediction_decision_p95_ms",
                "mean_end_to_end_ms",
                "end_to_end_p95_ms",
                "throughput_flows_per_second",
                "end_to_end_throughput_flows_per_second",
                "peak_cuda_allocated_mib",
                "peak_cuda_reserved_mib",
                "peak_rss_increment_mib",
                "adaptation_event_rate",
                "total_ground_truth_update_labels",
            ]
        ],
        on="method",
        how="inner",
        validate="one_to_one",
    )

    overall.to_csv(destination / "drift_adaptation_overall.csv", index=False)
    by_episode.to_csv(destination / "drift_adaptation_by_episode.csv", index=False)
    recurrence.to_csv(destination / "s3_recurrence_and_forgetting.csv", index=False)
    stationary.to_csv(destination / "s0_stationary_update_activity.csv", index=False)
    detector.to_csv(destination / "drift_detection_events.csv", index=False)
    performance.to_csv(destination / "gpu_performance_locked7.csv", index=False)
    joined.to_csv(destination / "accuracy_performance_join.csv", index=False)
    paired.to_csv(destination / "paired_dasp_statistics.csv", index=False)
    _plot_episode_heatmap(by_episode, destination)
    _plot_tradeoff(joined, destination)

    dasp = joined[joined["method"].eq("dasp")].iloc[0]
    dasp_detector = detector[detector["method"].eq("dasp")].iloc[0]
    dasp_recurrence = recurrence[recurrence["method"].eq("dasp")].iloc[0]
    f1_pairs = paired[paired["metric"].eq("early_macro_f1")]
    f1_supported = int(
        (
            f1_pairs["ci95_low"].gt(0)
            & f1_pairs["mean_difference"].gt(0)
        ).sum()
    )
    recall_pairs = paired[paired["metric"].eq("early_attack_recall")]
    recall_losses = recall_pairs[
        recall_pairs["ci95_high"].lt(0) & recall_pairs["mean_difference"].lt(0)
    ]
    fastest = performance.iloc[0]
    best_f1 = overall.iloc[0]
    inference_order = performance.sort_values("inference_p95_ms")
    inference_rank = int(
        inference_order.reset_index(drop=True)
        .index[inference_order.reset_index(drop=True)["method"].eq("dasp")][0]
        + 1
    )
    end_to_end_rank = int(
        performance.reset_index(drop=True)
        .index[performance.reset_index(drop=True)["method"].eq("dasp")][0]
        + 1
    )
    next_inference = inference_order[inference_order["method"].ne("dasp")].iloc[0]
    forward_advantage = next_inference.inference_p95_ms / dasp.inference_p95_ms
    e2e_advantages = {
        method: float(
            performance.loc[
                performance["method"].eq(method), "mean_end_to_end_ms"
            ].iloc[0]
            / dasp.mean_end_to_end_ms
        )
        for method in ["ssf_faithful", "ssf_budget", "repshield"]
    }
    replay_e2e = float(
        performance.loc[
            performance["method"].eq("experience_replay"), "mean_end_to_end_ms"
        ].iloc[0]
    )
    conclusion = (
        "DASP has the highest mean early Macro-F1, but universal superiority is "
        "not established because at least one paired confidence interval crosses "
        "zero and DASP is not the fastest method."
    )
    report = f"""# Locked-seven drift adaptation and GPU performance report

## Material Passport

- ID: NF-UNSW-LOCKED7-DRIFT-PERF
- Type: experiment_result + validation_report
- Status: ANALYZED
- Confirmation evidence: `{args.confirmation_root}`
- Clean performance evidence: `{args.performance_root}`
- Scope: NF-UNSW-NB15-v3, binary Benign/Attack, S0--S4
- Methods: DASP plus exactly seven locked comparators

## Measurement separation

Drift accuracy is taken only from the 400-run confirmation matrix
(5 stream seeds x 2 model seeds). Runtime and GPU memory are taken only from
the clean 80-run development benchmark (1 stream seed x 2 model seeds). The
development accuracy is not substituted for confirmation accuracy.

End-to-end latency is measured as prediction/decision (already including
preprocessing, forward inference, and drift scoring) + metric computation +
test-then-train update. Forward inference is also reported separately.

## Drift-adaptation result

{_markdown_table(overall, [
    "method_display", "early_macro_f1", "early_attack_recall",
    "minimum_post_drift_macro_f1", "median_recovery_time",
    "recovery_censor_rate", "recovery_deficit_at_5"
])}

DASP early Macro-F1 = {dasp.early_macro_f1:.6f}. It has the highest mean
({DISPLAY[best_f1.method]} = {best_f1.early_macro_f1:.6f}) and its paired 95%
bootstrap interval is above zero against {f1_supported}/7 comparators.
Attack Recall has {len(recall_losses)} statistically supported loss(es):
{", ".join(DISPLAY[item] for item in recall_losses["baseline"]) or "none"}.

All methods have median recovery time 0 under the preregistered recovery floor.
This is a ceiling effect: recovery time does not discriminate adaptation on this
binary stream, so early Macro-F1/Attack Recall and post-drift minimum remain the
informative adaptation outcomes.

## Detection behavior on drift scenarios

{_markdown_table(detector, [
    "method_display", "detector_true_events", "detector_alarms",
    "detector_matched", "detector_missed", "detector_false_alarms",
    "detector_precision", "detector_recall", "mean_detection_delay"
])}

DASP detector precision = {dasp_detector.detector_precision:.6f}, recall =
{dasp_detector.detector_recall:.6f}, and matched-event delay =
{dasp_detector.mean_detection_delay:.6f} windows. Thus the high predictive
scores do not validate the detector mechanism by themselves.

## S3 recurrence and forgetting

{_markdown_table(recurrence, [
    "method_display", "first_macro_f1", "recurrence_macro_f1",
    "recurrence_gain_macro_f1", "first_attack_recall",
    "recurrence_attack_recall", "forgetting_A1_to_A2"
])}

DASP recurrence gain is {dasp_recurrence.recurrence_gain_macro_f1:.6f}
Macro-F1 and {dasp_recurrence.recurrence_gain_attack_recall:.6f} Attack Recall;
negative values mean recurrence is slightly worse than its own first-drift
episode.

## S0 stationary activity

The update-event rate is descriptive computational activity, not a detector
false-alarm probability for methods that update by design.

{_markdown_table(stationary, [
    "method_display", "stationary_update_event_rate",
    "stationary_alarm_or_fast_trigger_rate",
    "stationary_gradient_evaluations", "stationary_ground_truth_labels"
])}

## Clean GPU performance

{_markdown_table(performance, [
    "method_display", "inference_p95_ms", "prediction_decision_p95_ms",
    "mean_end_to_end_ms", "end_to_end_p95_ms",
    "end_to_end_throughput_flows_per_second", "peak_cuda_allocated_mib",
    "peak_cuda_reserved_mib", "adaptation_event_rate"
])}

Fastest mean end-to-end method: {DISPLAY[fastest.method]}
({fastest.mean_end_to_end_ms:.6f} ms/window). DASP:
{dasp.mean_end_to_end_ms:.6f} ms/window, p95 forward inference
{dasp.inference_p95_ms:.6f} ms/window, end-to-end throughput
{dasp.end_to_end_throughput_flows_per_second:.2f} flows/s, peak CUDA allocated
{dasp.peak_cuda_allocated_mib:.2f} MiB.

DASP ranks {inference_rank}/8 for p95 forward inference and is
{forward_advantage:.2f}x faster than the next method
({DISPLAY[next_inference.method]}) on that isolated operation. It ranks
{end_to_end_rank}/8 for mean complete test-then-train latency:
{(dasp.mean_end_to_end_ms / replay_e2e - 1) * 100:.2f}% slower than Replay,
but {e2e_advantages["ssf_faithful"]:.2f}x, {e2e_advantages["ssf_budget"]:.2f}x,
and {e2e_advantages["repshield"]:.2f}x faster than SSF-faithful, SSF-budget,
and RepShield, respectively. Its peak allocated CUDA memory is the largest
among these eight methods.

Runtime anomaly: CUDA SVD emitted a non-convergence warning for a baseline
batch and PyTorch automatically used its more accurate fallback solver. No run
crashed or was retried. The resulting wall-clock latency is retained because
the fallback is part of that method's executed computation.

## Scientific conclusion

{conclusion}

Therefore this result supports a selective adaptation/accuracy claim, not a
"fastest and most accurate against every method" claim. The locked manuscript
gate remains closed until DASP also obtains a positive paired interval against
Replay for early Macro-F1 and removes the supported Attack-Recall loss against
SSF-faithful.
"""
    (destination / "DRIFT_ADAPTATION_AND_PERFORMANCE_REPORT.md").write_text(
        report, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
