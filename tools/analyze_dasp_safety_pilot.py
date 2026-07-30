from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOTS = {
    "nf-unsw-nb15-v3": Path("outputs_dasp_safety_pilot_unsw"),
    "nf-cse-cic-ids2018-v3": Path("outputs_dasp_safety_pilot_cse"),
    "nf-ton-iot-v3": Path("outputs_dasp_safety_pilot_ton"),
    "nf-bot-iot-v3": Path("outputs_dasp_safety_pilot_bot"),
    "cic-ids2017": Path("outputs_dasp_safety_pilot_cic"),
}
V2_ROOTS = {
    "nf-unsw-nb15-v3": Path("outputs_dasp_safety_v2_pilot_unsw"),
    "nf-cse-cic-ids2018-v3": Path("outputs_dasp_safety_v2_pilot_cse"),
    "nf-ton-iot-v3": Path("outputs_dasp_safety_v2_pilot_ton"),
    "nf-bot-iot-v3": Path("outputs_dasp_safety_v2_pilot_bot"),
    "cic-ids2017": Path("outputs_dasp_safety_v2_pilot_cic"),
}
METHODS = [
    "dasp",
    "dasp_safe_query",
    "dasp_safe_replay",
    "dasp_safe_oneclass",
    "dasp_safe",
    "dasp_safe_v2",
    "ssf_budget",
    "experience_replay",
]
DASP_METHODS = set(METHODS[:6])
KEYS = ["dataset", "scenario", "stream_seed", "model_seed"]
PRIMARY = ["mean_macro_f1", "mean_attack_recall", "mean_pr_auc", "mean_mcc"]


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def _collapse_streak(frame: pd.DataFrame) -> int:
    evaluated = frame.loc[~_bool_series(frame["warmup"])].copy()
    denominator = evaluated["attack_tn"] + evaluated["attack_fp"]
    benign = np.divide(
        evaluated["attack_tn"],
        denominator,
        out=np.full(len(evaluated), np.nan, dtype=float),
        where=denominator.to_numpy() > 0,
    )
    attack = evaluated["attack_recall"].to_numpy(dtype=float)
    flags = (benign <= 0.01) | (attack <= 0.01)
    current = maximum = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        maximum = max(maximum, current)
    return maximum


def _episode_metrics(frame: pd.DataFrame, scenario: str) -> dict[str, float]:
    evaluated = frame.loc[~_bool_series(frame["warmup"])].copy()
    pre = evaluated.loc[
        (evaluated["window"] >= 4) & (evaluated["window"] <= 11),
        "macro_f1",
    ]
    post = evaluated.loc[
        (evaluated["window"] >= 12) & (evaluated["window"] <= 16),
        "macro_f1",
    ]
    result = {
        "pre_drift_macro_f1": float(pre.mean()) if len(pre) else np.nan,
        "post_drift_macro_f1_at5": (
            float(post.mean()) if len(post) else np.nan
        ),
        "performance_drop_at5": (
            float(pre.mean() - post.min())
            if len(pre) and len(post)
            else np.nan
        ),
        "recovery_windows_90pct": np.nan,
        "detection_delay": np.nan,
        "missed_drift": False,
        "s0_false_alarms": 0,
    }
    if scenario == "S0":
        result["s0_false_alarms"] = int(
            _bool_series(evaluated["triggered"]).sum()
        )
        return result
    threshold = 0.9 * float(pre.mean()) if len(pre) else np.nan
    recovery = evaluated.loc[
        (evaluated["window"] >= 12)
        & (evaluated["macro_f1"] >= threshold),
        "window",
    ]
    if len(recovery):
        result["recovery_windows_90pct"] = float(
            int(recovery.iloc[0]) - 12
        )
    alarms = evaluated.loc[
        (evaluated["window"] >= 12)
        & _bool_series(evaluated["triggered"]),
        "window",
    ]
    if len(alarms):
        result["detection_delay"] = float(int(alarms.iloc[0]) - 12)
    else:
        result["missed_drift"] = True
    return result


def _paired_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    generator = np.random.default_rng(seed)
    draws = generator.choice(
        values, size=(10000, len(values)), replace=True
    ).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(
        np.quantile(draws, 0.975)
    )


def main() -> None:
    destination = Path("outputs_dasp_safety_v2_pilot_combined")
    destination.mkdir(parents=True, exist_ok=True)
    summary_parts: list[pd.DataFrame] = []
    for dataset, root in ROOTS.items():
        frame = pd.read_csv(root / "summary_by_run.csv")
        if set(frame["dataset"]) != {dataset}:
            raise ValueError(f"Dataset mismatch in {root}")
        summary_parts.append(frame)
        v2_frame = pd.read_csv(V2_ROOTS[dataset] / "summary_by_run.csv")
        if set(v2_frame["dataset"]) != {dataset}:
            raise ValueError(f"Dataset mismatch in {V2_ROOTS[dataset]}")
        summary_parts.append(v2_frame)
    runs = pd.concat(summary_parts, ignore_index=True)
    runs.to_csv(destination / "summary_by_run.csv", index=False)

    violations: list[dict[str, object]] = []
    expected = 5 * 10 * 3 * len(METHODS)
    if len(runs) != expected:
        violations.append(
            {"type": "matrix_size", "observed": len(runs), "expected": expected}
        )
    if runs.duplicated(KEYS + ["method"]).any():
        violations.append({"type": "duplicate_run_cell"})
    if set(runs["method"]) != set(METHODS):
        violations.append({"type": "method_set"})
    if not np.isfinite(runs[PRIMARY].to_numpy(dtype=float)).all():
        violations.append({"type": "nonfinite_primary_summary"})

    diagnostics: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    for row in runs.itertuples(index=False):
        run_dir = Path(row.run_dir)
        windows = pd.read_csv(run_dir / "windows.csv")
        config = yaml.safe_load(
            (run_dir / "config.yaml").read_text(encoding="utf-8")
        )
        local: list[str] = []
        if config["training"]["label_delay"] != 1:
            local.append("label_delay")
        if len(windows) != 36 or windows["window"].tolist() != list(range(36)):
            local.append("window_order")
        evaluated = windows.loc[~_bool_series(windows["warmup"])]
        if evaluated["ground_truth_update_labels"].max() > 256:
            local.append("label_budget")
        if (
            row.method in DASP_METHODS
            and _bool_series(evaluated["global_update"]).any()
        ):
            local.append("global_update")
        rollback = evaluated.get(
            "safe_rollback_exact", pd.Series(dtype=object)
        ).dropna()
        if (
            row.method in {"dasp_safe", "dasp_safe_v2"}
            and (rollback.astype(str).str.lower() == "false").any()
        ):
            local.append("rollback")
        collapse = _collapse_streak(windows)
        if row.method == "dasp_safe_v2" and collapse >= 3:
            local.append("class_collapse")
        if local:
            violations.append(
                {
                    "type": "run_integrity",
                    "dataset": row.dataset,
                    "scenario": row.scenario,
                    "stream_seed": int(row.stream_seed),
                    "method": row.method,
                    "violations": local,
                }
            )
        oneclass_skips = int(
            evaluated["selected_candidate"]
            .astype(str)
            .str.contains("oneclass")
            .sum()
        )
        query_oneclass = int(
            _bool_series(
                evaluated.get(
                    "safe_oneclass_revealed",
                    pd.Series(False, index=evaluated.index),
                ).fillna(False)
            ).sum()
        )
        safe_gate_passes = int(
            _bool_series(
                evaluated.get(
                    "safe_gate_passed",
                    pd.Series(False, index=evaluated.index),
                ).fillna(False)
            ).sum()
        )
        safe_rollbacks = int(
            _bool_series(
                evaluated.get(
                    "safe_rollback_exact",
                    pd.Series(False, index=evaluated.index),
                ).fillna(False)
            ).sum()
        )
        replay_minority_fractions: list[float] = []
        for value in evaluated.get(
            "v17_iwms_selected_per_class",
            pd.Series(dtype=object),
        ).dropna():
            counts = [
                int(item)
                for item in str(value).split("|")
                if str(item).strip()
            ]
            if counts and sum(counts):
                replay_minority_fractions.append(
                    min(counts) / sum(counts)
                )
        diagnostics.append(
            {
                **{key: getattr(row, key) for key in KEYS},
                "method": row.method,
                "collapse_streak": collapse,
                "oneclass_skips": oneclass_skips,
                "query_oneclass_reveals": query_oneclass,
                "safe_gate_passes": safe_gate_passes,
                "safe_rollbacks": safe_rollbacks,
                "mean_replay_minority_fraction": (
                    float(np.mean(replay_minority_fractions))
                    if replay_minority_fractions
                    else np.nan
                ),
                "mean_attack_rate": float(evaluated["attack_rate"].mean()),
                "detector_triggers": int(
                    _bool_series(evaluated["triggered"]).sum()
                ),
                "fast_warnings": int(
                    _bool_series(
                        evaluated.get(
                            "fast_update_warning",
                            pd.Series(False, index=evaluated.index),
                        ).fillna(False)
                    ).sum()
                ),
                "adapter_commits": int(
                    evaluated["accepted"].astype(str).isin(
                        ["dasp", "light"]
                    ).sum()
                ),
                "rollback_checks": int(len(rollback)),
                "integrity_violations": "|".join(local),
            }
        )
        episode_rows.append(
            {
                **{key: getattr(row, key) for key in KEYS},
                "method": row.method,
                **_episode_metrics(windows, row.scenario),
            }
        )
    diagnostic_frame = pd.DataFrame(diagnostics)
    episode_frame = pd.DataFrame(episode_rows)
    diagnostic_frame.to_csv(
        destination / "run_integrity_and_collapse.csv", index=False
    )
    (
        diagnostic_frame.groupby(["dataset", "method"], as_index=False)
        .agg(
            collapse_runs=(
                "collapse_streak",
                lambda values: int((values >= 3).sum()),
            ),
            oneclass_skips=("oneclass_skips", "sum"),
            query_oneclass_reveals=("query_oneclass_reveals", "sum"),
            safe_gate_passes=("safe_gate_passes", "sum"),
            safe_rollbacks=("safe_rollbacks", "sum"),
            mean_replay_minority_fraction=(
                "mean_replay_minority_fraction",
                "mean",
            ),
            mean_attack_rate=("mean_attack_rate", "mean"),
            detector_triggers=("detector_triggers", "sum"),
            fast_warnings=("fast_warnings", "sum"),
            adapter_commits=("adapter_commits", "sum"),
        )
        .to_csv(destination / "mechanism_diagnostics.csv", index=False)
    )
    episode_frame.to_csv(
        destination / "episode_performance.csv", index=False
    )

    aggregate = (
        runs.groupby(["dataset", "method"], as_index=False)
        .agg(
            macro_f1=("mean_macro_f1", "mean"),
            attack_recall=("mean_attack_recall", "mean"),
            pr_auc=("mean_pr_auc", "mean"),
            mcc=("mean_mcc", "mean"),
            fnr=("mean_fnr", "mean"),
            benign_recall_min=(
                "cumulative_benign_recall",
                "min",
            ),
            attack_recall_min=(
                "cumulative_attack_recall",
                "min",
            ),
            median_update_seconds=("median_update_seconds", "median"),
            peak_rss_bytes=("peak_rss_bytes", "mean"),
        )
        .merge(
            episode_frame.groupby(["dataset", "method"], as_index=False).agg(
                post_drift_macro_f1_at5=(
                    "post_drift_macro_f1_at5",
                    "mean",
                ),
                performance_drop_at5=("performance_drop_at5", "mean"),
                recovery_windows_90pct=(
                    "recovery_windows_90pct",
                    "mean",
                ),
                detection_delay=("detection_delay", "mean"),
                missed_drift_rate=("missed_drift", "mean"),
                s0_false_alarms=("s0_false_alarms", "sum"),
            ),
            on=["dataset", "method"],
            how="left",
        )
    )
    aggregate.to_csv(destination / "metrics_by_dataset_method.csv", index=False)

    paired_rows: list[dict[str, object]] = []
    for dataset in ROOTS:
        subset = runs.loc[runs["dataset"] == dataset]
        pivot = subset.pivot(index=KEYS, columns="method", values="mean_macro_f1")
        for method in METHODS:
            if method == "dasp_safe_v2":
                continue
            delta = (
                pivot["dasp_safe_v2"] - pivot[method]
            ).to_numpy(dtype=float)
            low, high = _paired_ci(delta, 9317 + METHODS.index(method))
            paired_rows.append(
                {
                    "dataset": dataset,
                    "comparator": method,
                    "n_cells": len(delta),
                    "mean_delta_macro_f1": float(np.mean(delta)),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(destination / "paired_macro_f1.csv", index=False)

    gate_rows: list[dict[str, object]] = []
    for dataset in ROOTS:
        table = aggregate.loc[aggregate["dataset"] == dataset].set_index(
            "method"
        )
        stronger = max(
            float(table.loc["ssf_budget", "macro_f1"]),
            float(table.loc["experience_replay", "macro_f1"]),
        )
        safe = float(table.loc["dasp_safe_v2", "macro_f1"])
        collapses = int(
            (
                (diagnostic_frame["dataset"] == dataset)
                & (diagnostic_frame["method"] == "dasp_safe_v2")
                & (diagnostic_frame["collapse_streak"] >= 3)
            ).sum()
        )
        row = {
            "dataset": dataset,
            "dasp_safe_macro_f1": safe,
            "stronger_ssf_er_macro_f1": stronger,
            "noninferiority_delta": safe - stronger,
            "noninferior_margin_0_03": safe - stronger >= -0.03,
            "collapse_runs": collapses,
            "min_benign_recall": float(
                table.loc["dasp_safe_v2", "benign_recall_min"]
            ),
            "min_attack_recall": float(
                table.loc["dasp_safe_v2", "attack_recall_min"]
            ),
        }
        row["dataset_passed"] = bool(
            row["noninferior_margin_0_03"]
            and collapses == 0
            and row["min_benign_recall"] > 0.05
            and row["min_attack_recall"] > 0.05
        )
        gate_rows.append(row)
    gate_frame = pd.DataFrame(gate_rows)
    gate_frame.to_csv(destination / "advancement_by_dataset.csv", index=False)
    gate = {
        "stage": "five_dataset_development_pilot",
        "expected_runs": expected,
        "completed_runs": len(runs),
        "integrity_violation_count": len(violations),
        "violations": violations,
        "all_datasets_passed": bool(gate_frame["dataset_passed"].all()),
        "confirmation_locked": True,
        "confirmation_run_executed": False,
    }
    gate["pilot_passed"] = bool(
        len(runs) == expected
        and not violations
        and gate["all_datasets_passed"]
    )
    (destination / "pilot_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
