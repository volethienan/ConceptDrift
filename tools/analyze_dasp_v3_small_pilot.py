from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path("outputs_dasp_v3_small_pilot")
DATASETS = [
    "nf-unsw-nb15-v3",
    "nf-cse-cic-ids2018-v3",
    "nf-ton-iot-v3",
    "nf-bot-iot-v3",
    "cic-ids2017",
]
SCENARIOS = ["S0", "S1-Sudden", "S2-Gradual"]
METHODS = [
    "dasp_safe_v2",
    "dasp_v3_memory",
    "dasp_v3_query",
    "dasp_v3_backbone",
    "dasp_v3_checkpoint",
    "dasp_v3",
    "ssf_budget",
    "experience_replay",
]
V3_METHODS = set(METHODS[:6])
PRIMARY = [
    "mean_macro_f1",
    "mean_attack_recall",
    "mean_pr_auc",
    "mean_mcc",
]


def _bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq("true")


def _collapse_streak(frame: pd.DataFrame) -> int:
    evaluated = frame.loc[~_bool(frame["warmup"])]
    denominator = evaluated["attack_tn"] + evaluated["attack_fp"]
    benign_recall = np.divide(
        evaluated["attack_tn"],
        denominator,
        out=np.full(len(evaluated), np.nan),
        where=denominator.to_numpy() > 0,
    )
    flags = (benign_recall <= 0.01) | (
        evaluated["attack_recall"].to_numpy(dtype=float) <= 0.01
    )
    current = maximum = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        maximum = max(maximum, current)
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="outputs_dasp_v3_small_pilot"
    )
    args = parser.parse_args()
    root = Path(args.root)
    runs = pd.read_csv(root / "summary_by_run.csv")
    expected = len(DATASETS) * len(SCENARIOS) * len(METHODS)
    violations: list[dict[str, object]] = []
    keys = ["dataset", "scenario", "stream_seed", "model_seed", "method"]
    if len(runs) != expected:
        violations.append(
            {"type": "matrix_size", "expected": expected, "observed": len(runs)}
        )
    if runs.duplicated(keys).any():
        violations.append({"type": "duplicate_cell"})
    if set(runs["dataset"]) != set(DATASETS):
        violations.append({"type": "dataset_set"})
    if set(runs["scenario"]) != set(SCENARIOS):
        violations.append({"type": "scenario_set"})
    if set(runs["method"]) != set(METHODS):
        violations.append({"type": "method_set"})
    if not np.isfinite(runs[PRIMARY].to_numpy(dtype=float)).all():
        violations.append({"type": "nonfinite_primary_summary"})

    diagnostics: list[dict[str, object]] = []
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
        evaluated = windows.loc[~_bool(windows["warmup"])]
        if evaluated["ground_truth_update_labels"].max() > 256:
            local.append("label_budget")
        if row.method in V3_METHODS and _bool(evaluated["global_update"]).any():
            local.append("global_update")
        rollback = evaluated.get(
            "safe_rollback_exact", pd.Series(dtype=object)
        ).dropna()
        if (rollback.astype(str).str.lower() == "false").any():
            local.append("inexact_rollback")
        collapse = _collapse_streak(windows)
        if row.method == "dasp_v3" and collapse >= 3:
            local.append("class_collapse")
        if local:
            violations.append(
                {
                    "type": "run_integrity",
                    "dataset": row.dataset,
                    "scenario": row.scenario,
                    "method": row.method,
                    "violations": local,
                }
            )
        diagnostics.append(
            {
                "dataset": row.dataset,
                "scenario": row.scenario,
                "method": row.method,
                "collapse_streak": collapse,
                "integrity_violations": "|".join(local),
                "checkpoint_restores": int(
                    _bool(
                        evaluated.get(
                            "v3_checkpoint_restored",
                            pd.Series(False, index=evaluated.index),
                        ).fillna(False)
                    ).sum()
                ),
                "calibration_updates": int(
                    _bool(
                        evaluated.get(
                            "v3_calibration_applied",
                            pd.Series(False, index=evaluated.index),
                        ).fillna(False)
                    ).sum()
                ),
            }
        )
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_csv(
        root / "run_integrity_and_collapse.csv", index=False
    )

    aggregate = (
        runs.groupby(["dataset", "method"], as_index=False)
        .agg(
            macro_f1=("mean_macro_f1", "mean"),
            attack_recall=("mean_attack_recall", "mean"),
            pr_auc=("mean_pr_auc", "mean"),
            mcc=("mean_mcc", "mean"),
            fnr=("mean_fnr", "mean"),
            total_update_seconds=("total_update_seconds", "sum"),
            peak_rss_bytes=("peak_rss_bytes", "mean"),
            peak_cuda_bytes=("peak_cuda_allocated_bytes", "mean"),
        )
    )
    aggregate.to_csv(root / "metrics_by_dataset_method.csv", index=False)

    gates: list[dict[str, object]] = []
    for dataset in DATASETS:
        table = aggregate.loc[aggregate["dataset"] == dataset].set_index(
            "method"
        )
        stronger = max(
            float(table.loc["ssf_budget", "macro_f1"]),
            float(table.loc["experience_replay", "macro_f1"]),
        )
        v3_macro = float(table.loc["dasp_v3", "macro_f1"])
        update_ratio = float(
            table.loc["dasp_v3", "total_update_seconds"]
            / max(
                float(table.loc["dasp_safe_v2", "total_update_seconds"]),
                1e-12,
            )
        )
        rss_ratio = float(
            table.loc["dasp_v3", "peak_rss_bytes"]
            / max(float(table.loc["dasp_safe_v2", "peak_rss_bytes"]), 1.0)
        )
        cuda_ratio = float(
            table.loc["dasp_v3", "peak_cuda_bytes"]
            / max(float(table.loc["dasp_safe_v2", "peak_cuda_bytes"]), 1.0)
        )
        collapse_runs = int(
            (
                (diagnostic_frame["dataset"] == dataset)
                & (diagnostic_frame["method"] == "dasp_v3")
                & (diagnostic_frame["collapse_streak"] >= 3)
            ).sum()
        )
        row = {
            "dataset": dataset,
            "dasp_v3_macro_f1": v3_macro,
            "stronger_ssf_er_macro_f1": stronger,
            "noninferiority_delta": v3_macro - stronger,
            "noninferior_margin_0_05": v3_macro - stronger >= -0.05,
            "collapse_runs": collapse_runs,
            "total_update_ratio_v3_v2": update_ratio,
            "rss_ratio_v3_v2": rss_ratio,
            "cuda_ratio_v3_v2": cuda_ratio,
        }
        row["dataset_passed"] = bool(
            row["noninferior_margin_0_05"]
            and collapse_runs == 0
            and update_ratio <= 2.0
            and rss_ratio <= 1.25
            and cuda_ratio <= 1.25
        )
        gates.append(row)
    gate_frame = pd.DataFrame(gates)
    gate_frame.to_csv(root / "advancement_by_dataset.csv", index=False)
    gate = {
        "stage": "dasp_v3_small_real_data_pilot",
        "expected_runs": expected,
        "completed_runs": len(runs),
        "integrity_violation_count": len(violations),
        "violations": violations,
        "all_datasets_passed": bool(gate_frame["dataset_passed"].all()),
        "full_1200_unlocked": False,
        "confirmation_4000_locked": True,
    }
    gate["small_pilot_passed"] = bool(
        len(runs) == expected
        and not violations
        and gate["all_datasets_passed"]
    )
    gate["full_1200_unlocked"] = gate["small_pilot_passed"]
    (root / "small_pilot_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(gate_frame.to_string(index=False))
    print(json.dumps(gate, indent=2))
    if not gate["small_pilot_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
