from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dasp_nids.config import load_config
from dasp_nids.runner import StreamingExperiment


SCENARIOS = ["S0", "S1", "S3", "S5"]
STREAM_SEEDS = [9101, 9102, 9103]
MODEL_SEED = 51
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


def _maximum_collapse_streak(windows: pd.DataFrame) -> int:
    evaluated = windows.loc[~windows["warmup"].astype(bool)].copy()
    benign_denominator = evaluated["attack_tn"] + evaluated["attack_fp"]
    benign_recall = np.divide(
        evaluated["attack_tn"],
        benign_denominator,
        out=np.full(len(evaluated), np.nan, dtype=float),
        where=benign_denominator.to_numpy() > 0,
    )
    attack_recall = evaluated["attack_recall"].to_numpy(dtype=float)
    collapsed = (benign_recall <= 0.01) | (attack_recall <= 0.01)
    maximum = current = 0
    for value in collapsed:
        current = current + 1 if bool(value) else 0
        maximum = max(maximum, current)
    return maximum


def main() -> None:
    base = load_config("configs/dasp_safety_mock.yaml")
    if base.training.label_delay != 1:
        raise RuntimeError("Mock protocol requires one-window label delay")
    root = Path("outputs_dasp_safety_mock")
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    expected = len(SCENARIOS) * len(STREAM_SEEDS) * len(METHODS)
    completed = 0
    for scenario in SCENARIOS:
        for stream_seed in STREAM_SEEDS:
            for method in METHODS:
                config = copy.deepcopy(base)
                config.data.scenario = scenario
                config.data.random_seed = stream_seed
                config.data.stream_seed = stream_seed
                config.training.model_seed = MODEL_SEED
                config.dasp.baseline = method
                config.output.directory = str(root / scenario)
                config.output.run_name = (
                    f"{method}_stream{stream_seed}_model{MODEL_SEED}"
                )
                result = StreamingExperiment(config).run()
                windows = result.windows
                evaluated = windows.loc[~windows["warmup"].astype(bool)]
                run_violations: list[str] = []
                if len(evaluated) != (
                    config.data.n_windows - config.data.warmup_windows
                ):
                    run_violations.append("missing_evaluated_windows")
                if not np.isfinite(
                    evaluated[["macro_f1", "mcc", "balanced_accuracy"]]
                    .to_numpy(dtype=float)
                ).all():
                    run_violations.append("nonfinite_primary_metric")
                if evaluated["ground_truth_update_labels"].max() > (
                    config.baselines.update_label_budget
                ):
                    run_violations.append("label_budget_exceeded")
                if (
                    method in DASP_METHODS
                    and evaluated["global_update"].astype(bool).any()
                ):
                    run_violations.append("global_update_detected")
                rollback = evaluated.get(
                    "safe_rollback_exact", pd.Series(dtype=object)
                ).dropna()
                if (
                    method in {"dasp_safe", "dasp_safe_v2"}
                    and (rollback.astype(str).str.lower() == "false").any()
                ):
                    run_violations.append("inexact_rollback")
                collapse_streak = _maximum_collapse_streak(windows)
                if method == "dasp_safe_v2" and collapse_streak >= 3:
                    run_violations.append("three_window_class_collapse")
                if run_violations:
                    violations.append(
                        {
                            "scenario": scenario,
                            "stream_seed": stream_seed,
                            "method": method,
                            "violations": run_violations,
                        }
                    )
                record = dict(result.summary)
                record.update(
                    {
                        "scenario": scenario,
                        "stream_seed": stream_seed,
                        "model_seed": MODEL_SEED,
                        "method": method,
                        "run_dir": str(result.output_dir),
                        "maximum_class_collapse_streak": collapse_streak,
                        "integrity_violations": "|".join(run_violations),
                    }
                )
                records.append(record)
                completed += 1
                pd.DataFrame(records).to_csv(
                    root / "summary_by_run.partial.csv", index=False
                )
                print(
                    f"completed {completed}/{expected} "
                    f"scenario={scenario} stream={stream_seed} "
                    f"method={method} "
                    f"macro={result.summary['mean_macro_f1']:.6f} "
                    f"collapse_streak={collapse_streak}",
                    flush=True,
                )

    frame = pd.DataFrame(records)
    keys = ["scenario", "stream_seed", "model_seed", "method"]
    matrix_complete = bool(
        len(frame) == expected and not frame.duplicated(keys).any()
    )
    final_rows = frame.loc[frame["method"] == "dasp_safe_v2"]
    gate = {
        "stage": "synthetic_safety_mock",
        "real_dataset_used": False,
        "confirmation_seed_used": False,
        "label_delay": 1,
        "expected_runs": expected,
        "completed_runs": len(frame),
        "matrix_complete": matrix_complete,
        "integrity_violation_count": len(violations),
        "violations": violations,
        "dasp_safe_v2_min_cumulative_benign_recall": float(
            final_rows["cumulative_benign_recall"].min()
        ),
        "dasp_safe_v2_min_cumulative_attack_recall": float(
            final_rows["cumulative_attack_recall"].min()
        ),
    }
    gate["mock_passed"] = bool(
        matrix_complete
        and not violations
        and gate["dasp_safe_v2_min_cumulative_benign_recall"] > 0.05
        and gate["dasp_safe_v2_min_cumulative_attack_recall"] > 0.05
    )
    frame.to_csv(root / "summary_by_run.csv", index=False)
    (root / "mock_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    (root / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "stage": "synthetic_safety_mock",
                "scenarios": SCENARIOS,
                "stream_seeds": STREAM_SEEDS,
                "model_seeds": [MODEL_SEED],
                "methods": METHODS,
                "expected_runs": expected,
                "completed_runs": len(frame),
                "real_dataset_used": False,
                "confirmation_seed_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2), flush=True)
    if not gate["mock_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
