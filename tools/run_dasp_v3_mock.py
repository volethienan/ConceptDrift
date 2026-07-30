from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dasp_nids.config import load_config
from dasp_nids.runner import StreamingExperiment


SCENARIOS = ["S0", "S1", "S3", "S5"]
STREAM_SEEDS = [9401, 9402, 9403]
MODEL_SEED = 81
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
DASP_METHODS = set(METHODS[:6])


def _bool(values: pd.Series) -> pd.Series:
    return values if values.dtype == bool else values.astype(str).str.lower().eq("true")


def _collapse_streak(windows: pd.DataFrame) -> int:
    evaluated = windows.loc[~_bool(windows["warmup"])]
    denominator = evaluated["attack_tn"] + evaluated["attack_fp"]
    benign = np.divide(
        evaluated["attack_tn"],
        denominator,
        out=np.full(len(evaluated), np.nan),
        where=denominator.to_numpy() > 0,
    )
    flags = (benign <= 0.01) | (
        evaluated["attack_recall"].to_numpy(dtype=float) <= 0.01
    )
    current = maximum = 0
    for flag in flags:
        current = current + 1 if bool(flag) else 0
        maximum = max(maximum, current)
    return maximum


def main() -> None:
    base = load_config("configs/dasp_safety_mock.yaml")
    if base.training.label_delay != 1:
        raise RuntimeError("DASP-v3 mock requires label_delay=1")
    root = Path("outputs_dasp_v3_mock")
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []
    expected = len(SCENARIOS) * len(STREAM_SEEDS) * len(METHODS)
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
                evaluated = result.windows.loc[
                    ~_bool(result.windows["warmup"])
                ]
                local: list[str] = []
                if not np.isfinite(
                    evaluated[
                        ["macro_f1", "balanced_accuracy", "mcc"]
                    ].to_numpy(dtype=float)
                ).all():
                    local.append("nonfinite_metric")
                if evaluated["ground_truth_update_labels"].max() > 64:
                    local.append("label_budget")
                if (
                    method in DASP_METHODS
                    and _bool(evaluated["global_update"]).any()
                ):
                    local.append("global_update")
                rollback = evaluated.get(
                    "safe_rollback_exact", pd.Series(dtype=object)
                ).dropna()
                if (rollback.astype(str).str.lower() == "false").any():
                    local.append("inexact_rollback")
                collapse = _collapse_streak(result.windows)
                if method == "dasp_v3" and collapse >= 3:
                    local.append("class_collapse")
                if local:
                    violations.append(
                        {
                            "scenario": scenario,
                            "stream_seed": stream_seed,
                            "method": method,
                            "violations": local,
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
                        "collapse_streak": collapse,
                        "integrity_violations": "|".join(local),
                    }
                )
                rows.append(record)
                pd.DataFrame(rows).to_csv(
                    root / "summary_by_run.partial.csv", index=False
                )
                print(
                    f"completed {len(rows)}/{expected} scenario={scenario} "
                    f"stream={stream_seed} method={method} "
                    f"macro={result.summary['mean_macro_f1']:.6f} "
                    f"collapse={collapse}",
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "summary_by_run.csv", index=False)
    keys = ["scenario", "stream_seed", "model_seed", "method"]
    complete = bool(
        len(frame) == expected and not frame.duplicated(keys).any()
    )
    v2 = frame.loc[frame["method"] == "dasp_safe_v2"]
    v3 = frame.loc[frame["method"] == "dasp_v3"]
    cost = {
        "latency_ratio_v3_v2": float(
            v3["median_update_seconds"].median()
            / max(v2["median_update_seconds"].median(), 1e-12)
        ),
        "total_update_ratio_v3_v2": float(
            v3["total_update_seconds"].sum()
            / max(v2["total_update_seconds"].sum(), 1e-12)
        ),
        "rss_ratio_v3_v2": float(
            v3["peak_rss_bytes"].mean()
            / max(v2["peak_rss_bytes"].mean(), 1.0)
        ),
        "cuda_ratio_v3_v2": float(
            v3["peak_cuda_allocated_bytes"].mean()
            / max(v2["peak_cuda_allocated_bytes"].mean(), 1.0)
        ),
    }
    gate = {
        "stage": "dasp_v3_synthetic_mock",
        "expected_runs": expected,
        "completed_runs": len(frame),
        "matrix_complete": complete,
        "integrity_violation_count": len(violations),
        "violations": violations,
        **cost,
        "real_dataset_used": False,
        "confirmation_seed_used": False,
    }
    gate["mock_passed"] = bool(
        complete
        and not violations
        and cost["total_update_ratio_v3_v2"] <= 2.0
        and cost["rss_ratio_v3_v2"] <= 1.25
        and cost["cuda_ratio_v3_v2"] <= 1.25
    )
    (root / "mock_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2), flush=True)
    if not gate["mock_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
