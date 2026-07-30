from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dasp_nids.config import load_config
from dasp_nids.runner import StreamingExperiment


SCENARIOS = ["S0", "S1", "S3", "S5"]
STREAM_SEEDS = [9501, 9502]
MODEL_SEED = 91
METHODS = [
    "dasp_v3_query",
    "dasp_v4_replay",
    "dasp_v4_prior",
    "dasp_v4",
    "experience_replay",
    "ssf_budget",
]
DASP_METHODS = set(METHODS[:4])


def _bool(values: pd.Series) -> pd.Series:
    return (
        values
        if values.dtype == bool
        else values.astype(str).str.lower().eq("true")
    )


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


def _paired_delta(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
) -> dict[str, float]:
    keys = ["scenario", "stream_seed", "model_seed"]
    left = frame.loc[
        frame["method"] == candidate, [*keys, metric]
    ].rename(columns={metric: "candidate"})
    right = frame.loc[
        frame["method"] == reference, [*keys, metric]
    ].rename(columns={metric: "reference"})
    paired = left.merge(right, on=keys, validate="one_to_one")
    delta = paired["candidate"] - paired["reference"]
    return {
        "n_pairs": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "minimum_delta": float(delta.min()),
        "maximum_delta": float(delta.max()),
    }


def main() -> None:
    base = load_config("configs/dasp_v4_mock.yaml")
    if base.training.label_delay != 1:
        raise RuntimeError("DASP-v4 mock requires label_delay=1")
    root = Path("outputs_dasp_v4_mock")
    root.mkdir(parents=True, exist_ok=True)
    partial_path = root / "summary_by_run.partial.csv"
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        rows = partial.to_dict(orient="records")
    else:
        rows: list[dict[str, object]] = []
    completed_keys = {
        (
            str(row["scenario"]),
            int(row["stream_seed"]),
            int(row["model_seed"]),
            str(row["method"]),
        )
        for row in rows
    }
    violations: list[dict[str, object]] = []
    expected = len(SCENARIOS) * len(STREAM_SEEDS) * len(METHODS)
    started = time.perf_counter()

    for scenario in SCENARIOS:
        for stream_seed in STREAM_SEEDS:
            for method in METHODS:
                run_key = (
                    scenario,
                    int(stream_seed),
                    int(MODEL_SEED),
                    method,
                )
                if run_key in completed_keys:
                    print(
                        f"resumed {len(rows)}/{expected} "
                        f"scenario={scenario} stream={stream_seed} "
                        f"method={method}",
                        flush=True,
                    )
                    continue
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
                metrics = evaluated[
                    ["macro_f1", "pr_auc", "mcc", "balanced_accuracy"]
                ].to_numpy(dtype=float)
                if not np.isfinite(metrics).all():
                    local.append("nonfinite_metric")
                if evaluated["ground_truth_update_labels"].max() > 64:
                    local.append("label_budget")
                if (
                    method in DASP_METHODS
                    and _bool(evaluated["global_update"]).any()
                ):
                    local.append("global_update")
                collapse = _collapse_streak(result.windows)
                if method == "dasp_v4" and collapse >= 3:
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
                completed_keys.add(run_key)
                pd.DataFrame(rows).to_csv(
                    partial_path, index=False
                )
                print(
                    f"completed {len(rows)}/{expected} "
                    f"scenario={scenario} stream={stream_seed} "
                    f"method={method} "
                    f"f1={result.summary['mean_macro_f1']:.6f} "
                    f"pr={result.summary['mean_pr_auc']:.6f} "
                    f"mcc={result.summary['mean_mcc']:.6f} "
                    f"update={result.summary['total_update_seconds']:.3f}s "
                    f"collapse={collapse}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "summary_by_run.csv", index=False)
    violations = [
        {
            "scenario": str(row.scenario),
            "stream_seed": int(row.stream_seed),
            "method": str(row.method),
            "violations": str(row.integrity_violations).split("|"),
        }
        for row in frame.itertuples()
        if isinstance(row.integrity_violations, str)
        and row.integrity_violations
    ]
    keys = ["scenario", "stream_seed", "model_seed", "method"]
    complete = bool(
        len(frame) == expected and not frame.duplicated(keys).any()
    )
    aggregates = (
        frame.groupby("method", as_index=False)
        .agg(
            macro_f1=("mean_macro_f1", "mean"),
            pr_auc=("mean_pr_auc", "mean"),
            mcc=("mean_mcc", "mean"),
            update_seconds=("total_update_seconds", "mean"),
            gradient_evaluations=("total_gradient_evaluations", "mean"),
            collapse_streak=("collapse_streak", "max"),
        )
        .sort_values("macro_f1", ascending=False)
    )
    aggregates.to_csv(root / "method_summary.csv", index=False)

    paired: dict[str, dict[str, dict[str, float]]] = {}
    for candidate in ["dasp_v4_replay", "dasp_v4_prior", "dasp_v4"]:
        paired[candidate] = {}
        for reference in ["dasp_v3_query", "experience_replay", "ssf_budget"]:
            for metric in ["mean_macro_f1", "mean_pr_auc", "mean_mcc"]:
                paired[candidate][f"{metric}_vs_{reference}"] = (
                    _paired_delta(frame, candidate, reference, metric)
                )

    v4 = aggregates.set_index("method").loc["dasp_v4"]
    v4_replay = aggregates.set_index("method").loc["dasp_v4_replay"]
    v3 = aggregates.set_index("method").loc["dasp_v3_query"]
    er = aggregates.set_index("method").loc["experience_replay"]
    cost = {
        "update_ratio_v4_v3_query": float(
            v4["update_seconds"] / max(v3["update_seconds"], 1e-12)
        ),
        "update_ratio_v4_er": float(
            v4["update_seconds"] / max(er["update_seconds"], 1e-12)
        ),
        "update_ratio_v4_replay_v3_query": float(
            v4_replay["update_seconds"]
            / max(v3["update_seconds"], 1e-12)
        ),
        "gradient_ratio_v4_v3_query": float(
            v4["gradient_evaluations"]
            / max(v3["gradient_evaluations"], 1e-12)
        ),
    }
    integrity_passed = bool(complete and not violations)
    replay_improvement_passed = bool(
        paired["dasp_v4_replay"][
            "mean_macro_f1_vs_dasp_v3_query"
        ]["mean_delta"]
        > 0.0
        and paired["dasp_v4_replay"][
            "mean_pr_auc_vs_dasp_v3_query"
        ]["mean_delta"]
        > 0.0
        and paired["dasp_v4_replay"][
            "mean_mcc_vs_dasp_v3_query"
        ]["mean_delta"]
        > 0.0
        and paired["dasp_v4_replay"][
            "mean_macro_f1_vs_dasp_v3_query"
        ]["minimum_delta"]
        >= -0.02
        and cost["update_ratio_v4_replay_v3_query"] <= 1.5
    )
    improvement_passed = bool(
        paired["dasp_v4"]["mean_macro_f1_vs_dasp_v3_query"][
            "mean_delta"
        ]
        > 0.0
        and paired["dasp_v4"]["mean_pr_auc_vs_dasp_v3_query"][
            "mean_delta"
        ]
        >= -0.01
        and paired["dasp_v4"]["mean_mcc_vs_dasp_v3_query"][
            "mean_delta"
        ]
        > 0.0
        and paired["dasp_v4"]["mean_macro_f1_vs_dasp_v3_query"][
            "minimum_delta"
        ]
        >= -0.02
        and cost["update_ratio_v4_v3_query"] <= 1.5
    )
    gate = {
        "stage": "dasp_v4_synthetic_mock",
        "expected_runs": expected,
        "completed_runs": len(frame),
        "matrix_complete": complete,
        "integrity_violation_count": len(violations),
        "violations": violations,
        "integrity_passed": integrity_passed,
        "replay_core_improvement_passed": replay_improvement_passed,
        "improvement_passed": improvement_passed,
        "paired_deltas": paired,
        **cost,
        "duration_seconds": float(time.perf_counter() - started),
        "real_dataset_used": False,
        "confirmation_seed_used": False,
        "selected_mock_candidate": (
            "dasp_v4_replay" if replay_improvement_passed else None
        ),
        "pilot_unlocked": False,
    }
    (root / "mock_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(aggregates.to_string(index=False), flush=True)
    print(json.dumps(gate, indent=2), flush=True)
    if not integrity_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
