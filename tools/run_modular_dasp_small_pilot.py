from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path

import yaml

from dasp_nids.config import load_config
from dasp_nids.runner import StreamingExperiment

DATASETS = [
    "cic-ids2017",
    "nf-bot-iot-v3",
    "nf-cse-cic-ids2018-v3",
    "nf-ton-iot-v3",
    "nf-unsw-nb15-v3",
]

SCENARIOS = ["S1-Sudden", "S2-Gradual", "S4-Recurrent"]

METHODS = [
    "online_mlp",
    "experience_replay",
    "online_ewc",
    "agem",
    "ssf",
    "repshield",
    "cdda_md",
    "modular_dasp",
]

STREAM_SEED = 2001
MODEL_SEED = 91
PREPARED_ROOT = Path("prepared_streams_final")
OUTPUT_ROOT = Path("outputs_pilot_small")
BASE_CONFIG = Path("configs/modular_dasp_main.yaml")


def _configure(dataset: str, scenario: str, method: str):
    cfg = copy.deepcopy(load_config(BASE_CONFIG))
    stream_path = PREPARED_ROOT / dataset / scenario / f"stream_seed{STREAM_SEED}.npz"
    if not stream_path.exists():
        raise FileNotFoundError(stream_path)
    cfg.data.source = "prepared_npz"
    cfg.data.path = str(stream_path)
    cfg.data.scenario = scenario
    cfg.data.random_seed = STREAM_SEED
    cfg.data.stream_seed = STREAM_SEED
    cfg.training.model_seed = MODEL_SEED
    cfg.training.device = "cpu"
    cfg.training.num_threads = 1
    cfg.dasp.baseline = method
    cfg.output.directory = str(OUTPUT_ROOT / dataset / scenario)
    cfg.output.run_name = f"{method}_stream{STREAM_SEED}_model{MODEL_SEED}"
    return cfg


def _load_existing_summary(dataset: str, scenario: str, method: str) -> dict[str, object] | None:
    summary_path = (
        OUTPUT_ROOT
        / dataset
        / scenario
        / f"{method}_stream{STREAM_SEED}_model{MODEL_SEED}"
        / "summary.json"
    )
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _row(dataset: str, scenario: str, method: str, summary: dict[str, object]) -> dict[str, object]:
    keys = [
        "mean_macro_f1",
        "cumulative_macro_f1",
        "cumulative_attack_recall",
        "cumulative_fnr",
        "mean_online_forgetting",
        "window_stability_macro_f1",
        "total_update_seconds",
        "median_update_seconds",
        "prediction_decision_throughput_flows_per_second",
        "model_parameter_bytes",
        "memory_payload_bytes",
        "adaptation_triggers",
        "dasp_commits",
        "online_commits",
        "global_update_count",
        "total_gradient_evaluations",
        "total_ground_truth_update_labels",
    ]
    output = {
        "dataset": dataset,
        "scenario": scenario,
        "stream_seed": STREAM_SEED,
        "model_seed": MODEL_SEED,
        "method": method,
    }
    for key in keys:
        output[key] = summary.get(key)
    return output


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    total = len(DATASETS) * len(SCENARIOS) * len(METHODS)
    done = 0
    started_all = time.perf_counter()
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            for method in METHODS:
                done += 1
                cached = _load_existing_summary(dataset, scenario, method)
                if cached is not None:
                    print(f"[{done}/{total}] cached {dataset} {scenario} {method}", flush=True)
                    rows.append(_row(dataset, scenario, method, cached))
                    continue
                print(f"[{done}/{total}] running {dataset} {scenario} {method}", flush=True)
                started = time.perf_counter()
                cfg = _configure(dataset, scenario, method)
                result = StreamingExperiment(cfg).run()
                elapsed = time.perf_counter() - started
                print(
                    f"[{done}/{total}] completed {dataset} {scenario} {method} "
                    f"macro_f1={result.summary.get('mean_macro_f1')} seconds={elapsed:.1f}",
                    flush=True,
                )
                rows.append(_row(dataset, scenario, method, result.summary))
                _write_rows(rows)
    _write_rows(rows)
    (OUTPUT_ROOT / "pilot_metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "datasets": DATASETS,
                "scenarios": SCENARIOS,
                "methods": METHODS,
                "stream_seed": STREAM_SEED,
                "model_seed": MODEL_SEED,
                "total_runs": total,
                "elapsed_seconds": time.perf_counter() - started_all,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(OUTPUT_ROOT)


def _write_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = OUTPUT_ROOT / "pilot_summary.csv"
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
