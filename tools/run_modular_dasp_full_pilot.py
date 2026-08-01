from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path

import pandas as pd

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
STREAM_SEEDS = [2001, 2002, 2003]
MODEL_SEEDS = [91, 92, 93]

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

PREPARED_ROOT = Path("prepared_streams_final")
OUTPUT_ROOT = Path("outputs_full_v6_core")
BASE_CONFIG = Path("configs/modular_dasp_main.yaml")


SUMMARY_KEYS = [
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
    "peak_rss_bytes",
    "run_start_rss_bytes",
    "peak_rss_increment_bytes",
    "adaptation_triggers",
    "dasp_commits",
    "online_commits",
    "global_update_count",
    "total_gradient_evaluations",
    "total_ground_truth_update_labels",
]


def _configure(dataset: str, scenario: str, stream_seed: int, model_seed: int, method: str):
    cfg = copy.deepcopy(load_config(BASE_CONFIG))
    stream_path = PREPARED_ROOT / dataset / scenario / f"stream_seed{stream_seed}.npz"
    if not stream_path.exists():
        raise FileNotFoundError(stream_path)
    cfg.data.source = "prepared_npz"
    cfg.data.path = str(stream_path)
    cfg.data.scenario = scenario
    cfg.data.random_seed = stream_seed
    cfg.data.stream_seed = stream_seed
    cfg.training.model_seed = model_seed
    cfg.training.device = "cpu"
    cfg.training.num_threads = 1
    cfg.dasp.baseline = method
    cfg.output.directory = str(OUTPUT_ROOT / dataset / scenario)
    cfg.output.run_name = f"{method}_stream{stream_seed}_model{model_seed}"
    return cfg


def _summary_path(dataset: str, scenario: str, stream_seed: int, model_seed: int, method: str) -> Path:
    return (
        OUTPUT_ROOT
        / dataset
        / scenario
        / f"{method}_stream{stream_seed}_model{model_seed}"
        / "summary.json"
    )


def _load_existing_summary(
    dataset: str, scenario: str, stream_seed: int, model_seed: int, method: str
) -> dict[str, object] | None:
    path = _summary_path(dataset, scenario, stream_seed, model_seed, method)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row(
    dataset: str,
    scenario: str,
    stream_seed: int,
    model_seed: int,
    method: str,
    summary: dict[str, object],
    elapsed_seconds: float | None,
) -> dict[str, object]:
    output = {
        "dataset": dataset,
        "scenario": scenario,
        "stream_seed": stream_seed,
        "model_seed": model_seed,
        "method": method,
        "elapsed_seconds": elapsed_seconds,
    }
    for key in SUMMARY_KEYS:
        output[key] = summary.get(key)
    return output


def _mean_sem(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return math.nan, math.nan
    sem = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(sem)


def _write_analysis(rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_ROOT / "full_summary.csv", index=False)

    method_rows: list[dict[str, object]] = []
    for method, group in df.groupby("method", sort=False):
        macro_mean, macro_sem = _mean_sem(group["mean_macro_f1"])
        update_mean, update_sem = _mean_sem(group["total_update_seconds"])
        throughput_mean, throughput_sem = _mean_sem(
            group["prediction_decision_throughput_flows_per_second"]
        )
        grad_mean, grad_sem = _mean_sem(group["total_gradient_evaluations"])
        memory_mean, memory_sem = _mean_sem(group["memory_payload_bytes"])
        rss_mean, rss_sem = _mean_sem(group["peak_rss_increment_bytes"])
        method_rows.append(
            {
                "method": method,
                "runs": len(group),
                "mean_macro_f1": macro_mean,
                "sem_macro_f1": macro_sem,
                "mean_update_seconds": update_mean,
                "sem_update_seconds": update_sem,
                "mean_throughput_flows_per_second": throughput_mean,
                "sem_throughput_flows_per_second": throughput_sem,
                "mean_gradient_evaluations": grad_mean,
                "sem_gradient_evaluations": grad_sem,
                "mean_memory_payload_bytes": memory_mean,
                "sem_memory_payload_bytes": memory_sem,
                "mean_peak_rss_increment_bytes": rss_mean,
                "sem_peak_rss_increment_bytes": rss_sem,
                "mean_global_update_count": float(
                    pd.to_numeric(group["global_update_count"], errors="coerce").fillna(0).mean()
                ),
            }
        )
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(OUTPUT_ROOT / "method_summary.csv", index=False)

    key_cols = ["dataset", "scenario", "stream_seed", "model_seed"]
    dasp = df[df["method"] == "modular_dasp"]
    ssf = df[df["method"] == "ssf"]
    paired = dasp.merge(ssf, on=key_cols, suffixes=("_modular_dasp", "_ssf"))
    paired["macro_f1_gap_vs_ssf"] = (
        paired["mean_macro_f1_modular_dasp"] - paired["mean_macro_f1_ssf"]
    )
    paired["update_speedup_vs_ssf"] = (
        paired["total_update_seconds_ssf"] / paired["total_update_seconds_modular_dasp"]
    )
    paired["gradient_reduction_vs_ssf"] = (
        paired["total_gradient_evaluations_ssf"]
        / paired["total_gradient_evaluations_modular_dasp"]
    )
    paired.to_csv(OUTPUT_ROOT / "modular_dasp_vs_ssf.csv", index=False)
    if paired.empty:
        analysis = {
            "total_runs": int(len(df)),
            "expected_runs": len(DATASETS)
            * len(SCENARIOS)
            * len(STREAM_SEEDS)
            * len(MODEL_SEEDS)
            * len(METHODS),
            "paired_modular_dasp_vs_ssf_runs": 0,
        }
        (OUTPUT_ROOT / "analysis.json").write_text(
            json.dumps(analysis, indent=2), encoding="utf-8"
        )
        return

    dataset_summary = (
        paired.groupby("dataset")
        .agg(
            mean_macro_f1_modular_dasp=("mean_macro_f1_modular_dasp", "mean"),
            mean_macro_f1_ssf=("mean_macro_f1_ssf", "mean"),
            mean_gap_vs_ssf=("macro_f1_gap_vs_ssf", "mean"),
            wins_vs_ssf=("macro_f1_gap_vs_ssf", lambda values: int((values > 0).sum())),
            mean_update_speedup_vs_ssf=("update_speedup_vs_ssf", "mean"),
            mean_gradient_reduction_vs_ssf=("gradient_reduction_vs_ssf", "mean"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(OUTPUT_ROOT / "dataset_summary_vs_ssf.csv", index=False)

    scenario_summary = (
        paired.groupby("scenario")
        .agg(
            mean_macro_f1_modular_dasp=("mean_macro_f1_modular_dasp", "mean"),
            mean_macro_f1_ssf=("mean_macro_f1_ssf", "mean"),
            mean_gap_vs_ssf=("macro_f1_gap_vs_ssf", "mean"),
            wins_vs_ssf=("macro_f1_gap_vs_ssf", lambda values: int((values > 0).sum())),
            mean_update_speedup_vs_ssf=("update_speedup_vs_ssf", "mean"),
            mean_gradient_reduction_vs_ssf=("gradient_reduction_vs_ssf", "mean"),
        )
        .reset_index()
    )
    scenario_summary.to_csv(OUTPUT_ROOT / "scenario_summary_vs_ssf.csv", index=False)

    analysis = {
        "total_runs": int(len(df)),
        "expected_runs": len(DATASETS)
        * len(SCENARIOS)
        * len(STREAM_SEEDS)
        * len(MODEL_SEEDS)
        * len(METHODS),
        "paired_modular_dasp_vs_ssf_runs": int(len(paired)),
        "modular_dasp_mean_macro_f1": float(paired["mean_macro_f1_modular_dasp"].mean()),
        "ssf_mean_macro_f1": float(paired["mean_macro_f1_ssf"].mean()),
        "mean_gap_vs_ssf": float(paired["macro_f1_gap_vs_ssf"].mean()),
        "wins_vs_ssf": int((paired["macro_f1_gap_vs_ssf"] > 0).sum()),
        "within_0_01_vs_ssf": int((paired["macro_f1_gap_vs_ssf"].abs() <= 0.01).sum()),
        "mean_update_speedup_vs_ssf": float(paired["update_speedup_vs_ssf"].mean()),
        "mean_gradient_reduction_vs_ssf": float(paired["gradient_reduction_vs_ssf"].mean()),
        "modular_dasp_global_update_sum": int(
            pd.to_numeric(dasp["global_update_count"], errors="coerce").fillna(0).sum()
        ),
        "worst_gap_vs_ssf": float(paired["macro_f1_gap_vs_ssf"].min()),
        "worst_case": paired.loc[
            paired["macro_f1_gap_vs_ssf"].idxmin(), key_cols
        ].to_dict(),
    }
    (OUTPUT_ROOT / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    total = len(DATASETS) * len(SCENARIOS) * len(STREAM_SEEDS) * len(MODEL_SEEDS) * len(METHODS)
    done = 0
    started_all = time.perf_counter()
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            for stream_seed in STREAM_SEEDS:
                for model_seed in MODEL_SEEDS:
                    for method in METHODS:
                        done += 1
                        cached = _load_existing_summary(
                            dataset, scenario, stream_seed, model_seed, method
                        )
                        if cached is not None:
                            print(
                                f"[{done}/{total}] cached {dataset} {scenario} stream={stream_seed} model={model_seed} {method}",
                                flush=True,
                            )
                            rows.append(
                                _row(
                                    dataset,
                                    scenario,
                                    stream_seed,
                                    model_seed,
                                    method,
                                    cached,
                                    None,
                                )
                            )
                            continue
                        cfg = _configure(dataset, scenario, stream_seed, model_seed, method)
                        print(
                            f"[{done}/{total}] running {dataset} {scenario} stream={stream_seed} model={model_seed} {method}",
                            flush=True,
                        )
                        started = time.perf_counter()
                        result = StreamingExperiment(cfg).run()
                        elapsed = time.perf_counter() - started
                        rows.append(
                            _row(
                                dataset,
                                scenario,
                                stream_seed,
                                model_seed,
                                method,
                                result.summary,
                                elapsed,
                            )
                        )
                        _write_analysis(rows)
                        print(
                            f"[{done}/{total}] done {method} macro={result.summary.get('mean_macro_f1'):.4f} update_s={result.summary.get('total_update_seconds'):.3f} elapsed_s={elapsed:.1f}",
                            flush=True,
                        )
    _write_analysis(rows)
    print(f"completed {len(rows)}/{total} in {time.perf_counter() - started_all:.1f}s")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
