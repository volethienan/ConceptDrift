from __future__ import annotations

import argparse
import copy
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

PRIMARY_SCENARIOS = [
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
]

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

FULL_PILOT_STREAM_SEEDS = [2001, 2002, 2003]
FULL_PILOT_MODEL_SEEDS = [91]
CONFIRMATION_STREAM_SEEDS = [2001, 2002, 2003, 2101, 2102]
CONFIRMATION_MODEL_SEEDS = [91, 92]

PREPARED_ROOT = Path("prepared_streams_final")
BASE_CONFIG = Path("configs/modular_dasp_main.yaml")
CORE_REUSE_ROOT = Path("outputs_full_v6_core")
FULL_PILOT_ROOT = Path("outputs_full_v6_10scenario_pilot")
CONFIRMATION_ROOT = Path("outputs_confirmation_v6_10scenario")

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


def _mode_spec(mode: str) -> tuple[Path, list[int], list[int], list[Path]]:
    if mode == "full_pilot":
        return (
            FULL_PILOT_ROOT,
            FULL_PILOT_STREAM_SEEDS,
            FULL_PILOT_MODEL_SEEDS,
            [CORE_REUSE_ROOT],
        )
    if mode == "confirmation":
        return (
            CONFIRMATION_ROOT,
            CONFIRMATION_STREAM_SEEDS,
            CONFIRMATION_MODEL_SEEDS,
            [FULL_PILOT_ROOT, CORE_REUSE_ROOT],
        )
    raise ValueError(f"unknown mode: {mode}")


def _run_name(method: str, stream_seed: int, model_seed: int) -> str:
    return f"{method}_stream{stream_seed}_model{model_seed}"


def _summary_path(
    root: Path,
    dataset: str,
    scenario: str,
    stream_seed: int,
    model_seed: int,
    method: str,
) -> Path:
    return root / dataset / scenario / _run_name(method, stream_seed, model_seed) / "summary.json"


def _load_summary(
    roots: list[Path],
    dataset: str,
    scenario: str,
    stream_seed: int,
    model_seed: int,
    method: str,
) -> tuple[dict[str, object], Path] | None:
    for root in roots:
        path = _summary_path(root, dataset, scenario, stream_seed, model_seed, method)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path
    return None


def _configure(
    output_root: Path,
    dataset: str,
    scenario: str,
    stream_seed: int,
    model_seed: int,
    method: str,
):
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
    cfg.output.directory = str(output_root / dataset / scenario)
    cfg.output.run_name = _run_name(method, stream_seed, model_seed)
    return cfg


def _row(
    dataset: str,
    scenario: str,
    stream_seed: int,
    model_seed: int,
    method: str,
    summary: dict[str, object],
    elapsed_seconds: float | None,
    source: str,
) -> dict[str, object]:
    output = {
        "dataset": dataset,
        "scenario": scenario,
        "stream_seed": stream_seed,
        "model_seed": model_seed,
        "method": method,
        "elapsed_seconds": elapsed_seconds,
        "summary_source": source,
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


def _write_analysis(
    output_root: Path,
    rows: list[dict[str, object]],
    expected_runs: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "full_summary.csv", index=False)
    if df.empty:
        return

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
                    pd.to_numeric(group["global_update_count"], errors="coerce")
                    .fillna(0)
                    .mean()
                ),
            }
        )
    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(output_root / "method_summary.csv", index=False)

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
    paired.to_csv(output_root / "modular_dasp_vs_ssf.csv", index=False)

    if paired.empty:
        analysis = {
            "total_runs": int(len(df)),
            "expected_runs": int(expected_runs),
            "paired_modular_dasp_vs_ssf_runs": 0,
        }
        (output_root / "analysis.json").write_text(
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
    dataset_summary.to_csv(output_root / "dataset_summary_vs_ssf.csv", index=False)

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
    scenario_summary.to_csv(output_root / "scenario_summary_vs_ssf.csv", index=False)

    analysis = {
        "total_runs": int(len(df)),
        "expected_runs": int(expected_runs),
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
    (output_root / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )


def run_protocol(mode: str) -> None:
    output_root, stream_seeds, model_seeds, reuse_roots = _mode_spec(mode)
    output_root.mkdir(parents=True, exist_ok=True)
    lookup_roots = [output_root, *reuse_roots]
    rows: list[dict[str, object]] = []
    total = (
        len(DATASETS)
        * len(PRIMARY_SCENARIOS)
        * len(stream_seeds)
        * len(model_seeds)
        * len(METHODS)
    )
    done = 0
    started_all = time.perf_counter()
    for dataset in DATASETS:
        for scenario in PRIMARY_SCENARIOS:
            for stream_seed in stream_seeds:
                for model_seed in model_seeds:
                    for method in METHODS:
                        done += 1
                        cached = _load_summary(
                            lookup_roots,
                            dataset,
                            scenario,
                            stream_seed,
                            model_seed,
                            method,
                        )
                        if cached is not None:
                            summary, source_path = cached
                            print(
                                f"[{done}/{total}] cached {dataset} {scenario} stream={stream_seed} model={model_seed} {method} from {source_path.parent.parent.parent.name}",
                                flush=True,
                            )
                            rows.append(
                                _row(
                                    dataset,
                                    scenario,
                                    stream_seed,
                                    model_seed,
                                    method,
                                    summary,
                                    None,
                                    str(source_path),
                                )
                            )
                            continue
                        cfg = _configure(
                            output_root, dataset, scenario, stream_seed, model_seed, method
                        )
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
                                "computed",
                            )
                        )
                        _write_analysis(output_root, rows, total)
                        print(
                            f"[{done}/{total}] done {method} macro={result.summary.get('mean_macro_f1'):.4f} update_s={result.summary.get('total_update_seconds'):.3f} elapsed_s={elapsed:.1f}",
                            flush=True,
                        )
    _write_analysis(output_root, rows, total)
    print(f"completed {len(rows)}/{total} in {time.perf_counter() - started_all:.1f}s")
    print(output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full_pilot", "confirmation"],
        required=True,
    )
    args = parser.parse_args()
    run_protocol(args.mode)


if __name__ == "__main__":
    main()
