from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from dasp_nids.config import load_config
from dasp_nids.revision import (
    REVISION_CONDITIONS,
    equal_label_pools,
    select_training_indices,
)
from dasp_nids.runner import StreamingExperiment


warnings.filterwarnings(
    "ignore",
    message="A single label was found in 'y_true' and 'y_pred'.*",
    category=UserWarning,
    module="sklearn.metrics._classification",
)


DATASETS = [
    "cic-ids2017",
    "nf-bot-iot-v3",
    "nf-cse-cic-ids2018-v3",
    "nf-ton-iot-v3",
    "nf-unsw-nb15-v3",
]
REDUCED_SCENARIOS = ["S1-Sudden", "S2-Gradual", "S7-Local-High"]
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
    method
    for method, condition in REVISION_CONDITIONS.items()
    if condition.identifier.startswith("A")
]
STREAM_SEED = 2001
MODEL_SEED = 91
CONFIRMATION_STREAM_SEEDS = [2001, 2002, 2003, 2101, 2102]
CONFIRMATION_MODEL_SEEDS = [91, 92]
SMOKE_DATASET = "nf-ton-iot-v3"
SMOKE_SCENARIO = "S1-Sudden"
PREPARED_ROOT = Path("prepared_streams_final")
BASE_CONFIG = Path("configs/revision_p0.yaml")
DEPENDENCY_LOCK = Path("requirements-revision-lock.txt")
OUTPUT_ROOT = Path("outputs_revision_v1")
STREAM_DIGESTS: dict[Path, str] = {}
LABEL_DEGENERACY_CACHE: dict[tuple[str, str, int], dict[str, object]] = {}

SUMMARY_FIELDS = [
    "mean_macro_f1",
    "cumulative_macro_f1",
    "cumulative_attack_recall",
    "cumulative_fnr",
    "mean_online_forgetting",
    "window_stability_macro_f1",
    "total_update_seconds",
    "median_update_seconds",
    "prediction_decision_throughput_flows_per_second",
    "total_gradient_evaluations",
    "total_parameter_gradient_element_evaluations",
    "total_ground_truth_update_labels",
    "total_labels_exposed",
    "total_new_training_labels_used",
    "revision_repair_count",
    "peak_rss_increment_bytes",
]


def _mode_spec(
    mode: str,
) -> tuple[Path, list[str], list[str], list[int], list[int], bool]:
    if mode == "smoke":
        return (
            OUTPUT_ROOT / "smoke_v1",
            [SMOKE_DATASET],
            [SMOKE_SCENARIO],
            [STREAM_SEED],
            [MODEL_SEED],
            True,
        )
    if mode == "reduced_pilot":
        return (
            OUTPUT_ROOT / "reduced_pilot_v1",
            DATASETS,
            REDUCED_SCENARIOS,
            [STREAM_SEED],
            [MODEL_SEED],
            False,
        )
    if mode == "full_p0":
        return (
            OUTPUT_ROOT / "full_p0_v1",
            DATASETS,
            PRIMARY_SCENARIOS,
            CONFIRMATION_STREAM_SEEDS,
            CONFIRMATION_MODEL_SEEDS,
            False,
        )
    raise ValueError(f"unknown revision mode: {mode}")


def _run_name(method: str, stream_seed: int, model_seed: int) -> str:
    return f"{method}_stream{stream_seed}_model{model_seed}"


def _canonical_config(config) -> dict[str, object]:
    return yaml.safe_load(yaml.safe_dump(config.to_dict(), sort_keys=True))


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_digest(path: Path) -> str:
    resolved = path.resolve()
    if resolved not in STREAM_DIGESTS:
        STREAM_DIGESTS[resolved] = _digest_file(resolved)
    return STREAM_DIGESTS[resolved]


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _git_status() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--short"], text=True, encoding="utf-8"
    )
    return [line for line in output.splitlines() if line]


def _source_digests() -> dict[str, str]:
    paths = [
        Path("pyproject.toml"),
        Path("src/dasp_nids/config.py"),
        Path("src/dasp_nids/revision.py"),
        Path("src/dasp_nids/runner.py"),
        Path("tools/run_revision_experiments.py"),
        BASE_CONFIG,
    ]
    return {path.as_posix(): _digest_file(path) for path in paths}


def _configure(
    output_root: Path,
    dataset: str,
    scenario: str,
    method: str,
    event_matched: bool,
    stream_seed: int = STREAM_SEED,
    model_seed: int = MODEL_SEED,
):
    config = copy.deepcopy(load_config(BASE_CONFIG))
    stream_path = PREPARED_ROOT / dataset / scenario / f"stream_seed{stream_seed}.npz"
    if not stream_path.exists():
        raise FileNotFoundError(stream_path)
    config.data.source = "prepared_npz"
    config.data.path = str(stream_path)
    config.data.scenario = scenario
    config.data.random_seed = stream_seed
    config.data.stream_seed = stream_seed
    config.training.model_seed = model_seed
    config.training.device = "cpu"
    config.training.num_threads = 1
    config.dasp.baseline = method
    config.revision.enabled = True
    config.revision.dataset_key = dataset
    config.revision.event_matched_diagnostics = bool(
        event_matched and method == "revision_a1_combined"
    )
    config.output.directory = str(output_root / dataset / scenario)
    config.output.run_name = _run_name(method, stream_seed, model_seed)
    return config


def _run_paths(
    output_root: Path,
    dataset: str,
    scenario: str,
    method: str,
    stream_seed: int,
    model_seed: int,
) -> tuple[Path, Path, Path]:
    run_dir = output_root / dataset / scenario / _run_name(method, stream_seed, model_seed)
    return run_dir, run_dir / "summary.json", run_dir / "config.yaml"


def _summary_row(
    dataset: str,
    scenario: str,
    method: str,
    summary: dict[str, object],
    elapsed_seconds: float | None,
    source: str,
    stream_seed: int,
    model_seed: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "scenario": scenario,
        "stream_seed": stream_seed,
        "model_seed": model_seed,
        "condition": REVISION_CONDITIONS[method].identifier,
        "method": method,
        "elapsed_seconds": elapsed_seconds,
        "summary_source": source,
    }
    for field in SUMMARY_FIELDS:
        row[field] = summary.get(field)
    return row


def _write_run_provenance(
    config,
    run_dir: Path,
    provenance_context: dict[str, object],
) -> None:
    stream_path = Path(config.data.path)
    artifacts = {}
    for name in ("config.yaml", "summary.json", "windows.csv"):
        path = run_dir / name
        if path.exists():
            artifacts[name] = {
                "bytes": path.stat().st_size,
                "sha256": _digest_file(path),
            }
    payload = {
        "protocol_version": config.revision.protocol_version,
        "condition": REVISION_CONDITIONS[config.dasp.baseline].identifier,
        "method": config.dasp.baseline,
        "dataset": config.revision.dataset_key,
        "scenario": config.data.scenario,
        "stream_seed": config.data.stream_seed,
        "model_seed": config.training.model_seed,
        "repository_commit": provenance_context["repository_commit"],
        "working_tree_status": provenance_context["working_tree_status"],
        "source_sha256": provenance_context["source_sha256"],
        "stream_path": str(stream_path),
        "stream_sha256": _cached_digest(stream_path),
        "artifacts": artifacts,
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _label_degeneracy(
    dataset: str,
    scenario: str,
    stream_seed: int,
) -> dict[str, object]:
    cache_key = (dataset, scenario, stream_seed)
    if cache_key in LABEL_DEGENERACY_CACHE:
        return LABEL_DEGENERACY_CACHE[cache_key]
    stream_path = PREPARED_ROOT / dataset / scenario / f"stream_seed{stream_seed}.npz"
    counts = {
        "full_window_one_class": 0,
        "query_one_class": 0,
        "repair_candidate_one_class": 0,
        "gate_one_class": 0,
        "balanced_training_one_class": 0,
    }
    examples: list[dict[str, object]] = []
    with np.load(stream_path, allow_pickle=False) as payload:
        labels = np.asarray(payload["y"])
    for window_index in range(4, len(labels)):
        window_y = torch.from_numpy(labels[window_index]).long()
        pools = equal_label_pools(
            len(window_y),
            512,
            384,
            128,
            dataset,
            scenario,
            stream_seed,
            window_index,
            "1.1",
        )
        training = select_training_indices(
            window_y,
            pools.repair_candidates,
            128,
            pools.selection_seed,
            label_balanced=True,
        )
        groups = {
            "full_window_one_class": torch.arange(len(window_y)),
            "query_one_class": pools.query,
            "repair_candidate_one_class": pools.repair_candidates,
            "gate_one_class": pools.gate,
            "balanced_training_one_class": training,
        }
        for key, indices in groups.items():
            if len(torch.unique(window_y[indices])) < 2:
                counts[key] += 1
                if len(examples) < 20:
                    examples.append({"window": window_index, "pool": key})
    result = {
        "post_warmup_windows": int(len(labels) - 4),
        **counts,
        "examples": examples,
    }
    LABEL_DEGENERACY_CACHE[cache_key] = result
    return result


def _finite_summary(summary: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for field in (
        "mean_macro_f1",
        "cumulative_macro_f1",
        "total_update_seconds",
        "prediction_decision_throughput_flows_per_second",
    ):
        value = summary.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"non-finite summary field {field}={value!r}")
    return errors


def _validate_cell(
    output_root: Path,
    dataset: str,
    scenario: str,
    stream_seed: int,
    model_seed: int,
) -> dict[str, object]:
    errors: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, object]] = {}
    for method in METHODS:
        run_dir, summary_path, config_path = _run_paths(
            output_root,
            dataset,
            scenario,
            method,
            stream_seed,
            model_seed,
        )
        windows_path = run_dir / "windows.csv"
        if not summary_path.exists() or not config_path.exists() or not windows_path.exists():
            errors.append(f"missing artifacts for {method}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries[method] = summary
        errors.extend(f"{method}: {error}" for error in _finite_summary(summary))
        frame = pd.read_csv(windows_path)
        frames[method] = frame[~frame["warmup"].astype(bool)].set_index("window")

    if len(frames) == len(METHODS):
        reference = frames["revision_a1_combined"]
        for method, frame in frames.items():
            common = reference.index.intersection(frame.index)
            for column in (
                "query_index_digest",
                "repair_candidate_index_digest",
                "gate_index_digest",
            ):
                if not reference.loc[common, column].equals(frame.loc[common, column]):
                    errors.append(f"{method}: mismatched {column}")
            if not (pd.to_numeric(frame["labels_exposed_total"], errors="coerce") == 512).all():
                errors.append(f"{method}: labels_exposed_total is not always 512")
            if not (
                pd.to_numeric(frame["repair_candidate_labels_exposed"], errors="coerce") == 384
            ).all():
                errors.append(f"{method}: repair candidate exposure is not always 384")
            if not (pd.to_numeric(frame["gate_labels_exposed"], errors="coerce") == 128).all():
                errors.append(f"{method}: gate exposure is not always 128")

        routing_methods = [method for method in METHODS if method != "revision_a10_ssf"]
        for method in routing_methods:
            frame = frames[method]
            common = reference.index.intersection(frame.index)
            if not reference.loc[common, "training_index_digest"].equals(
                frame.loc[common, "training_index_digest"]
            ):
                errors.append(f"{method}: mismatched common training indices")
            both_repair = (
                reference.loc[common, "revision_repair_applied"].astype(bool)
                & frame.loc[common, "revision_repair_applied"].astype(bool)
            )
            repair_windows = common[both_repair.to_numpy()]
            if len(repair_windows) and not reference.loc[
                repair_windows, "replay_index_digest"
            ].equals(frame.loc[repair_windows, "replay_index_digest"]):
                errors.append(f"{method}: mismatched replay indices on common repairs")

            condition = REVISION_CONDITIONS[method]
            repaired = frame[frame["revision_repair_applied"].astype(bool)]
            if repaired.empty:
                continue
            active_count = pd.to_numeric(
                repaired["active_representation_module_count"], errors="coerce"
            )
            if not (active_count == condition.routing_k).all():
                errors.append(f"{method}: active module count differs from k")
            inactive_change = pd.to_numeric(
                repaired["inactive_module_max_abs_change"], errors="coerce"
            )
            if not (inactive_change == 0.0).all():
                errors.append(f"{method}: inactive module changed")
            required_columns = (
                "optimizer_included_parameter_fraction",
                "active_representation_parameter_fraction",
                "changed_parameter_fraction",
                "classifier_parameter_fraction",
                "parameter_gradient_element_evaluations",
            )
            for column in required_columns:
                if column not in repaired or repaired[column].isna().any():
                    errors.append(f"{method}: missing locality field {column}")

        ssf = frames["revision_a10_ssf"]
        if not (pd.to_numeric(ssf["labels_used_for_gating"], errors="coerce") == 0).all():
            errors.append("revision_a10_ssf: gate labels were algorithmically used")
        if not (pd.to_numeric(ssf["pseudo_labels_used"], errors="coerce") == 0).all():
            errors.append("revision_a10_ssf: pseudo labels were used")
        if not (pd.to_numeric(ssf["new_training_labels_used"], errors="coerce") <= 128).all():
            errors.append("revision_a10_ssf: optimization label budget exceeded")

        combined = frames["revision_a1_combined"]
        if "event_matched_complete" in combined.columns:
            completed = combined[combined["event_matched_complete"] == True]
            if completed.empty:
                errors.append("revision_a1_combined: no completed event-matched diagnostic")

    degeneracy = _label_degeneracy(dataset, scenario, stream_seed)
    warnings_list = [
        f"{key}={value}"
        for key, value in degeneracy.items()
        if key.endswith("_one_class") and int(value) > 0
    ]
    return {
        "dataset": dataset,
        "scenario": scenario,
        "stream_seed": stream_seed,
        "model_seed": model_seed,
        "expected_methods": len(METHODS),
        "completed_methods": len(frames),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings_list,
        "label_degeneracy": degeneracy,
    }


def _write_root_metadata(
    output_root: Path,
    mode: str,
    expected_runs: int,
) -> dict[str, object]:
    environment = {
        "captured_at": "2026-09-01",
        "mode": mode,
        "expected_runs": expected_runs,
        "protocol_version": "1.1",
        "base_config": str(BASE_CONFIG),
        "base_config_sha256": _digest_file(BASE_CONFIG),
        "dependency_lock": str(DEPENDENCY_LOCK),
        "dependency_lock_sha256": _digest_file(DEPENDENCY_LOCK),
        "repository_commit": _git_commit(),
        "working_tree_status": _git_status(),
        "source_sha256": _source_digests(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "torch_threads_per_run": 1,
        "command": f"python tools/run_revision_experiments.py --mode {mode} --resume",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "execution_environment.json"
    if metadata_path.exists():
        locked_environment = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in (
            "mode",
            "expected_runs",
            "protocol_version",
            "base_config_sha256",
            "dependency_lock_sha256",
            "repository_commit",
            "source_sha256",
        ):
            if locked_environment.get(key) != environment.get(key):
                raise RuntimeError(f"locked execution metadata mismatch for {key}")
        return locked_environment
    metadata_path.write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    return environment


def _write_progress(
    output_root: Path,
    rows: list[dict[str, object]],
    validations: list[dict[str, object]],
    expected_runs: int,
) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "full_summary.csv", index=False)
    if not frame.empty:
        numeric = [field for field in SUMMARY_FIELDS if field in frame]
        method_summary = (
            frame.groupby(["condition", "method"], as_index=False)[numeric]
            .mean(numeric_only=True)
            .merge(
                frame.groupby(["condition", "method"], as_index=False)
                .size()
                .rename(columns={"size": "runs"}),
                on=["condition", "method"],
            )
        )
        method_summary.to_csv(output_root / "method_summary.csv", index=False)
    validation = {
        "expected_runs": expected_runs,
        "completed_runs": len(frame),
        "complete": len(frame) == expected_runs,
        "passed": len(frame) == expected_runs and all(item["passed"] for item in validations),
        "cells": validations,
    }
    (output_root / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )


def _write_gate_report(output_root: Path, mode: str) -> None:
    validation = json.loads((output_root / "validation.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output_root / "method_summary.csv")
    warning_cells = [cell for cell in validation["cells"] if cell.get("warnings")]
    lines = [
        f"# {mode.replace('_', ' ').title()} Validation Report",
        "",
        "Status date: September 1, 2026.",
        "",
        f"- Expected runs: {validation['expected_runs']}",
        f"- Completed runs: {validation['completed_runs']}",
        f"- Validation passed: {validation['passed']}",
        f"- Validated cells: {len(validation['cells'])}",
        f"- Cells with label-pool warnings: {len(warning_cells)}",
        "",
        "## Method Runtime Summary",
        "",
        "| Condition | Runs | Mean Macro-F1 | Mean update seconds | Mean throughput |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("condition").itertuples(index=False):
        lines.append(
            f"| {row.condition} | {int(row.runs)} | {row.mean_macro_f1:.6f} | "
            f"{row.total_update_seconds:.3f} | "
            f"{row.prediction_decision_throughput_flows_per_second:.1f} |"
        )
    lines.extend(["", "## Label-Degeneracy Audit", ""])
    if not warning_cells:
        lines.append("No one-class full, query, repair-candidate, gate, or balanced-training pool was found.")
    else:
        for cell in warning_cells:
            lines.append(
                f"- `{cell['dataset']} / {cell['scenario']} / "
                f"stream={cell['stream_seed']} / model={cell['model_seed']}`: "
                + ", ".join(cell["warnings"])
            )
    lines.append("")
    if mode == "full_p0":
        lines.append(
            "This is the locked primary P0 confirmation. Outcome-driven changes require a new protocol version and rerun of affected conditions."
        )
    else:
        lines.append(
            "This gate is a correctness and runtime diagnostic. Its predictive differences must not be used to alter the locked P0 routing, label, replay, gating, or weighting definitions."
        )
    (output_root / "GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_artifact_manifest(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "artifact_sha256.csv":
            continue
        rows.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _digest_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "artifact_sha256.csv", index=False)


def run_mode(mode: str, resume: bool) -> None:
    (
        output_root,
        datasets,
        scenarios,
        stream_seeds,
        model_seeds,
        event_matched,
    ) = _mode_spec(mode)
    expected_runs = (
        len(datasets)
        * len(scenarios)
        * len(stream_seeds)
        * len(model_seeds)
        * len(METHODS)
    )
    provenance_context = _write_root_metadata(output_root, mode, expected_runs)
    if mode == "reduced_pilot":
        smoke_validation_path = OUTPUT_ROOT / "smoke_v1" / "validation.json"
        if not smoke_validation_path.exists():
            raise RuntimeError("Gate 2 validation is missing; refusing to start Gate 3")
        smoke_validation = json.loads(smoke_validation_path.read_text(encoding="utf-8"))
        if not smoke_validation.get("passed"):
            raise RuntimeError("Gate 2 validation did not pass; refusing to start Gate 3")
    if mode == "full_p0":
        pilot_validation_path = OUTPUT_ROOT / "reduced_pilot_v1" / "validation.json"
        if not pilot_validation_path.exists():
            raise RuntimeError("Gate 3 validation is missing; refusing to start Gate 4")
        pilot_validation = json.loads(pilot_validation_path.read_text(encoding="utf-8"))
        if not pilot_validation.get("passed"):
            raise RuntimeError("Gate 3 validation did not pass; refusing to start Gate 4")

    rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    completed = 0
    started_all = time.perf_counter()
    for dataset in datasets:
        for scenario in scenarios:
            for stream_seed in stream_seeds:
                for model_seed in model_seeds:
                    for method in METHODS:
                        completed += 1
                        config = _configure(
                            output_root,
                            dataset,
                            scenario,
                            method,
                            event_matched,
                            stream_seed,
                            model_seed,
                        )
                        run_dir, summary_path, config_path = _run_paths(
                            output_root,
                            dataset,
                            scenario,
                            method,
                            stream_seed,
                            model_seed,
                        )
                        if resume and summary_path.exists() and config_path.exists():
                            cached_config = yaml.safe_load(
                                config_path.read_text(encoding="utf-8")
                            )
                            if cached_config != _canonical_config(config):
                                raise RuntimeError(f"cached config mismatch: {run_dir}")
                            summary = json.loads(summary_path.read_text(encoding="utf-8"))
                            rows.append(
                                _summary_row(
                                    dataset,
                                    scenario,
                                    method,
                                    summary,
                                    None,
                                    "cached",
                                    stream_seed,
                                    model_seed,
                                )
                            )
                            _write_run_provenance(config, run_dir, provenance_context)
                            print(
                                f"[{completed}/{expected_runs}] cached {dataset} {scenario} "
                                f"stream={stream_seed} model={model_seed} {method}",
                                flush=True,
                            )
                            continue
                        print(
                            f"[{completed}/{expected_runs}] running {dataset} {scenario} "
                            f"stream={stream_seed} model={model_seed} {method}",
                            flush=True,
                        )
                        started = time.perf_counter()
                        result = StreamingExperiment(config).run()
                        elapsed = time.perf_counter() - started
                        rows.append(
                            _summary_row(
                                dataset,
                                scenario,
                                method,
                                result.summary,
                                elapsed,
                                "computed",
                                stream_seed,
                                model_seed,
                            )
                        )
                        _write_run_provenance(config, run_dir, provenance_context)
                        _write_progress(output_root, rows, validations, expected_runs)
                        print(
                            f"[{completed}/{expected_runs}] done {method} "
                            f"macro={result.summary['mean_macro_f1']:.6f} elapsed_s={elapsed:.1f}",
                            flush=True,
                        )
                    validations.append(
                        _validate_cell(
                            output_root,
                            dataset,
                            scenario,
                            stream_seed,
                            model_seed,
                        )
                    )
                    _write_progress(output_root, rows, validations, expected_runs)
                    if not validations[-1]["passed"]:
                        raise RuntimeError(
                            f"validation failed for {dataset}/{scenario}/"
                            f"stream={stream_seed}/model={model_seed}: "
                            + "; ".join(validations[-1]["errors"])
                        )
    _write_progress(output_root, rows, validations, expected_runs)
    _write_gate_report(output_root, mode)
    _write_artifact_manifest(output_root)
    print(
        f"completed mode={mode} runs={len(rows)}/{expected_runs} "
        f"elapsed_s={time.perf_counter() - started_all:.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "reduced_pilot", "full_p0"],
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_mode(args.mode, args.resume)


if __name__ == "__main__":
    main()
