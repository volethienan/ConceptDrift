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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from dasp_nids.config import load_config
from dasp_nids.revision import REVISION_CONDITIONS, index_digest
from dasp_nids.runner import StreamingExperiment


DATASETS = [
    "cic-ids2017",
    "nf-bot-iot-v3",
    "nf-cse-cic-ids2018-v3",
    "nf-ton-iot-v3",
    "nf-unsw-nb15-v3",
]
SCENARIOS = ["S1-Sudden", "S2-Gradual", "S7-Local-High"]
STREAM_SEEDS = [2001, 2002, 2003]
MODEL_SEED = 91
PERMUTATION_SEEDS = [3101, 3102, 3103]
NATURAL_MODEL_SEEDS = [91, 92, 93]
NATURAL_STREAM_SEED = 4101

COMPONENT_METHODS = [
    "revision_b1_no_replay",
    "revision_b2_no_weighting",
    "revision_b3_no_gate",
    "revision_b4_uniform_selection",
    "revision_b5_flat_global",
]
PERMUTATION_METHODS = [
    "revision_a1_combined",
    "revision_a6_all",
    "revision_a9_classifier",
]
NATURAL_METHODS = [
    "revision_a1_combined",
    "revision_a10_ssf",
    "experience_replay",
    "online_mlp",
]
NATURAL_IDENTIFIERS = {
    "experience_replay": "N3",
    "online_mlp": "N4",
}

PREPARED_ROOT = Path("prepared_streams_final")
NATURAL_STREAM = Path(
    "prepared_streams_revision/nf-ton-iot-v3/natural_chronological_v1.npz"
)
BASE_CONFIG = Path("configs/revision_p0.yaml")
DEPENDENCY_LOCK = Path("requirements-revision-lock.txt")
OUTPUT_ROOT = Path("outputs_revision_v1")
P0_ROOT = OUTPUT_ROOT / "full_p0_v1"
STREAM_DIGESTS: dict[Path, str] = {}

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
    "model_parameter_count",
]


@dataclass(frozen=True)
class RunSpec:
    family: str
    dataset: str
    scenario: str
    stream_seed: int
    model_seed: int
    method: str
    permutation_seed: int | None = None

    @property
    def identifier(self) -> str:
        if self.method in REVISION_CONDITIONS:
            return REVISION_CONDITIONS[self.method].identifier
        return NATURAL_IDENTIFIERS[self.method]


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
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_status() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--short"],
        text=True,
        encoding="utf-8",
    )
    return output.splitlines()


def _source_digests() -> dict[str, str]:
    paths = [
        Path("pyproject.toml"),
        Path("src/dasp_nids/config.py"),
        Path("src/dasp_nids/data.py"),
        Path("src/dasp_nids/model.py"),
        Path("src/dasp_nids/revision.py"),
        Path("src/dasp_nids/runner.py"),
        Path("tools/prepare_revision_natural_stream.py"),
        Path("tools/run_revision_p1.py"),
        BASE_CONFIG,
        DEPENDENCY_LOCK,
    ]
    return {path.as_posix(): _digest_file(path) for path in paths}


def _mode_spec(mode: str) -> tuple[Path, list[RunSpec], int]:
    if mode in {"component_smoke", "components"}:
        datasets = ["nf-ton-iot-v3"] if mode == "component_smoke" else DATASETS
        scenarios = ["S1-Sudden"] if mode == "component_smoke" else SCENARIOS
        seeds = [2001] if mode == "component_smoke" else STREAM_SEEDS
        specs = [
            RunSpec("components", dataset, scenario, seed, MODEL_SEED, method)
            for dataset in datasets
            for scenario in scenarios
            for seed in seeds
            for method in COMPONENT_METHODS
        ]
        root = OUTPUT_ROOT / (
            "p1_component_smoke_v2" if mode == "component_smoke" else "p1_components_v1"
        )
        return root, specs, len(COMPONENT_METHODS)
    if mode in {"permutation_smoke", "permutations"}:
        datasets = ["nf-ton-iot-v3"] if mode == "permutation_smoke" else DATASETS
        scenarios = ["S1-Sudden"] if mode == "permutation_smoke" else SCENARIOS
        seeds = [2001] if mode == "permutation_smoke" else STREAM_SEEDS
        permutation_seeds = [3101] if mode == "permutation_smoke" else PERMUTATION_SEEDS
        specs = [
            RunSpec(
                "permutations",
                dataset,
                scenario,
                seed,
                MODEL_SEED,
                method,
                permutation_seed,
            )
            for dataset in datasets
            for scenario in scenarios
            for seed in seeds
            for permutation_seed in permutation_seeds
            for method in PERMUTATION_METHODS
        ]
        root = OUTPUT_ROOT / (
            "p1_permutation_smoke_v2"
            if mode == "permutation_smoke"
            else "p1_feature_permutation_v1"
        )
        return root, specs, len(PERMUTATION_METHODS)
    if mode in {"natural_smoke", "natural"}:
        model_seeds = [91] if mode == "natural_smoke" else NATURAL_MODEL_SEEDS
        specs = [
            RunSpec(
                "natural",
                "nf-ton-iot-v3-natural",
                "Natural-Chronological",
                NATURAL_STREAM_SEED,
                model_seed,
                method,
            )
            for model_seed in model_seeds
            for method in NATURAL_METHODS
        ]
        root = OUTPUT_ROOT / (
            "p1_natural_smoke_v2" if mode == "natural_smoke" else "p1_natural_v1"
        )
        return root, specs, len(NATURAL_METHODS)
    raise ValueError(f"unknown P1 mode: {mode}")


def _run_name(spec: RunSpec) -> str:
    suffix = "" if spec.permutation_seed is None else f"_perm{spec.permutation_seed}"
    return (
        f"{spec.method}_stream{spec.stream_seed}_model{spec.model_seed}{suffix}"
    )


def _run_dir(output_root: Path, spec: RunSpec) -> Path:
    return output_root / spec.dataset / spec.scenario / _run_name(spec)


def _stream_path(spec: RunSpec) -> Path:
    if spec.family == "natural":
        return NATURAL_STREAM
    return (
        PREPARED_ROOT
        / spec.dataset
        / spec.scenario
        / f"stream_seed{spec.stream_seed}.npz"
    )


def _configure(output_root: Path, spec: RunSpec):
    config = copy.deepcopy(load_config(BASE_CONFIG))
    stream_path = _stream_path(spec)
    if not stream_path.exists():
        raise FileNotFoundError(stream_path)
    config.data.source = "prepared_npz"
    config.data.path = str(stream_path)
    config.data.scenario = spec.scenario
    config.data.random_seed = spec.stream_seed
    config.data.stream_seed = spec.stream_seed
    config.training.model_seed = spec.model_seed
    config.training.device = "cpu"
    config.training.num_threads = 1
    config.dasp.baseline = spec.method
    config.revision.enabled = True
    config.revision.dataset_key = spec.dataset
    config.revision.event_matched_diagnostics = False
    config.revision.feature_permutation_seed = spec.permutation_seed
    config.revision.common_label_exposure_baselines = spec.family == "natural"
    if spec.method == "revision_b5_flat_global":
        config.model.architecture = "sequential"
        config.model.hidden_dims = [184, 40] if spec.dataset == "cic-ids2017" else [136, 96]
    else:
        config.model.architecture = "modular"
        config.model.hidden_dims = [128, 64]
    config.output.directory = str(output_root / spec.dataset / spec.scenario)
    config.output.run_name = _run_name(spec)
    return config


def _canonical_config(config) -> dict[str, object]:
    return yaml.safe_load(yaml.safe_dump(config.to_dict(), sort_keys=True))


def _summary_row(spec: RunSpec, summary: dict[str, object], source: str) -> dict[str, object]:
    row: dict[str, object] = {
        "family": spec.family,
        "dataset": spec.dataset,
        "scenario": spec.scenario,
        "stream_seed": spec.stream_seed,
        "model_seed": spec.model_seed,
        "permutation_seed": spec.permutation_seed,
        "condition": spec.identifier,
        "method": spec.method,
        "summary_source": source,
    }
    for field in SUMMARY_FIELDS:
        row[field] = summary.get(field)
    return row


def _write_run_provenance(
    config,
    spec: RunSpec,
    run_dir: Path,
    context: dict[str, object],
) -> None:
    artifacts = {}
    for name in ("config.yaml", "summary.json", "windows.csv"):
        path = run_dir / name
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": _digest_file(path),
        }
    stream_path = Path(config.data.path)
    payload = {
        "protocol_version": config.revision.protocol_version,
        "family": spec.family,
        "condition": spec.identifier,
        "method": spec.method,
        "dataset": spec.dataset,
        "scenario": spec.scenario,
        "stream_seed": spec.stream_seed,
        "model_seed": spec.model_seed,
        "permutation_seed": spec.permutation_seed,
        "repository_commit": context["repository_commit"],
        "working_tree_status": context["working_tree_status"],
        "source_sha256": context["source_sha256"],
        "stream_path": str(stream_path),
        "stream_sha256": _cached_digest(stream_path),
        "artifacts": artifacts,
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _write_root_metadata(output_root: Path, mode: str, expected_runs: int) -> dict[str, object]:
    environment = {
        "captured_at": datetime.now().astimezone().isoformat(),
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
        "command": f"python tools/run_revision_p1.py --mode {mode} --resume",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "execution_environment.json"
    if path.exists():
        locked = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "mode",
            "expected_runs",
            "protocol_version",
            "base_config_sha256",
            "dependency_lock_sha256",
            "repository_commit",
            "source_sha256",
        ):
            if locked.get(key) != environment.get(key):
                raise RuntimeError(f"locked P1 execution metadata mismatch for {key}")
        return locked
    path.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    return environment


def _finite_summary(summary: dict[str, object]) -> list[str]:
    errors = []
    for field in (
        "mean_macro_f1",
        "cumulative_macro_f1",
        "total_update_seconds",
        "prediction_decision_throughput_flows_per_second",
    ):
        value = summary.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"non-finite {field}={value!r}")
    return errors


def _load_run(output_root: Path, spec: RunSpec) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    run_dir = _run_dir(output_root, spec)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(run_dir / "windows.csv")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return summary, frame[~frame["warmup"].astype(bool)].set_index("window"), config


def _p0_reference(spec: RunSpec, method: str) -> pd.DataFrame:
    run_dir = (
        P0_ROOT
        / spec.dataset
        / spec.scenario
        / f"{method}_stream{spec.stream_seed}_model{spec.model_seed}"
    )
    frame = pd.read_csv(run_dir / "windows.csv")
    return frame[~frame["warmup"].astype(bool)].set_index("window")


def _validate_group(output_root: Path, specs: list[RunSpec]) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, object]] = {}
    configs: dict[str, dict[str, object]] = {}
    for spec in specs:
        run_dir = _run_dir(output_root, spec)
        missing_artifact = False
        for name in ("summary.json", "windows.csv", "config.yaml", "provenance.json"):
            if not (run_dir / name).exists():
                errors.append(f"{spec.method}: missing {name}")
                missing_artifact = True
        if missing_artifact:
            continue
        summary, frame, config = _load_run(output_root, spec)
        summaries[spec.method] = summary
        frames[spec.method] = frame
        configs[spec.method] = config
        errors.extend(f"{spec.method}: {error}" for error in _finite_summary(summary))
        if not (pd.to_numeric(frame["labels_exposed_total"], errors="coerce") == 512).all():
            errors.append(f"{spec.method}: labels_exposed_total != 512")
        if not (pd.to_numeric(frame["repair_candidate_labels_exposed"], errors="coerce") == 384).all():
            errors.append(f"{spec.method}: repair exposure != 384")
        if not (pd.to_numeric(frame["gate_labels_exposed"], errors="coerce") == 128).all():
            errors.append(f"{spec.method}: gate exposure != 128")

    if len(frames) == len(specs):
        reference_method = specs[0].method
        reference = frames[reference_method]
        for method, frame in frames.items():
            common = reference.index.intersection(frame.index)
            for column in (
                "query_index_digest",
                "repair_candidate_index_digest",
                "gate_index_digest",
            ):
                if not reference.loc[common, column].equals(frame.loc[common, column]):
                    errors.append(f"{method}: mismatched {column}")

        family = specs[0].family
        if family == "components":
            p0 = _p0_reference(specs[0], "revision_a1_combined")
            for method, frame in frames.items():
                common = p0.index.intersection(frame.index)
                for column in (
                    "query_index_digest",
                    "repair_candidate_index_digest",
                    "gate_index_digest",
                ):
                    if not p0.loc[common, column].equals(frame.loc[common, column]):
                        errors.append(f"{method}: differs from P0 {column}")
            b1 = frames["revision_b1_no_replay"]
            repaired_b1 = b1[b1["revision_repair_applied"].astype(bool)]
            if b1["revision_replay_enabled"].astype(bool).any():
                errors.append("B1: replay policy was enabled")
            if not repaired_b1.empty and not (
                pd.to_numeric(
                    repaired_b1["revision_replay_size"],
                    errors="coerce",
                )
                == 0
            ).all():
                errors.append("B1: replay was not zero")
            b2 = frames["revision_b2_no_weighting"]
            if not (pd.to_numeric(b2["revision_new_sample_weight"], errors="coerce") == 1.0).all():
                errors.append("B2: new-sample weight was not one")
            b3 = frames["revision_b3_no_gate"]
            if b3["revision_gate_repair"].astype(bool).any():
                errors.append("B3: performance-gate repair occurred")
            if (pd.to_numeric(b3["labels_used_for_gating"], errors="coerce") != 0).any():
                errors.append("B3: gate labels were algorithmically used")
            repaired_b3 = b3[b3["revision_repair_applied"].astype(bool)]
            if not repaired_b3.empty and not repaired_b3["triggered"].astype(bool).all():
                errors.append("B3: repair occurred without stored MMD trigger")
            b4 = frames["revision_b4_uniform_selection"]
            if not (b4["revision_selection_policy"] == "uniform_label_blind").all():
                errors.append("B4: selection was not uniform label-blind")
            b5 = frames["revision_b5_flat_global"]
            if "mmd_layer_1" in b5.columns or "mmd_layer_2" not in b5.columns:
                errors.append("B5: drift was not computed from only the final flat layer")
            repaired_b5 = b5[b5["revision_repair_applied"].astype(bool)]
            if not repaired_b5.empty and not (
                pd.to_numeric(
                    repaired_b5["active_representation_module_count"],
                    errors="coerce",
                )
                == 2
            ).all():
                errors.append("B5: global flat representation update did not include both layers")
            expected_parameters = 22466 if specs[0].dataset == "cic-ids2017" else 20610
            if int(summaries["revision_b5_flat_global"].get("model_parameter_count", -1)) != expected_parameters:
                errors.append("B5: parameter count mismatch")
            b5_model = configs["revision_b5_flat_global"]["model"]
            if b5_model["architecture"] != "sequential":
                errors.append("B5: architecture is not sequential")
        elif family == "permutations":
            digests = {
                str(frame["feature_permutation_digest"].dropna().iloc[0])
                for frame in frames.values()
            }
            if len(digests) != 1:
                errors.append("feature permutation digest differs across methods")
            seeds = {
                int(pd.to_numeric(frame["feature_permutation_seed"]).dropna().iloc[0])
                for frame in frames.values()
            }
            if seeds != {int(specs[0].permutation_seed)}:
                errors.append("feature permutation seed mismatch")
        elif family == "natural":
            for method, frame in frames.items():
                if (pd.to_numeric(frame["new_training_labels_used"], errors="coerce") > 128).any():
                    errors.append(f"{method}: natural optimization budget exceeded")
            for method in ("experience_replay", "online_mlp"):
                frame = frames[method]
                if not (frame["label_query_policy"] == "common_512_uniform_128").all():
                    errors.append(f"{method}: common-exposure policy missing")

    first = specs[0]
    return {
        "family": first.family,
        "dataset": first.dataset,
        "scenario": first.scenario,
        "stream_seed": first.stream_seed,
        "model_seed": first.model_seed,
        "permutation_seed": first.permutation_seed,
        "expected_methods": len(specs),
        "completed_methods": len(frames),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _group_key(spec: RunSpec) -> tuple[object, ...]:
    return (
        spec.family,
        spec.dataset,
        spec.scenario,
        spec.stream_seed,
        spec.model_seed,
        spec.permutation_seed,
    )


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
        summary = (
            frame.groupby(["condition", "method"], as_index=False)[numeric]
            .mean(numeric_only=True)
            .merge(
                frame.groupby(["condition", "method"], as_index=False)
                .size()
                .rename(columns={"size": "runs"}),
                on=["condition", "method"],
            )
        )
        summary.to_csv(output_root / "method_summary.csv", index=False)
    validation = {
        "expected_runs": expected_runs,
        "completed_runs": len(frame),
        "complete": len(frame) == expected_runs,
        "passed": len(frame) == expected_runs and all(item["passed"] for item in validations),
        "groups": validations,
    }
    (output_root / "validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )


def _write_component_analysis(output_root: Path) -> None:
    components = pd.read_csv(output_root / "full_summary.csv")
    p0 = pd.read_csv(P0_ROOT / "full_summary.csv")
    keys = ["dataset", "scenario", "stream_seed", "model_seed"]
    reference = p0[
        (p0["method"] == "revision_a1_combined")
        & p0["scenario"].isin(SCENARIOS)
        & p0["stream_seed"].isin(STREAM_SEEDS)
        & (p0["model_seed"] == MODEL_SEED)
    ][keys + ["mean_macro_f1", "total_update_seconds"]]
    merged = components.merge(reference, on=keys, suffixes=("", "_a1"))
    merged["macro_f1_delta_vs_a1"] = merged["mean_macro_f1"] - merged["mean_macro_f1_a1"]
    merged["update_seconds_delta_vs_a1"] = merged["total_update_seconds"] - merged["total_update_seconds_a1"]
    merged.to_csv(output_root / "paired_components_vs_a1.csv", index=False)

    a6 = p0[
        (p0["method"] == "revision_a6_all")
        & p0["scenario"].isin(SCENARIOS)
        & p0["stream_seed"].isin(STREAM_SEEDS)
        & (p0["model_seed"] == MODEL_SEED)
    ][keys + ["mean_macro_f1", "total_update_seconds"]]
    b5 = components[components["condition"] == "B5"].merge(
        a6,
        on=keys,
        suffixes=("_b5", "_a6"),
    )
    b5["macro_f1_delta_b5_minus_a6"] = b5["mean_macro_f1_b5"] - b5["mean_macro_f1_a6"]
    b5.to_csv(output_root / "paired_b5_vs_a6.csv", index=False)


def _write_permutation_analysis(output_root: Path) -> None:
    permuted = pd.read_csv(output_root / "full_summary.csv")
    permuted["feature_order"] = permuted["permutation_seed"].map(
        lambda value: f"perm_{int(value)}"
    )
    p0 = pd.read_csv(P0_ROOT / "full_summary.csv")
    original = p0[
        p0["method"].isin(PERMUTATION_METHODS)
        & p0["scenario"].isin(SCENARIOS)
        & p0["stream_seed"].isin(STREAM_SEEDS)
        & (p0["model_seed"] == MODEL_SEED)
    ].copy()
    original["family"] = "permutations"
    original["permutation_seed"] = np.nan
    original["feature_order"] = "original"
    combined = pd.concat([original, permuted], ignore_index=True, sort=False)
    combined.to_csv(output_root / "feature_order_combined_with_p0.csv", index=False)
    order_summary = (
        combined.groupby(["method", "feature_order"], as_index=False)
        .agg(
            runs=("mean_macro_f1", "size"),
            mean_macro_f1=("mean_macro_f1", "mean"),
            std_macro_f1=("mean_macro_f1", "std"),
            minimum_macro_f1=("mean_macro_f1", "min"),
            maximum_macro_f1=("mean_macro_f1", "max"),
        )
    )
    order_summary.to_csv(output_root / "feature_order_summary.csv", index=False)


def _write_gate_report(output_root: Path, mode: str) -> None:
    validation = json.loads((output_root / "validation.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output_root / "method_summary.csv")
    lines = [
        f"# {mode.replace('_', ' ').title()} Validation Report",
        "",
        f"Status date: {datetime.now().astimezone().date().isoformat()}.",
        "",
        f"- Expected runs: {validation['expected_runs']}",
        f"- Completed runs: {validation['completed_runs']}",
        f"- Validation passed: {validation['passed']}",
        f"- Validated groups: {len(validation['groups'])}",
        f"- Failed groups: {sum(not group['passed'] for group in validation['groups'])}",
        "",
        "## Method Summary",
        "",
        "| Condition | Runs | Mean Macro-F1 | Mean update seconds |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.sort_values("condition").itertuples(index=False):
        lines.append(
            f"| {row.condition} | {int(row.runs)} | {row.mean_macro_f1:.6f} | "
            f"{row.total_update_seconds:.3f} |"
        )
    (output_root / "GATE_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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
    output_root, specs, methods_per_group = _mode_spec(mode)
    if not (P0_ROOT / "validation.json").exists():
        raise RuntimeError("P0 validation is missing; refusing to start P1")
    p0_validation = json.loads((P0_ROOT / "validation.json").read_text(encoding="utf-8"))
    if not p0_validation.get("passed"):
        raise RuntimeError("P0 validation did not pass; refusing to start P1")
    if specs[0].family == "natural" and not NATURAL_STREAM.exists():
        raise RuntimeError(
            "natural stream is missing; run tools/prepare_revision_natural_stream.py first"
        )
    context = _write_root_metadata(output_root, mode, len(specs))
    rows: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    groups: dict[tuple[object, ...], list[RunSpec]] = {}
    for spec in specs:
        groups.setdefault(_group_key(spec), []).append(spec)
    completed = 0
    started_all = time.perf_counter()
    for group_specs in groups.values():
        if len(group_specs) != methods_per_group:
            raise RuntimeError("invalid P1 group size")
        for spec in group_specs:
            completed += 1
            config = _configure(output_root, spec)
            run_dir = _run_dir(output_root, spec)
            summary_path = run_dir / "summary.json"
            config_path = run_dir / "config.yaml"
            windows_path = run_dir / "windows.csv"
            if resume and summary_path.exists() and config_path.exists() and windows_path.exists():
                cached = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if cached != _canonical_config(config):
                    raise RuntimeError(f"cached config mismatch: {run_dir}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rows.append(_summary_row(spec, summary, "cached"))
                _write_run_provenance(config, spec, run_dir, context)
                print(f"[{completed}/{len(specs)}] cached {_run_name(spec)}", flush=True)
                continue
            print(f"[{completed}/{len(specs)}] running {_run_name(spec)}", flush=True)
            result = StreamingExperiment(config).run()
            rows.append(_summary_row(spec, result.summary, "computed"))
            _write_run_provenance(config, spec, run_dir, context)
            _write_progress(output_root, rows, validations, len(specs))
            print(
                f"[{completed}/{len(specs)}] done {_run_name(spec)} "
                f"macro={result.summary['mean_macro_f1']:.6f}",
                flush=True,
            )
        validation = _validate_group(output_root, group_specs)
        validations.append(validation)
        _write_progress(output_root, rows, validations, len(specs))
        if not validation["passed"]:
            raise RuntimeError("P1 validation failed: " + "; ".join(validation["errors"]))
    if specs[0].family == "components" and mode == "components":
        _write_component_analysis(output_root)
    if specs[0].family == "permutations" and mode == "permutations":
        _write_permutation_analysis(output_root)
    _write_progress(output_root, rows, validations, len(specs))
    _write_gate_report(output_root, mode)
    _write_artifact_manifest(output_root)
    print(
        f"completed mode={mode} runs={len(rows)}/{len(specs)} "
        f"elapsed_s={time.perf_counter() - started_all:.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "component_smoke",
            "components",
            "permutation_smoke",
            "permutations",
            "natural_smoke",
            "natural",
        ],
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_mode(args.mode, args.resume)


if __name__ == "__main__":
    main()
