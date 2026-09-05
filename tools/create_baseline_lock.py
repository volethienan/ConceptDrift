from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_CODE_PATHS = [
    "pyproject.toml",
    "configs/modular_dasp_main.yaml",
    "tools/run_modular_dasp_protocol.py",
    "src/dasp_nids/analysis.py",
    "src/dasp_nids/baselines.py",
    "src/dasp_nids/config.py",
    "src/dasp_nids/data.py",
    "src/dasp_nids/drift.py",
    "src/dasp_nids/memory.py",
    "src/dasp_nids/model.py",
    "src/dasp_nids/runner.py",
]

PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "torch",
    "river",
    "psutil",
    "pytest",
]


def _run_git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _resolve_summary_path(repo_root: Path, row: dict[str, str]) -> Path:
    source = row["summary_source"].strip()
    if source and source != "computed":
        return (repo_root / source).resolve()
    run_name = f"{row['method']}_stream{row['stream_seed']}_model{row['model_seed']}"
    return (
        repo_root
        / "outputs_confirmation_v6_10scenario"
        / row["dataset"]
        / row["scenario"]
        / run_name
        / "summary.json"
    ).resolve()


def _load_matrix(repo_root: Path) -> tuple[list[dict[str, str]], list[Path]]:
    summary_csv = repo_root / "outputs_confirmation_v6_10scenario" / "full_summary.csv"
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4000:
        raise RuntimeError(f"Expected 4000 matrix rows, found {len(rows)}")

    evidence_paths: set[Path] = set()
    matrix_rows: list[dict[str, str]] = []
    for row in rows:
        summary_path = _resolve_summary_path(repo_root, row)
        run_directory = summary_path.parent
        run_files = {
            "config": run_directory / "config.yaml",
            "summary": summary_path,
            "windows": run_directory / "windows.csv",
        }
        for file_path in run_files.values():
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            evidence_paths.add(file_path.resolve())

        prepared_base = (
            repo_root
            / "prepared_streams_final"
            / row["dataset"]
            / row["scenario"]
            / f"stream_seed{row['stream_seed']}"
        )
        prepared_npz = prepared_base.with_suffix(".npz")
        prepared_manifest = prepared_base.with_suffix(".manifest.json")
        for file_path in (prepared_npz, prepared_manifest):
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            evidence_paths.add(file_path.resolve())

        source_root = summary_path.relative_to(repo_root).parts[0]
        matrix_rows.append(
            {
                "dataset": row["dataset"],
                "scenario": row["scenario"],
                "stream_seed": row["stream_seed"],
                "model_seed": row["model_seed"],
                "method": row["method"],
                "source_root": source_root,
                "config_path": _relative(run_files["config"], repo_root),
                "summary_path": _relative(run_files["summary"], repo_root),
                "windows_path": _relative(run_files["windows"], repo_root),
                "prepared_npz_path": _relative(prepared_npz, repo_root),
                "prepared_manifest_path": _relative(prepared_manifest, repo_root),
            }
        )

    confirmation_root = repo_root / "outputs_confirmation_v6_10scenario"
    for file_path in confirmation_root.iterdir():
        if file_path.is_file():
            evidence_paths.add(file_path.resolve())
    for relative_path in DEFAULT_CODE_PATHS:
        file_path = repo_root / relative_path
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        evidence_paths.add(file_path.resolve())
    return matrix_rows, sorted(evidence_paths, key=lambda path: _relative(path, repo_root))


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "NOT_INSTALLED"
    return versions


def create_lock(repo_root: Path, lock_directory: Path) -> None:
    repo_root = repo_root.resolve()
    lock_directory = lock_directory.resolve()
    lock_directory.mkdir(parents=True, exist_ok=True)

    matrix_rows, evidence_paths = _load_matrix(repo_root)
    evidence_records: list[dict[str, str | int]] = []
    aggregate_digest = hashlib.sha256()
    total_bytes = 0
    for file_path in evidence_paths:
        relative_path = _relative(file_path, repo_root)
        size_bytes = file_path.stat().st_size
        file_digest = _sha256(file_path)
        total_bytes += size_bytes
        aggregate_digest.update(
            f"{relative_path}\t{size_bytes}\t{file_digest}\n".encode("utf-8")
        )
        evidence_records.append(
            {"path": relative_path, "size_bytes": size_bytes, "sha256": file_digest}
        )

    with (lock_directory / "matrix_runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)

    with (lock_directory / "evidence_sha256.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(evidence_records)

    git_status = _run_git(repo_root, "status", "--porcelain=v1")
    (lock_directory / "git_status_at_lock.txt").write_text(
        git_status + ("\n" if git_status else ""), encoding="utf-8"
    )

    source_root_counts: dict[str, int] = {}
    for row in matrix_rows:
        source_root = row["source_root"]
        source_root_counts[source_root] = source_root_counts.get(source_root, 0) + 1

    unique_cells = {
        (row["dataset"], row["scenario"], row["stream_seed"], row["model_seed"])
        for row in matrix_rows
    }
    unique_streams = {
        (row["dataset"], row["scenario"], row["stream_seed"])
        for row in matrix_rows
    }
    manifest = {
        "lock_version": "1.0",
        "lock_name": "pre-revision-4000-run-confirmation",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "repository": {
            "remote_origin": _run_git(repo_root, "remote", "get-url", "origin"),
            "commit": _run_git(repo_root, "rev-parse", "HEAD"),
            "branch": _run_git(repo_root, "branch", "--show-current"),
            "working_tree_clean": not bool(git_status),
            "working_tree_status_file": "git_status_at_lock.txt",
        },
        "matrix": {
            "rows": len(matrix_rows),
            "paired_cells": len(unique_cells),
            "prepared_streams": len(unique_streams),
            "methods": sorted({row["method"] for row in matrix_rows}),
            "source_root_run_counts": source_root_counts,
            "mapping_file": "matrix_runs.csv",
        },
        "evidence": {
            "file_count": len(evidence_records),
            "total_bytes": total_bytes,
            "aggregate_sha256": aggregate_digest.hexdigest(),
            "file_manifest": "evidence_sha256.csv",
        },
        "inspection_environment": {
            "note": (
                "This records the environment used to create the lock and is not "
                "asserted to be the original 4000-run execution environment."
            ),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": _package_versions(),
        },
    }
    (lock_directory / "baseline_lock.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def verify_lock(repo_root: Path, lock_directory: Path) -> None:
    repo_root = repo_root.resolve()
    lock_directory = lock_directory.resolve()
    manifest_path = lock_directory / "baseline_lock.json"
    evidence_path = lock_directory / "evidence_sha256.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with evidence_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))

    aggregate_digest = hashlib.sha256()
    missing_paths: list[str] = []
    changed_paths: list[str] = []
    total_bytes = 0
    for record in records:
        relative_path = record["path"]
        expected_size = int(record["size_bytes"])
        expected_digest = record["sha256"]
        file_path = repo_root / relative_path
        if not file_path.is_file():
            missing_paths.append(relative_path)
            continue
        actual_size = file_path.stat().st_size
        actual_digest = _sha256(file_path)
        total_bytes += actual_size
        if actual_size != expected_size or actual_digest != expected_digest:
            changed_paths.append(relative_path)
        aggregate_digest.update(
            f"{relative_path}\t{actual_size}\t{actual_digest}\n".encode("utf-8")
        )

    expected_aggregate = manifest["evidence"]["aggregate_sha256"]
    actual_aggregate = aggregate_digest.hexdigest()
    if missing_paths or changed_paths or actual_aggregate != expected_aggregate:
        details = {
            "missing_count": len(missing_paths),
            "changed_count": len(changed_paths),
            "missing_paths": missing_paths[:20],
            "changed_paths": changed_paths[:20],
            "expected_aggregate_sha256": expected_aggregate,
            "actual_aggregate_sha256": actual_aggregate,
        }
        raise RuntimeError(json.dumps(details, indent=2))
    print(
        json.dumps(
            {
                "status": "PASS",
                "file_count": len(records),
                "total_bytes": total_bytes,
                "aggregate_sha256": actual_aggregate,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--lock-directory",
        type=Path,
        default=Path("revision/baseline_lock_2026-09-01"),
    )
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    lock_directory = (
        arguments.lock_directory
        if arguments.lock_directory.is_absolute()
        else arguments.repo_root / arguments.lock_directory
    )
    if arguments.verify:
        verify_lock(arguments.repo_root, lock_directory)
    else:
        create_lock(arguments.repo_root, lock_directory)


if __name__ == "__main__":
    main()
