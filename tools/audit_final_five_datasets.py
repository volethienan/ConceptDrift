from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from dasp_nids.data import NF_V3_NON_FEATURE_COLUMNS
from tools.build_controlled_cic_streams import CIC_SPEC
from tools.build_controlled_nf_streams import DATASETS, _sha256


ORDER = [
    "nf-unsw-nb15-v3",
    "nf-cse-cic-ids2018-v3",
    "nf-ton-iot-v3",
    "nf-bot-iot-v3",
    "cic-ids2017",
]


def _duplicate_count(path: Path, size: int) -> int:
    values = np.memmap(path, mode="r+", dtype=np.uint64, shape=(size,))
    values.sort()
    duplicates = int(np.sum(values[1:] == values[:-1])) if size > 1 else 0
    del values
    return duplicates


def _binary_label(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return numeric.to_numpy(dtype=np.int8)
    text = values.astype(str).str.strip().str.casefold()
    return (~text.eq("benign")).to_numpy(dtype=np.int8)


def _audit_files(
    dataset_key: str,
    files: list[Path],
    total_rows: int,
    feature_columns: list[str],
    label_column: str,
    subtype_column: str,
    timestamp_column: str | None,
    output_dir: Path,
    chunksize: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_hash_path = output_dir / f"{dataset_key}.row_hashes.u64"
    feature_label_hash_path = (
        output_dir / f"{dataset_key}.feature_label_hashes.u64"
    )
    row_hashes = np.memmap(
        row_hash_path, mode="w+", dtype=np.uint64, shape=(total_rows,)
    )
    feature_label_hashes = np.memmap(
        feature_label_hash_path,
        mode="w+",
        dtype=np.uint64,
        shape=(total_rows,),
    )
    label_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    infinite: Counter[str] = Counter()
    minimum = np.full(len(feature_columns), np.inf, dtype=float)
    maximum = np.full(len(feature_columns), -np.inf, dtype=float)
    direct_proxy = np.ones(len(feature_columns), dtype=bool)
    timestamp_inversions = 0
    timestamp_pairs = 0
    previous_timestamp: float | None = None
    file_rows: dict[str, int] = {}
    offset = 0
    for path in files:
        count = 0
        for chunk_index, chunk in enumerate(
            pd.read_csv(path, chunksize=chunksize, low_memory=False)
        ):
            chunk.columns = [str(value).strip() for value in chunk.columns]
            size = len(chunk)
            labels = _binary_label(chunk[label_column])
            label_counts.update(
                ["Attack" if value else "Benign" for value in labels]
            )
            subtype_values = (
                chunk[subtype_column]
                .astype(str)
                .str.strip()
                .str.casefold()
                .str.replace("\ufffd", "-", regex=False)
            )
            subtype_counts.update(subtype_values.tolist())

            numeric = (
                chunk[feature_columns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64, copy=True)
            )
            missing_counts = chunk[feature_columns].isna().sum()
            for column, value in missing_counts[missing_counts > 0].items():
                missing[column] += int(value)
            inf_mask = np.isinf(numeric)
            for index in np.flatnonzero(inf_mask.any(axis=0)):
                infinite[feature_columns[index]] += int(
                    inf_mask[:, index].sum()
                )
            finite = np.isfinite(numeric)
            for index in range(len(feature_columns)):
                values = numeric[finite[:, index], index]
                if len(values):
                    minimum[index] = min(minimum[index], float(values.min()))
                    maximum[index] = max(maximum[index], float(values.max()))
                if (
                    not finite[:, index].all()
                    or not np.array_equal(
                        numeric[:, index], labels.astype(float)
                    )
                ):
                    direct_proxy[index] = False

            row_hashes[offset : offset + size] = (
                pd.util.hash_pandas_object(chunk, index=False)
                .to_numpy(dtype=np.uint64)
            )
            feature_label = chunk[feature_columns].copy()
            feature_label["__binary_label"] = labels
            feature_label_hashes[offset : offset + size] = (
                pd.util.hash_pandas_object(feature_label, index=False)
                .to_numpy(dtype=np.uint64)
            )

            if timestamp_column is not None:
                timestamps = pd.to_numeric(
                    chunk[timestamp_column], errors="coerce"
                ).to_numpy(dtype=float)
                valid = timestamps[np.isfinite(timestamps)]
                if len(valid):
                    if previous_timestamp is not None:
                        timestamp_inversions += int(
                            valid[0] < previous_timestamp
                        )
                        timestamp_pairs += 1
                    timestamp_inversions += int(
                        np.sum(valid[1:] < valid[:-1])
                    )
                    timestamp_pairs += max(0, len(valid) - 1)
                    previous_timestamp = float(valid[-1])
            offset += size
            count += size
            print(
                f"audit dataset={dataset_key} file={path.name} "
                f"chunk={chunk_index + 1} rows={count}",
                flush=True,
            )
        file_rows[path.name] = count
    if offset != total_rows:
        raise ValueError(
            f"{dataset_key}: expected {total_rows} rows, observed {offset}"
        )
    row_hashes.flush()
    feature_label_hashes.flush()
    del row_hashes
    del feature_label_hashes
    exact_duplicates = _duplicate_count(row_hash_path, total_rows)
    feature_label_duplicates = _duplicate_count(
        feature_label_hash_path, total_rows
    )
    constants = [
        feature_columns[index]
        for index in range(len(feature_columns))
        if np.isfinite(minimum[index])
        and minimum[index] == maximum[index]
    ]
    proxies = [
        feature_columns[index]
        for index in range(len(feature_columns))
        if direct_proxy[index]
    ]
    return {
        "dataset_key": dataset_key,
        "files": [str(path.resolve()) for path in files],
        "sha256": {path.name: _sha256(path) for path in files},
        "schema_columns": len(feature_columns)
        + (1 if label_column == subtype_column else 2),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "binary_label_column": label_column,
        "attack_subtype_column": subtype_column,
        "rows": total_rows,
        "file_rows": file_rows,
        "label_counts": dict(label_counts),
        "attack_family_counts": dict(subtype_counts),
        "missing": dict(missing),
        "infinite": dict(infinite),
        "exact_duplicate_hash_count": exact_duplicates,
        "exact_duplicate_hash_fraction": exact_duplicates / total_rows,
        "feature_label_duplicate_hash_count": feature_label_duplicates,
        "feature_label_duplicate_hash_fraction": (
            feature_label_duplicates / total_rows
        ),
        "constant_features": constants,
        "direct_binary_label_proxies": proxies,
        "timestamp_column": timestamp_column,
        "timestamp_adjacent_inversions": timestamp_inversions,
        "timestamp_adjacent_pairs": timestamp_pairs,
        "timestamp_inversion_fraction": (
            timestamp_inversions / timestamp_pairs
            if timestamp_pairs
            else None
        ),
        "session_ground_truth_available": False,
        "duplicate_hash_note": (
            "Counts use 64-bit pandas row hashes; collision probability is "
            "negligible but hashes are not a byte-for-byte duplicate proof."
        ),
    }


def _nf_audit(
    dataset_key: str, output_dir: Path, chunksize: int
) -> dict[str, object]:
    spec = DATASETS[dataset_key]
    path = Path(spec.path)
    header = [
        str(value).strip()
        for value in pd.read_csv(path, nrows=0).columns
    ]
    features = [
        column
        for column in header
        if column not in NF_V3_NON_FEATURE_COLUMNS
        and not column.startswith("__")
    ]
    return _audit_files(
        dataset_key,
        [path],
        spec.total_rows,
        features,
        "Label",
        "Attack",
        "FLOW_START_MILLISECONDS",
        output_dir,
        chunksize,
    )


def _cic_audit(output_dir: Path, chunksize: int) -> dict[str, object]:
    paths = sorted(Path(CIC_SPEC.path).glob("*.csv"))
    schemas = [
        [
            str(value).strip()
            for value in pd.read_csv(path, nrows=0).columns
        ]
        for path in paths
    ]
    if any(columns != schemas[0] for columns in schemas[1:]):
        raise ValueError("CIC-IDS2017 resolved schemas differ")
    features = [column for column in schemas[0] if column != "Label"]
    result = _audit_files(
        "cic-ids2017",
        paths,
        CIC_SPEC.total_rows,
        features,
        "Label",
        "Label",
        None,
        output_dir,
        chunksize,
    )
    result["schema_specific_notes"] = [
        "The raw header repeats Fwd Header Length; pandas resolves the second occurrence as Fwd Header Length.1.",
        "No IP addresses, flow ID, session ID, or timestamp are present in the supplied CSV files.",
        "Capture-day files are retained separately in provenance and are not forced into the NF-v3 schema.",
    ]
    return result


def _write_report(results: list[dict[str, object]], destination: Path) -> None:
    lines = [
        "# Final five-dataset audit",
        "",
        "| Dataset | Rows | Features | Benign | Attack | Missing | Inf | Exact duplicate hashes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        labels = result["label_counts"]
        lines.append(
            f"| {result['dataset_key']} | {result['rows']:,} | "
            f"{result['feature_count']} | {labels.get('Benign', 0):,} | "
            f"{labels.get('Attack', 0):,} | "
            f"{sum(result['missing'].values()):,} | "
            f"{sum(result['infinite'].values()):,} | "
            f"{result['exact_duplicate_hash_count']:,} |"
        )
    lines.extend(
        [
            "",
            "All preprocessing artifacts must fit imputation and scaling only "
            "on W0-W3. Attack subtype is retained for stream construction and "
            "subgroup audit only; it is never included in model features.",
            "",
            "CIC-IDS2017 has no session or timestamp field in the supplied "
            "schema. Its controlled-stream builder therefore enforces source-"
            "row and duplicate-group disjointness, while explicitly avoiding "
            "a claim of ground-truth session disjointness.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/final_five_dataset")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--datasets", default=",".join(ORDER))
    args = parser.parse_args()
    destination = Path(args.output_dir)
    selected = [
        value.strip() for value in args.datasets.split(",") if value.strip()
    ]
    unknown = set(selected).difference(ORDER)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    results: list[dict[str, object]] = []
    for dataset_key in selected:
        result = (
            _cic_audit(destination, args.chunksize)
            if dataset_key == "cic-ids2017"
            else _nf_audit(dataset_key, destination, args.chunksize)
        )
        results.append(result)
        (destination / f"{dataset_key}_audit.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (destination / "audit_all.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(results, destination / "AUDIT_REPORT.md")


if __name__ == "__main__":
    main()
