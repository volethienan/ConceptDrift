from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from dasp_nids.data import NF_V3_NON_FEATURE_COLUMNS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(value: pd.Series) -> pd.Series:
    return value.astype(str).str.strip().str.casefold()


def _duplicate_count(parts: list[np.ndarray]) -> int:
    hashes = np.concatenate(parts)
    return int(len(hashes) - len(np.unique(hashes)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-file audit for the NF-UNSW-NB15-v3 pilot."
    )
    parser.add_argument(
        "--input",
        default="data/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/nf_unsw_pilot/audit.json",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(source)

    header = pd.read_csv(source, nrows=0).columns.astype(str).tolist()
    feature_columns = [
        column
        for column in header
        if column not in NF_V3_NON_FEATURE_COLUMNS
        and not column.startswith("__")
    ]
    required = {
        "Label",
        "Attack",
        "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "PROTOCOL",
    }
    missing_required = required.difference(header)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    rows = 0
    label_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    missing_counts = Counter({name: 0 for name in header})
    infinity_counts = Counter({name: 0 for name in feature_columns})
    finite_min = {name: np.inf for name in feature_columns}
    finite_max = {name: -np.inf for name in feature_columns}
    numeric_non_missing = Counter({name: 0 for name in feature_columns})
    raw_hash_parts: list[np.ndarray] = []
    feature_label_hash_parts: list[np.ndarray] = []
    session_hash_parts: list[np.ndarray] = []
    label_attack_mismatches = 0
    direct_label_proxy_matches = Counter({name: 0 for name in feature_columns})
    direct_inverse_proxy_matches = Counter(
        {name: 0 for name in feature_columns}
    )
    timestamp_min = np.inf
    timestamp_max = -np.inf
    adjacent_time_pairs = 0
    adjacent_time_inversions = 0
    previous_timestamp: float | None = None
    observed_dtypes: dict[str, set[str]] = {name: set() for name in header}

    for chunk_index, chunk in enumerate(
        pd.read_csv(source, chunksize=args.chunksize, low_memory=False)
    ):
        rows += len(chunk)
        for name, dtype in chunk.dtypes.items():
            observed_dtypes[str(name)].add(str(dtype))
        missing_counts.update(
            {
                str(name): int(value)
                for name, value in chunk.isna().sum().items()
            }
        )

        labels = _normalise(chunk["Label"])
        families = _normalise(chunk["Attack"])
        label_counts.update(labels.tolist())
        family_counts.update(families.tolist())
        numeric_labels = pd.to_numeric(
            chunk["Label"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if (
            np.any(~np.isfinite(numeric_labels))
            or not np.all(np.isin(numeric_labels, [0.0, 1.0]))
        ):
            raise ValueError(
                "NF-UNSW Label must contain only finite binary 0/1 values"
            )
        binary_label = numeric_labels.astype(np.int8)
        binary_family = (families != "benign").to_numpy(dtype=np.int8)
        label_attack_mismatches += int(
            np.count_nonzero(binary_label != binary_family)
        )

        numeric = chunk[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        values = numeric.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        infinity = np.isinf(values)
        for index, name in enumerate(feature_columns):
            column = values[:, index]
            mask = finite[:, index]
            infinity_counts[name] += int(np.count_nonzero(infinity[:, index]))
            numeric_non_missing[name] += int(np.count_nonzero(mask))
            if np.any(mask):
                finite_min[name] = min(
                    finite_min[name], float(np.min(column[mask]))
                )
                finite_max[name] = max(
                    finite_max[name], float(np.max(column[mask]))
                )
            comparable = mask & np.isin(column, [0.0, 1.0])
            if np.any(comparable):
                direct_label_proxy_matches[name] += int(
                    np.count_nonzero(
                        comparable
                        & (column == binary_label.astype(np.float64))
                    )
                )
                direct_inverse_proxy_matches[name] += int(
                    np.count_nonzero(
                        comparable
                        & (column == 1.0 - binary_label.astype(np.float64))
                    )
                )

        raw_hash_parts.append(
            pd.util.hash_pandas_object(chunk, index=False).to_numpy(
                dtype=np.uint64
            )
        )
        feature_label = numeric.copy()
        feature_label["__binary_label"] = binary_label
        feature_label_hash_parts.append(
            pd.util.hash_pandas_object(
                feature_label, index=False
            ).to_numpy(dtype=np.uint64)
        )
        session_hash_parts.append(
            pd.util.hash_pandas_object(
                chunk[
                    [
                        "IPV4_SRC_ADDR",
                        "IPV4_DST_ADDR",
                        "L4_SRC_PORT",
                        "L4_DST_PORT",
                        "PROTOCOL",
                    ]
                ],
                index=False,
            ).to_numpy(dtype=np.uint64)
        )

        timestamp = pd.to_numeric(
            chunk["FLOW_START_MILLISECONDS"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        finite_time = timestamp[np.isfinite(timestamp)]
        if len(finite_time):
            timestamp_min = min(timestamp_min, float(finite_time.min()))
            timestamp_max = max(timestamp_max, float(finite_time.max()))
            if previous_timestamp is not None:
                adjacent_time_pairs += 1
                adjacent_time_inversions += int(
                    finite_time[0] < previous_timestamp
                )
            if len(finite_time) > 1:
                adjacent_time_pairs += len(finite_time) - 1
                adjacent_time_inversions += int(
                    np.count_nonzero(np.diff(finite_time) < 0)
                )
            previous_timestamp = float(finite_time[-1])
        print(
            f"audit chunk={chunk_index + 1} cumulative_rows={rows}",
            flush=True,
        )

    constant_features = [
        name
        for name in feature_columns
        if numeric_non_missing[name] > 0
        and finite_min[name] == finite_max[name]
    ]
    all_missing_features = [
        name for name in feature_columns if numeric_non_missing[name] == 0
    ]
    direct_proxies = []
    for name in feature_columns:
        comparable_rows = rows - missing_counts[name] - infinity_counts[name]
        if comparable_rows <= 0:
            continue
        direct_rate = direct_label_proxy_matches[name] / comparable_rows
        inverse_rate = direct_inverse_proxy_matches[name] / comparable_rows
        if max(direct_rate, inverse_rate) >= 0.999:
            direct_proxies.append(
                {
                    "feature": name,
                    "direct_match_rate": direct_rate,
                    "inverse_match_rate": inverse_rate,
                }
            )

    session_hashes = np.concatenate(session_hash_parts)
    report = {
        "source": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "file_count": 1,
        },
        "schema": {
            "rows": rows,
            "column_count": len(header),
            "feature_count_before_constant_removal": len(feature_columns),
            "columns": header,
            "observed_chunk_dtypes": {
                name: sorted(values)
                for name, values in observed_dtypes.items()
            },
            "binary_label_column": "Label",
            "attack_subtype_column": "Attack",
            "excluded_non_feature_columns": sorted(
                NF_V3_NON_FEATURE_COLUMNS
            ),
        },
        "labels": {
            "normalisation": {"Benign": 0, "Attack": 1},
            "raw_label_counts": dict(label_counts),
            "binary_counts": {
                "Benign": int(label_counts.get("0", 0)),
                "Attack": int(label_counts.get("1", 0)),
            },
            "attack_family_counts": dict(
                sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))
            ),
            "label_attack_mismatches": label_attack_mismatches,
        },
        "quality": {
            "missing_total": int(sum(missing_counts.values())),
            "missing_by_column": {
                name: int(count)
                for name, count in missing_counts.items()
                if count
            },
            "infinity_total_features": int(sum(infinity_counts.values())),
            "infinity_by_feature": {
                name: int(count)
                for name, count in infinity_counts.items()
                if count
            },
            "exact_duplicate_rows": _duplicate_count(raw_hash_parts),
            "duplicate_feature_label_rows": _duplicate_count(
                feature_label_hash_parts
            ),
            "constant_features": constant_features,
            "all_missing_features": all_missing_features,
        },
        "leakage": {
            "direct_or_inverse_binary_label_proxies": direct_proxies,
            "excluded_label_columns": ["Label", "Attack"],
            "excluded_identifiers": [
                "IPV4_SRC_ADDR",
                "IPV4_DST_ADDR",
            ],
            "excluded_time_columns": [
                "FLOW_START_MILLISECONDS",
                "FLOW_END_MILLISECONDS",
            ],
            "note": (
                "Attack subtype is retained only in the frozen stream "
                "manifest and is never exposed to the classifier."
            ),
        },
        "split_capability": {
            "by_file": {
                "possible": False,
                "reason": "The dataset is supplied as one CSV file.",
            },
            "by_time": {
                "possible": True,
                "column": "FLOW_START_MILLISECONDS",
                "minimum": (
                    int(timestamp_min) if np.isfinite(timestamp_min) else None
                ),
                "maximum": (
                    int(timestamp_max) if np.isfinite(timestamp_max) else None
                ),
                "adjacent_inversion_fraction": (
                    adjacent_time_inversions / adjacent_time_pairs
                    if adjacent_time_pairs
                    else None
                ),
                "input_is_monotonic": adjacent_time_inversions == 0,
            },
            "by_session": {
                "possible": True,
                "explicit_session_id": False,
                "candidate_key": [
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                    "L4_SRC_PORT",
                    "L4_DST_PORT",
                    "PROTOCOL",
                ],
                "unique_directional_5tuples": int(
                    len(np.unique(session_hashes))
                ),
                "warning": (
                    "A 5-tuple is a derived grouping key, not a supplied "
                    "ground-truth session identifier."
                ),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
