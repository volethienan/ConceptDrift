from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tools.build_controlled_nf_streams import (
    CANONICAL_SCENARIOS,
    HASH_SALT,
    RANK_SALT,
    DatasetSpec,
    _build_stream,
    _requirements,
    _sha256,
    _splitmix64,
    _write_stream,
)

CIC_ORDER_BLOCK_ROWS = np.int64(10_000)
CIC_BLOCK_SALT = np.uint64(0xD1B54A32D192ED03)


CIC_SPEC = DatasetSpec(
    name="CIC-IDS2017",
    path="data/CIC-IDS2017",
    total_rows=2_830_743,
    attack_rows=557_646,
    family_counts={
        "benign": 2_273_097,
        "ddos": 128_027,
        "portscan": 158_930,
        "bot": 1_966,
        "infiltration": 36,
        "web attack - brute force": 1_507,
        "web attack - xss": 652,
        "web attack - sql injection": 21,
        "ftp-patator": 7_938,
        "ssh-patator": 5_897,
        "dos slowloris": 5_796,
        "dos slowhttptest": 5_499,
        "dos hulk": 231_073,
        "dos goldeneye": 10_293,
        "heartbleed": 11,
    },
    concept_a={"dos hulk": 0.70, "ddos": 0.30},
    concept_b={"portscan": 0.60, "ddos": 0.40},
    concept_c={
        "dos hulk": 0.20,
        "ddos": 0.30,
        "portscan": 0.40,
        "dos goldeneye": 0.05,
        "dos slowloris": 0.05,
    },
    stable={"dos hulk": 0.70, "ddos": 0.30},
    local_a={"ddos": 1.0},
    local_b={"portscan": 1.0},
)


def _normalise_labels(values: pd.Series) -> np.ndarray:
    return (
        values.astype(str)
        .str.strip()
        .str.casefold()
        .str.replace("\ufffd", "-", regex=False)
        .to_numpy(dtype=str)
    )


def _schema(paths: list[Path]) -> list[str]:
    schemas: list[list[str]] = []
    for path in paths:
        columns = [
            str(value).strip()
            for value in pd.read_csv(path, nrows=0, low_memory=False).columns
        ]
        schemas.append(columns)
    if any(columns != schemas[0] for columns in schemas[1:]):
        raise ValueError("CIC-IDS2017 CSV files do not share one resolved schema")
    if "Label" not in schemas[0]:
        raise ValueError("CIC-IDS2017 Label column was not found")
    return [column for column in schemas[0] if column != "Label"]


def _collect_pools(
    paths: list[Path],
    feature_columns: list[str],
    stream_seeds: list[int],
    required: dict[str, int],
    chunksize: int,
) -> tuple[
    dict[tuple[str, int], dict[str, np.ndarray]],
    Counter[str],
    dict[str, int],
]:
    n_slots = len(stream_seeds)
    takes = {
        family: int(
            # Candidate-only reserve for indivisible duplicate groups. The
            # emitted windows still contain exactly the protocol quota.
            np.ceil((2.00 if family == "benign" else 1.14) * count)
        )
        for family, count in required.items()
    }
    collected: dict[tuple[str, int], dict[str, list[np.ndarray]]] = {
        (family, slot): {
            "x": [],
            "source_ids": [],
            "session_ids": [],
            "timestamps": [],
            "rank": [],
        }
        for family in required
        for slot in range(n_slots)
    }
    observed: Counter[str] = Counter()
    per_file_rows: dict[str, int] = {}
    row_offset = 0
    for file_index, path in enumerate(paths):
        file_rows = 0
        for chunk_index, chunk in enumerate(
            pd.read_csv(path, chunksize=chunksize, low_memory=False)
        ):
            chunk.columns = [str(value).strip() for value in chunk.columns]
            row_ids = np.arange(
                row_offset, row_offset + len(chunk), dtype=np.uint64
            )
            local_rows = np.arange(
                file_rows, file_rows + len(chunk), dtype=np.int64
            )
            row_offset += len(chunk)
            file_rows += len(chunk)
            families = _normalise_labels(chunk["Label"])
            counts = Counter(families.tolist())
            for family in required:
                observed[family] += counts.get(family, 0)

            # CIC-IDS2017 has neither flow IDs nor timestamps in these files.
            # Treating every identical feature row over an entire capture as
            # one "session" creates enormous, scientifically unjustified
            # groups. Bound the exact-feature duplicate proxy by a contiguous
            # 10k-row source-order block within each CSV.
            feature_hashes = pd.util.hash_pandas_object(
                chunk[feature_columns], index=False
            ).to_numpy(dtype=np.uint64)
            block_ids = (
                np.uint64(file_index) * np.uint64(1_000_000)
                + (
                    local_rows.astype(np.uint64)
                    // np.uint64(CIC_ORDER_BLOCK_ROWS)
                )
            )
            group_hashes = _splitmix64(
                feature_hashes
                ^ _splitmix64(block_ids ^ CIC_BLOCK_SALT)
            )
            assignment = _splitmix64(group_hashes ^ HASH_SALT)
            slots = assignment % np.uint64(n_slots)
            ranks = _splitmix64(row_ids ^ RANK_SALT)
            union = np.zeros(len(chunk), dtype=bool)
            masks: dict[tuple[str, int], np.ndarray] = {}
            for family in required:
                expected = CIC_SPEC.family_counts[family] / n_slots
                probability = min(
                    1.0,
                    (takes[family] / max(1.0, expected)) * 1.25 + 0.01,
                )
                if family != "benign":
                    probability = 1.0
                rank_limit = np.uint64(
                    min((2**64) - 1, int(probability * ((2**64) - 1)))
                )
                family_mask = families == family
                for slot in range(n_slots):
                    mask = (
                        family_mask
                        & (slots == np.uint64(slot))
                        & (ranks <= rank_limit)
                    )
                    if np.any(mask):
                        masks[(family, slot)] = mask
                        union |= mask
            if np.any(union):
                numeric = (
                    chunk.loc[union, feature_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .to_numpy(dtype=np.float32, copy=True)
                )
                positions = np.flatnonzero(union)
                map_to_selected = np.full(len(chunk), -1, dtype=np.int64)
                map_to_selected[positions] = np.arange(len(positions))
                timestamps = (
                    np.int64(file_index) * np.int64(1_000_000_000)
                    + local_rows
                )
                for key, mask in masks.items():
                    local = map_to_selected[np.flatnonzero(mask)]
                    collected[key]["x"].append(numeric[local])
                    collected[key]["source_ids"].append(
                        row_ids[mask].astype(np.int64)
                    )
                    collected[key]["session_ids"].append(group_hashes[mask])
                    collected[key]["timestamps"].append(timestamps[mask])
                    collected[key]["rank"].append(ranks[mask])
            print(
                f"scan CIC file={path.name} chunk={chunk_index + 1} "
                f"rows={file_rows}",
                flush=True,
            )
        per_file_rows[path.name] = file_rows

    if row_offset != CIC_SPEC.total_rows:
        raise ValueError(
            f"CIC-IDS2017 expected {CIC_SPEC.total_rows} rows, got {row_offset}"
        )
    for family in required:
        if observed[family] != CIC_SPEC.family_counts[family]:
            raise ValueError(
                f"CIC-IDS2017/{family}: expected "
                f"{CIC_SPEC.family_counts[family]}, got {observed[family]}"
            )

    merged_by_key: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}
    for key, values in collected.items():
        if not values["x"]:
            raise ValueError(f"CIC-IDS2017/{key}: no rows collected")
        merged_by_key[key] = {
            name: np.concatenate(parts) for name, parts in values.items()
        }

    group_family_count: Counter[int] = Counter()
    for family in required:
        group_family_count.update(
            np.unique(
                np.concatenate(
                    [
                        merged_by_key[(family, slot)]["session_ids"]
                        for slot in range(n_slots)
                    ]
                )
            ).tolist()
        )
    ambiguous_groups = np.asarray(
        [
            group
            for group, count in group_family_count.items()
            if count > 1
        ],
        dtype=np.uint64,
    )
    balanced_by_key: dict[
        tuple[str, int], dict[str, np.ndarray]
    ] = {}
    for family in required:
        merged_family = {
            name: np.concatenate(
                [
                    merged_by_key[(family, slot)][name]
                    for slot in range(n_slots)
                ]
            )
            for name in merged_by_key[(family, 0)]
        }
        if len(ambiguous_groups):
            keep = ~np.isin(
                merged_family["session_ids"], ambiguous_groups
            )
            merged_family = {
                name: values[keep]
                for name, values in merged_family.items()
            }
        groups, inverse = np.unique(
            merged_family["session_ids"], return_inverse=True
        )
        sizes = np.bincount(inverse, minlength=len(groups))
        tiebreak = _splitmix64(groups ^ HASH_SALT)
        group_order = np.lexsort((tiebreak, -sizes))
        loads = np.zeros(n_slots, dtype=np.int64)
        group_slot = np.empty(len(groups), dtype=np.int16)
        for group_index in group_order:
            slot = int(np.argmin(loads))
            group_slot[int(group_index)] = slot
            loads[slot] += int(sizes[int(group_index)])
        target = takes[family]
        while np.any(loads < target):
            receiver = int(np.argmin(loads))
            candidates: list[tuple[int, int, int]] = []
            for group_index, donor in enumerate(group_slot.tolist()):
                if donor == receiver:
                    continue
                size = int(sizes[group_index])
                if loads[donor] - size < target:
                    continue
                imbalance = abs(
                    int(loads[receiver] + size)
                    - int(loads[donor] - size)
                )
                candidates.append((imbalance, size, group_index))
            if not candidates:
                break
            _, size, group_index = min(candidates)
            donor = int(group_slot[group_index])
            group_slot[group_index] = receiver
            loads[donor] -= size
            loads[receiver] += size
        row_slots = group_slot[inverse]
        for slot in range(n_slots):
            indexes = np.flatnonzero(row_slots == slot)
            balanced_by_key[(family, slot)] = {
                name: values[indexes]
                for name, values in merged_family.items()
            }
    merged_by_key = balanced_by_key

    pools: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for key, merged in merged_by_key.items():
        order = np.argsort(merged["rank"], kind="stable")
        take = takes[key[0]]
        if len(order) < take:
            raise ValueError(
                f"CIC-IDS2017/{key}: collected {len(order)}, needs {take}"
            )
        chosen = order[:take]
        pools[key] = {
            name: array[chosen]
            for name, array in merged.items()
            if name != "rank"
        }
    return pools, observed, per_file_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-audited controlled CIC-IDS2017 streams."
    )
    parser.add_argument(
        "--scenarios", default=",".join(CANONICAL_SCENARIOS)
    )
    parser.add_argument(
        "--stream-seeds",
        default="2001,2002,2003,2101,2102,2103,2104,2105",
    )
    parser.add_argument("--output-dir", default="prepared_streams_final")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--n-windows", type=int, default=36)
    parser.add_argument("--warmup-windows", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    scenarios = tuple(
        value.strip()
        for value in args.scenarios.split(",")
        if value.strip()
    )
    unknown = set(scenarios).difference(CANONICAL_SCENARIOS)
    if unknown:
        raise ValueError(f"Unsupported CIC scenarios: {sorted(unknown)}")
    stream_seeds = [
        int(value.strip())
        for value in args.stream_seeds.split(",")
        if value.strip()
    ]
    if len(stream_seeds) != len(set(stream_seeds)):
        raise ValueError("stream seeds must be unique")
    root = Path(args.output_dir)
    expected = [
        root / "cic-ids2017" / scenario / f"stream_seed{seed}.npz"
        for scenario in scenarios
        for seed in stream_seeds
    ]
    if args.resume and all(
        path.exists() and path.with_suffix(".manifest.json").exists()
        for path in expected
    ):
        print("resume CIC-IDS2017: all artifacts exist", flush=True)
        return

    paths = sorted(Path(CIC_SPEC.path).glob("*.csv"))
    if len(paths) != 8:
        raise FileNotFoundError(
            f"Expected eight CIC-IDS2017 files, found {len(paths)}"
        )
    feature_columns = _schema(paths)
    attack_count, benign_count, required = _requirements(
        CIC_SPEC,
        scenarios,
        args.window_size,
        args.n_windows,
        0.05,
    )
    pools, observed, per_file_rows = _collect_pools(
        paths,
        feature_columns,
        stream_seeds,
        required,
        args.chunksize,
    )

    for left_index, left_seed in enumerate(stream_seeds):
        left_slot = left_index
        left_ids = set(
            np.concatenate(
                [
                    pools[(family, left_slot)]["source_ids"]
                    for family in required
                ]
            ).tolist()
        )
        left_groups = set(
            np.concatenate(
                [
                    pools[(family, left_slot)]["session_ids"]
                    for family in required
                ]
            ).tolist()
        )
        for right_index in range(left_index + 1, len(stream_seeds)):
            right_seed = stream_seeds[right_index]
            right_ids = set(
                np.concatenate(
                    [
                        pools[(family, right_index)]["source_ids"]
                        for family in required
                    ]
                ).tolist()
            )
            right_groups = set(
                np.concatenate(
                    [
                        pools[(family, right_index)]["session_ids"]
                        for family in required
                    ]
                ).tolist()
            )
            if left_ids.intersection(right_ids):
                raise ValueError(
                    f"CIC data seeds {left_seed}/{right_seed} share rows"
                )
            if left_groups.intersection(right_groups):
                raise ValueError(
                    f"CIC data seeds {left_seed}/{right_seed} share "
                    "duplicate groups"
                )

    checksums = {path.name: _sha256(path) for path in paths}
    for slot, seed in enumerate(stream_seeds):
        for scenario in scenarios:
            arrays, _ = _build_stream(
                CIC_SPEC,
                feature_columns,
                pools,
                {},
                None,
                slot,
                seed,
                scenario,
                args.window_size,
                args.n_windows,
                args.warmup_windows,
                0.05,
                4,
                0.70,
                0.30,
                0.20,
                0.10,
                0.10,
                0.05,
                "enforce",
            )
            destination = (
                root
                / "cic-ids2017"
                / scenario
                / f"stream_seed{seed}.npz"
            )
            manifest = {
                "protocol_version": "2.0",
                "dataset_key": "cic-ids2017",
                "dataset_name": CIC_SPEC.name,
                "source_files": [path.name for path in paths],
                "source_sha256": checksums,
                "source_rows": CIC_SPEC.total_rows,
                "source_rows_by_file": per_file_rows,
                "observed_required_family_counts": dict(observed),
                "scenario": scenario,
                "stream_seed": seed,
                "window_size": args.window_size,
                "n_windows": args.n_windows,
                "warmup_windows": args.warmup_windows,
                "attack_count_per_window": attack_count,
                "benign_count_per_window": benign_count,
                "realized_attack_rate": attack_count / args.window_size,
                "feature_count": len(feature_columns),
                "feature_names": feature_columns,
                "unique_source_rows": int(
                    len(np.unique(arrays["source_ids"]))
                ),
                "overlap_within_stream": 0,
                "duplicate_group_overlap_within_stream": 0,
                "cross_seed_overlap": 0,
                "seed_assignment": {
                    "method": (
                        "exact-feature hash x 10000-row source-order block; "
                        "whole-group deterministic load balancing"
                    ),
                    "salt": int(HASH_SALT),
                    "selection_basis": (
                        "label-blind quota feasibility; no model outcome"
                    ),
                },
                "preprocessing_fit_windows": list(
                    range(args.warmup_windows)
                ),
                "concept_definitions": {
                    "A": CIC_SPEC.concept_a,
                    "B": CIC_SPEC.concept_b,
                    "C": CIC_SPEC.concept_c,
                    "A_prime": CIC_SPEC.concept_a,
                    "local_stable": CIC_SPEC.stable,
                    "local_A": CIC_SPEC.local_a,
                    "local_B": CIC_SPEC.local_b,
                },
                "limitations": [
                    "No source/destination IP, flow ID, timestamp, or ground-truth session identifier is present.",
                    "An exact-feature hash bounded to a contiguous 10000-row source-order block is used only as a local duplicate proxy, not as a claimed session ID.",
                    "Source-row overlap is ruled out, but true session-disjointness cannot be verified for CIC-IDS2017.",
                    "Attack families are partially confounded with capture-day CSV files.",
                ],
            }
            _write_stream(destination, arrays, manifest)
            print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
