"""Build controlled CIC-IDS2017 streams under the DASP-v2 protocol.

The archived CIC-IDS2017 streams (protocol 2.0) were produced by a builder that
is no longer available, so this is a new, separately versioned protocol. It
shares the scenario recipes, window assembly, segment partitioning and
preprocessing of ``build_controlled_nf_streams.py`` and only replaces how
source rows are read, cleaned and assigned to stream seeds.

CIC-IDS2017 MachineLearningCVE files carry no IP address, port pair, flow ID
or timestamp, so true sessions cannot be reconstructed. The protocol therefore:

1. removes feature vectors that occur with more than one label (label noise);
2. keeps only the first occurrence of each exact feature vector, so an
   identical flow can never appear in two windows, segments or seeds;
3. groups the remaining rows of a family into contiguous source-order blocks
   of ``--block-rows`` deduplicated rows, a temporal-neighbourhood proxy that
   is kept whole within one seed and one stream segment.

``timestamps`` holds the global source-order ordinal of a row (files in
capture order), not a time. These limitations are written to each manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.build_controlled_nf_streams import (  # noqa: E402
    CANONICAL_SCENARIOS,
    HASH_SALT,
    LOCAL_LEVELS,
    RANK_SALT,
    STRESS_SCENARIOS,
    DatasetSpec,
    _build_stream,
    _requirements,
    _sha256,
    _splitmix64,
    _write_stream,
)

PROTOCOL_VERSION = "3.0-cic"
GROUP_SALT = np.uint64(0xD1B54A32D192ED03)

# Capture order (3-7 July 2017). Friday afternoon PortScan precedes DDoS.
FILE_ORDER = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)

# Raw label counts are validated exactly; the binary prevalence is the raw
# capture prevalence, as in the archived protocol (807 attacks per 4,096).
CIC_IDS2017 = DatasetSpec(
    name="CIC-IDS2017",
    path="data/CIC-IDS2017",
    total_rows=2_830_743,
    attack_rows=557_646,
    family_counts={
        "benign": 2_273_097,
        "dos hulk": 231_073,
        "portscan": 158_930,
        "ddos": 128_027,
        "dos goldeneye": 10_293,
        "dos slowloris": 5_796,
    },
    concept_a={"dos hulk": 0.7, "ddos": 0.3},
    concept_b={"portscan": 0.6, "ddos": 0.4},
    concept_c={
        "dos hulk": 0.2,
        "ddos": 0.3,
        "portscan": 0.4,
        "dos goldeneye": 0.05,
        "dos slowloris": 0.05,
    },
    stable={"dos hulk": 0.7, "ddos": 0.3},
    local_a={"ddos": 1.0},
    local_b={"portscan": 1.0},
)

LIMITATIONS = [
    "No source/destination IP, flow ID, timestamp or ground-truth session "
    "identifier is present in the MachineLearningCVE files.",
    "Contiguous source-order blocks are a temporal-neighbourhood proxy, not "
    "sessions; session-disjointness cannot be verified.",
    "timestamps are global source-order ordinals, not times.",
    "Attack families are confounded with capture-day files.",
    "CIC-IDS2017 has documented labelling and feature-extraction errors "
    "(Engelen et al., 2021); labels are used as published.",
]


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="latin-1", low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Label" not in frame.columns:
        raise ValueError(f"{path}: missing Label column")
    return frame


def _family(frame: pd.DataFrame) -> np.ndarray:
    return frame["Label"].astype(str).str.strip().str.casefold().to_numpy(dtype=str)


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")


def scan_rows(source_dir: Path, file_order: tuple[str, ...]) -> dict[str, object]:
    """First pass: labels, exact-feature hashes and source positions."""

    hashes, families, files, positions = [], [], [], []
    feature_columns: list[str] | None = None
    rows_by_file: dict[str, int] = {}
    for file_index, name in enumerate(file_order):
        frame = _read(source_dir / name)
        features = _features(frame)
        columns = list(features.columns)
        if feature_columns is None:
            feature_columns = columns
        elif columns != feature_columns:
            raise ValueError(f"{name}: feature columns differ from {file_order[0]}")
        hashes.append(pd.util.hash_pandas_object(features, index=False).to_numpy(np.uint64))
        families.append(_family(frame))
        files.append(np.full(len(frame), file_index, dtype=np.int16))
        positions.append(np.arange(len(frame), dtype=np.int64))
        rows_by_file[name] = len(frame)
        print(f"scan file={name} rows={len(frame)}", flush=True)
    assert feature_columns is not None
    return {
        "feature_columns": feature_columns,
        "hash": np.concatenate(hashes),
        "family": np.concatenate(families),
        "file": np.concatenate(files),
        "position": np.concatenate(positions),
        "rows_by_file": rows_by_file,
    }


def clean_rows(scan: dict[str, object]) -> dict[str, object]:
    """Drop label-conflicting vectors, then keep first exact occurrences."""

    hashes = scan["hash"]
    families = scan["family"]
    frame = pd.DataFrame({"hash": hashes, "family": families})
    labels_per_hash = frame.groupby("hash", sort=False)["family"].nunique()
    conflicting = labels_per_hash.index[labels_per_hash.to_numpy() > 1].to_numpy(np.uint64)
    conflict_mask = np.isin(hashes, conflicting)
    first = ~frame.duplicated("hash", keep="first").to_numpy()
    keep = first & ~conflict_mask
    return {
        "keep": keep,
        "conflicting_vectors": int(len(conflicting)),
        "conflicting_rows": int(conflict_mask.sum()),
        "duplicate_rows_removed": int((~first & ~conflict_mask).sum()),
    }


MIN_BLOCKS_PER_SEED_SEGMENT = 25


def family_block_rows(clean_rows: int, block_rows: int, n_slots: int) -> int:
    """Rows per block: at most ``block_rows`` and fine enough for balancing.

    Whole blocks are balanced across seeds and then across the four stream
    segments, so each (seed, segment) needs enough blocks to meet its quota.
    """

    return max(1, min(int(block_rows), clean_rows // (n_slots * 4 * MIN_BLOCKS_PER_SEED_SEGMENT)))


def block_ids(
    scan: dict[str, object],
    keep: np.ndarray,
    families: list[str],
    block_rows: int,
    n_slots: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Group id per row: family x capture file x block of deduplicated rows.

    Blocks count deduplicated rows of one family in source order, so sparse
    families are not split into many one-row blocks. Rows outside ``keep``
    or outside ``families`` get id 0 and are never selected.
    """

    labels = scan["family"]
    files = scan["file"].astype(np.uint64)
    ids = np.zeros(len(labels), dtype=np.uint64)
    sizes: dict[str, int] = {}
    for family_index, family in enumerate(sorted(families)):
        rows = np.flatnonzero(keep & (labels == family))
        sizes[family] = family_block_rows(len(rows), block_rows, n_slots)
        block = np.arange(len(rows), dtype=np.uint64) // np.uint64(sizes[family])
        key = (np.uint64(family_index + 1) << np.uint64(48)) ^ (files[rows] << np.uint64(32)) ^ block
        ids[rows] = _splitmix64(key ^ GROUP_SALT)
    return ids, sizes


def assign_slots(
    spec: DatasetSpec,
    scan: dict[str, object],
    keep: np.ndarray,
    required: dict[str, int],
    n_slots: int,
    block_rows: int,
    attack_margin: float,
    benign_margin: float,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[str, dict[str, int]]]:
    """Balance whole blocks across seeds; return global row ids per slot."""

    families = scan["family"]
    row_ids = np.arange(len(families), dtype=np.uint64)
    ranks = _splitmix64(row_ids ^ RANK_SALT)
    group_ids, block_sizes = block_ids(scan, keep, list(required), block_rows, n_slots)
    selected: dict[tuple[str, int], np.ndarray] = {}
    usage: dict[str, dict[str, int]] = {}
    for family in sorted(required):
        rows = np.flatnonzero(keep & (families == family))
        groups, inverse = np.unique(group_ids[rows], return_inverse=True)
        sizes = np.bincount(inverse, minlength=len(groups))
        tiebreak = _splitmix64(groups ^ HASH_SALT)
        order = np.lexsort((tiebreak, -sizes))
        margin = benign_margin if family == "benign" else attack_margin
        take = int(np.ceil(margin * required[family]))
        loads = np.zeros(n_slots, dtype=np.int64)
        group_slot = np.empty(len(groups), dtype=np.int64)
        for group in order:
            slot = int(np.argmin(loads))
            group_slot[group] = slot
            loads[slot] += int(sizes[group])
        if loads.min() < take:
            raise ValueError(
                f"{spec.name}/{family}: {len(rows)} clean rows in {len(groups)} blocks "
                f"cannot give {n_slots} seeds {take} rows each (min load {loads.min()})"
            )
        row_slot = group_slot[inverse]
        for slot in range(n_slots):
            members = rows[row_slot == slot]
            members = members[np.argsort(ranks[members], kind="stable")][:take]
            selected[(family, slot)] = np.sort(members)
        usage[family] = {
            "clean_rows": int(len(rows)),
            "rows_per_block": int(block_sizes[family]),
            "blocks": int(len(groups)),
            "per_seed_take": take,
            "required_per_stream": int(required[family]),
        }
    return selected, usage


def extract(
    source_dir: Path,
    file_order: tuple[str, ...],
    scan: dict[str, object],
    selected: dict[tuple[str, int], np.ndarray],
    keep: np.ndarray,
    block_rows: int,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Second pass: features for selected rows only."""

    wanted = np.unique(np.concatenate(list(selected.values())))
    features = np.empty((len(wanted), len(scan["feature_columns"])), dtype=np.float32)
    lookup = np.full(len(scan["family"]), -1, dtype=np.int64)
    lookup[wanted] = np.arange(len(wanted))
    offset = 0
    for name in file_order:
        frame = _read(source_dir / name)
        count = len(frame)
        local = wanted[(wanted >= offset) & (wanted < offset + count)]
        if len(local):
            values = _features(frame).to_numpy(dtype=np.float64)[local - offset]
            features[lookup[local]] = values.astype(np.float32)
        offset += count
        print(f"extract file={name} rows={len(local)}", flush=True)
    n_slots = len({slot for _, slot in selected})
    group_ids, _ = block_ids(scan, keep, sorted({family for family, _ in selected}), block_rows, n_slots)
    # session_ids carry the same block ids used for seed assignment, so the
    # shared stream assembly keeps every block inside one segment.
    return {
        key: {
            "x": features[lookup[rows]],
            "source_ids": rows.astype(np.int64),
            "session_ids": group_ids[rows],
            "timestamps": rows.astype(np.int64),
        }
        for key, rows in selected.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", default=CIC_IDS2017.path)
    parser.add_argument("--scenarios", default=",".join(CANONICAL_SCENARIOS))
    parser.add_argument("--output-dir", default="prepared_streams_v2")
    parser.add_argument("--stream-seeds", default="3001,3002,3003,3004,3005")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--n-windows", type=int, default=36)
    parser.add_argument("--warmup-windows", type=int, default=4)
    parser.add_argument("--block-rows", type=int, default=1000)
    parser.add_argument("--attack-margin", type=float, default=1.10)
    parser.add_argument("--benign-margin", type=float, default=2.00)
    args = parser.parse_args()
    build(args)


def build(args, spec: DatasetSpec = CIC_IDS2017, file_order: tuple[str, ...] = FILE_ORDER) -> None:
    scenarios = tuple(value.strip() for value in args.scenarios.split(",") if value.strip())
    unknown = [value for value in scenarios if value not in CANONICAL_SCENARIOS]
    if unknown or any(value in STRESS_SCENARIOS for value in scenarios):
        raise ValueError(f"CIC protocol supports canonical scenarios only: {unknown}")
    seeds = [int(value) for value in args.stream_seeds.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("stream seeds must be unique and non-empty")
    if args.block_rows < 1 or args.attack_margin < 1.0 or args.benign_margin < 1.0:
        raise ValueError("block rows must be positive and margins at least 1")
    source_dir = Path(args.source_dir)
    source_sha256 = {name: _sha256(source_dir / name) for name in file_order}

    scan = scan_rows(source_dir, file_order)
    observed = Counter(scan["family"].tolist())
    if len(scan["family"]) != spec.total_rows:
        raise ValueError(f"{spec.name}: expected {spec.total_rows} rows, observed {len(scan['family'])}")
    for family, expected in spec.family_counts.items():
        if observed[family] != expected:
            raise ValueError(f"{spec.name}/{family}: expected {expected}, observed {observed[family]}")
    cleaning = clean_rows(scan)
    attack_count, benign_count, required = _requirements(
        spec, scenarios, args.window_size, args.n_windows, 0.05
    )
    selected, usage = assign_slots(
        spec, scan, cleaning["keep"], required, len(seeds),
        args.block_rows, args.attack_margin, args.benign_margin,
    )
    pools = extract(source_dir, file_order, scan, selected, cleaning["keep"], args.block_rows)

    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            for family in required:
                a, b = pools[(family, left)], pools[(family, right)]
                if np.intersect1d(a["source_ids"], b["source_ids"]).size:
                    raise ValueError(f"seeds {seeds[left]}/{seeds[right]} share source rows")
                if np.intersect1d(a["session_ids"], b["session_ids"]).size:
                    raise ValueError(f"seeds {seeds[left]}/{seeds[right]} share blocks")

    output_root = Path(args.output_dir)
    for slot, stream_seed in enumerate(seeds):
        for scenario in scenarios:
            arrays, _ = _build_stream(
                spec, scan["feature_columns"], pools, {}, None, slot, stream_seed,
                scenario, args.window_size, args.n_windows, args.warmup_windows,
                0.05, 4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "off",
            )
            if len(np.unique(scan["hash"][arrays["source_ids"].reshape(-1)])) != arrays["source_ids"].size:
                raise ValueError(f"{scenario}/{stream_seed}: duplicate raw feature vectors in stream")
            manifest = {
                "protocol_version": PROTOCOL_VERSION,
                "builder": "tools/build_controlled_cic_streams.py",
                "dataset_key": "cic-ids2017",
                "dataset_name": spec.name,
                "source_files": list(file_order),
                "source_sha256": source_sha256,
                "source_rows": spec.total_rows,
                "source_rows_by_file": scan["rows_by_file"],
                "observed_required_family_counts": {f: int(observed[f]) for f in required},
                "cleaning": {k: v for k, v in cleaning.items() if k != "keep"},
                "family_usage": usage,
                "scenario": scenario,
                "stream_seed": stream_seed,
                "stream_seed_set": seeds,
                "window_size": args.window_size,
                "n_windows": args.n_windows,
                "warmup_windows": args.warmup_windows,
                "attack_count_per_window": attack_count,
                "benign_count_per_window": benign_count,
                "realized_attack_rate": attack_count / args.window_size,
                "local_fraction_of_attacks": LOCAL_LEVELS.get(scenario, 0.05),
                "feature_count": len(scan["feature_columns"]),
                "feature_names": scan["feature_columns"],
                "subtype_counts": dict(Counter(arrays["subtypes"].reshape(-1).tolist())),
                "unique_source_rows": int(len(np.unique(arrays["source_ids"]))),
                "overlap_within_stream": 0,
                "duplicate_feature_vectors_within_stream": 0,
                "block_overlap_across_segments": 0,
                "cross_seed_overlap": 0,
                "session_proxy": (
                    "contiguous block of deduplicated same-family rows per capture file; "
                    f"rows per block = min({args.block_rows}, clean rows / (seeds x 4 segments x "
                    f"{MIN_BLOCKS_PER_SEED_SEGMENT})), see family_usage"
                ),
                "seed_assignment": {
                    "method": "whole-block deterministic load balancing (largest first)",
                    "salt": int(HASH_SALT),
                    "selection_basis": "block size with label-blind hash tie-break; no model outcome",
                },
                "margins": {"attack": args.attack_margin, "benign": args.benign_margin},
                "preprocessing_fit_windows": list(range(args.warmup_windows)),
                "concept_definitions": {
                    "A": spec.concept_a, "B": spec.concept_b, "C": spec.concept_c,
                    "A_prime": spec.concept_a, "local_stable": spec.stable,
                    "local_A": spec.local_a, "local_B": spec.local_b,
                },
                "limitations": LIMITATIONS,
            }
            destination = output_root / "cic-ids2017" / scenario / f"stream_seed{stream_seed}.npz"
            _write_stream(destination, arrays, manifest)
            print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
