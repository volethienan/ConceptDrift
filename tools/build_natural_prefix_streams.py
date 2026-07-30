from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dasp_nids.data import NF_V3_NON_FEATURE_COLUMNS
from build_controlled_nf_streams import DATASETS


LUFLOW_FEATURES = [
    "avg_ipt",
    "bytes_in",
    "bytes_out",
    "dest_port",
    "entropy",
    "num_pkts_out",
    "num_pkts_in",
    "proto",
    "src_port",
    "total_entropy",
    "duration",
]


def _take_earliest(
    times: np.ndarray, source_ids: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take the lexicographically earliest (timestamp, original row id) rows."""

    if len(times) <= count:
        order = np.lexsort((source_ids, times))
        return times[order], source_ids[order]
    cutoff = np.partition(times, count - 1)[count - 1]
    earlier = np.flatnonzero(times < cutoff)
    tied = np.flatnonzero(times == cutoff)
    needed = count - len(earlier)
    tied_order = tied[np.argsort(source_ids[tied], kind="stable")[:needed]]
    selected = np.concatenate([earlier, tied_order])
    order = np.lexsort((source_ids[selected], times[selected]))
    selected = selected[order]
    return times[selected], source_ids[selected]


def _top_k_source_ids(
    files: list[Path],
    time_column: str,
    target_rows: int,
    chunksize: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    best_times = np.empty(0, dtype=np.int64)
    best_ids = np.empty(0, dtype=np.int64)
    row_offset = 0
    for path in files:
        for chunk in pd.read_csv(
            path,
            usecols=[time_column],
            chunksize=chunksize,
            low_memory=False,
        ):
            times = pd.to_numeric(chunk[time_column], errors="coerce").to_numpy()
            if np.any(~np.isfinite(times)):
                raise ValueError(f"Invalid {time_column} in {path}")
            times = times.astype(np.int64)
            ids = np.arange(
                row_offset, row_offset + len(chunk), dtype=np.int64
            )
            row_offset += len(chunk)
            combined_times = np.concatenate([best_times, times])
            combined_ids = np.concatenate([best_ids, ids])
            if len(combined_times) > target_rows:
                best_times, best_ids = _take_earliest(
                    combined_times, combined_ids, target_rows
                )
            else:
                best_times, best_ids = _take_earliest(
                    combined_times, combined_ids, len(combined_times)
                )
            print(
                f"time-scan file={path.name} rows={row_offset}",
                flush=True,
            )
    if len(best_ids) < target_rows:
        raise ValueError(
            f"Requested {target_rows} rows but only {len(best_ids)} exist"
        )
    return best_ids, best_times, row_offset


def _selected_mask(
    row_offset: int,
    chunk_rows: int,
    selected_sorted: np.ndarray,
) -> np.ndarray:
    ids = np.arange(row_offset, row_offset + chunk_rows, dtype=np.int64)
    positions = np.searchsorted(selected_sorted, ids)
    positions = np.minimum(positions, len(selected_sorted) - 1)
    return selected_sorted[positions] == ids


def _extract_nf(
    path: Path,
    selected_ids: np.ndarray,
    chunksize: int,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    features = [
        str(column)
        for column in header
        if column not in NF_V3_NON_FEATURE_COLUMNS
    ]
    usecols = [*features, "Label", "Attack", "FLOW_START_MILLISECONDS"]
    sorted_ids = np.sort(selected_ids)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subtypes: list[np.ndarray] = []
    ids_out: list[np.ndarray] = []
    times_out: list[np.ndarray] = []
    row_offset = 0
    for chunk in pd.read_csv(
        path, usecols=usecols, chunksize=chunksize, low_memory=False
    ):
        mask = _selected_mask(row_offset, len(chunk), sorted_ids)
        chunk_ids = np.arange(
            row_offset, row_offset + len(chunk), dtype=np.int64
        )
        row_offset += len(chunk)
        if not np.any(mask):
            continue
        selected = chunk.loc[mask]
        xs.append(
            selected[features]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        labels = pd.to_numeric(
            selected["Label"], errors="coerce"
        ).fillna(0)
        ys.append((labels.to_numpy() != 0).astype(np.int64))
        subtypes.append(
            selected["Attack"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .to_numpy(dtype=str)
        )
        ids_out.append(chunk_ids[mask])
        times_out.append(
            pd.to_numeric(
                selected["FLOW_START_MILLISECONDS"], errors="raise"
            ).to_numpy(dtype=np.int64)
        )
    return (
        features,
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(subtypes),
        np.concatenate(ids_out),
        np.concatenate(times_out),
    )


def _extract_luflow(
    files: list[Path],
    selected_ids: np.ndarray,
    chunksize: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sorted_ids = np.sort(selected_ids)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subtypes: list[np.ndarray] = []
    ids_out: list[np.ndarray] = []
    times_out: list[np.ndarray] = []
    row_offset = 0
    usecols = [*LUFLOW_FEATURES, "label", "time_start"]
    for path in files:
        for chunk in pd.read_csv(
            path, usecols=usecols, chunksize=chunksize, low_memory=False
        ):
            mask = _selected_mask(row_offset, len(chunk), sorted_ids)
            chunk_ids = np.arange(
                row_offset, row_offset + len(chunk), dtype=np.int64
            )
            row_offset += len(chunk)
            if not np.any(mask):
                continue
            selected = chunk.loc[mask]
            xs.append(
                selected[LUFLOW_FEATURES]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float32)
            )
            labels = selected["label"].astype(str).str.casefold()
            ys.append((labels != "benign").to_numpy(dtype=np.int64))
            subtypes.append(labels.to_numpy(dtype=str))
            ids_out.append(chunk_ids[mask])
            times_out.append(
                pd.to_numeric(
                    selected["time_start"], errors="raise"
                ).to_numpy(dtype=np.int64)
            )
    return (
        LUFLOW_FEATURES,
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(subtypes),
        np.concatenate(ids_out),
        np.concatenate(times_out),
    )


def _finalize(
    feature_names: list[str],
    raw_x: np.ndarray,
    y: np.ndarray,
    subtypes: np.ndarray,
    source_ids: np.ndarray,
    timestamps: np.ndarray,
    window_size: int,
    n_windows: int,
    warmup_windows: int,
) -> dict[str, np.ndarray]:
    order = np.lexsort((source_ids, timestamps))
    raw_x = raw_x[order]
    y = y[order]
    subtypes = subtypes[order]
    source_ids = source_ids[order]
    timestamps = timestamps[order]
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Natural stream timestamps are not monotone")
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("Natural stream contains duplicate source rows")
    shape = (n_windows, window_size)
    raw_x = raw_x.reshape(*shape, raw_x.shape[-1])
    y = y.reshape(shape)
    subtypes = subtypes.reshape(shape)
    source_ids = source_ids.reshape(shape)
    timestamps = timestamps.reshape(shape)
    warmup = raw_x[:warmup_windows].reshape(-1, raw_x.shape[-1])
    medians = np.nanmedian(warmup, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(raw_x), raw_x, medians)
    scaler = StandardScaler().fit(
        imputed[:warmup_windows].reshape(-1, imputed.shape[-1])
    )
    x = scaler.transform(imputed.reshape(-1, imputed.shape[-1])).reshape(
        imputed.shape
    )
    phases = np.full(n_windows, "early", dtype="<U12")
    phases[:warmup_windows] = "warmup"
    remaining = n_windows - warmup_windows
    middle = warmup_windows + remaining // 3
    late = warmup_windows + 2 * remaining // 3
    phases[middle:late] = "middle"
    phases[late:] = "late"
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.int64),
        "subtypes": subtypes.astype("<U64"),
        "source_ids": source_ids.astype(np.int64),
        "timestamps": timestamps.astype(np.int64),
        "phases": phases,
        "feature_names": np.asarray(feature_names, dtype="<U64"),
        "class_names": np.asarray(["Benign", "Attack"], dtype="<U16"),
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "imputation_median": medians.astype(np.float64),
    }


def _path_manifest_hash(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact chronological-prefix streams for five IDS datasets."
    )
    parser.add_argument("--output-dir", default="prepared_natural_streams")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--n-windows", type=int, default=96)
    parser.add_argument("--warmup-windows", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--datasets",
        default=",".join([*DATASETS, "luflow"]),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    target_rows = args.window_size * args.n_windows
    output_root = Path(args.output_dir)
    keys = [value for value in args.datasets.split(",") if value]
    for key in keys:
        destination = output_root / key / "natural_prefix.npz"
        manifest_path = destination.with_suffix(".manifest.json")
        if args.resume and destination.exists() and manifest_path.exists():
            print(f"resume dataset={key}", flush=True)
            continue
        if key == "luflow":
            files = sorted(Path("data/LUFlow").rglob("*.csv"))
            if not files:
                raise FileNotFoundError("data/LUFlow")
            time_column = "time_start"
            selected_ids, _, source_rows = _top_k_source_ids(
                files, time_column, target_rows, args.chunksize
            )
            extracted = _extract_luflow(
                files, selected_ids, args.chunksize
            )
            dataset_name = "LUFlow"
        else:
            spec = DATASETS[key]
            files = [Path(spec.path)]
            time_column = "FLOW_START_MILLISECONDS"
            selected_ids, _, source_rows = _top_k_source_ids(
                files, time_column, target_rows, args.chunksize
            )
            extracted = _extract_nf(
                files[0], selected_ids, args.chunksize
            )
            dataset_name = spec.name
        arrays = _finalize(
            *extracted,
            args.window_size,
            args.n_windows,
            args.warmup_windows,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        manifest = {
            "dataset_key": key,
            "dataset_name": dataset_name,
            "track": "exact_global_chronological_prefix",
            "source_files": len(files),
            "source_path_size_manifest_sha256": _path_manifest_hash(files),
            "source_rows": source_rows,
            "selected_rows": target_rows,
            "window_size": args.window_size,
            "n_windows": args.n_windows,
            "warmup_windows": args.warmup_windows,
            "first_timestamp": int(arrays["timestamps"][0, 0]),
            "last_timestamp": int(arrays["timestamps"][-1, -1]),
            "monotone_timestamps": True,
            "unique_source_rows": int(
                len(np.unique(arrays["source_ids"]))
            ),
            "attack_rate": float(arrays["y"].mean()),
            "preprocessing_fit_windows": list(range(args.warmup_windows)),
            "limitation": (
                "This is the exact earliest bounded prefix, not the complete "
                "dataset timeline."
            ),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
