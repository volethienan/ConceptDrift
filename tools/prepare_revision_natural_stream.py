from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dasp_nids.data import _numeric_feature_columns


RAW_PATH = Path("data/NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv")
OUTPUT_PATH = Path(
    "prepared_streams_revision/nf-ton-iot-v3/natural_chronological_v1.npz"
)
WINDOW_SIZE = 4096
N_WINDOWS = 36
WARMUP_WINDOWS = 4
TARGET_ROWS = WINDOW_SIZE * N_WINDOWS
TIME_COLUMN = "FLOW_START_MILLISECONDS"
LABEL_COLUMN = "Label"
SUBTYPE_COLUMN = "Attack"


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retain_earliest(
    current: dict[str, np.ndarray] | None,
    candidate: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if current is None:
        combined = candidate
    else:
        combined = {
            key: np.concatenate([current[key], candidate[key]], axis=0)
            for key in current
        }
    order = np.lexsort((combined["source_ids"], combined["timestamps"]))
    selected = order[:TARGET_ROWS]
    return {key: value[selected] for key, value in combined.items()}


def prepare(raw_path: Path, output_path: Path, chunksize: int) -> None:
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    feature_columns = _numeric_feature_columns(raw_path)
    required = [
        *feature_columns,
        LABEL_COLUMN,
        SUBTYPE_COLUMN,
        TIME_COLUMN,
    ]
    retained: dict[str, np.ndarray] | None = None
    total_rows = 0
    valid_rows = 0
    invalid_rows = 0
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(
        pd.read_csv(
            raw_path,
            usecols=required,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        source_ids = np.arange(total_rows, total_rows + len(chunk), dtype=np.int64)
        total_rows += len(chunk)
        timestamps = pd.to_numeric(chunk[TIME_COLUMN], errors="coerce")
        labels = pd.to_numeric(chunk[LABEL_COLUMN], errors="coerce")
        valid = timestamps.notna() & labels.notna()
        invalid_rows += int((~valid).sum())
        if not valid.any():
            continue
        valid_rows += int(valid.sum())
        numeric = chunk.loc[valid, feature_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        candidate = {
            "x": numeric.to_numpy(dtype=np.float32, copy=True),
            "y": (labels.loc[valid].to_numpy() != 0).astype(np.int64),
            "subtypes": chunk.loc[valid, SUBTYPE_COLUMN]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .to_numpy(dtype="U64"),
            "timestamps": timestamps.loc[valid].to_numpy(dtype=np.int64),
            "source_ids": source_ids[valid.to_numpy()],
        }
        retained = _retain_earliest(retained, candidate)
        if chunk_index % 25 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"chunks={chunk_index} rows={total_rows} "
                f"retained={len(retained['y'])} elapsed_s={elapsed:.1f}",
                flush=True,
            )
    if retained is None or len(retained["y"]) < TARGET_ROWS:
        found = 0 if retained is None else len(retained["y"])
        raise ValueError(f"required {TARGET_ROWS} valid rows but retained {found}")

    order = np.lexsort((retained["source_ids"], retained["timestamps"]))
    retained = {key: value[order] for key, value in retained.items()}
    raw_x = retained["x"]
    warmup_rows = WINDOW_SIZE * WARMUP_WINDOWS
    medians = np.nanmedian(raw_x[:warmup_rows], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    nonfinite_count = int((~np.isfinite(raw_x)).sum())
    raw_x = np.nan_to_num(raw_x, nan=medians, posinf=medians, neginf=medians)
    scaler = StandardScaler().fit(raw_x[:warmup_rows])
    scaled = scaler.transform(raw_x).astype(np.float32)

    x = scaled.reshape(N_WINDOWS, WINDOW_SIZE, len(feature_columns))
    y = retained["y"].reshape(N_WINDOWS, WINDOW_SIZE)
    subtypes = retained["subtypes"].reshape(N_WINDOWS, WINDOW_SIZE)
    timestamps = retained["timestamps"].reshape(N_WINDOWS, WINDOW_SIZE)
    source_ids = retained["source_ids"].reshape(N_WINDOWS, WINDOW_SIZE)
    phases = np.asarray(
        ["warmup" if index < WARMUP_WINDOWS else "chronological" for index in range(N_WINDOWS)]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        phases=phases,
        feature_names=np.asarray(feature_columns),
        class_names=np.asarray(["Benign", "Attack"]),
        subtypes=subtypes,
        source_ids=source_ids,
        timestamps=timestamps,
    )
    class_counts = np.bincount(y.reshape(-1), minlength=2)
    manifest = {
        "protocol_version": "1.1",
        "construction": "stable global timestamp sort; earliest valid prefix",
        "raw_path": str(raw_path),
        "raw_bytes": raw_path.stat().st_size,
        "raw_sha256": _digest_file(raw_path),
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _digest_file(output_path),
        "window_size": WINDOW_SIZE,
        "n_windows": N_WINDOWS,
        "warmup_windows": WARMUP_WINDOWS,
        "selected_rows": TARGET_ROWS,
        "rows_scanned": total_rows,
        "valid_rows_scanned": valid_rows,
        "invalid_timestamp_or_label_rows": invalid_rows,
        "nonfinite_feature_values_imputed": nonfinite_count,
        "feature_count": len(feature_columns),
        "feature_names": feature_columns,
        "earliest_timestamp": int(timestamps.reshape(-1)[0]),
        "latest_timestamp": int(timestamps.reshape(-1)[-1]),
        "benign_rows": int(class_counts[0]),
        "attack_rows": int(class_counts[1]),
        "per_window_attack_rows": y.sum(axis=1).astype(int).tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--chunksize", type=int, default=65536)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.chunksize)


if __name__ == "__main__":
    main()
