from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


@dataclass
class StreamWindow:
    x: torch.Tensor
    y: torch.Tensor
    index: int
    phase: str
    subtypes: np.ndarray | None = None
    source_ids: np.ndarray | None = None
    start_time: int | None = None
    end_time: int | None = None


@dataclass
class PreparedStream:
    windows: list[StreamWindow]
    feature_names: list[str]
    class_names: list[str]


def load_prepared_stream(
    path: str | Path, drop_features: list[str] | tuple[str, ...] = ()
) -> PreparedStream:
    """Load a frozen, leakage-audited stream artifact.

    Prepared artifacts are generated before model training and contain the
    warm-up-only scaled features, labels, phase names, source row identifiers,
    and timestamps.  The loader validates the rectangular window layout and
    does not reshuffle any samples.  ``drop_features`` removes the named
    columns (shortcut ablation); names the stream does not have are ignored.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Prepared stream was not found: {source}")
    with np.load(source, allow_pickle=False) as payload:
        required = {"x", "y", "phases", "feature_names", "class_names"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Prepared stream is missing arrays: {sorted(missing)}")
        x = np.asarray(payload["x"], dtype=np.float32)
        y = np.asarray(payload["y"], dtype=np.int64)
        phases = np.asarray(payload["phases"]).astype(str)
        feature_names = np.asarray(payload["feature_names"]).astype(str).tolist()
        class_names = np.asarray(payload["class_names"]).astype(str).tolist()
        subtypes = (
            np.asarray(payload["subtypes"]).astype(str)
            if "subtypes" in payload.files
            else None
        )
        source_ids = (
            np.asarray(payload["source_ids"], dtype=np.int64)
            if "source_ids" in payload.files
            else None
        )
        timestamps = (
            np.asarray(payload["timestamps"], dtype=np.int64)
            if "timestamps" in payload.files
            else None
        )

    if x.ndim != 3:
        raise ValueError("Prepared x must have shape [windows, samples, features]")
    if y.shape != x.shape[:2]:
        raise ValueError("Prepared y must match the first two x dimensions")
    if len(phases) != len(x):
        raise ValueError("Prepared phases must contain one value per window")
    if len(feature_names) != x.shape[2]:
        raise ValueError("Prepared feature_names do not match x feature dimension")
    if drop_features:
        keep = [i for i, name in enumerate(feature_names) if name not in set(drop_features)]
        if not keep:
            raise ValueError("drop_features would remove every feature")
        x = np.ascontiguousarray(x[:, :, keep])
        feature_names = [feature_names[i] for i in keep]
    for name, values in {
        "subtypes": subtypes,
        "source_ids": source_ids,
        "timestamps": timestamps,
    }.items():
        if values is not None and values.shape != y.shape:
            raise ValueError(f"Prepared {name} must have shape {y.shape}")

    windows: list[StreamWindow] = []
    for index in range(len(x)):
        window_times = None if timestamps is None else timestamps[index]
        windows.append(
            StreamWindow(
                x=_tensor_from_numpy(x[index], torch.float32),
                y=_tensor_from_numpy(y[index], torch.int64),
                index=index,
                phase=str(phases[index]),
                subtypes=None if subtypes is None else subtypes[index],
                source_ids=None if source_ids is None else source_ids[index],
                start_time=(
                    None if window_times is None else int(np.min(window_times))
                ),
                end_time=(
                    None if window_times is None else int(np.max(window_times))
                ),
            )
        )
    return PreparedStream(windows, feature_names, class_names)


NF_V3_NON_FEATURE_COLUMNS = {
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "Label",
    "Attack",
}


def _tensor_from_numpy(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert contiguous arrays even when an old Torch meets NumPy 2 at runtime."""
    contiguous = np.ascontiguousarray(value)
    return torch.frombuffer(memoryview(contiguous), dtype=dtype).clone().reshape(contiguous.shape)


def _campaign_centers(n_features: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    centers = {"benign": np.zeros(n_features)}
    for name in ["hulk", "ddos", "goldeneye", "slow", "ftp", "ssh"]:
        centers[name] = rng.normal(0, 0.55, n_features)
    centers["hulk"][:6] += 2.2
    centers["ddos"][6:12] += 2.4
    centers["goldeneye"][12:18] += 2.0
    centers["slow"][18:24] += 2.0
    centers["ftp"][::2] += 1.8
    centers["ssh"][1::2] += 1.8
    return centers


def _scenario_mix(scenario: str, window: int, n_windows: int) -> tuple[dict[str, float], str]:
    midpoint = n_windows // 2
    pre = window < midpoint
    if scenario == "S0":
        return {"hulk": 1.0}, "stable"
    if scenario == "S1":
        return ({"hulk": 1.0}, "pre") if pre else ({"ddos": 0.4, "goldeneye": 0.3, "slow": 0.3}, "post")
    if scenario == "S2":
        alpha = window / max(1, n_windows - 1)
        return {"hulk": 1.0 - alpha, "ddos": alpha}, "gradual"
    if scenario == "S3":
        third = max(1, n_windows // 3)
        if window < third:
            return {"hulk": 1.0}, "A1"
        if window < 2 * third:
            return {"ddos": 1.0}, "B"
        return {"hulk": 1.0}, "A2"
    if scenario == "S4":
        # Dual-campaign drift: DoS and Brute Force both change subtype, while
        # remaining distinct coarse classes. Previously both labels used the
        # same FTP/SSH center, making the three-class task internally
        # unidentifiable and inflating the meaning of Macro-F1 comparisons.
        return ({"hulk": 1.0}, "pre") if pre else ({"ddos": 1.0}, "post")
    if scenario == "S5":
        return (
            ({"hulk": 1.0}, "A")
            if pre
            else ({"ddos": 0.4, "goldeneye": 0.3, "slow": 0.3}, "B_evasive")
        )
    raise ValueError(f"Unknown scenario: {scenario}")


def make_synthetic_stream(
    scenario: str = "S1",
    n_windows: int = 12,
    window_size: int = 4096,
    n_features: int = 32,
    seed: int = 42,
    task: str = "multiclass",
) -> PreparedStream:
    """Controlled stream with stable coarse labels and changing attack campaigns."""
    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be 'binary' or 'multiclass'")
    rng = np.random.default_rng(seed)
    centers = _campaign_centers(n_features, rng)
    windows: list[StreamWindow] = []
    for window_idx in range(n_windows):
        dos_mix, phase = _scenario_mix(scenario, window_idx, n_windows)
        counts = rng.multinomial(window_size, [0.70, 0.15, 0.15])
        labels = np.repeat(np.arange(3), counts)
        rng.shuffle(labels)
        x = np.empty((window_size, n_features), dtype=np.float32)
        subtypes = np.empty(window_size, dtype=object)
        dos_names = list(dos_mix)
        dos_probs = np.asarray([dos_mix[name] for name in dos_names], dtype=float)
        dos_probs /= dos_probs.sum()
        for i, label in enumerate(labels):
            if label == 0:
                subtype = "benign"
            elif label == 1:
                subtype = str(rng.choice(dos_names, p=dos_probs))
            else:
                if scenario == "S4":
                    subtype = "ftp" if window_idx < n_windows // 2 else "ssh"
                else:
                    subtype = "ftp"
            mean = centers[subtype].copy()
            if (
                scenario == "S5"
                and window_idx >= n_windows // 2
                and label != 0
            ):
                # Synthetic fallback for the real-flow S5 stress track:
                # both attack classes move sharply toward the benign region.
                # The NF-v3 generator performs model-independent, audited
                # hard-example selection instead of altering feature values.
                mean *= 0.15
            x[i] = rng.normal(mean, 1.0).astype(np.float32)
            subtypes[i] = subtype
        windows.append(
            StreamWindow(
                x=_tensor_from_numpy(x, torch.float32),
                y=_tensor_from_numpy(
                    (
                        (labels != 0).astype(np.int64)
                        if task == "binary"
                        else labels.astype(np.int64)
                    ),
                    torch.int64,
                ),
                index=window_idx,
                phase=phase,
                subtypes=subtypes,
            )
        )
    class_names = (
        ["Benign", "Attack"]
        if task == "binary"
        else ["Benign", "DoS", "Brute Force"]
    )
    return PreparedStream(
        windows,
        [f"f{i}" for i in range(n_features)],
        class_names,
    )


def _coarse_label(value: object) -> tuple[str, str]:
    text = str(value).strip()
    lower = text.lower()
    if "benign" in lower:
        return "Benign", "benign"
    if any(token in lower for token in ["dos", "ddos", "hulk", "goldeneye", "slowloris", "slowhttptest", "heartbleed"]):
        return "DoS", lower
    if any(token in lower for token in ["patator", "brute", "ftp", "ssh"]):
        return "Brute Force", lower
    return "Other Attack", lower


def load_cic_csv(
    paths: str | Path | Iterable[str | Path],
    window_size: int,
    label_column: str = "Label",
    seed: int = 42,
) -> PreparedStream:
    """Compatibility wrapper for small CIC-style files.

    For the multi-gigabyte NF-v3 datasets, use :func:`load_nf_v3_stream` so only
    the requested chronological prefix is held in memory.
    """
    if isinstance(paths, (str, Path)):
        candidate = Path(paths)
        file_paths = sorted(candidate.glob("*.csv")) if candidate.is_dir() else [candidate]
    else:
        file_paths = [Path(p) for p in paths]
    if not file_paths:
        raise FileNotFoundError("No CIC CSV files were found")
    frames = [pd.read_csv(path, low_memory=False) for path in file_paths]
    frame = pd.concat(frames, ignore_index=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    resolved_label = next((c for c in frame.columns if c.lower() == label_column.strip().lower()), None)
    if resolved_label is None:
        raise ValueError(f"Label column {label_column!r} was not found")
    mapped = frame[resolved_label].map(_coarse_label)
    frame["__coarse_label"] = mapped.map(lambda item: item[0])
    frame["__subtype"] = mapped.map(lambda item: item[1])
    frame = frame[frame["__coarse_label"].isin(["Benign", "DoS", "Brute Force"])].reset_index(drop=True)

    excluded_tokens = ("flow id", "source ip", "destination ip", "timestamp", "label", "__")
    feature_columns = [
        c for c in frame.columns
        if not any(token in c.lower() for token in excluded_tokens)
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns are available after filtering")
    values = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = values.iloc[: min(window_size, len(values))].median(numeric_only=True)
    values = values.fillna(medians).fillna(0.0).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(values.iloc[: min(window_size, len(values))])
    values = scaler.transform(values).astype(np.float32)
    class_names = ["Benign", "Brute Force", "DoS"]
    label_map = {name: idx for idx, name in enumerate(class_names)}
    labels = frame["__coarse_label"].map(label_map).to_numpy(dtype=np.int64)

    windows: list[StreamWindow] = []
    for idx, start in enumerate(range(0, len(frame) - window_size + 1, window_size)):
        stop = start + window_size
        windows.append(
            StreamWindow(
                x=_tensor_from_numpy(values[start:stop], torch.float32),
                y=_tensor_from_numpy(labels[start:stop], torch.int64),
                index=idx,
                phase="chronological",
                subtypes=frame["__subtype"].iloc[start:stop].to_numpy(),
            )
        )
    if not windows:
        raise ValueError(f"Dataset has fewer than one full window ({window_size} rows)")
    return PreparedStream(windows, feature_columns, class_names)


def _resolve_csv_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidate = Path(paths)
        return sorted(candidate.glob("*.csv")) if candidate.is_dir() else [candidate]
    return [Path(path) for path in paths]


def _numeric_feature_columns(path: Path) -> list[str]:
    sample = pd.read_csv(path, nrows=256, low_memory=False)
    columns: list[str] = []
    for column in sample.columns:
        if column in NF_V3_NON_FEATURE_COLUMNS or str(column).startswith("__"):
            continue
        converted = pd.to_numeric(sample[column], errors="coerce")
        if converted.notna().any():
            columns.append(str(column))
    if not columns:
        raise ValueError(f"No numeric feature columns found in {path}")
    return columns


def load_nf_v3_stream(
    paths: str | Path | Iterable[str | Path],
    window_size: int,
    n_windows: int,
    warmup_windows: int = 3,
    label_column: str = "Label",
    subtype_column: str = "Attack",
    task: str = "binary",
    start_row: int = 0,
    chunksize: int = 65536,
    sort_by_time: bool = False,
    time_column: str = "FLOW_START_MILLISECONDS",
    include_partial_window: bool = False,
) -> PreparedStream:
    """Load a bounded or full NF-v3 stream with leakage-safe preprocessing.

    The scaler is fitted only on declared warm-up windows.  This makes the
    preprocessing boundary explicit and prevents future-window leakage. Set
    ``n_windows=0`` to consume every available row. When ``sort_by_time`` is
    enabled, the complete source is stably sorted before ``start_row`` and
    ``n_windows`` are applied; otherwise a bounded request would select a
    physical CSV prefix rather than a globally chronological segment.
    """
    file_paths = _resolve_csv_paths(paths)
    if not file_paths or any(not path.exists() for path in file_paths):
        raise FileNotFoundError("One or more NF-v3 CSV files were not found")
    if n_windows > 0 and n_windows <= warmup_windows:
        raise ValueError("n_windows must be greater than warmup_windows")
    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be 'binary' or 'multiclass'")

    feature_columns = _numeric_feature_columns(file_paths[0])
    required = [*feature_columns, label_column, subtype_column]
    if sort_by_time and time_column not in required:
        required.append(time_column)
    target_rows = window_size * n_windows if n_windows > 0 else None
    remaining_skip = 0 if sort_by_time else max(0, start_row)
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    subtype_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    collected = 0

    for path in file_paths:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [column for column in required if column not in header]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        for chunk in pd.read_csv(path, usecols=required, chunksize=chunksize, low_memory=False):
            if remaining_skip:
                drop = min(remaining_skip, len(chunk))
                chunk = chunk.iloc[drop:]
                remaining_skip -= drop
                if chunk.empty:
                    continue
            take = (
                len(chunk)
                if target_rows is None or sort_by_time
                else min(len(chunk), target_rows - collected)
            )
            if take <= 0:
                break
            chunk = chunk.iloc[:take]
            numeric = chunk[feature_columns].apply(pd.to_numeric, errors="coerce")
            feature_parts.append(numeric.to_numpy(dtype=np.float32, copy=True))
            raw_label = pd.to_numeric(chunk[label_column], errors="coerce").fillna(0)
            if task == "binary":
                labels = (raw_label.to_numpy() != 0).astype(np.int64)
            else:
                labels = np.array([_coarse_label(value)[0] for value in chunk[subtype_column]], dtype=object)
            label_parts.append(labels)
            subtype_parts.append(chunk[subtype_column].astype(str).str.lower().to_numpy())
            if sort_by_time:
                timestamps = pd.to_numeric(chunk[time_column], errors="coerce")
                if timestamps.isna().any():
                    raise ValueError(f"Invalid timestamps found in {time_column}")
                time_parts.append(timestamps.to_numpy(dtype=np.int64, copy=True))
            collected += take
        if target_rows is not None and not sort_by_time and collected >= target_rows:
            break

    required_rows = (
        target_rows + max(0, start_row)
        if sort_by_time and target_rows is not None
        else target_rows
    )
    if required_rows is not None and collected < required_rows:
        raise ValueError(f"Requested {target_rows} rows but only {collected} were available")
    raw_x = np.concatenate(feature_parts, axis=0)
    raw_y = np.concatenate(label_parts, axis=0)
    subtypes = np.concatenate(subtype_parts, axis=0)
    timestamps = np.concatenate(time_parts, axis=0) if sort_by_time else None
    if timestamps is not None:
        order = np.argsort(timestamps, kind="stable")
        raw_x = raw_x[order]
        raw_y = raw_y[order]
        subtypes = subtypes[order]
        timestamps = timestamps[order]
        selection_start = max(0, start_row)
        selection_stop = (
            None if target_rows is None else selection_start + target_rows
        )
        raw_x = raw_x[selection_start:selection_stop]
        raw_y = raw_y[selection_start:selection_stop]
        subtypes = subtypes[selection_start:selection_stop]
        timestamps = timestamps[selection_start:selection_stop]
        collected = len(raw_x)

    if collected < window_size * warmup_windows:
        raise ValueError("Full stream is smaller than the declared warm-up prefix")

    medians = np.nanmedian(raw_x[: window_size * warmup_windows], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    raw_x = np.nan_to_num(raw_x, nan=medians, posinf=medians, neginf=medians)
    scaler = StandardScaler()
    scaler.fit(raw_x[: window_size * warmup_windows])
    values = scaler.transform(raw_x).astype(np.float32)

    if task == "binary":
        encoded = raw_y.astype(np.int64)
        class_names = ["Benign", "Attack"]
    else:
        class_names = ["Benign", "Brute Force", "DoS", "Other Attack"]
        label_map = {name: idx for idx, name in enumerate(class_names)}
        encoded = np.array([label_map[str(value)] for value in raw_y], dtype=np.int64)

    full_windows, remainder = divmod(collected, window_size)
    effective_windows = (
        n_windows
        if n_windows > 0
        else full_windows + int(include_partial_window and remainder > 0)
    )
    windows: list[StreamWindow] = []
    for idx in range(effective_windows):
        start = idx * window_size
        stop = min(start + window_size, collected)
        if stop - start < window_size and not include_partial_window:
            break
        windows.append(
            StreamWindow(
                x=_tensor_from_numpy(values[start:stop], torch.float32),
                y=_tensor_from_numpy(encoded[start:stop], torch.int64),
                index=idx,
                phase="warmup" if idx < warmup_windows else "chronological",
                subtypes=subtypes[start:stop],
                start_time=(None if timestamps is None else int(timestamps[start])),
                end_time=(None if timestamps is None else int(timestamps[stop - 1])),
            )
        )
    return PreparedStream(windows, feature_columns, class_names)


def load_luflow_stream(
    paths: str | Path | Iterable[str | Path],
    window_size: int,
    n_windows: int,
    warmup_windows: int = 3,
    start_row: int = 0,
    chunksize: int = 65536,
) -> PreparedStream:
    """Load LUFlow files in lexical date order as a bounded binary stream."""
    file_paths = _resolve_csv_paths(paths)
    if len(file_paths) == 1 and file_paths[0].is_dir():
        file_paths = sorted(file_paths[0].rglob("*.csv"))
    elif isinstance(paths, (str, Path)) and Path(paths).is_dir():
        file_paths = sorted(Path(paths).rglob("*.csv"))
    if not file_paths:
        raise FileNotFoundError("No LUFlow CSV files were found")
    feature_columns = [
        "avg_ipt", "bytes_in", "bytes_out", "dest_port", "entropy",
        "num_pkts_out", "num_pkts_in", "proto", "src_port",
        "total_entropy", "duration",
    ]
    target_rows = window_size * n_windows
    remaining_skip = max(0, start_row)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subtypes: list[np.ndarray] = []
    collected = 0
    for path in file_paths:
        usecols = [*feature_columns, "label"]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            if remaining_skip:
                drop = min(remaining_skip, len(chunk))
                chunk = chunk.iloc[drop:]
                remaining_skip -= drop
                if chunk.empty:
                    continue
            take = min(len(chunk), target_rows - collected)
            if take <= 0:
                break
            chunk = chunk.iloc[:take]
            xs.append(chunk[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32))
            label_text = chunk["label"].astype(str).str.lower()
            ys.append((label_text != "benign").to_numpy(np.int64))
            subtypes.append(label_text.to_numpy())
            collected += take
        if collected >= target_rows:
            break
    if collected < target_rows:
        raise ValueError(f"Requested {target_rows} rows but only {collected} were available")
    raw_x = np.concatenate(xs)
    medians = np.nanmedian(raw_x[: window_size * warmup_windows], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    raw_x = np.nan_to_num(raw_x, nan=medians, posinf=medians, neginf=medians)
    scaler = StandardScaler().fit(raw_x[: window_size * warmup_windows])
    values = scaler.transform(raw_x).astype(np.float32)
    labels = np.concatenate(ys)
    subtype_values = np.concatenate(subtypes)
    windows = []
    for idx in range(n_windows):
        start, stop = idx * window_size, (idx + 1) * window_size
        windows.append(StreamWindow(
            x=_tensor_from_numpy(values[start:stop], torch.float32),
            y=_tensor_from_numpy(labels[start:stop], torch.int64),
            index=idx,
            phase="warmup" if idx < warmup_windows else "chronological",
            subtypes=subtype_values[start:stop],
        ))
    return PreparedStream(windows, feature_columns, ["Benign", "Attack"])
