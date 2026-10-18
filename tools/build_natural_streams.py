"""Build chronological NetFlow-v3 streams that span the whole capture.

The earlier natural stream kept only the earliest 147,456 flows, a short slice
at the start of one capture. Here every flow is ordered by
FLOW_START_MILLISECONDS (ties by source row), the ordered capture is cut into
``n_windows`` contiguous segments with equal numbers of flows, and each window
is a label-blind uniform sample of ``window_size`` flows from its segment, kept
in time order. All seeds are drawn together without replacement, so streams of
different seeds share no flow. Class ratios are whatever the capture holds in
each period; nothing is rebalanced. Imputation and scaling are fitted on the
warm-up windows only.

    python tools/build_natural_streams.py --datasets nf-ton-iot-v3 \
        --stream-seeds 3001,3002,3003 --output-dir prepared_streams_v2/natural
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from tools.build_controlled_nf_streams import DATASETS, _feature_columns, _sha256, _write_stream  # noqa: E402

PROTOCOL_VERSION = 'natural-2.0'
TIME_COLUMN = 'FLOW_START_MILLISECONDS'
LIMITATIONS = [
    'Windows are uniform subsamples of equal-count time segments; flows of one session '
    'are not kept together and inter-flow timing inside a window is not preserved.',
    'Segments have equal numbers of flows, not equal durations; busy periods are compressed.',
    'Drift is whatever the capture contains; no scenario is imposed and no drift is guaranteed.',
]


def scan(path: Path, chunksize: int) -> tuple[np.ndarray, np.ndarray, int]:
    times, labels = [], []
    rows = 0
    for chunk in pd.read_csv(path, usecols=[TIME_COLUMN, 'Label'], chunksize=chunksize):
        times.append(pd.to_numeric(chunk[TIME_COLUMN], errors='coerce').to_numpy(float))
        labels.append(pd.to_numeric(chunk['Label'], errors='coerce').to_numpy(float))
        rows += len(chunk)
        print(f'scan rows={rows}', flush=True)
    return np.concatenate(times), np.concatenate(labels), rows


def select(times: np.ndarray, labels: np.ndarray, n_windows: int, window_size: int,
           seeds: list[int], sample_seed: int) -> tuple[dict[int, np.ndarray], list[dict], int]:
    """Row ids per seed, shape (n_windows, window_size), in time order."""

    valid = np.isfinite(times) & np.isin(labels, (0, 1))
    rows = np.flatnonzero(valid)
    order = rows[np.lexsort((rows, times[rows]))]
    segments = np.array_split(order, n_windows)
    need = window_size * len(seeds)
    if min(len(segment) for segment in segments) < need:
        raise ValueError(f'a segment has {min(map(len, segments))} flows; {len(seeds)} seeds need {need}')
    rng = np.random.default_rng(sample_seed)
    position = np.empty(len(times), dtype=np.int64)
    position[order] = np.arange(len(order))
    chosen = {seed: [] for seed in seeds}
    summary = []
    for index, segment in enumerate(segments):
        picked = rng.choice(segment, size=need, replace=False)
        for slot, seed in enumerate(seeds):
            part = picked[slot * window_size:(slot + 1) * window_size]
            chosen[seed].append(part[np.argsort(position[part], kind='stable')])
        summary.append({'window': index, 'segment_flows': int(len(segment)),
                        'start_ms': int(times[segment[0]]), 'end_ms': int(times[segment[-1]]),
                        'segment_attack_rate': float(labels[segment].mean())})
    return {seed: np.stack(parts) for seed, parts in chosen.items()}, summary, int((~valid).sum())


def extract(path: Path, features: list[str], wanted: np.ndarray, chunksize: int):
    wanted = np.unique(wanted)
    x = np.empty((len(wanted), len(features)), dtype=np.float32)
    subtype = np.empty(len(wanted), dtype='<U48')
    offset = 0
    for chunk in pd.read_csv(path, usecols=[*features, 'Attack'], chunksize=chunksize, low_memory=False):
        local = wanted[(wanted >= offset) & (wanted < offset + len(chunk))]
        if len(local):
            positions = np.searchsorted(wanted, local)
            x[positions] = chunk[features].apply(pd.to_numeric, errors='coerce').to_numpy(np.float64)[local - offset]
            subtype[positions] = chunk['Attack'].astype(str).str.strip().str.casefold().to_numpy()[local - offset]
        offset += len(chunk)
        print(f'extract rows={offset}', flush=True)
    return wanted, x, subtype


def build(args) -> None:
    seeds = [int(value) for value in args.stream_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('stream seeds must be unique and non-empty')
    output_root = Path(args.output_dir)
    for dataset in [value.strip() for value in args.datasets.split(',') if value.strip()]:
        spec = DATASETS[dataset]
        path = Path(spec.path)
        features = _feature_columns(path)
        times, labels, rows = scan(path, args.chunksize)
        if rows != spec.total_rows:
            raise ValueError(f'{dataset}: expected {spec.total_rows} rows, read {rows}')
        sample_seed = int.from_bytes(dataset.encode(), 'little') % (2**32) ^ args.sample_salt
        chosen, segments, invalid = select(times, labels, args.n_windows, args.window_size, seeds, sample_seed)
        wanted, raw, subtypes = extract(path, features, np.concatenate([v.reshape(-1) for v in chosen.values()]),
                                        args.chunksize)
        source_sha256 = _sha256(path)
        for seed, row_ids in chosen.items():
            index = np.searchsorted(wanted, row_ids)
            values = raw[index]
            warmup = values[:args.warmup_windows].reshape(-1, values.shape[-1])
            medians = np.nanmedian(np.where(np.isfinite(warmup), warmup, np.nan), axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            imputed = np.where(np.isfinite(values), values, medians)
            scaler = StandardScaler().fit(imputed[:args.warmup_windows].reshape(-1, imputed.shape[-1]))
            scaled = scaler.transform(imputed.reshape(-1, imputed.shape[-1])).reshape(imputed.shape)
            y = labels[row_ids].astype(np.int64)
            arrays = {
                'x': scaled.astype(np.float32), 'y': y, 'subtypes': subtypes[index].astype('<U48'),
                'source_ids': row_ids.astype(np.int64), 'timestamps': times[row_ids].astype(np.int64),
                'phases': np.asarray(['warmup' if w < args.warmup_windows else 'chronological'
                                      for w in range(args.n_windows)], dtype='<U24'),
                'feature_names': np.asarray(features, dtype='<U64'),
                'class_names': np.asarray(['Benign', 'Attack'], dtype='<U16'),
                'scaler_mean': scaler.mean_.astype(np.float64), 'scaler_scale': scaler.scale_.astype(np.float64),
                'imputation_median': medians.astype(np.float64)}
            manifest = {
                'protocol_version': PROTOCOL_VERSION, 'builder': 'tools/build_natural_streams.py',
                'dataset_key': dataset, 'dataset_name': spec.name, 'source_path': str(path.resolve()),
                'source_sha256': source_sha256, 'source_rows': rows, 'invalid_rows_excluded': invalid,
                'scenario': 'Natural-Chronological', 'stream_seed': seed, 'stream_seed_set': seeds,
                'sample_seed': sample_seed, 'window_size': args.window_size, 'n_windows': args.n_windows,
                'warmup_windows': args.warmup_windows, 'feature_count': len(features),
                'attack_rate_per_window': [round(float(v), 6) for v in y.mean(axis=1)],
                'subtype_counts': dict(Counter(arrays['subtypes'].reshape(-1).tolist())),
                'segments': segments, 'cross_seed_overlap': 0,
                'preprocessing_fit_windows': list(range(args.warmup_windows)), 'limitations': LIMITATIONS}
            destination = output_root / dataset / 'Natural-Chronological' / f'stream_seed{seed}.npz'
            _write_stream(destination, arrays, manifest)
            print(f'wrote {destination}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--stream-seeds', required=True)
    parser.add_argument('--output-dir', default='prepared_streams_v2/natural')
    parser.add_argument('--window-size', type=int, default=4096)
    parser.add_argument('--n-windows', type=int, default=36)
    parser.add_argument('--warmup-windows', type=int, default=4)
    parser.add_argument('--sample-salt', type=int, default=20261018)
    parser.add_argument('--chunksize', type=int, default=500_000)
    build(parser.parse_args())


if __name__ == '__main__':
    main()
