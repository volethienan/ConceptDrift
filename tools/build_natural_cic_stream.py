"""Build chronological CIC-IDS2017 streams that span the whole capture week.

Counterpart of ``build_natural_streams.py`` for the MachineLearningCVE files,
which carry no timestamp. Rows are ordered by capture file (Monday to Friday
afternoon, as in ``build_controlled_cic_streams.FILE_ORDER``) and then by row
position inside the file. CICFlowMeter writes a flow when it ends, so row
order follows flow-end time; the published attack schedule is reproduced
(Wednesday: slowloris, Slowhttptest, Hulk, GoldenEye, Heartbleed). The ordered
capture is cut into ``n_windows`` equal-count segments and each window is a
label-blind uniform sample of ``window_size`` flows from its segment, kept in
order. Seeds are drawn together without replacement. Nothing is deduplicated
or rebalanced, so bursts of identical attack flows keep their natural density.
Imputation and scaling are fitted on the warm-up windows only.

    python tools/build_natural_cic_stream.py --stream-seeds 4001,4002,4003 \
        --output-dir prepared_streams_v2/natural
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from tools.build_controlled_cic_streams import CIC_IDS2017, FILE_ORDER, _family, _features, _read  # noqa: E402
from tools.build_controlled_nf_streams import _sha256, _write_stream  # noqa: E402

PROTOCOL_VERSION = 'natural-2.0-cic'
DATASET = 'cic-ids2017'
LIMITATIONS = [
    'The MachineLearningCVE files have no timestamp; order is capture file, then row '
    'position (flow-end order). timestamps holds the global source-order ordinal, not a time.',
    'Windows are uniform subsamples of equal-count segments; sessions are not kept together.',
    'Segments have equal numbers of flows, not equal durations; busy periods are compressed.',
    'Labels are used as published; CIC-IDS2017 has documented labelling and feature-extraction '
    'errors (Engelen et al., 2021). Feature vectors that occur with both labels are kept.',
    'Drift is whatever the capture contains; no scenario is imposed.',
]


def build(args) -> None:
    seeds = [int(value) for value in args.stream_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('stream seeds must be unique and non-empty')
    source = Path(args.source_dir)
    families, starts, columns = [], [0], None
    for name in FILE_ORDER:
        frame = _read(source / name)
        names = [c for c in frame.columns if c != 'Label']
        if columns is None:
            columns = names
        elif names != columns:
            raise ValueError(f'{name}: feature columns differ')
        families.append(_family(frame))
        starts.append(starts[-1] + len(frame))
        print(f'scan {name} rows={len(frame)}', flush=True)
    family = np.concatenate(families)
    if len(family) != CIC_IDS2017.total_rows:
        raise ValueError(f'expected {CIC_IDS2017.total_rows} rows, read {len(family)}')
    labels = (family != 'benign').astype(np.int64)
    if int(labels.sum()) != CIC_IDS2017.attack_rows:
        raise ValueError(f'expected {CIC_IDS2017.attack_rows} attack rows, read {int(labels.sum())}')

    segments = np.array_split(np.arange(len(family)), args.n_windows)
    need = args.window_size * len(seeds)
    if min(map(len, segments)) < need:
        raise ValueError(f'a segment has {min(map(len, segments))} rows; {len(seeds)} seeds need {need}')
    sample_seed = int.from_bytes(DATASET.encode(), 'little') % (2**32) ^ args.sample_salt
    rng = np.random.default_rng(sample_seed)
    chosen = {seed: [] for seed in seeds}
    summary = []
    for index, segment in enumerate(segments):
        picked = rng.choice(segment, size=need, replace=False)
        for slot, seed in enumerate(seeds):
            chosen[seed].append(np.sort(picked[slot * args.window_size:(slot + 1) * args.window_size]))
        files = sorted({FILE_ORDER[int(np.searchsorted(starts, row, side='right')) - 1] for row in (segment[0], segment[-1])})
        summary.append({'window': index, 'segment_rows': int(len(segment)), 'start_ordinal': int(segment[0]),
                        'end_ordinal': int(segment[-1]), 'files': files,
                        'segment_attack_rate': float(labels[segment].mean())})
    chosen = {seed: np.stack(parts) for seed, parts in chosen.items()}

    wanted = np.unique(np.concatenate([v.reshape(-1) for v in chosen.values()]))
    raw = np.empty((len(wanted), len(columns)), dtype=np.float64)
    for file_index, name in enumerate(FILE_ORDER):
        local = wanted[(wanted >= starts[file_index]) & (wanted < starts[file_index + 1])]
        if len(local):
            values = _features(_read(source / name)).to_numpy(np.float64)
            raw[np.searchsorted(wanted, local)] = values[local - starts[file_index]]
        print(f'extract {name}', flush=True)
    source_sha256 = {name: _sha256(source / name) for name in FILE_ORDER}

    for seed, row_ids in chosen.items():
        index = np.searchsorted(wanted, row_ids)
        values = raw[index]
        warmup = values[:args.warmup_windows].reshape(-1, values.shape[-1])
        medians = np.nanmedian(np.where(np.isfinite(warmup), warmup, np.nan), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        imputed = np.where(np.isfinite(values), values, medians)
        scaler = StandardScaler().fit(imputed[:args.warmup_windows].reshape(-1, imputed.shape[-1]))
        scaled = scaler.transform(imputed.reshape(-1, imputed.shape[-1])).reshape(imputed.shape)
        y = labels[row_ids]
        arrays = {
            'x': scaled.astype(np.float32), 'y': y, 'subtypes': family[row_ids].astype('<U48'),
            'source_ids': row_ids.astype(np.int64), 'timestamps': row_ids.astype(np.int64),
            'phases': np.asarray(['warmup' if w < args.warmup_windows else 'chronological'
                                  for w in range(args.n_windows)], dtype='<U24'),
            'feature_names': np.asarray(columns, dtype='<U64'),
            'class_names': np.asarray(['Benign', 'Attack'], dtype='<U16'),
            'scaler_mean': scaler.mean_.astype(np.float64), 'scaler_scale': scaler.scale_.astype(np.float64),
            'imputation_median': medians.astype(np.float64)}
        manifest = {
            'protocol_version': PROTOCOL_VERSION, 'builder': 'tools/build_natural_cic_stream.py',
            'dataset_key': DATASET, 'dataset_name': CIC_IDS2017.name, 'source_dir': str(source.resolve()),
            'source_sha256': source_sha256, 'source_rows': int(len(family)), 'file_order': list(FILE_ORDER),
            'scenario': 'Natural-Chronological', 'stream_seed': seed, 'stream_seed_set': seeds,
            'sample_seed': sample_seed, 'window_size': args.window_size, 'n_windows': args.n_windows,
            'warmup_windows': args.warmup_windows, 'feature_count': len(columns),
            'attack_rate_per_window': [round(float(v), 6) for v in y.mean(axis=1)],
            'subtype_counts': dict(Counter(arrays['subtypes'].reshape(-1).tolist())),
            'segments': summary, 'cross_seed_overlap': 0,
            'preprocessing_fit_windows': list(range(args.warmup_windows)), 'limitations': LIMITATIONS}
        destination = Path(args.output_dir) / DATASET / 'Natural-Chronological' / f'stream_seed{seed}.npz'
        _write_stream(destination, arrays, manifest)
        print(f'wrote {destination}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stream-seeds', required=True)
    parser.add_argument('--source-dir', default=CIC_IDS2017.path)
    parser.add_argument('--output-dir', default='prepared_streams_v2/natural')
    parser.add_argument('--window-size', type=int, default=4096)
    parser.add_argument('--n-windows', type=int, default=36)
    parser.add_argument('--warmup-windows', type=int, default=4)
    parser.add_argument('--sample-salt', type=int, default=20261023)
    build(parser.parse_args())


if __name__ == '__main__':
    main()
