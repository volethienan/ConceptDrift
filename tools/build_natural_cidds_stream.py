"""Build chronological CIDDS-001 (OpenStack) streams, a second held-out test.

CIDDS-001 (Ring et al., 2017) is NetFlow from an emulated small business
network on OpenStack. Its attacks (DoS, port scans, some of them nmap -T 1
paranoid scans running for hours, ping scans, SSH brute force) were all
launched in the first two capture weeks (2017-03-15 to 2017-03-28); weeks
three and four are benign only, so the stream uses weeks one and two. Flows
are ordered by "Date first seen" (ties by file position), cut into
``n_windows`` equal-count segments, and each window is a label-blind uniform
sample of ``window_size`` flows from its segment, kept in time order. Seeds
are drawn together without replacement; nothing is deduplicated or
rebalanced. A flow is an attack when attackType is not '---' (attacker and
victim flows both). Features are behaviour only: no IP address or
timestamp. Scaling is fitted on the warm-up windows only.

    python tools/build_natural_cidds_stream.py --stream-seeds 5001,5002,5003 \
        --output-dir prepared_streams_v2/natural
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from tools.build_controlled_nf_streams import _sha256, _write_stream  # noqa: E402

PROTOCOL_VERSION = 'natural-2.0-cidds'
DATASET = 'cidds-001'
FILES = ('CIDDS-001-internal-week1.csv', 'CIDDS-001-internal-week2.csv')
DEFAULT_SOURCE = Path(r'C:\Users\AnThiwn\Desktop\Q!\data\CIDDS-001\CIDDS-001\traffic\OpenStack')
FEATURES = ['duration', 'src_port', 'dst_port', 'packets', 'bytes', 'flows', 'tos', 'bytes_per_packet',
            'proto_tcp', 'proto_udp', 'proto_icmp', 'proto_other',
            'flag_urg', 'flag_ack', 'flag_psh', 'flag_rst', 'flag_syn', 'flag_fin']
CHUNK = 1_000_000
LIMITATIONS = [
    'Weeks 1-2 of the OpenStack traffic only: weeks 3-4 contain no attack.',
    'Windows are uniform subsamples of equal-count time segments; sessions are not kept together.',
    'Segments have equal numbers of flows, not equal durations; nights and weekends are compressed.',
    'Labels as published: attack = attackType other than ---, attacker and victim flows alike.',
    'Public IP addresses are anonymised in the source; no address is used as a feature.',
]


def _number(column: pd.Series) -> np.ndarray:
    """CIDDS writes large counts as '1.2 M'."""
    text = column.astype(str).str.strip()
    scale = np.where(text.str.endswith('M'), 1e6, np.where(text.str.endswith('K'), 1e3,
                     np.where(text.str.endswith('G'), 1e9, 1.0)))
    return pd.to_numeric(text.str.rstrip('MKG').str.strip(), errors='coerce').to_numpy(np.float64) * scale


def _features(frame: pd.DataFrame) -> np.ndarray:
    proto = frame['Proto'].astype(str).str.strip().str.upper()
    flags = frame['Flags'].astype(str).str.strip().str.pad(6, side='right', fillchar='.')
    packets, size = _number(frame['Packets']), _number(frame['Bytes'])
    out = np.column_stack([
        pd.to_numeric(frame['Duration'], errors='coerce').to_numpy(np.float64),
        pd.to_numeric(frame['Src Pt'], errors='coerce').to_numpy(np.float64),
        pd.to_numeric(frame['Dst Pt'], errors='coerce').to_numpy(np.float64),
        packets, size, _number(frame['Flows']),
        pd.to_numeric(frame['Tos'], errors='coerce').to_numpy(np.float64),
        np.where(packets > 0, size / np.maximum(packets, 1), 0.0),
        (proto == 'TCP').to_numpy(float), (proto == 'UDP').to_numpy(float), (proto == 'ICMP').to_numpy(float),
        (~proto.isin(['TCP', 'UDP', 'ICMP'])).to_numpy(float),
        *[(flags.str[i] != '.').to_numpy(float) for i in range(6)],
    ])
    return out


def _scan(path: Path):
    times, attack = [], []
    for chunk in pd.read_csv(path, usecols=['Date first seen', 'attackType'], chunksize=CHUNK, dtype=str):
        times.append(pd.to_datetime(chunk['Date first seen'].str.strip(), format='%Y-%m-%d %H:%M:%S.%f')
                     .to_numpy('datetime64[ms]').astype(np.int64))
        attack.append(chunk['attackType'].str.strip().to_numpy())
    return np.concatenate(times), np.concatenate(attack).astype('<U16')


def build(args) -> None:
    seeds = [int(value) for value in args.stream_seeds.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('stream seeds must be unique and non-empty')
    source = Path(args.source_dir)
    times, family, starts = [], [], [0]
    for name in FILES:
        t, a = _scan(source / name)
        times.append(t)
        family.append(a)
        starts.append(starts[-1] + len(t))
        print(f'scan {name} rows={len(t)}', flush=True)
    times, family = np.concatenate(times), np.concatenate(family)
    family = np.where(family == '---', 'benign', family)
    labels = (family != 'benign').astype(np.int64)
    order = np.lexsort((np.arange(len(times)), times))  # time, then file position

    segments = np.array_split(order, args.n_windows)
    need = args.window_size * len(seeds)
    if min(map(len, segments)) < need:
        raise ValueError(f'a segment has {min(map(len, segments))} rows; {len(seeds)} seeds need {need}')
    sample_seed = int.from_bytes(DATASET.encode(), 'little') % (2**32) ^ args.sample_salt
    rng = np.random.default_rng(sample_seed)
    chosen = {seed: [] for seed in seeds}
    summary = []
    for index, segment in enumerate(segments):
        picked = rng.choice(len(segment), size=need, replace=False)
        for slot, seed in enumerate(seeds):
            part = np.sort(picked[slot * args.window_size:(slot + 1) * args.window_size])
            chosen[seed].append(segment[part])  # time order inside the window
        summary.append({'window': index, 'segment_rows': int(len(segment)),
                        'start_ms': int(times[segment[0]]), 'end_ms': int(times[segment[-1]]),
                        'segment_attack_rate': float(labels[segment].mean())})
    chosen = {seed: np.stack(parts) for seed, parts in chosen.items()}

    wanted = np.unique(np.concatenate([v.reshape(-1) for v in chosen.values()]))
    raw = np.empty((len(wanted), len(FEATURES)), dtype=np.float64)
    for file_index, name in enumerate(FILES):
        offset = starts[file_index]
        for number, chunk in enumerate(pd.read_csv(source / name, chunksize=CHUNK, dtype=str)):
            lo = offset + number * CHUNK
            hi = lo + len(chunk)
            local = wanted[(wanted >= lo) & (wanted < hi)]
            if len(local):
                raw[np.searchsorted(wanted, local)] = _features(chunk.iloc[local - lo])
        print(f'extract {name}', flush=True)
    source_sha256 = {name: _sha256(source / name) for name in FILES}

    for seed, row_ids in chosen.items():
        values = raw[np.searchsorted(wanted, row_ids)]
        warmup = values[:args.warmup_windows].reshape(-1, values.shape[-1])
        medians = np.nanmedian(np.where(np.isfinite(warmup), warmup, np.nan), axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        imputed = np.where(np.isfinite(values), values, medians)
        scaler = StandardScaler().fit(imputed[:args.warmup_windows].reshape(-1, imputed.shape[-1]))
        scaler.scale_ = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
        scaled = scaler.transform(imputed.reshape(-1, imputed.shape[-1])).reshape(imputed.shape)
        y = labels[row_ids]
        arrays = {
            'x': scaled.astype(np.float32), 'y': y, 'subtypes': family[row_ids].astype('<U48'),
            'source_ids': row_ids.astype(np.int64), 'timestamps': times[row_ids].astype(np.int64),
            'phases': np.asarray(['warmup' if w < args.warmup_windows else 'chronological'
                                  for w in range(args.n_windows)], dtype='<U24'),
            'feature_names': np.asarray(FEATURES, dtype='<U64'),
            'class_names': np.asarray(['Benign', 'Attack'], dtype='<U16'),
            'scaler_mean': scaler.mean_.astype(np.float64), 'scaler_scale': scaler.scale_.astype(np.float64),
            'imputation_median': medians.astype(np.float64)}
        manifest = {
            'protocol_version': PROTOCOL_VERSION, 'builder': 'tools/build_natural_cidds_stream.py',
            'dataset_key': DATASET, 'dataset_name': 'CIDDS-001 (OpenStack, weeks 1-2)',
            'source_dir': str(source.resolve()), 'source_sha256': source_sha256, 'source_rows': int(len(times)),
            'file_order': list(FILES), 'scenario': 'Natural-Chronological', 'stream_seed': seed,
            'stream_seed_set': seeds, 'sample_seed': sample_seed, 'window_size': args.window_size,
            'n_windows': args.n_windows, 'warmup_windows': args.warmup_windows, 'feature_count': len(FEATURES),
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
    parser.add_argument('--source-dir', default=str(DEFAULT_SOURCE))
    parser.add_argument('--output-dir', default='prepared_streams_v2/natural')
    parser.add_argument('--window-size', type=int, default=4096)
    parser.add_argument('--n-windows', type=int, default=36)
    parser.add_argument('--warmup-windows', type=int, default=4)
    parser.add_argument('--sample-salt', type=int, default=20261110)
    build(parser.parse_args())


if __name__ == '__main__':
    main()
