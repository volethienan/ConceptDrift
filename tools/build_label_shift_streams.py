"""Build controlled label-shift streams: class prevalence changes, attack families do not.

Every window draws attacks from the dataset's concept-A family mixture, so
p(x | y) is held fixed while p(y) follows a schedule. Seeds share no source
row or session (pools come from the NF builder's seed assignment); sessions
are kept inside one stream segment. Imputation and scaling are fitted on the
warm-up windows only.

Schedules for the minority class share m(t) (attacks in NF-UNSW, peak 25%,
since three disjoint seeds exhaust 'exploits' above that; benign in NF-BoT,
peak 12%), with warm-up and windows 4-11 at the dataset base rate:

* L1-Burst: high share in windows 12-19, base afterwards;
* L2-Ramp:  linear rise to the high share at window 23, linear return to base at 35;
* L3-Swing: alternate high/low every four windows from window 12.

    python tools/build_label_shift_streams.py --datasets nf-unsw-nb15-v3 \
        --stream-seeds 2001,2002,2003 --output-dir prepared_streams_v2/label_shift
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from tools.build_controlled_nf_streams import (  # noqa: E402
    DATASETS, _allocate, _collect_pools, _sha256, _splitmix64, _write_stream)

PROTOCOL_VERSION = 'label-shift-1.0'
SCENARIOS = ('L1-Burst', 'L2-Ramp', 'L3-Swing')
# Minority class and its share: base (dataset prevalence), high and low.
SHIFT = {
    'nf-unsw-nb15-v3': {'minority': 'attack', 'base': 221 / 4096, 'high': 0.25, 'low': 0.01},
    'nf-bot-iot-v3': {'minority': 'benign', 'base': 13 / 4096, 'high': 0.12, 'low': 13 / 4096},
}
SEGMENTS = ((0, 4), (4, 12), (12, 24), (24, 36))


def minority_share(scenario: str, window: int, base: float, high: float, low: float) -> float:
    if window < 12:
        return base
    if scenario == 'L1-Burst':
        return high if window < 20 else base
    if scenario == 'L2-Ramp':
        return base + (high - base) * (window - 11) / 12 if window <= 23 else high - (high - base) * (window - 23) / 12
    if scenario == 'L3-Swing':
        return high if ((window - 12) // 4) % 2 == 0 else low
    raise ValueError(f'unknown label-shift scenario {scenario}')


def window_counts(spec, dataset: str, scenario: str, window_size: int, n_windows: int) -> list[dict[str, int]]:
    """Per-window family counts: benign plus the fixed concept-A attack mixture."""

    shift = SHIFT[dataset]
    counts = []
    for window in range(n_windows):
        share = minority_share(scenario, window, shift['base'], shift['high'], shift['low'])
        minority = int(round(window_size * share))
        attacks = minority if shift['minority'] == 'attack' else window_size - minority
        recipe = dict(_allocate(attacks, spec.concept_a))
        recipe['benign'] = window_size - attacks
        counts.append({family: count for family, count in recipe.items() if count > 0})
    return counts


def partition(pool: dict[str, np.ndarray], needs: list[int], salt: int) -> list[dict[str, np.ndarray]]:
    """Assign whole sessions to segments, largest first, proportional to need."""

    sessions, inverse = np.unique(pool['session_ids'].astype(np.uint64), return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(sessions))
    order = np.lexsort((_splitmix64(sessions ^ np.uint64(salt)), -sizes))
    targets = np.asarray(needs, dtype=float)
    assigned = np.zeros(len(needs))
    segment_of = np.full(len(sessions), -1)
    active = np.flatnonzero(targets > 0)
    for session in order:
        segment = active[np.argmin(assigned[active] / targets[active])]
        segment_of[session] = segment
        assigned[segment] += sizes[session]
    rows = segment_of[inverse]
    parts = []
    for index, need in enumerate(needs):
        selected = np.flatnonzero(rows == index)
        if len(selected) < need:
            raise ValueError(f'segment {index}: {len(selected)} rows, needs {need}')
        parts.append({name: values[selected] for name, values in pool.items()})
    return parts


def assemble(spec, dataset, scenario, pools, slot, seed, features, window_size, n_windows, warmup):
    counts = window_counts(spec, dataset, scenario, window_size, n_windows)
    families = sorted({family for recipe in counts for family in recipe})
    segment_pools = {}
    salt = sum((i + 1) * ord(c) for i, c in enumerate(scenario))
    for family in families:
        needs = [sum(counts[w].get(family, 0) for w in range(*bounds)) for bounds in SEGMENTS]
        segment_pools[family] = partition(pools[(family, slot)], needs, salt)
    cursor = Counter()
    rng = np.random.default_rng(seed * 1_000_003 + salt)
    parts = {key: [] for key in ('x', 'y', 'subtypes', 'source_ids', 'session_ids', 'timestamps')}
    for window, recipe in enumerate(counts):
        segment = next(i for i, (lo, hi) in enumerate(SEGMENTS) if lo <= window < hi)
        chunks = []
        for family in sorted(recipe):
            pool = segment_pools[family][segment]
            start = cursor[(family, segment)]
            stop = start + recipe[family]
            if stop > len(pool['x']):
                raise ValueError(f'{dataset}/{scenario}/{seed}/{family}: segment {segment} exhausted')
            cursor[(family, segment)] = stop
            chunks.append(({name: values[start:stop] for name, values in pool.items()}, family))
        order = rng.permutation(window_size)
        stack = lambda name: np.concatenate([chunk[name] for chunk, _ in chunks])[order]
        parts['x'].append(stack('x'))
        parts['source_ids'].append(stack('source_ids'))
        parts['session_ids'].append(stack('session_ids'))
        parts['timestamps'].append(stack('timestamps'))
        labels = np.concatenate([np.full(len(chunk['x']), int(family != 'benign')) for chunk, family in chunks])
        names = np.concatenate([np.full(len(chunk['x']), family, dtype='<U48') for chunk, family in chunks])
        parts['y'].append(labels[order])
        parts['subtypes'].append(names[order])
    arrays = {key: np.stack(values) for key, values in parts.items()}
    if len(np.unique(arrays['source_ids'])) != arrays['source_ids'].size:
        raise ValueError(f'{dataset}/{scenario}/{seed}: source row overlap')
    segment_sessions = [set(arrays['session_ids'][lo:hi].reshape(-1).tolist()) for lo, hi in SEGMENTS]
    if any(a & b for i, a in enumerate(segment_sessions) for b in segment_sessions[i + 1:]):
        raise ValueError(f'{dataset}/{scenario}/{seed}: session overlap across segments')
    raw = arrays['x']
    warm = raw[:warmup].reshape(-1, raw.shape[-1])
    medians = np.nanmedian(np.where(np.isfinite(warm), warm, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = np.where(np.isfinite(raw), raw, medians)
    scaler = StandardScaler().fit(imputed[:warmup].reshape(-1, imputed.shape[-1]))
    arrays['x'] = scaler.transform(imputed.reshape(-1, imputed.shape[-1])).reshape(imputed.shape).astype(np.float32)
    arrays['y'] = arrays['y'].astype(np.int64)
    arrays['session_ids'] = arrays['session_ids'].astype(np.uint64)
    arrays.update({
        'phases': np.asarray(['warmup' if w < warmup else f'prior_{arrays["y"][w].mean():.3f}'
                              for w in range(n_windows)], dtype='<U24'),
        'feature_names': np.asarray(features, dtype='<U64'),
        'class_names': np.asarray(['Benign', 'Attack'], dtype='<U16'),
        'scaler_mean': scaler.mean_.astype(np.float64), 'scaler_scale': scaler.scale_.astype(np.float64),
        'imputation_median': medians.astype(np.float64)})
    return arrays, counts


def build(args) -> None:
    seeds = [int(value) for value in args.stream_seeds.split(',') if value.strip()]
    for dataset in [value.strip() for value in args.datasets.split(',') if value.strip()]:
        spec = DATASETS[dataset]
        required = {}
        for scenario in SCENARIOS:
            totals = Counter()
            for recipe in window_counts(spec, dataset, scenario, args.window_size, args.n_windows):
                totals.update(recipe)
            for family, count in totals.items():
                required[family] = max(required.get(family, 0), count)
        features, pools, _, _, observed = _collect_pools(spec, seeds, required, {}, {}, 3.0,
                                                         args.chunksize, 0, 0)
        source_sha256 = _sha256(Path(spec.path))
        for slot, seed in enumerate(seeds):
            for scenario in SCENARIOS:
                arrays, counts = assemble(spec, dataset, scenario, pools, slot, seed, features,
                                          args.window_size, args.n_windows, args.warmup_windows)
                manifest = {
                    'protocol_version': PROTOCOL_VERSION, 'builder': 'tools/build_label_shift_streams.py',
                    'dataset_key': dataset, 'dataset_name': spec.name, 'source_sha256': source_sha256,
                    'observed_required_family_counts': dict(observed), 'scenario': scenario,
                    'stream_seed': seed, 'stream_seed_set': seeds, 'window_size': args.window_size,
                    'n_windows': args.n_windows, 'warmup_windows': args.warmup_windows,
                    'minority_class': SHIFT[dataset]['minority'], 'shift_levels': SHIFT[dataset],
                    'attack_rate_per_window': [round(float(v), 6) for v in arrays['y'].mean(axis=1)],
                    'attack_families_fixed_to': spec.concept_a, 'feature_count': len(features),
                    'subtype_counts': dict(Counter(arrays['subtypes'].reshape(-1).tolist())),
                    'overlap_within_stream': 0, 'session_overlap_across_segments': 0,
                    'preprocessing_fit_windows': list(range(args.warmup_windows)),
                    'limitations': ['Seed assignment is computed for this seed set only; rows may also '
                                    'appear in other builds of the same dataset with different seed sets.']}
                destination = Path(args.output_dir) / dataset / scenario / f'stream_seed{seed}.npz'
                _write_stream(destination, arrays, manifest)
                print(f'wrote {destination}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--stream-seeds', default='2001,2002,2003')
    parser.add_argument('--output-dir', default='prepared_streams_v2/label_shift')
    parser.add_argument('--window-size', type=int, default=4096)
    parser.add_argument('--n-windows', type=int, default=36)
    parser.add_argument('--warmup-windows', type=int, default=4)
    parser.add_argument('--chunksize', type=int, default=200_000)
    build(parser.parse_args())


if __name__ == '__main__':
    main()
