from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

import tools.build_natural_streams as natural
from tools.build_controlled_nf_streams import DatasetSpec
from tools.run_original_review import inspect_prepared


@pytest.fixture()
def capture(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    rows = 36 * 300
    # Shuffled source order: the builder must sort by start time itself.
    times = rng.permutation(rows) * 1000 + 1_700_000_000_000
    label = (times > np.quantile(times, 0.6)).astype(int)  # attacks only late in the capture
    frame = pd.DataFrame({'FLOW_START_MILLISECONDS': times, 'FLOW_END_MILLISECONDS': times + 5,
                          'IPV4_SRC_ADDR': '10.0.0.1', 'IPV4_DST_ADDR': '10.0.0.2',
                          'IN_BYTES': rng.normal(size=rows) + 3 * label, 'OUT_PKTS': rng.normal(size=rows),
                          'Label': label, 'Attack': np.where(label == 1, 'DoS', 'Benign')})
    frame.loc[7, 'IN_BYTES'] = np.inf
    path = tmp_path / 'capture.csv'
    frame.to_csv(path, index=False)
    spec = DatasetSpec('synthetic-nf', str(path), rows, int(label.sum()), {}, {}, {}, {}, {}, {}, {})
    monkeypatch.setitem(natural.DATASETS, 'synthetic-nf', spec)
    return tmp_path, frame


def arguments(output, seeds='1,2,3'):
    return Namespace(datasets='synthetic-nf', stream_seeds=seeds, output_dir=str(output), window_size=64,
                     n_windows=36, warmup_windows=4, sample_salt=11, chunksize=1000)


def load(root, seed):
    path = root / 'synthetic-nf' / 'Natural-Chronological' / f'stream_seed{seed}.npz'
    return path, dict(np.load(path)), json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))


def test_streams_span_the_capture_in_time_order_and_seeds_are_disjoint(capture):
    root, frame = capture
    natural.build(arguments(root / 'out'))
    seen = set()
    for seed in (1, 2, 3):
        path, arrays, manifest = load(root / 'out', seed)
        inspect_prepared(path, chronological=True)
        stamps = arrays['timestamps'].reshape(-1)
        assert np.all(np.diff(stamps) >= 0)
        for window, segment in enumerate(manifest['segments']):
            assert segment['start_ms'] <= arrays['timestamps'][window].min()
            assert arrays['timestamps'][window].max() <= segment['end_ms']
        ids = set(arrays['source_ids'].reshape(-1).tolist())
        assert not seen & ids
        seen |= ids
        assert np.array_equal(arrays['y'].reshape(-1), frame.Label.to_numpy()[arrays['source_ids'].reshape(-1)])
        assert arrays['y'][:12].sum() == 0 and arrays['y'][-8:].min() == 1  # natural prevalence kept
        assert np.isfinite(arrays['x']).all()
        warmup = arrays['x'][:4].reshape(-1, arrays['x'].shape[-1])
        assert np.allclose(warmup.mean(axis=0), 0, atol=1e-5)


def test_build_is_deterministic(capture):
    root, _ = capture
    natural.build(arguments(root / 'a'))
    natural.build(arguments(root / 'b'))
    for seed in (1, 2, 3):
        first, second = load(root / 'a', seed)[1], load(root / 'b', seed)[1]
        assert all(np.array_equal(first[key], second[key]) for key in first)


def test_too_many_seeds_for_a_segment_fails(capture):
    root, _ = capture
    with pytest.raises(ValueError, match='segment'):
        natural.build(arguments(root / 'c', seeds='1,2,3,4,5'))
