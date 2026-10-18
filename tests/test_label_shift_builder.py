from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

import tools.build_label_shift_streams as shift
from tools.build_controlled_nf_streams import DatasetSpec

WINDOW = 64


@pytest.fixture()
def source(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    counts = {'benign': 9000, 'dos': 7000, 'scan': 500}
    frames = []
    for family, count in counts.items():
        frames.append(pd.DataFrame({
            'FLOW_START_MILLISECONDS': rng.integers(0, 10**9, count),
            'FLOW_END_MILLISECONDS': 0, 'IPV4_SRC_ADDR': [f'10.{family}.{i}' for i in range(count)],
            'IPV4_DST_ADDR': '10.0.0.1', 'L4_SRC_PORT': rng.integers(1, 60000, count),
            'L4_DST_PORT': 80, 'PROTOCOL': 6, 'IN_BYTES': rng.normal(size=count) + (family != 'benign'),
            'OUT_PKTS': rng.normal(size=count), 'Label': int(family != 'benign'),
            'Attack': 'Benign' if family == 'benign' else family}))
    frame = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=1)
    path = tmp_path / 'nf.csv'
    frame.to_csv(path, index=False)
    spec = DatasetSpec('synthetic-nf', str(path), len(frame), counts['dos'] + counts['scan'],
                       counts, {'dos': 1.0}, {'scan': 1.0}, {'scan': 1.0}, {'dos': 1.0}, {'dos': 1.0}, {'scan': 1.0})
    monkeypatch.setitem(shift.DATASETS, 'synthetic-nf', spec)
    monkeypatch.setitem(shift.SHIFT, 'synthetic-nf', {'minority': 'attack', 'base': 0.10, 'high': 0.50, 'low': 0.02})
    return tmp_path, spec


def run(root, seeds='1,2'):
    shift.build(Namespace(datasets='synthetic-nf', stream_seeds=seeds, output_dir=str(root / 'out'),
                          window_size=WINDOW, n_windows=36, warmup_windows=4, chunksize=5000))


def load(root, scenario, seed):
    path = root / 'out' / 'synthetic-nf' / scenario / f'stream_seed{seed}.npz'
    return dict(np.load(path)), json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))


def test_prevalence_follows_the_schedule_and_families_stay_fixed(source):
    root, spec = source
    run(root)
    for scenario in shift.SCENARIOS:
        arrays, manifest = load(root, scenario, 1)
        expected = [round(WINDOW * shift.minority_share(scenario, w, .10, .50, .02)) / WINDOW for w in range(36)]
        assert np.allclose(arrays['y'].mean(axis=1), expected, atol=1e-9)
        assert set(arrays['subtypes'][arrays['y'] == 1].tolist()) == {'dos'}
        assert manifest['attack_families_fixed_to'] == {'dos': 1.0}
        assert np.isfinite(arrays['x']).all()
    burst = load(root, 'L1-Burst', 1)[0]['y'].mean(axis=1)
    assert burst[12:20].min() > burst[:12].max() and burst[20:].max() < burst[12:20].min()


def test_seeds_and_segments_are_disjoint(source):
    root, _ = source
    run(root)
    ids = {seed: set(load(root, 'L3-Swing', seed)[0]['source_ids'].reshape(-1).tolist()) for seed in (1, 2)}
    assert not ids[1] & ids[2]
    sessions = load(root, 'L2-Ramp', 2)[0]['session_ids']
    parts = [set(sessions[lo:hi].reshape(-1).tolist()) for lo, hi in shift.SEGMENTS]
    assert all(not a & b for i, a in enumerate(parts) for b in parts[i + 1:])


def test_schedule_shapes():
    assert shift.minority_share('L2-Ramp', 23, .1, .5, .02) == pytest.approx(.5)
    assert shift.minority_share('L2-Ramp', 35, .1, .5, .02) == pytest.approx(.1)
    assert [shift.minority_share('L3-Swing', w, .1, .5, .02) for w in (12, 16, 20)] == [.5, .02, .5]
    with pytest.raises(ValueError):
        shift.minority_share('L9', 20, .1, .5, .02)
