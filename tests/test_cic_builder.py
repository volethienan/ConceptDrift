from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from tools.build_controlled_cic_streams import PROTOCOL_VERSION, build
from tools.build_controlled_nf_streams import DatasetSpec

FILES = ('day1.csv', 'day2.csv', 'day3.csv')


def write_source(directory, duplicate_benign=40):
    rng = np.random.default_rng(7)
    layout = {'day1.csv': {'benign': 4000}, 'day2.csv': {'benign': 3000, 'alpha': 1600},
              'day3.csv': {'benign': 2400, 'beta': 1600}}
    counts = {}
    planted = {}
    for name, families in layout.items():
        parts = []
        for family, count in families.items():
            values = rng.normal(size=(count, 5)) + (0 if family == 'benign' else 3)
            frame = pd.DataFrame(values, columns=[f' f{i}' for i in range(5)])
            frame[' Label'] = family.upper() if family == 'benign' else family
            parts.append(frame)
        frame = pd.concat(parts, ignore_index=True)
        if name == 'day1.csv':
            # Exact duplicates of benign rows and one vector labelled twice.
            frame = pd.concat([frame, frame.iloc[:duplicate_benign]], ignore_index=True)
            conflict = frame.iloc[[100]].copy()
            conflict[' Label'] = 'alpha'
            frame = pd.concat([frame, conflict], ignore_index=True)
            planted = {'duplicates': duplicate_benign, 'conflict_vector': frame.iloc[100, :5].to_numpy(dtype=float)}
        # A non-finite value outside the duplicated rows, as in Flow Bytes/s.
        frame.loc[60, ' f1'] = np.inf
        frame.to_csv(directory / name, index=False)
        for family, count in frame[' Label'].str.strip().str.casefold().value_counts().items():
            counts[family] = counts.get(family, 0) + int(count)
    return counts, planted


def spec_for(counts):
    attacks = sum(v for k, v in counts.items() if k != 'benign')
    return DatasetSpec(
        name='synthetic-cic', path='unused', total_rows=sum(counts.values()), attack_rows=attacks,
        family_counts=counts, concept_a={'alpha': 1.0}, concept_b={'beta': 1.0},
        concept_c={'alpha': 0.5, 'beta': 0.5}, stable={'alpha': 1.0},
        local_a={'beta': 1.0}, local_b={'alpha': 1.0})


def arguments(source, output, seeds='11,12', block_rows=40):
    return Namespace(source_dir=str(source), scenarios='S0,S1-Sudden,S4-Recurrent',
                     output_dir=str(output), stream_seeds=seeds, window_size=48, n_windows=36,
                     warmup_windows=4, block_rows=block_rows, attack_margin=1.10, benign_margin=1.20)


@pytest.fixture()
def built(tmp_path):
    source = tmp_path / 'raw'
    source.mkdir()
    counts, planted = write_source(source)
    spec = spec_for(counts)
    build(arguments(source, tmp_path / 'out'), spec, FILES)
    return tmp_path, source, spec, planted


def load(root, scenario, seed):
    path = root / 'out' / 'cic-ids2017' / scenario / f'stream_seed{seed}.npz'
    return dict(np.load(path)), json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))


def test_streams_are_seed_disjoint_deduplicated_and_conflict_free(built):
    root, source, _, planted = built
    raw = pd.concat([pd.read_csv(source / name) for name in FILES], ignore_index=True)
    ids = {}
    for seed in (11, 12):
        arrays, manifest = load(root, 'S0', seed)
        assert manifest['protocol_version'] == PROTOCOL_VERSION
        assert manifest['cleaning']['conflicting_rows'] == 2
        assert manifest['cleaning']['duplicate_rows_removed'] == planted['duplicates']
        source_ids = arrays['source_ids'].reshape(-1)
        assert len(np.unique(source_ids)) == len(source_ids)
        rows = raw.iloc[source_ids, :5].round(9)
        assert not rows.duplicated().any()
        assert not (rows.to_numpy() == np.round(planted['conflict_vector'], 9)).all(axis=1).any()
        ids[seed] = set(source_ids.tolist())
        assert np.isfinite(arrays['x']).all()
    assert not ids[11] & ids[12]


def test_blocks_stay_inside_one_segment_and_one_seed(built):
    root, *_ = built
    blocks = {}
    for seed in (11, 12):
        arrays, _ = load(root, 'S4-Recurrent', seed)
        sessions = arrays['session_ids']
        segments = [sessions[:4], sessions[4:12], sessions[12:24], sessions[24:]]
        sets = [set(part.reshape(-1).tolist()) for part in segments]
        assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
        blocks[seed] = set(sessions.reshape(-1).tolist())
    assert not blocks[11] & blocks[12]


def test_recipe_and_prevalence_follow_the_shared_protocol(built):
    root, _, spec, _ = built
    arrays, manifest = load(root, 'S1-Sudden', 11)
    attacks = manifest['attack_count_per_window']
    assert attacks == round(48 * spec.attack_rate)
    assert (arrays['y'].sum(axis=1) == attacks).all()
    assert set(arrays['subtypes'][:12].reshape(-1)) <= {'benign', 'alpha'}
    assert set(arrays['subtypes'][12:].reshape(-1)) <= {'benign', 'beta'}


def test_build_is_deterministic(tmp_path, built):
    root, source, spec, _ = built
    build(arguments(source, tmp_path / 'again'), spec, FILES)
    for scenario in ('S0', 'S1-Sudden', 'S4-Recurrent'):
        for seed in (11, 12):
            first = np.load(root / 'out' / 'cic-ids2017' / scenario / f'stream_seed{seed}.npz')
            second = np.load(tmp_path / 'again' / 'cic-ids2017' / scenario / f'stream_seed{seed}.npz')
            assert all(np.array_equal(first[k], second[k]) for k in first.files)


def test_infeasible_seed_count_fails_loudly(tmp_path, built):
    _, source, spec, _ = built
    with pytest.raises(ValueError, match='cannot give'):
        build(arguments(source, tmp_path / 'too_many', seeds='1,2,3,4,5,6'), spec, FILES)


def test_label_counts_are_validated(tmp_path, built):
    _, source, spec, _ = built
    wrong = DatasetSpec(**{**spec.__dict__, 'family_counts': {**spec.family_counts, 'alpha': 1}})
    with pytest.raises(ValueError, match='expected 1'):
        build(arguments(source, tmp_path / 'wrong'), wrong, FILES)
