from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from dasp_nids import sota_adapters
from dasp_nids.config import load_config
from dasp_nids.runner import StreamingExperiment, _label_blind_uniform_query_indices
from tools.analyze_original_review import load_completed
from tools.run_dasp_v2 import ALL_GRIDS, GRIDS, SOTA_GRIDS, config_id, jobs_for, parser, run

ADAPTERS = sorted(sota_adapters.ADAPTERS)


def config(tmp_path, method, name=None):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), name or method
    cfg.training.model_seed = 91
    cfg.dasp.baseline = method
    cfg.baselines.update_label_budget = 32
    return cfg


def test_every_adapter_gets_exactly_24_distinct_configurations():
    assert set(SOTA_GRIDS) == set(ADAPTERS)
    for method, grid in SOTA_GRIDS.items():
        assert len(grid) == 24
        assert len({config_id(method, overrides) for overrides in grid}) == 24


@pytest.mark.parametrize('method', ADAPTERS)
def test_adapter_reports_the_label_contract(tmp_path, method):
    cfg = config(tmp_path, method)
    result = StreamingExperiment(cfg).run()
    evaluated = result.windows[~result.windows.warmup.astype(bool)]
    assert set(evaluated.ground_truth_update_labels) == {cfg.baselines.update_label_budget}
    assert set(evaluated.label_query_policy) == {'uniform_label_blind'}
    summary = result.summary
    assert summary['total_ground_truth_update_labels'] == cfg.baselines.update_label_budget * len(evaluated)
    assert summary['warmup_labels'] == cfg.data.warmup_windows * cfg.data.window_size
    assert summary['adapter_version'] == sota_adapters.ADAPTER_VERSION


@pytest.mark.parametrize('method', ADAPTERS)
def test_predictions_never_depend_on_unqueried_labels(tmp_path, method, monkeypatch):
    """Flip every post-warm-up label outside the 128-query set: the adapter's
    predictions must not change (only the scores computed from them may)."""

    original = StreamingExperiment._load_stream

    def corrupted(self):
        stream = original(self)
        cfg = self.config
        windows = list(stream.windows[:cfg.data.warmup_windows])
        for window in stream.windows[cfg.data.warmup_windows:]:
            query = _label_blind_uniform_query_indices(
                len(window.x), cfg.baselines.update_label_budget,
                cfg.data.random_seed + window.index * 104729 + 17)
            keep = torch.zeros(len(window.y), dtype=torch.bool)
            keep[query] = True
            y = torch.where(keep, window.y, 1 - window.y)
            windows.append(dataclasses.replace(window, y=y))
        return dataclasses.replace(stream, windows=windows)

    clean = StreamingExperiment(config(tmp_path, method, 'clean')).run().windows
    monkeypatch.setattr(StreamingExperiment, '_load_stream', corrupted)
    flipped = StreamingExperiment(config(tmp_path, method, 'flipped')).run().windows
    assert list(clean.prediction_digest) == list(flipped.prediction_digest)
    evaluated = ~clean.warmup.astype(bool)
    assert not np.allclose(clean.macro_f1[evaluated], flipped.macro_f1[evaluated])


def test_update_interface_receives_only_queried_labels(tmp_path, monkeypatch):
    seen = []
    original = sota_adapters.Adapter.update

    def spy(self, x, query, y_query):
        seen.append((len(x), len(query), len(y_query)))
        return original(self, x, query, y_query)

    monkeypatch.setattr(sota_adapters.Adapter, 'update', spy)
    cfg = config(tmp_path, 'soul_adapted')
    StreamingExperiment(cfg).run()
    assert seen and all(n == cfg.data.window_size and q == y == cfg.baselines.update_label_budget
                        for n, q, y in seen)


def test_macro_f1_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    scores = rng.random(300) + y * 0.5
    threshold = sota_adapters.best_f1_threshold(scores, y)
    from sklearn.metrics import f1_score
    best = max(f1_score(y, (scores >= t).astype(int), average='macro') for t in np.unique(scores))
    assert f1_score(y, (scores >= threshold).astype(int), average='macro') == pytest.approx(best)
    assert sota_adapters.best_f1_threshold(scores, np.zeros(300)) is None
    assert sota_adapters.best_f1_threshold(np.ones(300), y) is None


def test_gpm_projection_removes_the_protected_subspace():
    rng = np.random.default_rng(1)
    activations = [rng.normal(size=(8, 3)) @ rng.normal(size=(3, 40))]  # rank 3
    bases = sota_adapters.update_gpm([], activations, [0.99])
    assert 1 <= bases[0].shape[1] <= 3
    gradient = torch.from_numpy(rng.normal(size=(5, 8)))
    projector = torch.from_numpy(bases[0] @ bases[0].T)
    projected = gradient - gradient @ projector
    assert torch.allclose(projected @ torch.from_numpy(bases[0]), torch.zeros(5, bases[0].shape[1], dtype=torch.float64), atol=1e-10)
    grown = sota_adapters.update_gpm(bases, [rng.normal(size=(8, 40))], [0.99])
    assert grown[0].shape[1] >= bases[0].shape[1] and grown[0].shape[1] <= 8


def test_stores_respect_their_budgets():
    store = sota_adapters.ClassBalancedStore(10, 2, 0)
    store.add(np.ones((100, 3)), np.r_[np.zeros(95), np.ones(5)].astype(int))
    x, y = store.arrays()
    assert len(store) == 10 and int((y == 0).sum()) == 5 and int((y == 1).sum()) == 5
    reservoir = sota_adapters.Reservoir(20, 0)
    reservoir.add(np.arange(200)[:, None].astype(float), np.zeros(200, dtype=int))
    assert len(reservoir) == 20 and reservoir.arrays()[0].max() > 20  # later samples get in


def test_crc_loss_needs_a_normal_anchor():
    features = torch.randn(6, 4, requires_grad=True)
    assert sota_adapters.crc_loss(features, torch.ones(6, dtype=torch.long), 0.02) is None
    loss = sota_adapters.crc_loss(features, torch.tensor([0, 0, 1, 1, 0, 1]), 0.02)
    loss.backward()
    assert torch.isfinite(loss) and features.grad is not None


def test_knee_picks_the_elbow_and_falls_back_to_the_largest_option():
    assert sota_adapters.knee_cluster_count([1, 2, 3, 4, 5], [100, 20, 15, 12, 10]) == 2
    assert sota_adapters.knee_cluster_count([1, 2], [5, 4]) == 2


def test_sota_dev_suite_is_separate_and_development_only(tmp_path):
    core = jobs_for(parser().parse_args(['--stage', 'dev', '--suite', 'core', '--output-root', str(tmp_path)]))
    assert {job['method'] for job in core} == set(GRIDS)
    sota = jobs_for(parser().parse_args(['--stage', 'dev', '--suite', 'sota', '--output-root', str(tmp_path)]))
    assert {job['method'] for job in sota} == set(SOTA_GRIDS)
    assert len(sota) == 4 * 24 * 2 * (10 + 8 + 3) * 3
    assert {job['stream_seed'] for job in sota} == {2001, 2002, 2003}
    with pytest.raises(ValueError, match='outside'):
        jobs_for(parser().parse_args(['--stage', 'dev', '--suite', 'core', '--methods', 'soul_adapted',
                                      '--output-root', str(tmp_path)]))


def test_smoke_suite_runs_every_adapter_under_one_budget(tmp_path):
    run(parser().parse_args(['--stage', 'smoke', '--suite', 'all', '--output-root', str(tmp_path), '--execute']))
    records = load_completed(tmp_path, 'smoke')
    assert {record['method'] for record in records} == set(ALL_GRIDS)
    labels = {record['method']: record['summary']['total_ground_truth_update_labels'] for record in records}
    assert len(set(labels.values())) == 1, labels


def test_final_plan_merges_core_and_sota_selections(tmp_path):
    def write(name, grids):
        path = tmp_path / name
        path.write_text(json.dumps({'selection': {m: {'overrides': g[5]} for m, g in grids.items()}}),
                        encoding='utf-8')
        return str(path)

    core, sota = write('core.json', GRIDS), write('sota.json', SOTA_GRIDS)
    args = parser().parse_args(['--stage', 'final', '--output-root', str(tmp_path), '--selection', core, sota])
    jobs = jobs_for(args)
    by_condition = {job['condition']: job for job in jobs}
    for method, grid in SOTA_GRIDS.items():
        assert by_condition[method]['method'] == method and by_condition[method]['overrides'] == grid[5]
    with pytest.raises(ValueError, match='lacks'):
        jobs_for(parser().parse_args(['--stage', 'final', '--output-root', str(tmp_path), '--selection', core]))
    with pytest.raises(ValueError, match='twice'):
        jobs_for(parser().parse_args(['--stage', 'final', '--output-root', str(tmp_path),
                                      '--selection', core, sota, sota]))
