"""Same-window fast lane in the runner (verified_propagation)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from dasp_nids.config import load_config
from dasp_nids.runner import StreamingExperiment

KINDS = ['uniform', 'representative', 'purity', 'uniform_knn', 'hierarchical']


def config(tmp_path, name, kind, fast=8, method='experience_replay'):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), name
    cfg.training.model_seed = 91
    cfg.dasp.baseline = method
    cfg.baselines.update_label_budget = 32
    cfg.baselines.fast_label_budget = fast
    cfg.baselines.fast_lane = kind
    cfg.baselines.fast_lane_clusters = 4
    cfg.baselines.fast_lane_test_labels = 2
    cfg.baselines.fast_lane_epsilon = 0.6
    cfg.baselines.fast_lane_delta = 0.25
    return cfg


def recorder(monkeypatch):
    fast, bulk, predictions = {}, {}, {}
    lane, query = StreamingExperiment._fast_lane, StreamingExperiment._query_indices

    def recording_lane(self, model, memory, window, metrics, n_classes):
        out = lane(self, model, memory, window, metrics, n_classes)
        run = self.config.output.run_name
        if window.index in self._fast_queries:
            fast.setdefault(run, {})[window.index] = self._fast_queries[window.index].copy()
        predictions.setdefault(run, {})[window.index] = self._last_predictions.copy()
        return out

    def recording_query(self, window):
        chosen = query(self, window)
        bulk.setdefault(self.config.output.run_name, {})[window.index] = chosen.numpy().copy()
        return chosen

    monkeypatch.setattr(StreamingExperiment, '_fast_lane', recording_lane)
    monkeypatch.setattr(StreamingExperiment, '_query_indices', recording_query)
    return fast, bulk, predictions


@pytest.mark.parametrize('kind', KINDS)
def test_fast_lane_reads_only_its_queries_and_splits_the_budget(tmp_path, kind, monkeypatch):
    """Flip every label outside the fast and bulk queries of a clean run: the
    window predictions (fused with the fast-lane alarms) must not change."""

    fast, bulk, predictions = recorder(monkeypatch)
    windows = StreamingExperiment(config(tmp_path, 'clean', kind)).run().windows
    evaluated = windows[~windows.warmup.astype(bool)]
    assert set(evaluated.fast_labels) == {8}
    assert set(evaluated.ground_truth_update_labels) == {24}
    for index, chosen in fast['clean'].items():
        assert len(set(chosen)) == 8
        if index in bulk['clean']:
            assert not set(chosen) & set(bulk['clean'][index])

    original = StreamingExperiment._load_stream

    def corrupted(self):
        stream = original(self)
        out = []
        for window in stream.windows:
            if window.index not in fast['clean']:
                out.append(window)
                continue
            keep = torch.zeros(len(window.y), dtype=torch.bool)
            keep[torch.from_numpy(fast['clean'][window.index])] = True
            if window.index in bulk['clean']:
                keep[torch.from_numpy(bulk['clean'][window.index])] = True
            out.append(dataclasses.replace(window, y=torch.where(keep, window.y, 1 - window.y)))
        return dataclasses.replace(stream, windows=out)

    monkeypatch.setattr(StreamingExperiment, '_load_stream', corrupted)
    StreamingExperiment(config(tmp_path, 'flipped', kind)).run()
    assert predictions['clean'].keys() == predictions['flipped'].keys()
    for index in predictions['clean']:
        np.testing.assert_array_equal(predictions['clean'][index], predictions['flipped'][index])


def test_zero_fast_budget_keeps_the_historical_run(tmp_path):
    base = StreamingExperiment(config(tmp_path, 'base', 'purity', fast=0)).run().windows
    assert 'fast_labels' not in base.columns
    assert set(base[~base.warmup.astype(bool)].ground_truth_update_labels) == {32}


def test_fast_lane_is_refused_where_unsupported(tmp_path):
    with pytest.raises(ValueError, match='fast_label_budget'):
        StreamingExperiment(config(tmp_path, 'x', 'purity', method='modular_dasp_v2')).run()


def test_iwms_replay_runs_and_is_marked(tmp_path):
    cfg = config(tmp_path, 'iwms', 'uniform', fast=0)
    cfg.baselines.replay_sampling = 'iwms'
    windows = StreamingExperiment(cfg).run().windows
    updated = windows[~windows.warmup.astype(bool) & windows.get('replay_sampling').notna()]
    assert len(updated) and set(updated.replay_sampling) == {'iwms'}


def test_iwms_picks_the_nearest_memory_sample():
    from dasp_nids.memory import ReservoirMemory
    from dasp_nids.model import OnlineMLP
    from dasp_nids.runner import _iwms_replay

    torch.manual_seed(0)
    model = OnlineMLP(4, 2, hidden_dims=(8,))
    memory = ReservoirMemory(10, 0)
    memory.add(torch.eye(4), torch.tensor([0, 1, 0, 1]))
    x, y = _iwms_replay(model, memory, torch.eye(4)[[2, 2, 3]], 3, torch.device('cpu'), 1)
    assert len(x) == 3 and set(y.tolist()) <= {0, 1}
    _, reps = model(torch.eye(4), return_representations=True)
    last = reps[list(reps)[-1]]
    target = torch.nn.functional.normalize(last, dim=1)
    expected = (target[[2]] @ target.T).argmax().item()
    assert any(torch.equal(row, torch.eye(4)[expected]) for row in x)


def test_verify_stage_plan():
    from tools.run_dasp_v2 import jobs_for, parser

    args = parser().parse_args(['--stage', 'verify', '--selection', 'results_snapshot/dev/dev_selection.json'])
    jobs = jobs_for(args)
    assert len(jobs) == 702 and {j['dataset'] for j in jobs} == {'nf-unsw-nb15-v3', 'nf-cse-cic-ids2018-v3', 'nf-ton-iot-v3'}
    held = jobs_for(parser().parse_args(['--stage', 'verify', '--heldout', '--selection',
                                         'results_snapshot/dev/dev_selection.json']))
    assert {j['dataset'] for j in held} == {'cic-ids2017'} and len(held) == 234
    budgets = {j['overrides'].get('baselines.update_label_budget') for j in jobs}
    assert budgets == {128}
