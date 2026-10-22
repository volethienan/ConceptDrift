from __future__ import annotations

import dataclasses

import pytest
import torch

from dasp_nids import dasp_v2
from dasp_nids.config import load_config
from dasp_nids.runner import StreamingExperiment, _label_blind_uniform_query_indices


def config(tmp_path, realign='attributed', name=None, accept=True):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), name or realign
    cfg.training.model_seed = 91
    cfg.dasp.baseline = 'modular_dasp_v2'
    cfg.baselines.update_label_budget = 32
    cfg.dasp_v2.attribution = 'pseudo'
    cfg.dasp_v2.routing = 'attributed'
    cfg.dasp_v2.attribution_min_class = 8
    cfg.dasp_v2.quarantine_samples = 64
    cfg.dasp_v2.realign = realign
    # Make the tiny synthetic stream exercise every branch.
    cfg.dasp_v2.realign_z = 0.5
    cfg.dasp_v2.realign_persistence = 1
    cfg.dasp_v2.realign_accept = accept
    cfg.dasp_v2.realign_class_check = accept
    return cfg


def test_estimate_map_inverts_a_group_affine_and_collapses():
    generator = torch.Generator().manual_seed(0)
    reference = torch.randn(4000, 6, generator=generator)
    labels = (torch.rand(4000, generator=generator) < 0.3).long()
    reference[labels == 1] += 2.0
    moments = {label: dasp_v2._robust_moments(reference[labels == label]) for label in (0, 1)}
    columns = torch.tensor([1, 3, 4])
    shifted = reference.clone()
    shifted[:, columns] = 3.0 * shifted[:, columns] - 5.0
    shifted[:, 4] = 7.0  # an exporter fault zeroes the feature to a constant
    mapped = dasp_v2.apply_maps(shifted, {'g': dasp_v2.estimate_map(shifted, labels, moments, columns, 32)})
    assert torch.allclose(mapped[:, [1, 3]], reference[:, [1, 3]], atol=0.15)
    assert torch.allclose(mapped[:, 4], reference[:, 4].median().expand(4000), atol=0.6)
    untouched = torch.tensor([0, 2, 5])
    assert torch.equal(mapped[:, untouched], shifted[:, untouched])
    assert dasp_v2.estimate_map(shifted, torch.full((4000,), 5), moments, columns, 32) is None


@pytest.mark.parametrize('realign', ['attributed', 'random', 'global'])
def test_realignment_runs_and_reports(tmp_path, realign):
    result = StreamingExperiment(config(tmp_path, realign)).run()
    evaluated = result.windows[~result.windows.warmup.astype(bool)]
    assert 'realign_active' in evaluated
    assert set(evaluated.ground_truth_update_labels) == {32}
    if realign == 'global':
        assert (evaluated.realign_parameters > 0).all()


@pytest.mark.parametrize('method', ['modular_dasp_v2', 'experience_replay'])
def test_realigned_predictions_never_read_unqueried_labels(tmp_path, monkeypatch, method):
    seen = []
    original = dasp_v2.prediction_logits

    def spy(experiment, model, x, labels=None):
        logits, diagnostics = original(experiment, model, x, labels)
        seen.append(logits.argmax(dim=1).clone())
        return logits, diagnostics

    monkeypatch.setattr(dasp_v2, 'prediction_logits', spy)
    # Acceptance off, so maps are applied, validated and reverted on this tiny stream.
    clean_cfg = config(tmp_path, name='clean', accept=False)
    clean_cfg.dasp.baseline = method
    windows = StreamingExperiment(clean_cfg).run().windows
    assert windows.realign_accepted.fillna('').ne('').any()
    clean, seen[:] = list(seen), []

    loader = StreamingExperiment._load_stream

    def corrupted(self):
        stream = loader(self)
        cfg = self.config
        windows = list(stream.windows[:cfg.data.warmup_windows])
        for window in stream.windows[cfg.data.warmup_windows:]:
            query = _label_blind_uniform_query_indices(len(window.x), cfg.baselines.update_label_budget,
                                                       cfg.data.random_seed + window.index * 104729 + 17)
            keep = torch.zeros(len(window.y), dtype=torch.bool)
            keep[query] = True
            windows.append(dataclasses.replace(window, y=torch.where(keep, window.y, 1 - window.y)))
        return dataclasses.replace(stream, windows=windows)

    monkeypatch.setattr(StreamingExperiment, '_load_stream', corrupted)
    flipped_cfg = config(tmp_path, name='flipped', accept=False)
    flipped_cfg.dasp.baseline = method
    StreamingExperiment(flipped_cfg).run()
    assert len(seen) == len(clean) and all(torch.equal(a, b) for a, b in zip(clean, seen))


def test_rejected_maps_leave_the_window_unchanged(tmp_path):
    cfg = config(tmp_path)
    cfg.dasp_v2.realign_z = 1e9  # never triggers
    result = StreamingExperiment(cfg).run()
    evaluated = result.windows[~result.windows.warmup.astype(bool)]
    assert (evaluated.realign_parameters == 0).all()
    assert evaluated.realign_accepted.fillna('').eq('').all()


def test_class_check_tells_measurement_change_from_new_attack_family():
    generator = torch.Generator().manual_seed(1)
    reference = torch.randn(6000, 4, generator=generator)
    labels = (torch.rand(6000, generator=generator) < 0.4).long()
    reference[labels == 1] += 3.0
    moments = {label: dasp_v2._robust_moments(reference[labels == label]) for label in (0, 1)}
    columns = torch.tensor([0, 1])
    # Measurement change: both classes rescaled the same way.
    measured = reference.clone()
    measured[:, columns] = measured[:, columns] * 4.0 + 1.0
    mapping = dasp_v2.estimate_map(measured, labels, moments, columns, 32)
    changed = (mapping[1] != 1) | (mapping[2] != 0)
    ok, checked = dasp_v2.class_consistent(dasp_v2.apply_maps(measured, {'g': mapping}), labels, moments,
                                           columns, changed, 32, 0.5)
    assert ok and checked == 2
    # New attack family: only the attack class moves.
    novel = reference.clone()
    novel[labels == 1, :2] = novel[labels == 1, :2] * 4.0 - 6.0
    mapping = dasp_v2.estimate_map(novel, labels, moments, columns, 32)
    changed = (mapping[1] != 1) | (mapping[2] != 0)
    ok, checked = dasp_v2.class_consistent(dasp_v2.apply_maps(novel, {'g': mapping}), labels, moments,
                                           columns, changed, 32, 0.5)
    assert not ok and checked == 2


def test_train_on_gate_refuses_utility_routing(tmp_path):
    cfg = config(tmp_path, realign='none')
    cfg.dasp_v2.train_on_gate = True
    cfg.dasp_v2.routing = 'utility'
    cfg.dasp_v2.gate_policy = 'off'  # every window repairs, so the router runs
    with pytest.raises(ValueError, match='utility routing'):
        StreamingExperiment(cfg).run()


@pytest.mark.parametrize('delay', [0, 1, 3])
def test_labels_of_a_window_arrive_only_after_delay_more_predictions(tmp_path, monkeypatch, delay):
    events = []
    evaluate, update = StreamingExperiment._evaluate, StreamingExperiment._update

    def spy_evaluate(self, model, window, n_classes):
        events.append(('predict', window.index))
        return evaluate(self, model, window, n_classes)

    def spy_update(self, model, memory, item, row):
        events.append(('labels', item.window.index))
        return update(self, model, memory, item, row)

    monkeypatch.setattr(StreamingExperiment, '_evaluate', spy_evaluate)
    monkeypatch.setattr(StreamingExperiment, '_update', spy_update)
    cfg = config(tmp_path, name=f'delay{delay}')
    cfg.training.label_delay = delay
    StreamingExperiment(cfg).run()
    order = {event: position for position, event in enumerate(events)}
    labelled = [index for kind, index in events if kind == 'labels']
    assert labelled == sorted(labelled) and labelled
    for index in labelled:
        later = ('predict', index + delay)
        if later in order:  # labels of `index` come after predicting index + delay ...
            assert order[('labels', index)] > order[later]
        if ('predict', index + delay + 1) in order:  # ... and before the next prediction
            assert order[('labels', index)] < order[('predict', index + delay + 1)]


def test_experience_replay_can_host_routed_recalibration(tmp_path):
    cfg = config(tmp_path, name='er_realign', accept=False)
    cfg.dasp.baseline = 'experience_replay'
    result = StreamingExperiment(cfg).run()
    evaluated = result.windows[~result.windows.warmup.astype(bool)]
    assert evaluated.realign_accepted.fillna('').ne('').any()
    assert 'realign_loss_with' in evaluated or evaluated.realign_active.notna().any()
    assert set(evaluated.ground_truth_update_labels) == {32}
