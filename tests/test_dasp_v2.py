from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from dasp_nids import dasp_v2
from dasp_nids.config import load_config
from dasp_nids.data import StreamWindow
from dasp_nids.drift import DriftDecision
from dasp_nids.memory import ReservoirMemory
from dasp_nids.model import ModularOnlineMLP
from dasp_nids.runner import PendingUpdate, StreamingExperiment


def config(tmp_path, name='v2'):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), name
    cfg.training.model_seed = 91
    cfg.dasp.baseline = 'modular_dasp_v2'
    cfg.baselines.update_label_budget = 32
    cfg.dasp_v2.gate_min_support = 4
    return cfg


def test_wilson_lower_bound_matches_reference_values():
    assert dasp_v2.wilson_lower_bound(20, 20, 1.96) == pytest.approx(0.8389, abs=1e-4)
    assert dasp_v2.wilson_lower_bound(19, 20, 1.96) == pytest.approx(0.7639, abs=1e-4)
    assert math.isnan(dasp_v2.wilson_lower_bound(0, 0, 1.96))


def test_gate_never_reads_missing_or_tiny_support_as_evidence():
    attacks_only = torch.ones(32, dtype=torch.long)
    result = dasp_v2.gate_assessment(attacks_only, attacks_only, 2, 'wilson', 20, .8, 1.96)
    assert result['assessed_classes'] == [1] and result['state'] == 'certified'
    y = torch.tensor([0, 0, 0] + [1] * 29)
    prediction = torch.tensor([1, 1, 1] + [1] * 29)  # all three benign wrong
    result = dasp_v2.gate_assessment(y, prediction, 2, 'wilson', 20, .8, 1.96)
    assert result['assessed_classes'] == [1] and not result['degraded']
    nothing = dasp_v2.gate_assessment(y[:10], prediction[:10], 2, 'wilson', 20, .8, 1.96)
    assert nothing['state'] == 'insufficient_support' and not nothing['degraded']


def test_wilson_gate_is_stricter_than_point_estimate():
    y = torch.ones(20, dtype=torch.long)
    prediction = y.clone()
    prediction[:3] = 0  # recall 0.85: point passes, Wilson cannot certify 0.80
    assert not dasp_v2.gate_assessment(y, prediction, 2, 'point', 20, .8, 1.96)['degraded']
    assert dasp_v2.gate_assessment(y, prediction, 2, 'wilson', 20, .8, 1.96)['degraded']


def test_logit_adjustment_uses_arrived_label_prior():
    memory = ReservoirMemory(64, 1)
    memory.add(torch.randn(30, 4), torch.tensor([0] * 27 + [1] * 3))
    adjustment = dasp_v2.class_prior_adjustment(memory, 2, 1.0)
    assert torch.allclose(adjustment.exp(), torch.tensor([28 / 32, 4 / 32]))
    assert dasp_v2.class_prior_adjustment(memory, 2, 0.0) is None


def test_linear_head_option_keeps_parameters_in_modules_and_default_is_unchanged():
    torch.manual_seed(1)
    legacy = ModularOnlineMLP(49, 2, [128, 64], 0.0, 4, 64)
    torch.manual_seed(1)
    default = ModularOnlineMLP(49, 2, [128, 64], 0.0, 4, 64, None)
    assert all(torch.equal(a, b) for a, b in zip(legacy.state_dict().values(), default.state_dict().values()))
    linear = ModularOnlineMLP(49, 2, [128, 64], 0.0, 4, 64, 0)
    head = sum(p.numel() for p in linear.classifier.parameters())
    assert head == 4 * 64 * 2 + 2
    assert head / sum(p.numel() for p in linear.parameters()) < 0.2


def make_update(tmp_path, labels, triggered=True, routing='utility', gate_policy='wilson'):
    cfg = config(tmp_path)
    cfg.dasp_v2.routing = routing
    cfg.dasp_v2.gate_policy = gate_policy
    torch.manual_seed(91)
    base = ModularOnlineMLP(16, 2, [16], n_modules=4, module_dim=8)
    x = torch.randn(96, 16)
    decision = DriftDecision({f'module_{i}': float(i == 2) for i in range(1, 5)}, {}, {}, 1., triggered, False)
    model = copy.deepcopy(base)
    memory = ReservoirMemory(128, 91)
    memory.add(x[:32], torch.arange(32) % 2)
    experiment = StreamingExperiment(cfg)
    row = {}
    dasp_v2.update(experiment, model, memory, PendingUpdate(StreamWindow(x, labels, 3, 'post'), decision, False), row)
    return base, model, row


def test_update_reads_only_queried_labels(tmp_path):
    y = torch.arange(96) % 2
    _, first, row = make_update(tmp_path, y)
    queried = json.loads(row['query_indices'])
    assert row['labels_exposed_total'] == len(queried) == 32
    assert row['new_training_labels_used'] + row['labels_used_for_gating'] == 32
    changed = y.clone()
    outside = torch.tensor(sorted(set(range(96)) - set(queried)))
    changed[outside] = 1 - changed[outside]
    _, second, _ = make_update(tmp_path, changed)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


def test_utility_routing_freezes_unselected_modules_and_charges_probes(tmp_path):
    base, model, row = make_update(tmp_path, torch.arange(96) % 2)
    chosen = set(filter(None, row['localized_repair_modules'].split('|')))
    assert len(chosen) <= 2
    for name, block in zip(model.layer_names, model.modules_list):
        reference = dict(zip(base.layer_names, base.modules_list))[name]
        same = all(torch.equal(a, b) for a, b in zip(block.parameters(), reference.parameters()))
        assert same == (name not in chosen)
    assert row['routing_probe_gradient_evaluations'] > 0
    phases = json.loads(row['gradient_phases'])
    assert set(phases) >= {'routing_probe', 'training'}
    assert set(json.loads(row['routing_gains'])) == set(model.layer_names)


def test_background_without_drift_updates_head_only(tmp_path):
    base, model, row = make_update(tmp_path, torch.arange(96) % 2, triggered=False, gate_policy='off')
    assert row['selected_candidate'] == 'background_classifier'
    assert all(torch.equal(a, b) for a, b in zip(model.modules_list.parameters(), base.modules_list.parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(model.classifier.parameters(), base.classifier.parameters()))


def test_end_to_end_run_respects_budget_and_reports_v2_summary(tmp_path):
    cfg = config(tmp_path)
    cfg.model.modular_head_dim = 0
    result = StreamingExperiment(cfg).run()
    evaluated = result.windows[~result.windows.warmup.astype(bool)]
    assert set(evaluated.ground_truth_update_labels) == {32}
    assert result.summary['dasp_v2_version'] == dasp_v2.VERSION
    assert result.summary['primary_comparison_scope'] == 'matched_total_query_budget'
    assert sum(result.summary['gate_state_counts'].values()) == len(evaluated)
    assert (tmp_path / 'v2' / 'provenance.json').is_file()
