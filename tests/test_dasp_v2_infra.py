from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from dasp_nids import runner as runner_module
from dasp_nids.baselines import (
    ErrorRateDriftDetector,
    RecalibratingErrorRateDetector,
    estimate_empirical_fisher,
)
from dasp_nids.config import load_config
from dasp_nids.model import OnlineMLP
from dasp_nids.runner import StreamingExperiment, _estimate_fisher
from tools.analyze_original_review import load_completed
from tools.run_original_review import jobs_for, parse_shard, parser, run, shard_jobs


def config(tmp_path, method, name=None):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), name or method
    cfg.training.model_seed = 91
    cfg.dasp.baseline = method
    cfg.baselines.update_label_budget = 32
    return cfg


def test_empirical_fisher_is_mean_of_per_example_squared_gradients():
    torch.manual_seed(3)
    model = OnlineMLP(6, 2, [5])
    x, y = torch.randn(9, 6), torch.arange(9) % 2
    fisher = estimate_empirical_fisher(model, (x, y), torch.device('cpu'))
    for name, parameter in model.named_parameters():
        expected = torch.zeros_like(parameter)
        for index in range(len(x)):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[index:index + 1]), y[index:index + 1]).backward()
            expected += parameter.grad.square() / len(x)
        assert torch.allclose(fisher[name], expected, atol=1e-7)


def test_empirical_fisher_survives_gradient_cancellation_that_zeroes_legacy_estimate():
    model = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    # Opposite labels at the same input: the batch-mean gradient cancels.
    x, y = torch.ones(2, 1), torch.tensor([0, 1])
    legacy = _estimate_fisher(model, (x, y), torch.device('cpu'))
    corrected = estimate_empirical_fisher(model, (x, y), torch.device('cpu'))
    assert float(legacy['weight'].sum()) == 0.0
    assert float(corrected['weight'].sum()) == pytest.approx(0.5)


def test_recalibrating_detector_alarms_again_after_recovery_unlike_legacy():
    errors = [0.05, 0.06, 0.05]
    stream = [0.40, 0.40, 0.40, 0.05, 0.05, 0.05, 0.30, 0.30, 0.30]
    legacy = ErrorRateDriftDetector.from_warmup(errors, 2.0, 3, 0.01)
    fixed = RecalibratingErrorRateDetector.from_warmup(errors, 2.0, 3, 0.01, 3)
    legacy_drifts = [legacy.update(value)[1] for value in stream]
    fixed_drifts, severities = zip(*[fixed.update(value)[1:] for value in stream])
    assert legacy_drifts == [False, False, True, False, False, False, False, False, False]
    assert list(fixed_drifts) == [False, False, True, False, False, False, False, False, True]
    assert severities[2] > 0.3


def test_cdda_v2_decides_only_from_queried_labels(tmp_path, monkeypatch):
    seen = []
    original = runner_module._labelled_error_rate

    def spy(model, x, y, device):
        seen.append((x.clone(), y.clone()))
        return original(model, x, y, device)

    monkeypatch.setattr(runner_module, '_labelled_error_rate', spy)
    cfg = config(tmp_path, 'cdda_md_v2')
    result = StreamingExperiment(cfg).run()
    evaluated = result.windows[~result.windows.warmup.astype(bool)]
    assert len(seen) == cfg.data.warmup_windows + len(evaluated)
    assert all(len(y) == cfg.baselines.update_label_budget for _, y in seen)
    assert set(evaluated.cdda_error_label_source) == {'queried_update_labels'}
    assert set(evaluated.ground_truth_update_labels) == {cfg.baselines.update_label_budget}


def test_ewc_v2_runs_under_the_budget_contract(tmp_path):
    ewc = config(tmp_path, 'online_ewc_v2')
    result = StreamingExperiment(ewc).run()
    evaluated = result.windows[~result.windows.warmup.astype(bool)]
    assert set(evaluated.ground_truth_update_labels) == {ewc.baselines.update_label_budget}
    assert set(evaluated.selected_candidate) == {'online_ewc_v2'}
    refreshed = evaluated.ewc_anchor_refreshed.astype(bool).tolist()
    interval = ewc.dasp.ewc_anchor_interval
    assert refreshed == [(index + 1) % interval == 0 for index in range(len(refreshed))]


def test_ewc_v2_penalty_restrains_updates_where_legacy_penalty_vanished(tmp_path):
    cfg = config(tmp_path, 'online_ewc_v2')
    cfg.dasp.ewc_lambda = 1000.0
    # Several steps away from the anchor, where the penalty has a gradient.
    cfg.training.batch_size, cfg.training.epochs_per_window = 8, 4
    torch.manual_seed(5)
    base = OnlineMLP(16, 2, [16])
    old_x, old_y = torch.randn(64, 16), torch.arange(64) % 2
    new_x, new_y = torch.randn(64, 16) + 2.0, 1 - torch.arange(64) % 2
    reference = {name: value.detach().clone() for name, value in base.named_parameters()}
    device = torch.device('cpu')
    moves = {}
    for label, estimator in (('legacy', _estimate_fisher), ('empirical', estimate_empirical_fisher)):
        model = OnlineMLP(16, 2, [16])
        model.load_state_dict(base.state_dict())
        fisher = estimator(model, (old_x, old_y), device)
        runner_module._train_online_ewc(model, new_x, new_y, reference, fisher, cfg, device, 1)
        moves[label] = sum(float((p.detach() - reference[n]).square().sum())
                           for n, p in model.named_parameters())
    assert moves['empirical'] < 0.5 * moves['legacy']


def test_shard_validation():
    assert parse_shard(None) is None
    assert parse_shard('2/8') == (2, 8)
    for value in ('0/4', '5/4', '1-4', 'a/b'):
        with pytest.raises(ValueError):
            parse_shard(value)


@pytest.mark.parametrize('stage', ['primary', 'routing', 'components'])
def test_shards_partition_whole_cells_and_keep_native_first(tmp_path, stage):
    args = parser().parse_args(['--stage', stage, '--output-root', str(tmp_path)])
    jobs = jobs_for(args)
    parts = [shard_jobs(jobs, (index, 7)) for index in range(1, 8)]
    flattened = [json.dumps(job, sort_keys=True) for part in parts for job in part]
    assert sorted(flattened) == sorted(json.dumps(job, sort_keys=True) for job in jobs)
    cells = [{(j['dataset'], j['scenario'], j['stream_seed'], j['model_seed']) for j in part} for part in parts]
    assert all(not a & b for i, a in enumerate(cells) for b in cells[i + 1:])


def test_sharded_execution_merges_for_analysis(tmp_path):
    from tools.run_original_review import _smoke_stream
    prepared = tmp_path / 'prepared'
    for seed in (2001, 2002):
        _smoke_stream(prepared / 'nf-ton-iot-v3' / 'S0' / f'stream_seed{seed}.npz')
    base = ['--stage', 'gate', '--datasets', 'nf-ton-iot-v3', '--scenarios', 'S0',
            '--stream-seeds', '2001', '2002', '--model-seeds', '91',
            '--prepared-root', str(prepared), '--output-root', str(tmp_path / 'out'), '--execute']
    for shard in ('1/2', '2/2'):
        run(parser().parse_args(base + ['--shard', shard]))
    records = load_completed(tmp_path / 'out', 'gate')
    assert len(records) == 4
    plan = json.loads((tmp_path / 'out' / 'gate_plan.json').read_text(encoding='utf-8'))
    assert plan['planned_runs'] == 4
    (tmp_path / 'out' / 'gate_completed.json').write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='both'):
        load_completed(tmp_path / 'out', 'gate')
