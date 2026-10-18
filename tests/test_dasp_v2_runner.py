from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tools.analyze_dasp_v2 import (cell_differences, compare_family, construct_eligible_cells,
                                   hierarchical_bootstrap, holm, select_configurations)
from tools.analyze_original_review import load_completed
from tools.run_dasp_v2 import GRIDS, apply_overrides, config_id, configure, jobs_for, parser, run


def test_every_method_gets_exactly_24_distinct_configurations():
    assert set(GRIDS) == {'modular_dasp_v2', 'online_mlp', 'experience_replay', 'agem',
                          'online_ewc_v2', 'ssf_budget', 'cdda_md_v2'}
    for method, grid in GRIDS.items():
        assert len(grid) == 24
        assert len({config_id(method, overrides) for overrides in grid}) == 24


def test_dev_plan_covers_only_development_data(tmp_path):
    args = parser().parse_args(['--stage', 'dev', '--output-root', str(tmp_path)])
    jobs = jobs_for(args)
    assert len(jobs) == 7 * 24 * 2 * 10 * 3
    assert {job['stream_seed'] for job in jobs} == {2001, 2002, 2003}
    assert {job['dataset'] for job in jobs} == {'nf-bot-iot-v3', 'nf-unsw-nb15-v3'}
    for bad in (['--stream-seeds', '3001'], ['--datasets', 'cic-ids2017']):
        with pytest.raises(ValueError):
            jobs_for(parser().parse_args(['--stage', 'dev', '--output-root', str(tmp_path), *bad]))


def test_overrides_reach_the_config_and_unknown_fields_fail(tmp_path):
    args = parser().parse_args(['--stage', 'smoke', '--output-root', str(tmp_path)])
    job = next(j for j in jobs_for(args) if j['method'] == 'modular_dasp_v2')
    cfg = configure(args, job)
    assert cfg.model.modular_head_dim == job['overrides']['model.modular_head_dim']
    assert cfg.baselines.update_label_budget == 128 and not cfg.original_review.enabled
    with pytest.raises(ValueError, match='unknown config field'):
        apply_overrides(cfg, {'dasp_v2.not_a_field': 1})


def test_smoke_runs_every_method_and_resume_verifies_receipts(tmp_path):
    base = ['--stage', 'smoke', '--output-root', str(tmp_path), '--execute']
    run(parser().parse_args(base))
    records = load_completed(tmp_path, 'smoke')
    assert {record['method'] for record in records} == set(GRIDS)
    labels = {record['method']: record['summary']['total_ground_truth_update_labels'] for record in records}
    assert len(set(labels.values())) == 1, labels
    with pytest.raises(FileExistsError):
        run(parser().parse_args(base))
    run(parser().parse_args(base + ['--resume']))
    summary = next(tmp_path.rglob('summary.json'))
    summary.write_text(summary.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='hashes'):
        run(parser().parse_args(base + ['--resume']))


def frame_for(values):
    rows = []
    for (condition, method, dataset, scenario), score in values.items():
        rows.append(dict(dataset=dataset, scenario=scenario, stream_seed=1, model_seed=91, method=method,
                         condition=condition, overrides=json.dumps({'k': condition}),
                         mean_macro_f1=score, gradient_cost=float(len(condition))))
    return pd.DataFrame(rows)


def test_selection_breaks_ties_by_cost_and_requires_complete_configurations():
    values = {}
    for dataset in ('a', 'b'):
        values[('cheap', 'm', dataset, 'S0')] = 0.900
        values[('expensive', 'm', dataset, 'S0')] = 0.901
        values[('worse', 'm', dataset, 'S0')] = 0.850
    chosen = select_configurations(frame_for(values))['m']
    assert chosen['condition'] == 'cheap' and chosen['tied_configurations'] == 2
    del values[('worse', 'm', 'b', 'S0')]
    with pytest.raises(ValueError, match='incomplete'):
        select_configurations(frame_for(values))


def test_hierarchical_bootstrap_resamples_datasets_before_scenarios():
    constant = pd.DataFrame({'dataset': ['a'] * 3 + ['b'] * 3, 'scenario': list('xyzxyz'),
                             'difference': [0.01] * 6})
    assert np.allclose(hierarchical_bootstrap(constant, 200), 0.01)
    split = constant.assign(difference=[0.05] * 3 + [-0.05] * 3)
    means = hierarchical_bootstrap(split, 2000)
    # Dataset-level resampling puts mass at both all-a and all-b draws.
    assert means.min() == pytest.approx(-0.05) and means.max() == pytest.approx(0.05)


def test_family_comparison_holm_and_construct_subset(tmp_path):
    values = {}
    for dataset in ('d1', 'd2', 'd3'):
        for index, scenario in enumerate(['S0', 'S1-Sudden', 'S2-Gradual', 'S3-Incremental']):
            values[('ref', 'r', dataset, scenario)] = 0.80
            values[('better', 'm', dataset, scenario)] = 0.82 + index * 0.001
            values[('same', 'n', dataset, scenario)] = 0.80
    frame = frame_for(values)
    report = tmp_path / 'validation_d1.json'
    report.write_text(json.dumps({'construct_summary': [
        {'scenario': 'S0', 'claim_eligible': True}, {'scenario': 'S1-Sudden', 'claim_eligible': True},
        {'scenario': 'S2-Gradual', 'claim_eligible': False}]}), encoding='utf-8')
    eligible = construct_eligible_cells(tmp_path)
    assert eligible == {('d1', 'S1-Sudden')}
    results = compare_family(frame, 'ref', ['better', 'same'], eligible, replicates=500)
    better = results[0]['all_cells']
    assert better['cells'] == 12 and better['ci95_low'] > 0 and better['positive_datasets'] == 3
    assert results[0]['construct_eligible_cells']['cells'] == 1
    assert 'holm_adjusted_p' in better
    with pytest.raises(ValueError, match='incomplete pairs'):
        cell_differences(frame[frame.dataset != 'd3'].pipe(lambda f: pd.concat(
            [f, frame[(frame.dataset == 'd3') & (frame.condition == 'ref')]])), 'ref', 'better')


def test_holm_matches_reference():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_phase_metrics_measure_response_and_recurrence():
    from tools.analyze_dasp_v2 import phase_metrics
    windows = pd.DataFrame({'window': range(4, 36), 'macro_f1': [0.9] * 8 + [0.5] * 4 + [0.8] * 8 + [0.6] * 4 + [0.9] * 8})
    recurrent = phase_metrics(windows, 'S4-Recurrent')
    assert recurrent['post_change_f1'] == pytest.approx((0.5 + 0.6) / 2)
    assert recurrent['recurrence_f1'] == pytest.approx(0.6)
    stationary = phase_metrics(windows, 'S0')
    assert np.isnan(stationary['post_change_f1']) and np.isnan(stationary['recurrence_f1'])
