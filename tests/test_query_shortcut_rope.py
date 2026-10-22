"""E6 query policies and sweep, E4 shortcut ablation, E15 Bayesian ROPE."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch

from dasp_nids.config import load_config
from dasp_nids.data import load_prepared_stream
from dasp_nids.feature_groups import NETFLOW_V3, semantic_groups
from dasp_nids.runner import (QUERY_POLICIES, StreamingExperiment, _label_blind_uniform_query_indices,
                              _score_query_indices)
from tools.analyze_dasp_v2 import analysis_frame, bayesian_signed_rank, paired_family
from tools.run_dasp_v2 import SHORTCUT_METHODS, SWEEP_METHODS, SWEEP_SETTINGS, jobs_for, parser

SCORES = np.array([0.02, 0.48, 0.97, 0.55, 0.10, 0.51, 0.90, 0.30, 0.49, 0.99])


def test_uniform_policy_is_the_historical_query():
    for budget in (0, 3, 10, 50):
        assert torch.equal(_score_query_indices(SCORES, budget, 7, 'uniform'),
                           _label_blind_uniform_query_indices(len(SCORES), budget, 7))


def test_uncertainty_takes_the_samples_nearest_the_threshold():
    chosen = _score_query_indices(SCORES, 3, 7, 'uncertainty')
    assert set(chosen.tolist()) == {1, 5, 8}  # 0.48, 0.51, 0.49
    shifted = _score_query_indices(SCORES, 2, 7, 'uncertainty', threshold=0.9)
    assert set(shifted.tolist()) == {6, 2}  # 0.90, 0.97


def test_mixed_is_half_uncertain_half_uniform_without_repeats():
    chosen = _score_query_indices(SCORES, 6, 7, 'mixed').tolist()
    assert len(chosen) == len(set(chosen)) == 6
    assert {1, 5, 8} <= set(chosen)


def test_pred_stratified_balances_and_fills_a_short_side():
    chosen = _score_query_indices(SCORES, 4, 7, 'pred_stratified').tolist()
    assert sum(SCORES[i] >= 0.5 for i in chosen) == 2
    rare = np.array([0.9] + [0.1] * 9)  # one predicted attack
    chosen = _score_query_indices(rare, 6, 7, 'pred_stratified').tolist()
    assert len(set(chosen)) == 6 and 0 in chosen


@pytest.mark.parametrize('policy', QUERY_POLICIES)
def test_every_policy_is_deterministic_and_respects_the_budget(policy):
    first = _score_query_indices(SCORES, 4, 11, policy)
    assert torch.equal(first, _score_query_indices(SCORES, 4, 11, policy))
    assert len(_score_query_indices(SCORES, 99, 11, policy)) == len(SCORES)
    with pytest.raises(ValueError):
        _score_query_indices(SCORES, 4, 11, 'oracle')


def config(tmp_path, method, policy, name):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory, cfg.output.run_name = str(tmp_path), name
    cfg.training.model_seed = 91
    cfg.dasp.baseline = method
    cfg.baselines.update_label_budget = 32
    cfg.baselines.query_policy = policy
    return cfg


@pytest.mark.parametrize('method', ['experience_replay', 'modular_dasp_v2', 'arf_adapted'])
def test_score_policies_never_read_unqueried_labels(tmp_path, method, monkeypatch):
    """Record the queries of a clean run, flip every post-warm-up label
    outside them: the prediction-time scores must not change."""

    queries, scores = {}, {}
    query, remember = StreamingExperiment._query_indices, StreamingExperiment._remember_query_scores

    def recording_query(self, window):
        chosen = query(self, window)
        queries.setdefault(self.config.output.run_name, {})[window.index] = chosen.clone()
        return chosen

    def recording_scores(self, window, attack_score, threshold=0.5):
        scores.setdefault(self.config.output.run_name, []).append(np.asarray(attack_score).copy())
        return remember(self, window, attack_score, threshold)

    monkeypatch.setattr(StreamingExperiment, '_query_indices', recording_query)
    monkeypatch.setattr(StreamingExperiment, '_remember_query_scores', recording_scores)
    clean = StreamingExperiment(config(tmp_path, method, 'mixed', 'clean')).run().windows
    evaluated = clean[~clean.warmup.astype(bool)]
    assert set(evaluated.label_query_policy) == {'mixed_label_blind'}
    assert set(evaluated.ground_truth_update_labels) == {32}

    original = StreamingExperiment._load_stream

    def corrupted(self):
        stream = original(self)
        windows = []
        for window in stream.windows:
            if window.index not in queries['clean']:
                windows.append(window)
                continue
            keep = torch.zeros(len(window.y), dtype=torch.bool)
            keep[queries['clean'][window.index]] = True
            windows.append(dataclasses.replace(window, y=torch.where(keep, window.y, 1 - window.y)))
        return dataclasses.replace(stream, windows=windows)

    monkeypatch.setattr(StreamingExperiment, '_load_stream', corrupted)
    StreamingExperiment(config(tmp_path, method, 'mixed', 'flipped')).run()
    assert len(scores['clean']) == len(scores['flipped'])
    for a, b in zip(scores['clean'], scores['flipped']):
        np.testing.assert_array_equal(a, b)


def test_score_policies_are_refused_for_methods_without_the_common_query(tmp_path):
    with pytest.raises(ValueError, match='common query'):
        StreamingExperiment(config(tmp_path, 'modular_dasp', 'mixed', 'x'))


def netflow_stream(path):
    names = [name for group in NETFLOW_V3.values() for name in group]
    x = np.arange(2 * 4 * len(names), dtype=np.float32).reshape(2, 4, len(names))
    np.savez(path, x=x, y=np.zeros((2, 4), dtype=np.int64), phases=np.array(['warmup', 'drift']),
             feature_names=np.array(names), class_names=np.array(['Benign', 'Attack']))
    return names, x


def test_dropped_features_leave_the_stream_and_their_groups(tmp_path):
    names, x = netflow_stream(tmp_path / 's.npz')
    dropped = ['MIN_TTL', 'MAX_TTL', 'L4_SRC_PORT', 'NOT_A_FEATURE']
    stream = load_prepared_stream(tmp_path / 's.npz', dropped)
    assert stream.feature_names == [n for n in names if n not in dropped]
    kept = [names.index(n) for n in stream.feature_names]
    np.testing.assert_array_equal(stream.windows[1].x.numpy(), x[1][:, kept])
    groups = semantic_groups(stream.feature_names, dropped)
    protocol = [stream.feature_names[i] for i in groups[3]]
    assert 'MAX_TTL' not in protocol and 'PROTOCOL' in protocol
    # A feature missing without being declared dropped is still a schema error.
    with pytest.raises(ValueError, match='schema'):
        semantic_groups(stream.feature_names)
    with pytest.raises(ValueError, match='every feature'):
        load_prepared_stream(tmp_path / 's.npz', names)


def selection_file(tmp_path):
    methods = ('modular_dasp_v2', 'experience_replay', 'online_mlp', 'ssf_budget', 'arf_adapted')
    path = tmp_path / 'dev_selection.json'
    path.write_text(json.dumps({'selection': {m: {'overrides': {'training.learning_rate': 0.002}}
                                              for m in methods}}), encoding='utf-8')
    return str(path)


def stage_jobs(tmp_path, stage, *extra):
    return jobs_for(parser().parse_args(['--stage', stage, '--output-root', str(tmp_path), '--selection',
                                         selection_file(tmp_path), '--seeds-per-dataset', '1', *extra]))


def test_sweep_plan_crosses_settings_with_the_selected_configurations(tmp_path):
    jobs = stage_jobs(tmp_path, 'sweep', '--datasets', 'nf-ton-iot-v3')
    cells = {(j['dataset'], j['scenario'], j['stream_seed']) for j in jobs}
    assert len(cells) == 3 + 1  # three controlled scenarios + the chronological stream
    per_cell = len(jobs) // len(cells)
    assert per_cell == (len(SWEEP_METHODS) - 1) * len(SWEEP_SETTINGS) + 1
    er = [j for j in jobs if j['condition'] == 'experience_replay@b32_d1_pred_stratified'][0]
    assert er['overrides'] == {'training.learning_rate': 0.002, 'baselines.update_label_budget': 32,
                               'training.label_delay': 1, 'baselines.query_policy': 'pred_stratified'}
    final = [j for j in jobs if j['condition'] == 'final@b128_d10_mixed'][0]
    assert final['method'] == 'modular_dasp_v2' and final['overrides']['dasp_v2.realign'] == 'attributed'
    assert [j['overrides'] for j in jobs if j['method'] == 'static'][0] == {}


def test_shortcut_plan_skips_sets_a_schema_lacks(tmp_path):
    jobs = stage_jobs(tmp_path, 'shortcut', '--datasets', 'cic-ids2017', 'nf-unsw-nb15-v3', '--skip-natural')
    cic = {j['condition'].split('@')[1] for j in jobs if j['dataset'] == 'cic-ids2017'}
    unsw = {j['condition'].split('@')[1] for j in jobs if j['dataset'] == 'nf-unsw-nb15-v3'}
    assert cic == {'none', 'ports', 'ports_ttl'} and unsw == {'none', 'ports', 'ttl', 'ports_ttl'}
    ports = [j for j in jobs if j['dataset'] == 'cic-ids2017' and j['condition'] == 'final@ports_ttl'][0]
    assert ports['overrides']['data.drop_features'] == ['Destination Port']
    assert {j['condition'].split('@')[0] for j in jobs} == set(SHORTCUT_METHODS)


def test_bayesian_signed_rank_separates_better_equivalent_and_worse():
    rng = np.random.default_rng(0)
    better = bayesian_signed_rank(0.05 + 0.005 * rng.standard_normal(20), rope=0.01, samples=4000)
    equal = bayesian_signed_rank(0.002 * rng.standard_normal(20), rope=0.01, samples=4000)
    worse = bayesian_signed_rank(-0.05 + 0.005 * rng.standard_normal(20), rope=0.01, samples=4000)
    assert better['p_condition_better'] > 0.99 and equal['p_practically_equivalent'] > 0.99
    assert worse['p_condition_worse'] > 0.99
    assert sum(v for k, v in better.items() if k.startswith('p_')) == pytest.approx(1.0)
    # Few cells: the prior pseudo-observation keeps the posterior uncertain.
    few = bayesian_signed_rank(np.array([0.05, 0.05]), rope=0.01, samples=4000)
    assert few['p_condition_better'] < 0.99 < better['p_condition_better']


def test_paired_family_reports_rope_and_subsets_absent_settings():
    rows = []
    for dataset in ('a', 'b'):
        for scenario in ('S0', 'S1-Sudden', 'S6-Combined'):
            for setting, delta in (('none', 0.0), ('ttl', -0.04)):
                if dataset == 'b' and setting == 'ttl':
                    continue
                rows.append(dict(stage='shortcut', dataset=dataset, scenario=scenario, stream_seed=1,
                                 model_seed=91, method='m', condition=f'm@{setting}', overrides={},
                                 summary={'mean_macro_f1': 0.9 + delta, 'cumulative_macro_f1': 0.9 + delta}))
    frame = analysis_frame(rows)
    assert set(frame.setting) == {'none', 'ttl'}
    with pytest.raises(ValueError, match='incomplete pairs'):
        paired_family(frame, [('m@none', 'm@ttl')], replicates=100)
    result = paired_family(frame, [('m@none', 'm@ttl')], replicates=100, subset_to_condition=True)[0]
    assert result['all_cells']['cells'] == 3
    assert result['all_cells']['mean_difference'] == pytest.approx(-0.04)
    assert 'bayes_signed_rank' in result['all_cells']
    assert isinstance(frame, pd.DataFrame)
