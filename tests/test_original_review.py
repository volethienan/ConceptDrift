from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dasp_nids.config import load_config
from dasp_nids.data import StreamWindow
from dasp_nids.drift import DriftDecision
from dasp_nids.memory import ReservoirMemory
from dasp_nids.model import ModularOnlineMLP
from dasp_nids.original_review import (gate_evidence, gradient_scope, route_control,
    update_original, identity)
from dasp_nids.runner import StreamingExperiment, PendingUpdate, _train_dasp
from tools.run_original_review import jobs_for, parser, flat_widths, inspect_prepared, run, _smoke_stream


def config(tmp_path,name='run'):
    cfg = load_config('configs/modular_dasp_smoke.yaml')
    cfg.output.directory,cfg.output.run_name = str(tmp_path),name
    cfg.training.model_seed = 91
    return cfg


def test_perfect_attack_only_gate_does_not_increase_repair_intensity():
    y = torch.ones(128,dtype=torch.long)
    new = gate_evidence(y,y,2,'supported_classes',.8)
    old = gate_evidence(y,y,2,'legacy_fixed_macro',.8)
    assert new['fixed_macro_f1'] == .5
    assert new['state'] == 'insufficient_support'
    assert not new['degraded'] and old['degraded']


def test_missing_benign_support_does_not_hide_actual_attack_errors():
    y = torch.ones(128,dtype=torch.long)
    assert gate_evidence(y,torch.zeros_like(y),2,'supported_classes',.8)['degraded']


def test_improved_path_matches_original_numerically_when_gate_policy_is_held_fixed(tmp_path):
    old = config(tmp_path,'legacy')
    old.original_review.enabled = False
    new = copy.deepcopy(old)
    new.original_review.enabled = True
    new.original_review.gate_policy = 'legacy_fixed_macro'
    new.output.run_name = 'corrected_compute_same_gate'
    first,second = StreamingExperiment(old).run(),StreamingExperiment(new).run()
    assert np.allclose(first.windows.macro_f1,second.windows.macro_f1,rtol=0,atol=1e-12)
    for name in ('modular_repair_epochs','modular_background_plasticity','memory_size','modular_replay_size'):
        assert getattr(old.dasp,name) == getattr(new.dasp,name)
    assert second.summary['original_review_version']


def test_gradient_requests_exclude_inactive_original_modules(tmp_path,monkeypatch):
    cfg = config(tmp_path)
    model = ModularOnlineMLP(16,2,[16],n_modules=4,module_dim=8)
    groups = model.adaptation_parameter_groups()
    active = {'module_1':.25,'module_2':0.,'module_3':0.,'module_4':0.,'classifier':1.}
    requested = []
    original_grad = torch.autograd.grad
    def instrument(loss,parameters,**kwargs):
        requested.extend(id(p) for p in parameters)
        return original_grad(loss,parameters,**kwargs)
    monkeypatch.setattr(torch.autograd,'grad',instrument)
    before = {id(p):p.detach().clone() for p in model.parameters()}
    counters = {}
    with gradient_scope(counters,'training'):
        calls = _train_dasp(model,torch.randn(16,16),torch.arange(16)%2,active,cfg,
            torch.device('cpu'),ReservoirMemory(16,91),strict_freeze=True,
            use_projection_memory=False,epochs=1)
    inactive = [p for n,params in groups.items() if active[n]==0 for p in params]
    assert not set(requested).intersection(id(p) for p in inactive)
    assert all(torch.equal(p,before[id(p)]) for p in inactive)
    included = sum(p.numel() for n,params in groups.items() if active[n]>0 for p in params)
    assert counters['training']['requested_elements'] == calls*included


def test_pipeline_counts_conflict_and_training_and_preserves_background(tmp_path):
    cfg = config(tmp_path)
    result = StreamingExperiment(cfg).run()
    rows = result.windows[~result.windows.warmup.astype(bool)]
    for _,row in rows.iterrows():
        phases = json.loads(row.original_gradient_phases)
        assert row.overall_gradient_evaluations == row.gradient_evaluations+row.conflict_gradient_evaluations
        assert row.conflict_gradient_evaluations == 2
        assert row.parameter_gradient_element_evaluations == sum(v['requested_elements'] for v in phases.values())
        assert row.inactive_module_max_abs_change == 0.
    assert (rows.gradient_evaluations > 0).all()
    assert cfg.dasp.eta_ceiling == 2. and (rows.original_effective_eta_ceiling==1.).all()
    assert result.summary['total_full_cycle_seconds'] > 0


def test_routing_controls_preserve_native_active_count_and_multiplier_multiset():
    native = {'m1':.2,'m2':.7,'m3':0.,'m4':.1,'classifier':1.}
    names = ['m1','m2','m3','m4']
    for policy in ('random','fixed','mmd','conflict'):
        first = route_control(native,names,{},dict(m1=.4,m2=.3),policy,91,1.)
        assert first == route_control(native,names,{},dict(m1=.4,m2=.3),policy,91,1.)
        assert sorted(first[n] for n in names) == sorted(native[n] for n in names)
        assert first['classifier'] == 1.


def test_matched_queries_are_causal_and_unqueried_labels_do_not_change_update(tmp_path):
    cfg = config(tmp_path)
    cfg.original_review.label_mode = 'matched'
    cfg.baselines.update_label_budget = 24
    torch.manual_seed(91)
    base = ModularOnlineMLP(16,2,[16],n_modules=4,module_dim=8)
    x,y = torch.randn(96,16),torch.arange(96)%2
    decision = DriftDecision({}, {}, {},0.,False,False)
    def run(labels):
        model = copy.deepcopy(base)
        memory = ReservoirMemory(128,91)
        memory.add(x[:32],y[:32])
        row = {}
        item = PendingUpdate(StreamWindow(x,labels,3,'post'),decision,False)
        update_original(StreamingExperiment(cfg),model,memory,item,row)
        return model,row
    first,row = run(y)
    queried = json.loads(row['original_query_indices'])
    gate = set(json.loads(row['original_gate_indices']))
    assert len(queried) == row['labels_exposed_total'] == 24
    assert len(set(queried))==24 and gate.issubset(queried)
    assert row['new_training_labels_used'] + row['labels_used_for_gating'] == 24
    changed = y.clone()
    outside = torch.tensor(sorted(set(range(96))-set(queried)))
    changed[outside] = 1-changed[outside]
    second,_ = run(changed)
    assert all(torch.equal(a,b) for a,b in zip(first.parameters(),second.parameters()))


def test_primary_plan_keeps_exact_original_4000_grid(tmp_path):
    args = parser().parse_args(['--stage','primary','--output-root',str(tmp_path)])
    jobs = jobs_for(args)
    assert len(jobs)==4000
    assert len({j['dataset'] for j in jobs})==5
    assert len({j['scenario'] for j in jobs})==10
    assert len({j['method'] for j in jobs})==8
    assert {j['stream_seed'] for j in jobs}=={2001,2002,2003,2101,2102}
    assert {j['model_seed'] for j in jobs}=={91,92}
    assert not any(j.get('event_schedule') for j in jobs)


@pytest.mark.parametrize('features',[49,78])
def test_flat_global_control_exactly_matches_modular_parameter_budget(features):
    model = ModularOnlineMLP(features,2,[128,64],n_modules=4,module_dim=64)
    target = sum(p.numel() for p in model.parameters())
    a,b = flat_widths(features,target)
    assert a*(features+3+b)+5*b+2 == target


def test_checkpoint_contains_real_model_memory_and_rng_and_roundtrips(tmp_path):
    cfg = config(tmp_path)
    cfg.output.save_checkpoints = True
    result = StreamingExperiment(cfg).run()
    payload = torch.load(result.output_dir/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert payload['completed_run']
    assert not payload['automatic_midstream_resume_supported']
    assert payload['model_state_dict'] and len(payload['memory'])>0
    assert payload['rng_torch'] is not None
    model = payload['model']
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in payload['model_state_dict'].items())


def test_review_knobs_do_not_change_other_baseline_predictions(tmp_path):
    cfg = config(tmp_path,'baseline_a')
    cfg.dasp.baseline = 'online_mlp'
    alternate = copy.deepcopy(cfg)
    alternate.output.run_name = 'baseline_b'
    alternate.original_review.routing = 'classifier'
    alternate.original_review.performance_gate_enabled = False
    alternate.original_review.label_mode = 'matched'
    first,second = StreamingExperiment(cfg).run(),StreamingExperiment(alternate).run()
    assert first.windows.macro_f1.tolist()==second.windows.macro_f1.tolist()
    assert 'original_review_version' not in first.summary


def test_cached_run_is_never_silently_overwritten(tmp_path):
    cfg = config(tmp_path)
    StreamingExperiment(cfg).run()
    with pytest.raises(FileExistsError):
        StreamingExperiment(cfg).run()


def test_preflight_rejects_invalid_labels_and_duplicate_ids(tmp_path):
    target = tmp_path/'invalid.npz'
    arrays = dict(x=np.ones((3,8,4)),y=np.zeros((3,8),dtype=int),
        phases=np.array(['a']*3),feature_names=np.array(['f']*4),class_names=np.array(['b','a']),
        source_ids=np.zeros((3,8),dtype=int))
    np.savez(target,**arrays)
    with pytest.raises(ValueError,match='source IDs'):
        inspect_prepared(target)


def test_server_resume_validates_run_identity_and_output_receipts(tmp_path):
    prepared = tmp_path/'prepared'
    _smoke_stream(prepared/'nf-ton-iot-v3'/'S0'/'stream_seed2001.npz')
    args = parser().parse_args(['--stage','gate','--datasets','nf-ton-iot-v3',
        '--scenarios','S0','--stream-seeds','2001','--model-seeds','91',
        '--prepared-root',str(prepared),'--output-root',str(tmp_path/'out'),'--execute'])
    run(args)
    args.resume = True
    run(args)
    args.no_locality_check = True
    with pytest.raises(ValueError,match='mismatch'):
        run(args)
    args.no_locality_check = False
    window_file = next((tmp_path/'out'/'gate').rglob('windows.csv'))
    window_file.write_text(window_file.read_text(encoding='utf-8')+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='hashes'):
        run(args)


def test_stage_plan_cannot_be_changed_after_results_exist(tmp_path):
    args = parser().parse_args(['--stage','primary','--output-root',str(tmp_path)])
    run(args)
    (tmp_path/'primary_completed.json').write_text('[]',encoding='utf-8')
    args.model_seeds = [91]
    with pytest.raises(ValueError,match='cannot change'):
        run(args)


def test_gpu_preflight_smoke_includes_exact_original_comparator_list(tmp_path):
    args = parser().parse_args(['--stage','smoke','--include-baselines','--output-root',str(tmp_path)])
    jobs = jobs_for(args)
    assert len(jobs)==23
    assert {j['method'] for j in jobs}=={'modular_dasp','ssf','repshield','cdda_md',
        'online_mlp','experience_replay','online_ewc','agem'}
