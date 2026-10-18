"""Corrections and reviewer controls for original modular DASP only.

No changes to A1, fast adapters, other baselines, or the primary experiment grid.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

VERSION = 'modular-dasp-original-review-20261018-v1'
ROUTES = ('native', 'random', 'fixed', 'mmd', 'conflict', 'all', 'classifier')
_counter = contextvars.ContextVar('original_dasp_gradient_counter', default=None)
_phase = contextvars.ContextVar('original_dasp_gradient_phase', default='training')


def count_gradient_request(parameters):
    counter = _counter.get()
    if counter is not None:
        phase = counter.setdefault(_phase.get(), {'calls': 0, 'requested_elements': 0})
        phase['calls'] += 1
        phase['requested_elements'] += sum(p.numel() for p in parameters)


@contextmanager
def gradient_scope(counter, phase):
    token = _counter.set(counter)
    phase_token = _phase.set(phase)
    try:
        yield
    finally:
        _phase.reset(phase_token)
        _counter.reset(token)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def identity(config):
    paths = sorted(Path(__file__).parent.glob('*.py'))
    root = Path(__file__).resolve().parents[2]
    inputs = {}
    for value in (config.data.path, config.original_review.event_schedule):
        if value:
            path = Path(value)
            if not path.is_file():
                raise ValueError('review runs require a prepared file or explicit raw file, not a directory')
            inputs[str(path)] = sha256(path)
    code = {str(p.relative_to(root)):sha256(p) for p in paths}
    for name in ('tools/run_original_review.py','tools/run_modular_dasp_protocol.py'):
        code[name] = sha256(root/name)
    return {'version': VERSION, 'code': code,
        'inputs': inputs, 'config': json.loads(json.dumps(config.to_dict())),
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'numpy': np.__version__, 'platform':platform.platform(),
        'cuda_build':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(torch.device(config.training.device)) if config.training.device.startswith('cuda') else None}


def gate_evidence(y, prediction, n_classes, policy, threshold):
    """Retain binary Macro-F1 as a diagnostic; missing support is never an error.

    observed-class errors remain actionable. Missing-class recall/FPR remains
    unknown. Background consolidation continues under original policy.
    """
    from .metrics import classification_metrics
    y, prediction = y.cpu(), prediction.cpu()
    fixed = float(classification_metrics(y.numpy(), prediction.numpy(), n_classes)['macro_f1'])
    counts = torch.bincount(y, minlength=n_classes)
    present = counts > 0
    recalls = [float((prediction[y == c] == c).float().mean())
               for c in range(n_classes) if bool(present[c])]
    complete = bool(present.all())
    if policy not in ('supported_classes', 'legacy_fixed_macro'):
        raise ValueError('unknown original DASP gate policy')
    # On a partial pool, assess only whether an observed class is failing.
    # Do not assert global stability from a perfect attack-only pool.
    degraded = fixed < threshold if complete or policy == 'legacy_fixed_macro' else (
        bool(recalls) and min(recalls) < threshold)
    return {'fixed_macro_f1': fixed, 'class_counts': counts.tolist(),
        'class_support_complete': complete, 'degraded': bool(degraded),
        'state': 'observed_degradation' if degraded else ('supported' if complete else 'insufficient_support')}


def route_control(native, names, requirements, conflicts, policy, seed, weight):
    if policy not in ROUTES:
        raise ValueError(f'unknown original DASP routing control: {policy}')
    native = dict(native)
    if policy == 'native':
        return native
    positive = sorted((float(native.get(n, 0.)) for n in names if native.get(n, 0.) > 0.), reverse=True)
    k = len(positive)
    if policy == 'classifier':
        chosen = []
    elif policy == 'all':
        chosen = list(names)
        positive = [float(np.mean(positive)) if positive else 1.] * len(names)
    elif policy == 'random':
        order = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed)).tolist()
        chosen = [names[i] for i in order[:k]]
    elif policy == 'fixed':
        chosen = names[:k]
    else:
        scores = requirements if policy == 'mmd' else {n: 1/(1+weight*conflicts.get(n, 0.)) for n in names}
        chosen = sorted(names, key=lambda n: (-float(scores.get(n, 0.)), names.index(n)))[:k]
    out = {n: 0. for n in names}
    out.update(dict(zip(chosen, positive)))
    out['classifier'] = native.get('classifier', 1.)
    return out


def _locality(model, before, plasticities):
    named = {id(p): n for n, p in model.named_parameters()}
    total, included, changed, inactive_max, representation, active_rep = 0, 0, 0, 0., 0, 0
    head = 0
    module_names = set(getattr(model, 'layer_names', []))
    for group, parameters in model.adaptation_parameter_groups().items():
        active = plasticities.get(group, 1.) > 0
        for p in parameters:
            size = p.numel()
            total += size
            included += size if active else 0
            if group not in module_names:
                head += size
            else:
                representation += size
                active_rep += size if active else 0
            if before is not None:
                difference = (p.detach().cpu()-before[named[id(p)]]).abs()
                changed += int((difference > 1e-12).sum())
                if not active:
                    inactive_max = max(inactive_max, float(difference.max()))
    if before is not None and inactive_max != 0.:
        raise RuntimeError('original DASP changed an inactive parameter')
    return {'optimizer_included_parameter_count': included,
        'optimizer_included_parameter_fraction': included/max(1,total),
        'representation_parameter_count': representation,
        'active_representation_parameter_fraction': active_rep/max(1,representation),
        'classifier_parameter_fraction': head/max(1,total),
        'changed_parameter_count': changed if before is not None else None,
        'inactive_module_max_abs_change': inactive_max if before is not None else None,
        'locality_verified': before is not None}


def update_original(experiment, model, memory, item, row):
    # Import at call time to keep runner's existing primitives and seed behavior.
    from . import runner as r
    cfg, review = experiment.config, experiment.config.original_review
    synchronize(experiment.device)
    started = time.perf_counter()
    if review.label_mode not in ('original', 'matched'):
        raise ValueError('label_mode must be original or matched')
    memory_probe, _ = r._split_memory_for_probe_and_guard(memory, cfg.dasp.memory_batch_size, experiment.device)
    if review.label_mode == 'original':
        tx, ty, gx, gy = r._split_train_guard(item.window, cfg.dasp.guard_ratio, cfg.data.random_seed)
        selection_exposure = len(ty)
        exposed = len(ty) + (len(gy) if review.performance_gate_enabled else 0)
        if review.balanced_selection:
            selected = r._balanced_tensor_sample(tx, ty, cfg.baselines.update_label_budget,
                torch.device('cpu'), cfg.data.random_seed+item.window.index*104729+17)
            tx, ty = selected
        else:
            selected = r._label_blind_uniform_query_indices(len(tx), cfg.baselines.update_label_budget,
                cfg.data.random_seed+item.window.index*104729+17)
            tx, ty = tx[selected], ty[selected]
    else:
        ids = r._label_blind_uniform_query_indices(len(item.window.x), cfg.baselines.update_label_budget,
            cfg.data.random_seed+item.window.index*104729+17)
        if len(ids) < 2:
            raise ValueError('matched DASP requires at least two labels for disjoint train/gate')
        gate_count = min(len(ids)-1, max(1, round(len(ids)*cfg.dasp.guard_ratio)))
        training_ids, gate_ids = ids[:-gate_count], ids[-gate_count:]
        tx, ty = item.window.x[training_ids], item.window.y[training_ids]
        gx, gy = item.window.x[gate_ids], item.window.y[gate_ids]
        exposed, selection_exposure = len(ids), len(training_ids)
        row['original_query_indices'] = json.dumps(ids.tolist())
        row['original_gate_indices'] = json.dumps(gate_ids.tolist())
    counters = {}
    with gradient_scope(counters, 'conflict'):
        conflicts = r._gradient_conflicts(model, tx[:cfg.dasp.memory_batch_size],
            ty[:cfg.dasp.memory_batch_size], memory_probe, experiment.device)
    gate = None
    if review.performance_gate_enabled:
        model.eval()
        with torch.no_grad():
            prediction = model(gx.to(experiment.device)).argmax(1).cpu()
        gate = gate_evidence(gy, prediction, int(model.classifier.out_features),
            review.gate_policy, cfg.dasp.v12_guard_min_macro_f1)
    emergency = bool(gate and gate['degraded'] and not item.decision.triggered)
    names = list(model.layer_names)
    if item.decision.triggered:
        raw = r._plasticities('modular_dasp', item.decision.requirements, conflicts, cfg.dasp.conflict_weight)
        ranked = sorted(((n, v) for n, v in raw.items() if v > 0.), key=lambda pair: pair[1], reverse=True)
        positive = dict(ranked[:max(1,cfg.drift.top_k)])
        if not positive and raw:
            n, v = max(raw.items(), key=lambda pair: pair[1])
            positive = {n: v}
        plasticities = {n: float(positive.get(n, 0.)) for n in names}
        epochs = cfg.dasp.modular_repair_epochs
        candidate = 'localized_drift_repair'
    else:
        active = sorted(names, key=lambda n: -conflicts.get(n, 0.))[:max(1,cfg.drift.top_k)]
        plasticities = {n: ((min(1.,max(cfg.dasp.v3_backbone_plasticity,conflicts.get(n,0.)))
            if emergency else cfg.dasp.modular_background_plasticity) if n in active else 0.) for n in names}
        epochs = cfg.dasp.modular_emergency_epochs if emergency else cfg.dasp.modular_repair_epochs
        candidate = 'semi_local_repair' if emergency else 'background_local_consolidation'
    plasticities['classifier'] = 1.
    if review.event_schedule:
        if not hasattr(experiment,'_original_event_schedule'):
            schedule = json.loads(Path(review.event_schedule).read_text(encoding='utf-8'))
            if schedule.get('version') != VERSION:
                raise ValueError('reference schedule version mismatch')
            signature = schedule.get('reference_signature')
            if signature and signature != {'stream_sha256':sha256(cfg.data.path),
                    'stream_seed':cfg.data.random_seed,'model_seed':cfg.training.model_seed}:
                raise ValueError('reference schedule belongs to a different paired stream/model cell')
            experiment._original_event_schedule = schedule
        schedule = experiment._original_event_schedule
        expected = schedule['events'].get(str(item.window.index))
        if expected is None:
            raise ValueError('reference schedule missing delayed window')
        plasticities = {n: float(expected['plasticities'].get(n, 0.)) for n in (*names,'classifier')}
        if not set(names).intersection(expected['plasticities']):
            if review.routing != 'all':
                raise ValueError('group names differ from reference; flat control requires all routing')
            values = [v for n,v in expected['plasticities'].items() if n != 'classifier' and v > 0]
            plasticities = {n: float(np.mean(values)) if values else 0. for n in names}
            plasticities['classifier'] = float(expected['plasticities'].get('classifier',1.))
        epochs = int(expected['epochs'])
        if epochs < 1 or not all(np.isfinite(v) and 0. <= v <= 1. for v in plasticities.values()):
            raise ValueError('invalid event schedule epochs or plasticities')
        candidate = expected['candidate']
        row['original_event_matched'] = True
    native = dict(plasticities)
    if review.routing_k is not None:
        if not 0 <= review.routing_k <= len(names):
            raise ValueError('routing_k outside module count')
        selected = sorted(names, key=lambda n:-native[n])[:review.routing_k]
        native = {n:(native[n] if n in selected else 0.) for n in names} | {'classifier':native['classifier']}
    seed = r.stable_seed('original-routing-control', cfg.data.random_seed,
        cfg.training.model_seed, item.window.index)
    plasticities = route_control(native, names, item.decision.requirements, conflicts,
        review.routing, seed, cfg.dasp.conflict_weight)
    training_cfg = r.copy.deepcopy(cfg)
    if not review.replay_enabled:
        training_cfg.dasp.modular_replay_size = 0
    if not review.weighting_enabled:
        training_cfg.dasp.modular_new_sample_weight = 1.
    bx, by, weights = r._localized_rehearsal_batch(tx, ty, memory, training_cfg, experiment.device)
    before = {n:p.detach().cpu().clone() for n,p in model.named_parameters()} if review.verify_locality else None
    with gradient_scope(counters, 'training'):
        calls = r._train_dasp(model, bx, by, plasticities, training_cfg, experiment.device,
            memory, eta_ceiling=review.effective_eta_ceiling, strict_freeze=True,
            sample_weights=weights, epochs=epochs, use_projection_memory=False)
    memory.add(tx, ty)
    row.update(_locality(model, before, plasticities))
    row.update({'original_review_version': VERSION, 'original_label_mode': review.label_mode,
        'labels_exposed_total': exposed, 'ground_truth_update_labels': exposed,
        'labels_used_for_selection': selection_exposure,
        'labels_used_for_gating': len(gy) if review.performance_gate_enabled else 0,
        'new_training_labels_used': len(ty), 'original_gate_state': gate['state'] if gate else 'disabled',
        'original_gate_class_counts': json.dumps(gate['class_counts']) if gate else '[]',
        'modular_calibration_macro_f1': gate['fixed_macro_f1'] if gate else np.nan,
        'original_emergency_observed': emergency,
        'original_emergency_repair': candidate=='semi_local_repair', 'original_epochs': epochs,
        'original_native_plasticities': json.dumps(native, sort_keys=True),
        'original_plasticities': json.dumps(plasticities, sort_keys=True),
        'original_routing_policy': review.routing, 'original_effective_eta_ceiling': review.effective_eta_ceiling,
        'original_route_has_positive_evidence': any(item.decision.requirements.get(n,0.)>0 for n in names),
        'localized_repair_modules': '|'.join(n for n in names if plasticities[n] > 0.),
        'localized_repair_module_count': sum(plasticities[n]>0. for n in names),
        'localized_rehearsal_size': len(by), 'localized_replay_size': len(by)-len(ty),
        'gradient_evaluations': calls, 'training_gradient_evaluations': calls,
        'conflict_gradient_evaluations': counters.get('conflict',{}).get('calls',0),
        'overall_gradient_evaluations': sum(c['calls'] for c in counters.values()),
        'parameter_gradient_element_evaluations': sum(c['requested_elements'] for c in counters.values()),
        'original_gradient_phases': json.dumps(counters,sort_keys=True),
        'accepted': 'dasp', 'selected_candidate': candidate, 'global_update': review.routing == 'all',
        'rss_bytes': int(experiment._process.memory_info().rss)})
    row.update({f'plasticity_{n}':v for n,v in plasticities.items()})
    row.update({f'conflict_{n}':v for n,v in conflicts.items()})
    synchronize(experiment.device)
    row['update_seconds'] = time.perf_counter()-started
    return 'dasp'


def annotate_outputs(experiment, rows, summary):
    if experiment.config.dasp.baseline != 'modular_dasp' or not experiment.config.original_review.enabled:
        return
    evaluated = [r for r in rows if not r['warmup']]
    summary.update({'original_review_version': VERSION,
        'algorithm_id': f'{VERSION}/{experiment.config.original_review.routing}/{experiment.config.original_review.label_mode}',
        'primary_comparison_scope': 'protocol_specific' if experiment.config.original_review.label_mode=='original' else 'matched_total_query_budget',
        'gradient_accounting_scope': 'post-warmup adaptation, training plus conflict; excludes warmup',
        'gradient_unit': 'requested parameter-gradient tensor elements; not FLOPs',
        'total_overall_gradient_evaluations': sum(int(r.get('overall_gradient_evaluations',0)) for r in evaluated),
        'total_conflict_gradient_evaluations': sum(int(r.get('conflict_gradient_evaluations',0)) for r in evaluated),
        'total_parameter_gradient_element_evaluations': sum(int(r.get('parameter_gradient_element_evaluations',0)) for r in evaluated),
        'total_full_cycle_seconds': sum(float(r.get('original_full_cycle_seconds',0.)) for r in rows),
        'terminal_update_seconds': sum(float(r.get('terminal_update_seconds',0.)) for r in rows),
        'locality_diagnostic_overhead_included_in_update_time': experiment.config.original_review.verify_locality,
        'total_update_seconds_scope': 'update routine only; reference refresh belongs to full_cycle',
        'frozen_parameters_do_not_imply_functional_or_causal_locality': True})
    tp, fn, fp, tn = [sum(int(r[k]) for r in evaluated) for k in ('attack_tp','attack_fn','attack_fp','attack_tn')]
    summary['cumulative_benign_fpr'] = fp/(fp+tn) if fp+tn else np.nan
    rates = [r['attack_fp']/(r['attack_fp']+r['attack_tn']) for r in evaluated if r['attack_fp']+r['attack_tn']]
    summary['mean_window_benign_fpr'] = float(np.mean(rates)) if rates else np.nan
    summary['operational_error_counts'] = dict(tp=tp,fn=fn,fp=fp,tn=tn)
