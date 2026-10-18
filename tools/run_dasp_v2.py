"""DASP-v2 development stage: equal 24-configuration grids per method.

Pre-registration sections 4-5. Every method runs under the common label-blind
128-label query contract on the development split only: NF-BoT-IoT-v3 and
NF-UNSW-NB15-v3, ten scenarios, stream seeds 2001-2003, model seed 91.
Confirmation streams are never read here.

    python tools/run_dasp_v2.py --stage dev --prepared-root prepared_streams_v2 \
        --output-root results/dasp_v2_dev --execute --resume --shard 1/8

Without --execute only the plan is written. Shards split whole paired cells.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from dasp_nids.config import load_config  # noqa: E402
from dasp_nids.original_review import identity, sha256  # noqa: E402
from dasp_nids.runner import StreamingExperiment  # noqa: E402
from tools.run_modular_dasp_protocol import PRIMARY_SCENARIOS  # noqa: E402
from tools.run_original_review import (  # noqa: E402
    _smoke_stream, inspect_prepared, parse_shard, shard_jobs, write_plan)

VERSION = 'dasp-v2-dev-runner-20261018-r0'
DEV_DATASETS = ('nf-bot-iot-v3', 'nf-unsw-nb15-v3')
DEV_STREAM_SEEDS = (2001, 2002, 2003)
DEV_MODEL_SEEDS = (91,)
STAGES = ('dev', 'screen', 'smoke')


def _grid(**axes):
    names = list(axes)
    return [dict(zip(names, values)) for values in itertools.product(*axes.values())]


# Pre-registration section 5: exactly 24 configurations per method.
GRIDS = {
    'modular_dasp_v2': [
        {'model.modular_head_dim': head, 'dasp_v2.gate_min_support': support,
         'dasp_v2.gate_threshold': threshold, 'dasp_v2.routing_top_k': top_k,
         'dasp_v2.logit_adjustment_tau': tau}
        for head in (0, 16) for support, threshold in ((20, .80), (20, .70), (40, .80))
        for top_k in (1, 2) for tau in (0.0, 1.0)],
    'online_mlp': _grid(**{'training.learning_rate': (2.5e-4, 5e-4, 1e-3, 2e-3),
                           'training.epochs_per_window': (1, 2, 4),
                           'training.weight_decay': (1e-4, 1e-3)}),
    'experience_replay': _grid(**{'training.learning_rate': (5e-4, 1e-3, 2e-3),
                                  'dasp.memory_batch_size': (64, 128, 256, 512),
                                  'training.epochs_per_window': (1, 2)}),
    'agem': _grid(**{'training.learning_rate': (5e-4, 1e-3, 2e-3),
                     'dasp.memory_batch_size': (64, 128, 256, 512),
                     'training.epochs_per_window': (1, 2)}),
    'online_ewc_v2': _grid(**{'dasp.ewc_lambda': (1.0, 10.0, 100.0, 1000.0),
                              'dasp.ewc_anchor_interval': (2, 4, 8),
                              'training.learning_rate': (5e-4, 1e-3)}),
    'ssf_budget': _grid(**{'baselines.ssf_lwf_lambda': (0.1, 0.3, 0.5, 1.0),
                           'baselines.ssf_new_sample_weight': (1.0, 3.0, 5.0),
                           'training.learning_rate': (5e-4, 1e-3)}),
    'cdda_md_v2': _grid(**{'baselines.cdda_error_z': (1.0, 2.0, 3.0),
                           'baselines.cdda_warning_patience': (1, 2, 3, 4),
                           'training.learning_rate': (5e-4, 1e-3)}),
}


# Fast iteration screen on development data: one default configuration per
# method, a no-update lower bound, and DASP-v2 variants that switch one
# component at a time (route_native = the old MMD+conflict routing inside v2).
_V2_LINEAR = {'model.modular_head_dim': 0}
SCREEN = {
    'static': ('static', {}),
    'online_mlp': ('online_mlp', {}),
    'experience_replay': ('experience_replay', {}),
    'agem': ('agem', {}),
    'online_ewc_v2': ('online_ewc_v2', {}),
    'ssf_budget': ('ssf_budget', {}),
    'cdda_md_v2': ('cdda_md_v2', {}),
    'v2_default_head': ('modular_dasp_v2', {}),
    'v2_linear': ('modular_dasp_v2', dict(_V2_LINEAR)),
    'v2_linear_route_all': ('modular_dasp_v2', _V2_LINEAR | {'dasp_v2.routing': 'all'}),
    'v2_linear_route_head': ('modular_dasp_v2', _V2_LINEAR | {'dasp_v2.routing': 'classifier'}),
    'v2_linear_route_native': ('modular_dasp_v2', _V2_LINEAR | {'dasp_v2.routing': 'native'}),
    'v2_linear_gate_off': ('modular_dasp_v2', _V2_LINEAR | {'dasp_v2.gate_policy': 'off'}),
    'v2_linear_tau0': ('modular_dasp_v2', _V2_LINEAR | {'dasp_v2.logit_adjustment_tau': 0.0}),
    # Round 2: one-component ablations on the historical head, the round-1 leader.
    'v2_route_all': ('modular_dasp_v2', {'dasp_v2.routing': 'all'}),
    'v2_route_head': ('modular_dasp_v2', {'dasp_v2.routing': 'classifier'}),
    'v2_route_native': ('modular_dasp_v2', {'dasp_v2.routing': 'native'}),
    'v2_gate_off': ('modular_dasp_v2', {'dasp_v2.gate_policy': 'off'}),
    'v2_tau0': ('modular_dasp_v2', {'dasp_v2.logit_adjustment_tau': 0.0}),
    'v2_head16': ('modular_dasp_v2', {'model.modular_head_dim': 16}),
    'v2_head32': ('modular_dasp_v2', {'model.modular_head_dim': 32}),
    'v2_k1': ('modular_dasp_v2', {'dasp_v2.routing_top_k': 1}),
    'v2_k3': ('modular_dasp_v2', {'dasp_v2.routing_top_k': 3}),
    # Round 3: which training ingredient carries the gain over replay baselines.
    'v2_no_replay': ('modular_dasp_v2', {'dasp_v2.replay_enabled': False}),
    'v2_uniform_replay': ('modular_dasp_v2', {'dasp_v2.balanced_replay': False}),
    'v2_weight1': ('modular_dasp_v2', {'dasp_v2.new_sample_weight': 1.0}),
    'v2_bg_none': ('modular_dasp_v2', {'dasp_v2.background': 'none'}),
    'v2_bg_all': ('modular_dasp_v2', {'dasp_v2.background': 'all', 'dasp_v2.routing': 'all'}),
    'v2_bg_all_tau0': ('modular_dasp_v2', {'dasp_v2.background': 'all', 'dasp_v2.routing': 'all',
                                           'dasp_v2.logit_adjustment_tau': 0.0}),
    # Round 4: label-shift correction from unlabeled windows.
    'v2_batch_la': ('modular_dasp_v2', {'dasp_v2.logit_adjustment_prior': 'batch'}),
    'v2_em_l1': ('modular_dasp_v2', {'dasp_v2.logit_adjustment_prior': 'batch',
                                     'dasp_v2.prior_correction': 'em', 'dasp_v2.prior_strength': 1.0}),
    'v2_em_l05': ('modular_dasp_v2', {'dasp_v2.logit_adjustment_prior': 'batch',
                                      'dasp_v2.prior_correction': 'em', 'dasp_v2.prior_strength': 0.5}),
    'v2_acc_l1': ('modular_dasp_v2', {'dasp_v2.logit_adjustment_prior': 'batch',
                                      'dasp_v2.prior_correction': 'acc', 'dasp_v2.prior_strength': 1.0}),
    'v2_acc_l05': ('modular_dasp_v2', {'dasp_v2.logit_adjustment_prior': 'batch',
                                       'dasp_v2.prior_correction': 'acc', 'dasp_v2.prior_strength': 0.5}),
    'v2_em_current_l1': ('modular_dasp_v2', {'dasp_v2.logit_adjustment_prior': 'batch',
                                             'dasp_v2.prior_correction': 'em', 'dasp_v2.prior_strength': 1.0,
                                             'dasp_v2.prior_source': 'current'}),
}
# Round 6: semantic feature-group modules on feature-local drift. Base recipe
# = batch-prior logit adjustment + ACC prior correction (lambda 0.5).
_V2_RECIPE = {'dasp_v2.logit_adjustment_prior': 'batch', 'dasp_v2.prior_correction': 'acc',
              'dasp_v2.prior_strength': 0.5}
_SEMANTIC = _V2_RECIPE | {'model.feature_grouping': 'semantic'}
SCREEN.update({
    'v2_contig_utility': ('modular_dasp_v2', dict(_V2_RECIPE)),
    'v2_sem_utility': ('modular_dasp_v2', dict(_SEMANTIC)),
    'v2_sem_utility_k1': ('modular_dasp_v2', _SEMANTIC | {'dasp_v2.routing_top_k': 1}),
    'v2_sem_native': ('modular_dasp_v2', _SEMANTIC | {'dasp_v2.routing': 'native'}),
    'v2_sem_native_k1': ('modular_dasp_v2', _SEMANTIC | {'dasp_v2.routing': 'native', 'dasp_v2.routing_top_k': 1}),
    'v2_sem_all': ('modular_dasp_v2', _SEMANTIC | {'dasp_v2.routing': 'all'}),
    'v2_sem_head': ('modular_dasp_v2', _SEMANTIC | {'dasp_v2.routing': 'classifier'}),
    'v2_contig_head': ('modular_dasp_v2', _V2_RECIPE | {'dasp_v2.routing': 'classifier'}),
    'v2_sem_full_every_window': ('modular_dasp_v2', _SEMANTIC | {'dasp_v2.routing': 'all',
                                                                 'dasp_v2.background': 'all'}),
})
FEATURE_SHIFT_SCENARIOS = tuple(f'{kind}-{group}' for kind in ('FS', 'FR')
                                for group in ('volume', 'timing', 'tcp', 'protocol'))
# Development datasets with a usable whole-capture chronological stream.
NATURAL_DEV_DATASETS = ('nf-unsw-nb15-v3',)
NATURAL_SCENARIO = 'Natural-Chronological'
LABEL_SHIFT_SCENARIOS = ('L1-Burst', 'L2-Ramp', 'L3-Swing')


def config_id(method, overrides):
    text = json.dumps(overrides, sort_keys=True)
    return f"{method}-{hashlib.sha256(text.encode()).hexdigest()[:10]}"


def apply_overrides(cfg, overrides):
    for dotted, value in overrides.items():
        section, name = dotted.split('.')
        target = getattr(cfg, section)
        if not hasattr(target, name):
            raise ValueError(f'unknown config field {dotted}')
        setattr(target, name, value)


def jobs_for(args):
    if args.stage == 'smoke':
        path = Path(args.output_root) / '_fixtures' / 'synthetic_smoke.npz'
        _smoke_stream(path)
        cells = [('synthetic-smoke', 'S1', 2001, 91, str(path))]
        grids = {method: grid[:1] for method, grid in GRIDS.items()}
    else:
        datasets = args.datasets or list(DEV_DATASETS)
        unknown = set(datasets) - set(DEV_DATASETS)
        if unknown:
            raise ValueError(f'not development datasets: {sorted(unknown)}')
        seeds = args.stream_seeds or list(DEV_STREAM_SEEDS)
        if set(seeds) - set(DEV_STREAM_SEEDS):
            raise ValueError('the development stage never reads confirmation seeds')
        scenarios = args.scenarios or PRIMARY_SCENARIOS
        cells = [(d, s, seed, model, str(Path(args.prepared_root) / d / s / f'stream_seed{seed}.npz'))
                 for d in datasets for s in scenarios for seed in seeds for model in DEV_MODEL_SEEDS]
        if args.natural:
            natural = [d for d in (args.datasets or NATURAL_DEV_DATASETS) if d in NATURAL_DEV_DATASETS]
            cells = [(d, NATURAL_SCENARIO, seed, model,
                      str(Path(args.prepared_root) / 'natural' / d / NATURAL_SCENARIO / f'stream_seed{seed}.npz'))
                     for d in natural for seed in seeds for model in DEV_MODEL_SEEDS]
        if args.feature_shift:
            cells = [(d, s, seed, model,
                      str(Path(args.prepared_root) / 'feature_shift' / d / s / f'stream_seed{seed}.npz'))
                     for d in datasets for s in FEATURE_SHIFT_SCENARIOS for seed in seeds
                     for model in DEV_MODEL_SEEDS]
        if args.label_shift:
            cells = [(d, s, seed, model,
                      str(Path(args.prepared_root) / 'label_shift' / d / s / f'stream_seed{seed}.npz'))
                     for d in datasets for s in LABEL_SHIFT_SCENARIOS for seed in seeds
                     for model in DEV_MODEL_SEEDS]
        grids = {method: grid for method, grid in GRIDS.items() if not args.methods or method in args.methods}
    if args.stage == 'screen':
        conditions = [(name, method, overrides) for name, (method, overrides) in SCREEN.items()
                      if not args.methods or name in args.methods]
    else:
        conditions = [(config_id(method, overrides), method, overrides)
                      for method, grid in grids.items() for overrides in grid]
    jobs = []
    for dataset, scenario, stream_seed, model_seed, stream in cells:
        for condition, method, overrides in conditions:
            jobs.append(dict(stage=args.stage, dataset=dataset, scenario=scenario,
                             stream_seed=stream_seed, model_seed=model_seed, method=method,
                             condition=condition, overrides=overrides, stream=stream))
    return jobs


def configure(args, job):
    cfg = copy.deepcopy(load_config(args.config))
    metadata = inspect_prepared(job['stream'])
    cfg.data.source, cfg.data.path = 'prepared_npz', str(Path(job['stream']).resolve())
    cfg.data.scenario = job['scenario']
    cfg.data.random_seed = cfg.data.stream_seed = job['stream_seed']
    cfg.data.n_windows, cfg.data.window_size, cfg.data.n_features = (
        metadata[key] for key in ('n_windows', 'window_size', 'n_features'))
    if args.stage == 'smoke':
        cfg.data.warmup_windows = cfg.drift.calibration_windows = 3
    cfg.training.model_seed, cfg.training.device = job['model_seed'], args.device
    cfg.training.num_threads = args.threads
    cfg.dasp.baseline = job['method']
    cfg.original_review.enabled = False
    cfg.baselines.update_label_budget = 128
    apply_overrides(cfg, job['overrides'])
    cfg.output.directory = str(Path(args.output_root) / job['stage'] / job['dataset'] / job['scenario'])
    cfg.output.run_name = f"{job['condition']}_stream{job['stream_seed']}_model{job['model_seed']}"
    cfg.output.save_checkpoints = False
    return cfg


def run_identity(cfg):
    record = identity(cfg)
    record['runner'] = {'version': VERSION, 'sha256': sha256(ROOT / 'tools' / 'run_dasp_v2.py')}
    return record


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    jobs = jobs_for(args)
    shard = parse_shard(args.shard)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    plan = {'version': VERSION, 'stage': args.stage, 'planned_runs': len(jobs), 'jobs': jobs,
            'grid_sizes': {method: len(grid) for method, grid in GRIDS.items()},
            'label_contract': 'label-blind uniform 128 queries per window for every method',
            'confirmation_data_read': False}
    write_plan(root, args.stage, plan)
    selected = shard_jobs(jobs, shard)
    completed_file = (root / f'{args.stage}_completed.json' if shard is None
                      else root / f'{args.stage}_completed.shard{shard[0]}of{shard[1]}.json')
    print(f'{len(jobs)} planned runs; this process: {len(selected)}', flush=True)
    if args.preflight:
        for stream in sorted({job['stream'] for job in selected}):
            inspect_prepared(stream)
        print('prepared inputs passed preflight', flush=True)
    if not args.execute:
        return
    completed = []
    for index, job in enumerate(selected, 1):
        cfg = configure(args, job)
        destination = Path(cfg.output.directory) / cfg.output.run_name
        provenance = run_identity(cfg)
        if (destination / 'COMPLETE').is_file():
            if not args.resume:
                raise FileExistsError(f'{destination}: completed run exists; pass --resume')
            if json.loads((destination / 'provenance.json').read_text(encoding='utf-8')) != provenance:
                raise ValueError(f'{destination}: code/data/config/environment mismatch')
            receipt = json.loads((destination / 'COMPLETE').read_text(encoding='utf-8'))
            if any(sha256(destination / name) != digest for name, digest in receipt['files'].items()):
                raise ValueError(f'{destination}: output hashes differ from the completion receipt')
            summary = json.loads((destination / 'summary.json').read_text(encoding='utf-8'))
        else:
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(f'{destination}: partial run; preserve it and use a fresh output root')
            result = StreamingExperiment(cfg).run()
            summary = result.summary
            (destination / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
            names = ['windows.csv', 'summary.json', 'config.yaml', 'provenance.json']
            receipt = {'version': VERSION, 'files': {name: sha256(destination / name) for name in names}}
            (destination / 'COMPLETE').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        completed.append({key: value for key, value in job.items()} | {'summary': summary})
        if index % 25 == 0 or index == len(selected):
            temporary = completed_file.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(completed, indent=2, default=str), encoding='utf-8')
            os.replace(temporary, completed_file)
        print(f"[{index}/{len(selected)}] {job['condition']} {job['dataset']}/{job['scenario']}"
              f" seed{job['stream_seed']} F1={summary['mean_macro_f1']:.6f}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--stage', required=True, choices=STAGES)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/modular_dasp_main.yaml')
    p.add_argument('--prepared-root', type=Path, default=Path('prepared_streams_v2'))
    p.add_argument('--output-root', type=Path, default=Path('results/dasp_v2_dev'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--shard', help='K/N: run only paired cells assigned to shard K of N')
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--scenarios', nargs='+', choices=PRIMARY_SCENARIOS)
    p.add_argument('--stream-seeds', nargs='+', type=int)
    p.add_argument('--methods', nargs='+', choices=sorted(set(GRIDS) | set(SCREEN)),
                   help='dev: method names; screen: condition names')
    p.add_argument('--natural', action='store_true',
                   help='use whole-capture chronological development streams instead of controlled scenarios')
    p.add_argument('--feature-shift', action='store_true',
                   help='use feature-group-local drift development streams (FS/FR scenarios)')
    p.add_argument('--label-shift', action='store_true',
                   help='use controlled label-shift development streams (prevalence changes, families fixed)')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
