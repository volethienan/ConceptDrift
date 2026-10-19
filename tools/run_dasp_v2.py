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
STAGES = ('dev', 'final', 'screen', 'smoke')


def _grid(**axes):
    names = list(axes)
    return [dict(zip(names, values)) for values in itertools.product(*axes.values())]


# Final method (chosen on development screens): semantic feature-group modules,
# class-conditional pseudo-label drift attribution steering the repair routing
# (top-1 module), class-balanced replay, batch-prior logit adjustment and ACC
# prior correction. STRICT keeps the same method with a blockwise head, so a
# repair changes only the attributed module, its head block and the last layer.
FINAL_METHOD = {
    'model.feature_grouping': 'semantic', 'dasp_v2.logit_adjustment_prior': 'batch',
    'dasp_v2.prior_correction': 'acc', 'dasp_v2.prior_strength': 0.5, 'dasp_v2.routing': 'attributed',
    'dasp_v2.routing_top_k': 1, 'dasp_v2.attribution': 'pseudo', 'dasp_v2.attribution_repr_weight': 0.0}

# Exactly 24 configurations per method, tuned on development data only.
GRIDS = {
    'modular_dasp_v2': [
        FINAL_METHOD | {'dasp_v2.logit_adjustment_tau': tau, 'dasp_v2.prior_strength': strength,
                        'dasp_v2.repair_epochs': epochs, 'training.learning_rate': lr}
        for tau in (0.5, 1.0) for strength in (0.0, 0.5) for epochs in (1, 2) for lr in (5e-4, 1e-3, 2e-3)],
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

# Budget-matched adaptations of published NIDS / streaming methods
# (src/dasp_nids/sota_adapters.py), 24 configurations each. A separate suite so
# the core development plan stays unchanged while it runs.
_CND_THRESHOLDS = ({'sota.cnd_threshold': 'quantile', 'sota.cnd_quantile': 0.95},
                   {'sota.cnd_threshold': 'quantile', 'sota.cnd_quantile': 0.99},
                   {'sota.cnd_threshold': 'labelled_f1'})
SOTA_GRIDS = {
    'aoc_ids_adapted': _grid(**{'training.learning_rate': (3e-3, 1e-2, 3e-2), 'sota.epochs': (1, 2),
                                'sota.aoc_flip_fraction': (0.0, 0.2), 'sota.aoc_temperature': (0.02, 0.1)}),
    'cnd_ids_adapted': [axes | threshold
                        for axes in _grid(**{'training.learning_rate': (5e-4, 1e-3), 'sota.epochs': (1, 3),
                                             'sota.cnd_latent_dim': (16, 30)})
                        for threshold in _CND_THRESHOLDS],
    'soul_adapted': _grid(**{'training.learning_rate': (3e-3, 1e-2, 3e-2), 'sota.epochs': (1, 3),
                             'sota.soul_alpha': (1.0, 9.0), 'sota.soul_gpm_threshold': (0.95, 0.99)}),
    'arf_adapted': _grid(**{'sota.arf_n_models': (5, 10), 'sota.arf_leaf_prediction': ('mc', 'nba'),
                            'sota.arf_grace_period': (50, 200), 'sota.arf_max_features': ('sqrt', '0.3', '0.6')}),
}
SUITES = {'core': GRIDS, 'sota': SOTA_GRIDS, 'all': GRIDS | SOTA_GRIDS}
ALL_GRIDS = SUITES['all']


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
# Round 10: subspace-local head. The first head layer is split into per-module
# column blocks; repairs update the attributed module, its block and the
# final layer; background updates touch only the final layer.
_BLOCK = _SEMANTIC | {'model.modular_head_type': 'blockwise', 'dasp_v2.routing': 'native'}
SCREEN.update({
    'blk_k2': ('modular_dasp_v2', dict(_BLOCK)),
    'blk_k1': ('modular_dasp_v2', _BLOCK | {'dasp_v2.routing_top_k': 1}),
    'blk_random_k1': ('modular_dasp_v2', _BLOCK | {'dasp_v2.routing': 'random', 'dasp_v2.routing_top_k': 1}),
    'blk_d128_f32_k1': ('modular_dasp_v2', _BLOCK | {'dasp_v2.routing_top_k': 1, 'model.module_dim': 128,
                                                     'model.modular_head_dim': 32}),
    'blk_d128_f16_k1': ('modular_dasp_v2', _BLOCK | {'dasp_v2.routing_top_k': 1, 'model.module_dim': 128,
                                                     'model.modular_head_dim': 16}),
    'blk_core_k1': ('modular_dasp_v2', _BLOCK | {'dasp_v2.routing_top_k': 1, 'dasp_v2.head_core_update': True}),
})
# Round 11: class-conditional attribution steering the routing.
_ATTR = _SEMANTIC | {'dasp_v2.routing': 'attributed', 'dasp_v2.routing_top_k': 1, 'dasp_v2.attribution': 'pseudo'}
SCREEN.update({
    'attr_pseudo_k1': ('modular_dasp_v2', dict(_ATTR)),
    'attr_pseudo_only_k1': ('modular_dasp_v2', _ATTR | {'dasp_v2.attribution_repr_weight': 0.0}),
    'attr_labels_k1': ('modular_dasp_v2', _ATTR | {'dasp_v2.attribution': 'labels'}),
    'blk_attr_pseudo_k1': ('modular_dasp_v2', _ATTR | {'model.modular_head_type': 'blockwise'}),
})
FEATURE_SHIFT_SCENARIOS = tuple(f'{kind}-{group}' for kind in ('FS', 'FR', 'XS', 'XR')
                                for group in ('volume', 'timing', 'tcp', 'protocol'))
# Round 8: output-local repairs (additive heads) and inference-time quarantine.
_ADDITIVE = _SEMANTIC | {'model.modular_head_type': 'additive', 'dasp_v2.routing': 'native',
                         'dasp_v2.routing_top_k': 1}
SCREEN.update({
    'add_native': ('modular_dasp_v2', dict(_ADDITIVE)),
    'add_random': ('modular_dasp_v2', _ADDITIVE | {'dasp_v2.routing': 'random'}),
    'add_all': ('modular_dasp_v2', _ADDITIVE | {'dasp_v2.routing': 'all'}),
    'add_bg_all': ('modular_dasp_v2', _ADDITIVE | {'dasp_v2.routing': 'all', 'dasp_v2.background': 'all'}),
    'add_quarantine': ('modular_dasp_v2', _ADDITIVE | {'dasp_v2.quarantine': True,
                                                       'dasp_v2.routing': 'quarantine'}),
    'add_quarantine_random': ('modular_dasp_v2', _ADDITIVE | {'dasp_v2.quarantine': True,
                                                              'dasp_v2.routing': 'random'}),
    'shared_quarantine': ('modular_dasp_v2', _SEMANTIC | {'dasp_v2.quarantine': True,
                                                          'dasp_v2.routing': 'quarantine',
                                                          'dasp_v2.routing_top_k': 1}),
    # Round 9: recalibrated input-space quarantine as a safety net for severe faults only.
    'shared_quarantine_z40': ('modular_dasp_v2', _SEMANTIC | {'dasp_v2.quarantine': True, 'dasp_v2.quarantine_z': 40.0,
                                                              'dasp_v2.routing': 'quarantine', 'dasp_v2.routing_top_k': 1}),
    'shared_quarantine_z80': ('modular_dasp_v2', _SEMANTIC | {'dasp_v2.quarantine': True, 'dasp_v2.quarantine_z': 80.0,
                                                              'dasp_v2.routing': 'quarantine', 'dasp_v2.routing_top_k': 1}),
})
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


DEV_TRACKS = ('controlled', 'feature_shift', 'label_shift')
FEATURE_SHIFT_MODERATE = tuple(f'{kind}-{group}' for kind in ('FS', 'FR')
                               for group in ('volume', 'timing', 'tcp', 'protocol'))
HELDOUT_DATASETS = ('nf-cse-cic-ids2018-v3', 'nf-ton-iot-v3', 'cic-ids2017')
CONFIRMATION_SEEDS = (3001, 3002, 3003, 3004, 3005)
LABEL_SHIFT_SEEDS = (3001, 3002, 3003)
FINAL_MODEL_SEEDS = (91, 92)
# Dev datasets also have confirmation seeds; reported separately as seen-dataset results.
DEV_CONFIRMATION_SEEDS = {'nf-bot-iot-v3': CONFIRMATION_SEEDS, 'nf-unsw-nb15-v3': (3001, 3002)}
NATURAL_HELDOUT = ('nf-cse-cic-ids2018-v3', 'nf-ton-iot-v3')
# Routed recalibration, added after the core development grid was frozen; its
# settings (dasp_v2.realign_* defaults) were fixed on development data only.
FINAL_REALIGN = {'dasp_v2.realign': 'attributed'}
FINAL_ABLATIONS = {
    'final_no_realign': {'dasp_v2.realign': 'none'},
    'final_realign_random': {'dasp_v2.realign': 'random'},
    'final_realign_global': {'dasp_v2.realign': 'global'},
    'final_realign_noaccept': {'dasp_v2.realign_accept': False},
    'final_strict_locality': {'model.modular_head_type': 'blockwise'},
    'final_route_random': {'dasp_v2.routing': 'random'},
    'final_route_head_only': {'dasp_v2.routing': 'classifier'},
    'final_route_all': {'dasp_v2.routing': 'all'},
    'final_contiguous_modules': {'model.feature_grouping': 'contiguous'},
    'final_no_logit_adjustment': {'dasp_v2.logit_adjustment_tau': 0.0},
    'final_no_prior_correction': {'dasp_v2.prior_correction': 'none'},
}


def stream_path(root, dataset, scenario, seed):
    root = Path(root)
    if scenario == NATURAL_SCENARIO:
        return str(root / 'natural' / dataset / scenario / f'stream_seed{seed}.npz')
    if scenario.startswith(('FS-', 'FR-', 'XS-', 'XR-')):
        return str(root / 'feature_shift' / dataset / scenario / f'stream_seed{seed}.npz')
    if scenario.startswith('L') and scenario[1].isdigit():
        return str(root / 'label_shift' / dataset / scenario / f'stream_seed{seed}.npz')
    return str(root / dataset / scenario / f'stream_seed{seed}.npz')


def dev_cells(args, datasets, seeds):
    tracks = args.tracks.split(',')
    scenarios = []
    if 'controlled' in tracks:
        scenarios += list(args.scenarios or PRIMARY_SCENARIOS)
    if 'feature_shift' in tracks:
        scenarios += list(FEATURE_SHIFT_MODERATE)
    if 'label_shift' in tracks:
        scenarios += list(LABEL_SHIFT_SCENARIOS)
    return [(d, s, seed, model, stream_path(args.prepared_root, d, s, seed))
            for d in datasets for s in scenarios for seed in seeds for model in DEV_MODEL_SEEDS]


def final_cells(args):
    cells = []
    scenario_sets = (list(PRIMARY_SCENARIOS) + list(FEATURE_SHIFT_SCENARIOS), CONFIRMATION_SEEDS), \
        (list(LABEL_SHIFT_SCENARIOS), LABEL_SHIFT_SEEDS)
    for dataset in HELDOUT_DATASETS:
        for scenarios, seeds in scenario_sets:
            cells += [(dataset, s, seed, m, stream_path(args.prepared_root, dataset, s, seed))
                      for s in scenarios for seed in seeds for m in FINAL_MODEL_SEEDS]
    for dataset in NATURAL_HELDOUT:
        cells += [(dataset, NATURAL_SCENARIO, seed, m, stream_path(args.prepared_root, dataset, NATURAL_SCENARIO, seed))
                  for seed in LABEL_SHIFT_SEEDS for m in FINAL_MODEL_SEEDS]
    for dataset, seeds in DEV_CONFIRMATION_SEEDS.items():
        cells += [(dataset, s, seed, m, stream_path(args.prepared_root, dataset, s, seed))
                  for s in PRIMARY_SCENARIOS for seed in seeds for m in FINAL_MODEL_SEEDS]
    if args.datasets:
        cells = [cell for cell in cells if cell[0] in args.datasets]
    return cells


def load_selection(paths):
    """Merge dev_selection.json files (core and sota suites); a method may
    appear in one file only."""

    selection = {}
    for path in paths:
        part = json.loads(Path(path).read_text(encoding='utf-8'))['selection']
        overlap = set(part) & set(selection)
        if overlap:
            raise ValueError(f'{path}: methods selected twice: {sorted(overlap)}')
        selection |= part
    return selection


def final_conditions(selection_paths):
    selection = load_selection(selection_paths)
    missing = set(ALL_GRIDS) - set(selection)
    if missing:
        raise ValueError(f'dev selection lacks {sorted(missing)}')
    final = selection['modular_dasp_v2']['overrides'] | FINAL_REALIGN
    conditions = [('final', 'modular_dasp_v2', final), ('static', 'static', {})]
    conditions += [(method, method, selection[method]['overrides'])
                   for method in ALL_GRIDS if method != 'modular_dasp_v2']
    conditions += [(name, 'modular_dasp_v2', final | delta) for name, delta in FINAL_ABLATIONS.items()]
    return conditions


def jobs_for(args):
    if args.stage == 'final':
        if not args.selection:
            raise ValueError('the final stage needs --selection pointing to dev_selection.json')
        cells, conditions = final_cells(args), final_conditions(args.selection)
        if args.methods:
            conditions = [c for c in conditions if c[0] in args.methods]
        return [dict(stage='final', dataset=d, scenario=s, stream_seed=seed, model_seed=m, method=method,
                     condition=name, overrides=overrides, stream=stream)
                for d, s, seed, m, stream in cells for name, method, overrides in conditions]
    if args.stage == 'smoke':
        path = Path(args.output_root) / '_fixtures' / 'synthetic_smoke.npz'
        _smoke_stream(path)
        cells = [('synthetic-smoke', 'S1', 2001, 91, str(path))]
        # The synthetic smoke stream has no documented feature schema, so the
        # smoke check uses contiguous modules; real streams keep semantic ones.
        grids = {method: [grid[0] | ({'model.feature_grouping': 'contiguous'}
                                     if 'model.feature_grouping' in grid[0] else {})]
                 for method, grid in SUITES[args.suite].items()}
    else:
        datasets = args.datasets or list(DEV_DATASETS)
        unknown = set(datasets) - set(DEV_DATASETS)
        if unknown:
            raise ValueError(f'not development datasets: {sorted(unknown)}')
        seeds = args.stream_seeds or list(DEV_STREAM_SEEDS)
        if set(seeds) - set(DEV_STREAM_SEEDS):
            raise ValueError('the development stage never reads confirmation seeds')
        if args.stage == 'dev':
            cells = dev_cells(args, datasets, seeds)
        else:
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
        grids = {method: grid for method, grid in SUITES[args.suite].items()
                 if not args.methods or method in args.methods}
        outside = set(args.methods or ()) - set(SUITES[args.suite])
        if args.stage == 'dev' and outside:
            raise ValueError(f'methods outside the {args.suite} suite: {sorted(outside)}')
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
            'grid_sizes': {method: len(grid) for method, grid in SUITES[args.suite].items()},
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
    p.add_argument('--suite', choices=sorted(SUITES), default='core',
                   help='dev/smoke: core methods (default), sota adapters, or all')
    p.add_argument('--methods', nargs='+', choices=sorted(set(ALL_GRIDS) | set(SCREEN) | set(FINAL_ABLATIONS)
                                                          | {'final', 'static'}),
                   help='dev: method names; screen: condition names; final: condition names')
    p.add_argument('--tracks', default=','.join(DEV_TRACKS),
                   help='dev stage: comma list of controlled, feature_shift (moderate FS/FR), label_shift')
    p.add_argument('--selection', type=Path, nargs='+',
                   help='final stage: dev_selection.json files from analyze_dasp_v2 (core and sota)')
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
