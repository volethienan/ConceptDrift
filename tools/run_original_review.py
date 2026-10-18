"""Original DASP: unchanged primary grid plus explicitly separate reviewer tests."""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import yaml

from dasp_nids.config import load_config
from dasp_nids.data import make_synthetic_stream
from dasp_nids.original_review import VERSION, identity, sha256
from dasp_nids.runner import StreamingExperiment
from tools.run_modular_dasp_protocol import (DATASETS, PRIMARY_SCENARIOS, METHODS,
    CONFIRMATION_STREAM_SEEDS, CONFIRMATION_MODEL_SEEDS)

ROUTING = ['native', 'random', 'fixed', 'mmd', 'conflict', 'all', 'classifier', 'k1', 'k2']
COMPONENTS = ['native', 'no_replay', 'no_weighting', 'no_gate', 'no_balance', 'all', 'flat_all']
SMOKE = list(dict.fromkeys(ROUTING + COMPONENTS + ['matched', 'gate_legacy']))
STAGES = ('primary', 'routing', 'components', 'gate', 'fairness', 'permutation', 'natural', 'smoke')


@lru_cache(maxsize=1024)
def _inspect_prepared(path_string, size, mtime_ns, chronological):
    path = Path(path_string)
    with np.load(path, allow_pickle=False) as arrays:
        for key in ('x','y','feature_names','class_names','phases'):
            if key not in arrays:
                raise ValueError(f'{path}: missing {key}')
        x, y = arrays['x'], arrays['y']
        if (x.ndim != 3 or y.shape != x.shape[:2] or not np.isfinite(x).all()
                or not np.issubdtype(y.dtype,np.integer) or len(arrays['feature_names']) != x.shape[2]
                or len(arrays['phases']) != len(x)):
            raise ValueError(f'{path}: invalid shapes/features/labels')
        if y.min() < 0 or y.max() >= len(arrays['class_names']):
            raise ValueError(f'{path}: labels outside class domain')
        if len(arrays['class_names']) != 2:
            raise ValueError('original manuscript design is binary; do not silently change task')
        if 'source_ids' in arrays:
            ids = arrays['source_ids']
            if ids.shape != y.shape or np.unique(ids).size != ids.size:
                raise ValueError(f'{path}: duplicate or invalid source IDs across windows')
        if chronological:
            if 'timestamps' not in arrays or 'source_ids' not in arrays:
                raise ValueError('chronological validation requires timestamps and source IDs')
            times = arrays['timestamps']
            if times.shape != y.shape or not np.isfinite(times).all() or np.any(np.diff(times.reshape(-1)) < 0):
                raise ValueError('natural stream must be globally ordered by actual timestamps')
        return {'n_windows':len(x), 'window_size':x.shape[1], 'n_features':x.shape[2],
                'sha256':sha256(path), 'source_ids_present':'source_ids' in arrays,
                'chronological_checked':chronological}


def inspect_prepared(path, chronological=False):
    path = Path(path).resolve()
    stat = path.stat()
    return _inspect_prepared(str(path), stat.st_size, stat.st_mtime_ns, chronological)


def flat_widths(n_features, target):
    # OnlineMLP(d,[a,b],2) has a*(d+3+b)+5*b+2 parameters.
    pairs = [(a,b) for a in range(4,513) for b in range(4,257)
             if a*(n_features+3+b)+5*b+2 == target]
    if not pairs:
        raise ValueError('no parameter-matched flat architecture in declared search range')
    return min(pairs,key=lambda p:(abs(p[0]-128)+abs(p[1]-64),p))


def configure(args, job):
    cfg = copy.deepcopy(load_config(args.config))
    metadata = inspect_prepared(job['stream'],job['stage']=='natural')
    cfg.data.source, cfg.data.path = 'prepared_npz',str(Path(job['stream']).resolve())
    cfg.data.scenario = job['scenario']
    cfg.data.random_seed = cfg.data.stream_seed = job['stream_seed']
    cfg.data.n_windows, cfg.data.window_size, cfg.data.n_features = (
        metadata[k] for k in ('n_windows','window_size','n_features'))
    cfg.training.model_seed, cfg.training.device = job['model_seed'],args.device
    if args.stage == 'smoke':
        cfg.data.warmup_windows = 3
        cfg.drift.calibration_windows = 3
    if cfg.data.warmup_windows >= cfg.data.n_windows:
        raise ValueError('warmup must leave evaluation windows')
    cfg.dasp.baseline = job['method']
    cfg.original_review.enabled = job['method']=='modular_dasp'
    cfg.output.directory = str(Path(args.output_root)/job['stage']/job['dataset']/job['scenario'])
    cfg.output.run_name = f"{job['condition']}_stream{job['stream_seed']}_model{job['model_seed']}"
    cfg.output.save_checkpoints = args.checkpoints
    cfg.original_review.verify_locality = not args.no_locality_check
    condition = job['condition']
    if condition in ('random','fixed','mmd','conflict','all','classifier'):
        cfg.original_review.routing = condition
    if condition in ('k1','k2'):
        cfg.original_review.routing_k = int(condition[1:])
    if condition == 'no_replay': cfg.original_review.replay_enabled = False
    if condition == 'no_weighting': cfg.original_review.weighting_enabled = False
    if condition == 'no_gate': cfg.original_review.performance_gate_enabled = False
    if condition == 'no_balance': cfg.original_review.balanced_selection = False
    if condition == 'matched': cfg.original_review.label_mode = 'matched'
    if condition == 'gate_legacy': cfg.original_review.gate_policy = 'legacy_fixed_macro'
    if condition == 'flat_all':
        modular = StreamingExperiment(cfg)._build_model(metadata['n_features'],2)
        target = sum(p.numel() for p in modular.parameters())
        cfg.model.architecture = 'sequential'
        cfg.model.hidden_dims = list(flat_widths(metadata['n_features'],target))
        cfg.original_review.routing = 'all'
    if job.get('permutation_seed') is not None:
        cfg.revision.dataset_key = job['dataset']
        cfg.revision.feature_permutation_seed = job['permutation_seed']
    if job.get('event_schedule'):
        cfg.original_review.event_schedule = job['event_schedule']
    return cfg


def _smoke_stream(path):
    if path.exists(): return
    stream = make_synthetic_stream('S1',8,128,12,2001,'binary')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,x=np.stack([w.x.numpy() for w in stream.windows]),
        y=np.stack([w.y.numpy() for w in stream.windows]),
        feature_names=np.array(stream.feature_names),class_names=np.array(stream.class_names),
        phases=np.array([w.phase for w in stream.windows]),
        source_ids=np.arange(8*128).reshape(8,128))


def jobs_for(args):
    jobs = []
    stage = args.stage
    datasets = args.datasets or DATASETS
    scenarios = args.scenarios or PRIMARY_SCENARIOS
    streams = args.stream_seeds or CONFIRMATION_STREAM_SEEDS
    models = args.model_seeds or CONFIRMATION_MODEL_SEEDS
    naturals = {}
    if stage == 'natural':
        for entry in args.natural_stream:
            dataset, separator, path = entry.partition('=')
            if not separator or not dataset or dataset in naturals:
                raise ValueError('--natural-stream requires unique dataset=path entries')
            naturals[dataset] = path
        if not naturals:
            raise ValueError('supply --natural-stream dataset=/path/to/chronological.npz')
        datasets,scenarios = list(naturals),['Natural-Chronological']
        if args.stream_seeds is None:
            streams = [4101]
    if stage == 'smoke':
        path = Path(args.output_root)/'_fixtures'/'synthetic_smoke.npz'
        _smoke_stream(path)
        datasets,scenarios,streams,models = ['synthetic-smoke'],['S1'],[2001],[91]
    for dataset in datasets:
        for scenario in scenarios:
            for stream_seed in streams:
                for model_seed in models:
                    stream = (str(path) if stage=='smoke' else naturals[dataset] if stage=='natural'
                        else str(Path(args.prepared_root)/dataset/scenario/f'stream_seed{stream_seed}.npz'))
                    if stage in ('primary','natural'):
                        conditions = [('native' if m=='modular_dasp' else m,m,None) for m in METHODS]
                    elif stage in ('routing','components','smoke'):
                        selected = ROUTING if stage=='routing' else COMPONENTS if stage=='components' else SMOKE
                        conditions = [(c,'modular_dasp',None) for c in selected]
                        if stage=='smoke' and args.include_baselines:
                            conditions += [(m,m,None) for m in METHODS if m!='modular_dasp']
                    elif stage=='fairness':
                        conditions = [('matched','modular_dasp',None),('ssf','ssf',None)]
                    elif stage=='gate':
                        conditions = [('native','modular_dasp',None),('gate_legacy','modular_dasp',None)]
                    else:
                        conditions = [(f'permutation_{p}','modular_dasp',p) for p in (None,701,702,703)]
                    for condition,method,permutation in conditions:
                        job = dict(stage=stage,dataset=dataset,scenario=scenario,stream_seed=stream_seed,
                            model_seed=model_seed,condition=condition,method=method,stream=stream,
                            permutation_seed=permutation)
                        if stage in ('routing','components','smoke') and method=='modular_dasp' and condition not in ('native','no_gate','matched','gate_legacy'):
                            job['event_schedule'] = str(Path(args.output_root)/stage/dataset/scenario/
                                f'native_stream{stream_seed}_model{model_seed}'/'event_schedule.json')
                        jobs.append(job)
    return jobs


def _write_schedule(path,frame,cfg):
    events = {}
    for _,row in frame[~frame.warmup.astype(bool)].iterrows():
        events[str(int(row['window']))] = dict(epochs=int(row.original_epochs),
            plasticities=json.loads(row.original_native_plasticities),candidate=row.selected_candidate)
    path.write_text(json.dumps({'scope':'offline event-matched diagnostic, not deployable oracle',
        'version':VERSION,'reference_signature':{'stream_sha256':sha256(cfg.data.path),
            'stream_seed':cfg.data.random_seed,'model_seed':cfg.training.model_seed},'events':events},indent=2),encoding='utf-8')


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.device.startswith('cuda') and args.execute and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install a matching CUDA-enabled torch build')
    jobs = jobs_for(args)
    root = Path(args.output_root)
    root.mkdir(parents=True,exist_ok=True)
    plan = {'version':VERSION, 'stage':args.stage, 'planned_runs':len(jobs),
        'primary_grid_unchanged':{'datasets':DATASETS,'scenarios':PRIMARY_SCENARIOS,'methods':METHODS,
            'stream_seeds':CONFIRMATION_STREAM_SEEDS,'model_seeds':CONFIRMATION_MODEL_SEEDS,'runs':4000},
        'jobs':jobs, 'python':platform.python_version(), 'torch':str(torch.__version__),
        'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'counts_are_not_evidence_of_superiority':True}
    plan_file = root/f'{args.stage}_plan.json'
    if plan_file.is_file() and (root/f'{args.stage}_completed.json').is_file():
        old = json.loads(plan_file.read_text(encoding='utf-8'))
        if old['jobs'] != jobs:
            raise ValueError('cannot change a plan with existing completed runs; use a fresh output root')
    plan_file.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(f'{len(jobs)} planned runs; {plan_file}',flush=True)
    if args.preflight:
        artifacts = {}
        for job in jobs:
            if job['stream'] not in artifacts:
                artifacts[job['stream']] = inspect_prepared(job['stream'],job['stage']=='natural')
        if args.stage=='natural' and len({v['sha256'] for v in artifacts.values()}) != len(artifacts):
            raise ValueError('the same capture cannot count as multiple natural datasets')
        (root/f'{args.stage}_preflight.json').write_text(json.dumps(artifacts,indent=2),encoding='utf-8')
        print(f'{len(artifacts)} prepared inputs passed preflight',flush=True)
    if not args.execute: return
    completed = []
    for index,job in enumerate(jobs,1):
        cfg = configure(args,job)
        destination = Path(cfg.output.directory)/cfg.output.run_name
        provenance = identity(cfg)
        summary_path, provenance_path = destination/'summary.json',destination/'provenance.json'
        if summary_path.exists():
            if not args.resume or not provenance_path.is_file():
                raise FileExistsError(f'{destination}: use --resume only for a matching completed run')
            if json.loads(provenance_path.read_text(encoding='utf-8')) != provenance:
                raise ValueError(f'{destination}: code/data/config/environment mismatch')
            if not (destination/'windows.csv').is_file() or not (destination/'COMPLETE').is_file():
                raise ValueError(f'{destination}: incomplete cache, use a fresh output root')
            receipt = json.loads((destination/'COMPLETE').read_text(encoding='utf-8'))
            if any(not (destination/name).is_file() or sha256(destination/name)!=digest
                   for name,digest in receipt['files'].items()):
                raise ValueError('cached output hashes differ from completed receipt')
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        else:
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError('partial run exists; preserve it and use a fresh output root')
            result = StreamingExperiment(cfg).run()
            summary = result.summary
            provenance_path.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
            if job['condition']=='native' and job['method']=='modular_dasp':
                _write_schedule(destination/'event_schedule.json',result.windows,cfg)
            names = ['windows.csv','summary.json','config.yaml','provenance.json']
            names += [n for n in ('event_schedule.json','checkpoint.pt') if (destination/n).is_file()]
            receipt = {'version':VERSION,'files':{name:sha256(destination/name) for name in names}}
            (destination/'COMPLETE').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        completed.append({**job,**summary})
        # Avoid rewriting a growing 4000-run aggregate on every single cell.
        # Receipts preserve every completed run; resume rebuilds this index.
        if index % 20 == 0 or index == len(jobs):
            temporary = root/f'{args.stage}_completed.tmp'
            temporary.write_text(json.dumps(completed,indent=2),encoding='utf-8')
            os.replace(temporary,root/f'{args.stage}_completed.json')
        print(f"[{index}/{len(jobs)}] {job['condition']} {job['dataset']}/{job['scenario']} F1={summary['mean_macro_f1']:.6f}",flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=STAGES)
    p.add_argument('--config',type=Path,default=ROOT/'configs/modular_dasp_main.yaml')
    p.add_argument('--prepared-root',type=Path,default=Path('prepared_streams_final'))
    p.add_argument('--output-root',type=Path,default=Path('outputs_original_review_v1'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--datasets',nargs='+',choices=DATASETS)
    p.add_argument('--scenarios',nargs='+',choices=PRIMARY_SCENARIOS)
    p.add_argument('--stream-seeds',nargs='+',type=int)
    p.add_argument('--model-seeds',nargs='+',type=int)
    p.add_argument('--natural-stream',action='append',default=[])
    p.add_argument('--execute',action='store_true',help='otherwise write the reviewable plan only')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoints',action='store_true')
    p.add_argument('--preflight',action='store_true',help='validate all distinct prepared inputs before training')
    p.add_argument('--no-locality-check',action='store_true',help='separate timing-only run, disables parameter snapshot diagnostics')
    p.add_argument('--include-baselines',action='store_true',help='smoke also exercises all seven original comparators')
    return p


if __name__=='__main__':
    run(parser().parse_args())
