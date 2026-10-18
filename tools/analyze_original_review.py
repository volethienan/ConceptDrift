"""Strict paired, cluster-aware analysis for the separate original-DASP stages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ['dataset','scenario','stream_seed','model_seed']


def operational_metrics(frame):
    frame = frame[~frame.warmup.astype(bool)]
    tp,fn,fp,tn = [int(frame[k].sum()) for k in ('attack_tp','attack_fn','attack_fp','attack_tn')]
    benign = frame.attack_fp+frame.attack_tn
    per_window = frame.loc[benign>0,'attack_fp']/benign[benign>0]
    attack_f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    benign_f1 = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    return {'tp':tp,'fn':fn,'fp':fp,'tn':tn,
        'cumulative_attack_recall':tp/(tp+fn) if tp+fn else np.nan,
        'cumulative_fnr':fn/(tp+fn) if tp+fn else np.nan,
        'cumulative_benign_fpr':fp/(fp+tn) if fp+tn else np.nan,
        'mean_window_benign_fpr':float(per_window.mean()) if len(per_window) else np.nan,
        'cumulative_binary_macro_f1_fixed_classes':(attack_f1+benign_f1)/2}


def paired(frame,reference,condition):
    left = frame[frame.condition==reference]
    right = frame[frame.condition==condition]
    for side in (left,right):
        if side.duplicated(KEYS).any():
            raise ValueError('duplicate paired cells')
    joined = left.merge(right,on=KEYS,suffixes=('_reference','_condition'),how='outer',indicator=True,validate='one_to_one')
    if not joined._merge.eq('both').all():
        raise ValueError(f'incomplete pairs for {reference} vs {condition}; no silent pair dropping')
    if not len(joined):
        raise ValueError('no pairs')
    joined['difference'] = joined.mean_macro_f1_condition-joined.mean_macro_f1_reference
    return joined


def cluster_statistics(pairs,columns,replicates=10000):
    values = pairs.groupby(columns,dropna=False).difference.mean().to_numpy(float)
    result = {'clusters':len(values),'cluster_columns':columns,'mean_difference':float(values.mean()),
        'paired_cells':len(pairs),'n_positive_clusters':int((values>0).sum()),
        'p_two_sided':None,'bootstrap_ci95_low':None,'bootstrap_ci95_high':None}
    if len(values)<2:
        result['limitation'] = 'insufficient independent clusters for inference'
        return result
    rng = np.random.default_rng(20261018)
    means = rng.choice(values,size=(replicates,len(values)),replace=True).mean(axis=1)
    result['bootstrap_ci95_low'],result['bootstrap_ci95_high'] = [float(v) for v in np.quantile(means,[.025,.975])]
    result['p_two_sided'] = 1. if np.allclose(values,0.) else float(wilcoxon(values,alternative='two-sided').pvalue)
    return result


def holm(contrasts,key):
    finite = [(index,item[key]['p_two_sided']) for index,item in enumerate(contrasts)
              if item[key]['p_two_sided'] is not None and np.isfinite(item[key]['p_two_sided'])]
    running = 0.
    for rank,(index,value) in enumerate(sorted(finite,key=lambda pair:pair[1])):
        running = max(running,min(1.,(len(finite)-rank)*value))
        contrasts[index][key]['holm_adjusted_p'] = running


def load_completed(root,stage):
    """Merge the single-process index or every shard index, never both."""
    single = root/f'{stage}_completed.json'
    shards = sorted(root.glob(f'{stage}_completed.shard*of*.json'))
    if single.is_file() and shards:
        raise ValueError('both unsharded and sharded completion indexes exist; keep one layout per output root')
    if not shards:
        return json.loads(single.read_text(encoding='utf-8'))
    counts = {path.name.rsplit('of',1)[1].split('.')[0] for path in shards}
    if len(counts) != 1:
        raise ValueError(f'shard indexes from different shard counts: {sorted(counts)}')
    expected = int(counts.pop())
    if len(shards) != expected:
        raise ValueError(f'{len(shards)}/{expected} shard indexes present')
    return [record for path in shards for record in json.loads(path.read_text(encoding='utf-8'))]


def analyze(root,stage,allow_incomplete=False):
    root = Path(root)
    plan = json.loads((root/f'{stage}_plan.json').read_text(encoding='utf-8'))
    completed = load_completed(root,stage)
    complete = len(completed)==plan['planned_runs']
    if not complete and not allow_incomplete:
        raise ValueError(f"stage incomplete: {len(completed)}/{plan['planned_runs']}")
    frame = pd.DataFrame(completed)
    if frame.duplicated(KEYS+['condition']).any():
        raise ValueError('duplicate run records')
    for index,row in frame.iterrows():
        directory = root/stage/row.dataset/row.scenario/f'{row.condition}_stream{row.stream_seed}_model{row.model_seed}'
        values = operational_metrics(pd.read_csv(directory/'windows.csv'))
        for name,value in values.items(): frame.loc[index,name] = value
        frame.loc[index,'requested_gradient_elements_comparable'] = row.get('total_parameter_gradient_element_evaluations') if row.method=='modular_dasp' else np.nan
    frame.to_csv(root/f'{stage}_operational_per_run.csv',index=False)
    reference = 'matched' if stage=='fairness' else 'permutation_None' if stage=='permutation' else 'native'
    contrasts = []
    if reference in set(frame.condition):
        for condition in sorted(set(frame.condition)-{reference}):
            pairs = paired(frame,reference,condition)
            pairs.to_csv(root/f'{stage}_paired_{reference}_vs_{condition}.csv',index=False)
            contrasts.append({'reference':reference,'condition':condition,
                'positive_difference_means':'condition better than reference',
                'dataset_scenario':cluster_statistics(pairs,['dataset','scenario']),
                'dataset_only_sensitivity':cluster_statistics(pairs,['dataset'])})
    for key in ('dataset_scenario','dataset_only_sensitivity'): holm(contrasts,key)
    results = {'stage':stage,'complete':complete,'runs':len(frame),'planned_runs':plan['planned_runs'],
        'diagnostic_only':stage=='smoke' or not complete,
        'assumptions':'dataset-scenario clusters share datasets; report dataset-only sensitivity and limited environments',
        'contrasts':contrasts,
        'estimands':'cumulative within run and mean-window rates separate; no pooled cross-dataset FPR headline',
        'forgetting_definition':'legacy running-best window F1 drop; not fixed-task catastrophic forgetting',
        'native_sota_reproduction_claimed':False,
        'no_efficiency_ratio_from_mixed_gradient_counter_scopes':True}
    (root/f'{stage}_analysis.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(f'{root}/{stage}_analysis.json: {len(contrasts)} contrasts; complete={complete}',flush=True)
    return results


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--allow-incomplete',action='store_true')
    args = parser.parse_args()
    analyze(args.output_root,args.stage,args.allow_incomplete)
