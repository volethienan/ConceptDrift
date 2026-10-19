"""DASP-v2 analysis: development selection and pre-registered statistics.

Selection (pre-registration section 4): for each method, the configuration
with the highest mean of run-level mean-window Macro-F1 over every
development cell; configurations within ``TIE_MARGIN`` of the best are tied
and the one with fewer gradient evaluations wins.

Statistics (section 7): the unit is a dataset x scenario cell averaged over
seeds. Uncertainty comes from a hierarchical bootstrap that resamples
datasets and then scenarios within each drawn dataset; two-sided Wilcoxon
tests over cells; Holm correction within a family. A construct-eligible
sensitivity analysis keeps only drift cells the stream validator classified
as boundary-impacting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_original_review import load_completed  # noqa: E402

TIE_MARGIN = 0.002
# Windows where the attack concept changes (stream builder recipes).
CHANGE_POINTS = {'S1-Sudden': (12,), 'S2-Gradual': (12,), 'S3-Incremental': (12,),
                 'S4-Recurrent': (12, 24), 'S5-Blip': (12, 14), 'S6-Combined': (12, 16, 24, 28),
                 'S7-Local-Low': (12,), 'S7-Local-Medium': (12,), 'S7-Local-High': (12,)}
# First window at which an earlier concept returns.
RECURRENCE_START = {'S4-Recurrent': 24, 'S5-Blip': 14, 'S6-Combined': 28}
RESPONSE_WINDOWS = 4
for _group in ('volume', 'timing', 'tcp', 'protocol'):
    for _persistent, _recurrent in (('FS', 'FR'), ('XS', 'XR')):
        CHANGE_POINTS[f'{_persistent}-{_group}'] = (12,)
        CHANGE_POINTS[f'{_recurrent}-{_group}'] = (12, 24)
        RECURRENCE_START[f'{_recurrent}-{_group}'] = 24


def phase_metrics(windows: pd.DataFrame, scenario: str) -> dict[str, float]:
    """Adaptation speed after concept changes, retention when a concept returns,
    and cumulative operational error rates over the evaluated windows."""

    evaluated = windows[~windows.warmup.astype(bool)] if 'warmup' in windows else windows
    counts = {}
    if {'attack_tp', 'attack_fn', 'attack_fp', 'attack_tn'} <= set(evaluated):
        tp, fn, fp, tn = (int(evaluated[k].sum()) for k in ('attack_tp', 'attack_fn', 'attack_fp', 'attack_tn'))
        counts = {'cumulative_attack_recall': tp / (tp + fn) if tp + fn else np.nan,
                  'cumulative_benign_fpr': fp / (fp + tn) if fp + tn else np.nan}
    f1 = evaluated.set_index('window').macro_f1

    def mean_from(start):
        values = f1.reindex(range(start, start + RESPONSE_WINDOWS)).dropna()
        return float(values.mean()) if len(values) else np.nan

    points = CHANGE_POINTS.get(scenario, ())
    post = [mean_from(point) for point in points]
    if 'optimizer_included_parameter_fraction' in evaluated:
        counts['optimizer_fraction'] = float(evaluated.optimizer_included_parameter_fraction.mean())
    return counts | {'post_change_f1': float(np.nanmean(post)) if post else np.nan,
                     'recurrence_f1': mean_from(RECURRENCE_START[scenario]) if scenario in RECURRENCE_START else np.nan}
CELL = ['dataset', 'scenario']
PAIR = ['dataset', 'scenario', 'stream_seed', 'model_seed']


def run_directory(root: Path, record: dict) -> Path:
    return (Path(root) / record['stage'] / record['dataset'] / record['scenario']
            / f"{record['condition']}_stream{record['stream_seed']}_model{record['model_seed']}")


def runs_frame(records: list[dict], root: Path | None = None) -> pd.DataFrame:
    rows = []
    for record in records:
        summary = record['summary']
        phases = ({} if root is None else
                  phase_metrics(pd.read_csv(run_directory(root, record) / 'windows.csv'), record['scenario']))
        cost = summary.get('total_overall_gradient_evaluations', summary.get('total_gradient_evaluations'))
        rows.append({key: record[key] for key in (*PAIR, 'method', 'condition')}
                    | {'overrides': json.dumps(record['overrides'], sort_keys=True),
                       'mean_macro_f1': float(summary['mean_macro_f1']),
                       'cumulative_macro_f1': float(summary.get('cumulative_macro_f1', np.nan)),
                       'gradient_cost': float(cost) if cost is not None else np.nan,
                       'cumulative_benign_fpr': summary.get('cumulative_benign_fpr', np.nan),
                       'cumulative_attack_recall': summary.get('cumulative_attack_recall', np.nan),
                       'total_update_seconds': summary.get('total_update_seconds', np.nan),
                       'repairs': summary.get('repairs', np.nan)} | phases)
    frame = pd.DataFrame(rows)
    if frame.duplicated([*PAIR, 'condition']).any():
        raise ValueError('duplicate run records')
    return frame


def select_configurations(frame: pd.DataFrame) -> dict[str, dict]:
    expected = frame[PAIR].drop_duplicates().shape[0]
    selection = {}
    for method, part in frame.groupby('method'):
        table = part.groupby(['condition', 'overrides']).agg(
            runs=('mean_macro_f1', 'size'), mean_macro_f1=('mean_macro_f1', 'mean'),
            gradient_cost=('gradient_cost', 'mean')).reset_index()
        incomplete = table[table.runs != expected]
        if len(incomplete):
            raise ValueError(f'{method}: incomplete configurations {incomplete.condition.tolist()}')
        best = table.mean_macro_f1.max()
        tied = table[table.mean_macro_f1 >= best - TIE_MARGIN]
        chosen = tied.sort_values(['gradient_cost', 'mean_macro_f1', 'condition'],
                                  ascending=[True, False, True]).iloc[0]
        selection[method] = {'condition': chosen.condition, 'overrides': json.loads(chosen.overrides),
                             'mean_macro_f1': float(chosen.mean_macro_f1),
                             'best_mean_macro_f1': float(best), 'tied_configurations': int(len(tied)),
                             'gradient_cost': float(chosen.gradient_cost), 'cells': int(expected)}
    return selection


def cell_differences(frame: pd.DataFrame, reference: str, condition: str,
                     column: str = 'mean_macro_f1') -> pd.DataFrame:
    """Paired run differences averaged to dataset x scenario cells."""

    left = frame[frame.condition == reference][[*PAIR, column]]
    right = frame[frame.condition == condition][[*PAIR, column]]
    joined = left.merge(right, on=PAIR, how='outer', suffixes=('_reference', '_condition'), indicator=True)
    if not joined._merge.eq('both').all() or not len(joined):
        raise ValueError(f'incomplete pairs for {reference} vs {condition}; no silent pair dropping')
    joined['difference'] = joined[f'{column}_condition'] - joined[f'{column}_reference']
    return joined.groupby(CELL, as_index=False).difference.mean()


def hierarchical_bootstrap(cells: pd.DataFrame, replicates: int = 10000, seed: int = 20261018) -> np.ndarray:
    """Resample datasets, then scenarios within each drawn dataset."""

    groups = [group.difference.to_numpy(float) for _, group in cells.groupby('dataset')]
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for index in range(replicates):
        drawn = rng.integers(len(groups), size=len(groups))
        values = [groups[g][rng.integers(len(groups[g]), size=len(groups[g]))] for g in drawn]
        means[index] = np.concatenate(values).mean()
    return means


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def contrast(cells: pd.DataFrame, replicates: int = 10000) -> dict:
    values = cells.difference.to_numpy(float)
    result = {'cells': int(len(values)), 'datasets': int(cells.dataset.nunique()),
              'mean_difference': float(values.mean()), 'positive_cells': int((values > 0).sum()),
              'positive_datasets': int((cells.groupby('dataset').difference.mean() > 0).sum()),
              'ci95_low': None, 'ci95_high': None, 'p_two_sided': None}
    if cells.dataset.nunique() >= 2 and len(values) >= 2:
        means = hierarchical_bootstrap(cells, replicates)
        result['ci95_low'], result['ci95_high'] = (float(v) for v in np.quantile(means, [.025, .975]))
        result['p_two_sided'] = 1.0 if np.allclose(values, 0) else float(wilcoxon(values).pvalue)
    else:
        result['limitation'] = 'fewer than two datasets: no inference'
    return result


def construct_eligible_cells(prepared_root: Path) -> set[tuple[str, str]]:
    """Drift cells (S0 excluded) the validator classified as boundary-impacting."""

    eligible = set()
    for report in sorted(Path(prepared_root).glob('validation_*.json')):
        data = json.loads(report.read_text(encoding='utf-8'))
        dataset = report.stem.removeprefix('validation_')
        for row in data['construct_summary']:
            if row['scenario'] != 'S0' and row['claim_eligible']:
                eligible.add((dataset, row['scenario']))
    if not eligible:
        raise ValueError(f'no validator construct reports under {prepared_root}')
    return eligible


def compare_family(frame: pd.DataFrame, reference: str, conditions: list[str],
                   eligible: set[tuple[str, str]] | None = None, replicates: int = 10000,
                   column: str = 'mean_macro_f1') -> list[dict]:
    """One Holm family: every condition against the same reference."""

    frame = frame[frame[column].notna()]
    results = []
    for condition in conditions:
        cells = cell_differences(frame, reference, condition, column)
        entry = {'reference': reference, 'condition': condition, 'metric': column,
                 'positive_difference_means': 'condition better than reference',
                 'all_cells': contrast(cells, replicates)}
        if eligible is not None:
            subset = cells[[tuple(key) in eligible for key in cells[CELL].itertuples(index=False)]]
            entry['construct_eligible_cells'] = (contrast(subset, replicates) if len(subset)
                                                 else {'cells': 0, 'limitation': 'no eligible cells'})
        results.append(entry)
    for key in ('all_cells', 'construct_eligible_cells'):
        tested = [(i, r[key]['p_two_sided']) for i, r in enumerate(results)
                  if key in r and r[key].get('p_two_sided') is not None]
        for (index, _), value in zip(tested, holm([p for _, p in tested])):
            results[index][key]['holm_adjusted_p'] = value
    return results


def analyze_dev(output_root: Path, allow_incomplete: bool = False) -> dict:
    plan = json.loads((output_root / 'dev_plan.json').read_text(encoding='utf-8'))
    records = load_completed(output_root, 'dev')
    if len(records) != plan['planned_runs'] and not allow_incomplete:
        raise ValueError(f"dev stage incomplete: {len(records)}/{plan['planned_runs']}")
    frame = runs_frame(records)
    frame.to_csv(output_root / 'dev_runs.csv', index=False)
    table = frame.groupby(['method', 'condition', 'overrides']).agg(
        runs=('mean_macro_f1', 'size'), mean_macro_f1=('mean_macro_f1', 'mean'),
        gradient_cost=('gradient_cost', 'mean'),
        cumulative_benign_fpr=('cumulative_benign_fpr', 'mean')).reset_index()
    table.sort_values(['method', 'mean_macro_f1'], ascending=[True, False]).to_csv(
        output_root / 'dev_configurations.csv', index=False)
    selection = select_configurations(frame)
    result = {'stage': 'dev', 'complete': len(records) == plan['planned_runs'],
              'selection_rule': f'max mean Macro-F1 over dev cells; ties within {TIE_MARGIN} -> fewer gradient evaluations',
              'selection': selection,
              'note': 'development data only; not evidence for H1-H5'}
    (output_root / 'dev_selection.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def analyze_screen(output_root: Path, prepared_root: Path, reference: str = 'v2_linear',
                   replicates: int = 5000) -> dict:
    """Development screen: where each condition stands, and paired contrasts."""

    records = load_completed(output_root, 'screen')
    plan = json.loads((output_root / 'screen_plan.json').read_text(encoding='utf-8'))
    frame = runs_frame(records, output_root)
    frame.to_csv(output_root / 'screen_runs.csv', index=False)
    frame['rank'] = frame.groupby(PAIR).mean_macro_f1.rank(ascending=False, method='average')
    by_dataset = frame.pivot_table(index='condition', columns='dataset', values='mean_macro_f1', aggfunc='mean')
    table = frame.groupby('condition').agg(
        runs=('mean_macro_f1', 'size'), mean_macro_f1=('mean_macro_f1', 'mean'),
        cumulative_macro_f1=('cumulative_macro_f1', 'mean'), mean_rank=('rank', 'mean'), cumulative_attack_recall=('cumulative_attack_recall', 'mean'),
        cumulative_benign_fpr=('cumulative_benign_fpr', 'mean'), gradient_cost=('gradient_cost', 'mean'),
        update_seconds=('total_update_seconds', 'mean'), repairs=('repairs', 'mean'),
        post_change_f1=('post_change_f1', 'mean'), recurrence_f1=('recurrence_f1', 'mean'))
    table = table.join(by_dataset).sort_values('mean_macro_f1', ascending=False)
    table.to_csv(output_root / 'screen_table.csv')
    frame.pivot_table(index=['dataset', 'scenario'], columns='condition', values='mean_macro_f1',
                      aggfunc='mean').to_csv(output_root / 'screen_by_scenario.csv')
    eligible = {cell for cell in construct_eligible_cells(prepared_root) if cell[0] in set(frame.dataset)}
    others = [condition for condition in table.index if condition != reference]
    metrics = ('mean_macro_f1', 'cumulative_macro_f1', 'post_change_f1', 'recurrence_f1')
    contrasts = {metric: compare_family(frame, reference, others, eligible, replicates, metric)
                 for metric in metrics if frame[metric].notna().any()}
    result = {'stage': 'screen', 'complete': len(records) == plan['planned_runs'], 'runs': len(records),
              'reference': reference, 'construct_eligible_dev_cells': sorted(map(list, eligible)),
              'table': json.loads(table.reset_index().to_json(orient='records')),
              'contrasts': contrasts, 'note': 'development data only; descriptive screen'}
    (output_root / 'screen_analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


HELDOUT = ('nf-cse-cic-ids2018-v3', 'nf-ton-iot-v3', 'cic-ids2017')
BASELINES = ('online_mlp', 'experience_replay', 'agem', 'online_ewc_v2', 'ssf_budget', 'cdda_md_v2', 'static')
ABLATIONS = ('final_strict_locality', 'final_route_random', 'final_route_head_only', 'final_route_all',
             'final_contiguous_modules', 'final_no_logit_adjustment', 'final_no_prior_correction')
GROUP_ORDER = ('volume', 'timing', 'tcp', 'protocol')


def track_of(scenario: str) -> str:
    if scenario == 'Natural-Chronological':
        return 'natural'
    if scenario.startswith(('FS-', 'FR-')):
        return 'feature_shift_moderate'
    if scenario.startswith(('XS-', 'XR-')):
        return 'feature_shift_severe'
    if scenario.startswith('L') and scenario[1].isdigit():
        return 'label_shift'
    return 'attack_family'


def attribution_table(root: Path, frame: pd.DataFrame) -> list[dict]:
    """Top-1 class-conditional attribution of the final method at shift onset."""

    rows = []
    for record in frame[(frame.condition == 'final') & frame.track.str.startswith('feature_shift')].itertuples():
        windows = pd.read_csv(run_directory(root, {'stage': 'final', 'dataset': record.dataset,
                                                   'scenario': record.scenario, 'condition': 'final',
                                                   'stream_seed': record.stream_seed,
                                                   'model_seed': record.model_seed}) / 'windows.csv')
        columns = [f'attr_cc_z_module_{i}' for i in range(1, 5)]
        if not set(columns) <= set(windows):
            continue
        onset = windows[windows.window.between(12, 15)][columns].astype(float).fillna(0.0).to_numpy()
        target = GROUP_ORDER.index(record.scenario.split('-')[1])
        rows.append({'dataset': record.dataset, 'track': record.track, 'group': GROUP_ORDER[target],
                     'top1': float(np.mean(onset.argmax(axis=1) == target))})
    if not rows:
        return []
    table = pd.DataFrame(rows).groupby(['dataset', 'track', 'group']).top1.mean().reset_index()
    return json.loads(table.to_json(orient='records'))


def analyze_final(output_root: Path, replicates: int = 10000, allow_incomplete: bool = False) -> dict:
    records = load_completed(output_root, 'final')
    plan = json.loads((output_root / 'final_plan.json').read_text(encoding='utf-8'))
    if len(records) != plan['planned_runs'] and not allow_incomplete:
        raise ValueError(f"final stage incomplete: {len(records)}/{plan['planned_runs']}")
    frame = runs_frame(records, output_root)
    frame['track'] = frame.scenario.map(track_of)
    frame['split'] = np.where(frame.dataset.isin(HELDOUT), 'held_out', 'dev_dataset_new_seeds')
    frame.to_csv(output_root / 'final_runs.csv', index=False)
    metrics = ['mean_macro_f1', 'cumulative_macro_f1', 'cumulative_attack_recall', 'cumulative_benign_fpr',
               'post_change_f1', 'recurrence_f1', 'gradient_cost']
    summary = frame.groupby(['split', 'track', 'condition'])[metrics].mean().reset_index()
    summary.to_csv(output_root / 'final_summary.csv', index=False)
    frame.pivot_table(index=['split', 'track', 'dataset'], columns='condition', values='mean_macro_f1') \
        .to_csv(output_root / 'final_by_dataset.csv')
    contrasts = {}
    for (split, track), part in frame.groupby(['split', 'track']):
        metric = 'cumulative_macro_f1' if track == 'natural' else 'mean_macro_f1'
        present = set(part.condition)
        families = {'baselines': [c for c in BASELINES if c in present],
                    'ablations': [c for c in ABLATIONS if c in present]}
        contrasts[f'{split}/{track}'] = {
            name: compare_family(part, 'final', conditions, None, replicates, metric)
            for name, conditions in families.items() if conditions and 'final' in present}
    locality = frame[frame.condition.isin(['final', 'final_strict_locality'])]
    result = {'stage': 'final', 'complete': len(records) == plan['planned_runs'], 'runs': len(records),
              'reference': 'final', 'positive_difference_means': 'condition better than final',
              'contrasts': contrasts, 'attribution_top1': attribution_table(output_root, frame),
              'note': 'held-out datasets were not used for any design or tuning decision'}
    if 'optimizer_fraction' in locality:
        result['locality'] = locality.groupby('condition').optimizer_fraction.mean().to_dict()
    (output_root / 'final_analysis.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stage', choices=('dev', 'screen', 'final'), default='dev')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--prepared-root', type=Path, default=Path('prepared_streams_v2'))
    parser.add_argument('--reference', default='v2_linear')
    parser.add_argument('--allow-incomplete', action='store_true')
    arguments = parser.parse_args()
    if arguments.stage == 'final':
        report = analyze_final(arguments.output_root, allow_incomplete=arguments.allow_incomplete)
        print(pd.read_csv(arguments.output_root / 'final_summary.csv').round(4).to_string(index=False))
    elif arguments.stage == 'screen':
        report = analyze_screen(arguments.output_root, arguments.prepared_root, arguments.reference)
        print(pd.DataFrame(report['table']).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    else:
        print(json.dumps(analyze_dev(arguments.output_root, arguments.allow_incomplete), indent=2))
