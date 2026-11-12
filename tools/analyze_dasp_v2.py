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
as boundary-impacting. Every contrast also reports a Bayesian signed-rank
test (Benavoli et al., JMLR 2017) with a region of practical equivalence of
+-ROPE Macro-F1, so ties can be stated as practical equivalence.
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
# Region of practical equivalence for the Bayesian signed-rank test (E15).
ROPE = 0.01
BAYES_SAMPLES = 20000
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
    CHANGE_POINTS[f'CX-{_group}'] = (12,)


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


def bayesian_signed_rank(values: np.ndarray, rope: float | None = None, samples: int = BAYES_SAMPLES,
                         prior: float = 0.5, seed: int = 20261022) -> dict:
    """Bayesian signed-rank test (Benavoli et al. 2017, as in baycomp).

    A Dirichlet process over the differences with ``prior`` weight on a
    pseudo-observation at 0; each posterior draw gives the probability that a
    Walsh average lies left of, inside or right of [-rope, rope]. Reported:
    the share of draws in which each region is the most probable one.
    """

    rope = ROPE if rope is None else rope
    values = np.asarray(values, dtype=float)
    points = np.concatenate(([0.0], values))
    weights = np.random.default_rng(seed).dirichlet(
        np.concatenate(([prior], np.ones(len(values)))), samples)
    sums = points[:, None] + points[None, :]
    regions = [(sums < -2 * rope), (sums >= -2 * rope) & (sums <= 2 * rope), (sums > 2 * rope)]
    probability = np.stack([((weights @ region.astype(float)) * weights).sum(axis=1) for region in regions], axis=1)
    winner = np.bincount(probability.argmax(axis=1), minlength=3) / samples
    return {'rope': rope, 'p_condition_worse': float(winner[0]), 'p_practically_equivalent': float(winner[1]),
            'p_condition_better': float(winner[2])}


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
    if len(values) >= 2:
        result['bayes_signed_rank'] = bayesian_signed_rank(values)
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


def paired_family(frame: pd.DataFrame, pairs: list[tuple[str, str]], column: str = 'mean_macro_f1',
                  replicates: int = 10000, subset_to_condition: bool = False) -> list[dict]:
    """One Holm family of (reference, condition) pairs. ``subset_to_condition``
    keeps only the reference runs whose cell the condition was planned for
    (shortcut sets that a dataset's schema lacks)."""

    frame = frame[frame[column].notna()]
    results = []
    for reference, condition in pairs:
        part = frame
        if subset_to_condition:
            keys = frame.loc[frame.condition == condition, PAIR].drop_duplicates()
            part = frame.merge(keys, on=PAIR)
        results.append({'reference': reference, 'condition': condition, 'metric': column,
                        'positive_difference_means': 'condition better than reference',
                        'all_cells': contrast(cell_differences(part, reference, condition, column), replicates)})
    tested = [(i, r['all_cells']['p_two_sided']) for i, r in enumerate(results)
              if r['all_cells'].get('p_two_sided') is not None]
    for (index, _), value in zip(tested, holm([p for _, p in tested])):
        results[index]['all_cells']['holm_adjusted_p'] = value
    return results


def analysis_frame(records: list[dict]) -> pd.DataFrame:
    """Runs of the sweep and shortcut stages, whose conditions are
    '<method>@<setting>'; natural streams are scored by cumulative Macro-F1."""

    frame = runs_frame(records)
    frame['method_name'], frame['setting'] = zip(*frame.condition.str.split('@', n=1))
    frame['track'] = frame.scenario.map(track_of)
    frame['score'] = np.where(frame.track == 'natural', frame.cumulative_macro_f1, frame.mean_macro_f1)
    return frame


def _load_stage(output_root: Path, stage: str, allow_incomplete: bool) -> list[dict]:
    plan = json.loads((output_root / f'{stage}_plan.json').read_text(encoding='utf-8'))
    records = load_completed(output_root, stage)
    if len(records) != plan['planned_runs'] and not allow_incomplete:
        raise ValueError(f"{stage} stage incomplete: {len(records)}/{plan['planned_runs']}")
    return records


def analyze_sweep(output_root: Path, replicates: int = 10000, allow_incomplete: bool = False) -> dict:
    """E6: does the ranking survive other label budgets, delays and query
    policies, and which policy spends a budget best?"""

    records = _load_stage(output_root, 'sweep', allow_incomplete)
    frame = analysis_frame(records)
    overrides = frame.overrides.map(json.loads)
    frame['budget'] = overrides.map(lambda o: o.get('baselines.update_label_budget', 0))
    frame['delay'] = overrides.map(lambda o: o.get('training.label_delay', -1))
    frame['policy'] = overrides.map(lambda o: o.get('baselines.query_policy', 'none'))
    frame.to_csv(output_root / 'sweep_runs.csv', index=False)
    metrics = ['score', 'cumulative_benign_fpr', 'cumulative_attack_recall', 'gradient_cost', 'total_update_seconds']
    summary = frame.groupby(['track', 'method_name', 'budget', 'delay', 'policy'])[metrics].mean().reset_index()
    summary.to_csv(output_root / 'sweep_summary.csv', index=False)
    contrasts = {}
    for track, part in frame.groupby('track'):
        column = 'cumulative_macro_f1' if track == 'natural' else 'mean_macro_f1'
        present = set(part.condition)
        settings = sorted({c.split('@', 1)[1] for c in present} - {'no_labels'})
        families = {}
        for rival in ('experience_replay', 'arf_adapted', 'online_mlp', 'ssf_budget'):
            pairs = [(f'final@{s}', f'{rival}@{s}') for s in settings
                     if {f'final@{s}', f'{rival}@{s}'} <= present]
            if pairs:
                families[f'{rival}_vs_final_per_setting'] = paired_family(part, pairs, column, replicates)
        for method in sorted(set(part.method_name) - {'static'}):
            pairs = [(f'{method}@{budget}_{delay}_uniform', f'{method}@{s}')
                     for s in settings for budget, delay, policy in [s.split('_', 2)]
                     if policy != 'uniform' and f'{method}@{budget}_{delay}_uniform' in present
                     and f'{method}@{s}' in present]
            if pairs:
                families[f'{method}_policy_vs_uniform'] = paired_family(part, pairs, column, replicates)
        contrasts[track] = families
    result = {'stage': 'sweep', 'runs': len(records), 'rope': ROPE,
              'positive_difference_means': 'condition better than reference', 'contrasts': contrasts,
              'note': 'dev-selected configurations, no re-tuning; every dataset is development data'}
    (output_root / 'sweep_analysis.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    return result


def run_level_family(frame: pd.DataFrame, pairs: list[tuple[str, str]], column: str,
                     replicates: int = 10000, seed: int = 20261109) -> list[dict]:
    """Paired run-level contrasts (pair = dataset, stream seed, model seed).

    Natural streams give one cell per dataset, so cell-level tests have three
    values and can never reach significance; here every paired run counts.
    Wilcoxon signed-rank per dataset and pooled, bootstrap CI of the mean
    difference, Holm across the family (pooled p values)."""

    rng = np.random.default_rng(seed)
    out = []
    for reference, condition in pairs:
        left = frame[frame.condition == reference][[*PAIR, column]]
        right = frame[frame.condition == condition][[*PAIR, column]]
        joined = left.merge(right, on=PAIR, how='outer', suffixes=('_ref', '_cond'), indicator=True)
        if not joined._merge.eq('both').all() or not len(joined):
            raise ValueError(f'incomplete pairs for {reference} vs {condition}; no silent pair dropping')
        joined['difference'] = joined[f'{column}_cond'] - joined[f'{column}_ref']

        def test(values):
            values = values.to_numpy(float)
            boot = rng.choice(values, size=(replicates, len(values))).mean(axis=1)
            p = 1.0 if np.allclose(values, 0) else float(wilcoxon(values).pvalue) if len(values) >= 6 else None
            return {'runs': int(len(values)), 'mean_difference': float(values.mean()),
                    'positive_runs': int((values > 0).sum()), 'ci95_low': float(np.quantile(boot, .025)),
                    'ci95_high': float(np.quantile(boot, .975)), 'p_two_sided': p,
                    'bayes_signed_rank': bayesian_signed_rank(values)}
        out.append({'reference': reference, 'condition': condition, 'metric': column,
                    'positive_difference_means': 'condition better than reference',
                    'pooled': test(joined.difference),
                    'per_dataset': {d: test(part.difference) for d, part in joined.groupby('dataset')}})
    tested = [(i, r['pooled']['p_two_sided']) for i, r in enumerate(out) if r['pooled']['p_two_sided'] is not None]
    for (index, _), value in zip(tested, holm([p for _, p in tested])):
        out[index]['pooled']['holm_adjusted_p'] = value
    for dataset in sorted({d for r in out for d in r['per_dataset']}):
        tested = [(i, r['per_dataset'][dataset]['p_two_sided']) for i, r in enumerate(out)
                  if r['per_dataset'].get(dataset, {}).get('p_two_sided') is not None]
        for (index, _), value in zip(tested, holm([p for _, p in tested])):
            out[index]['per_dataset'][dataset]['holm_adjusted_p'] = value
    return out


def window_confusion(root: Path, record: dict) -> dict[str, float]:
    windows = pd.read_csv(run_directory(root, record) / 'windows.csv')
    windows = windows[~windows.warmup.astype(bool)]
    tp, fn, fp, tn = (float(windows[f'attack_{k}'].sum()) for k in ('tp', 'fn', 'fp', 'tn'))
    return {'cumulative_benign_fpr': fp / (fp + tn) if fp + tn else np.nan,
            'cumulative_attack_recall': tp / (tp + fn) if tp + fn else np.nan}


VERIFY_METRICS = ['cumulative_macro_f1', 'cumulative_benign_fpr', 'cumulative_attack_recall', 'mean_macro_f1',
                  'total_update_seconds']


def confusion_frame(output_root: Path, records: list[dict]) -> pd.DataFrame:
    """Run frame whose cumulative FPR and recall are recounted from the
    post-warm-up confusion counts in windows.csv (not every learner's summary
    carries them)."""
    frame = analysis_frame(records)
    counts = [window_confusion(output_root, record) for record in records]
    keyed = pd.DataFrame([{key: r[key] for key in (*PAIR, 'condition')} | c for r, c in zip(records, counts)])
    frame = frame.drop(columns=['cumulative_benign_fpr', 'cumulative_attack_recall']).merge(keyed, on=[*PAIR, 'condition'])
    overrides = frame.overrides.map(json.loads)
    frame['delay'] = overrides.map(lambda o: o.get('training.label_delay', -1))
    frame['fast_budget'] = overrides.map(lambda o: o.get('baselines.fast_label_budget', 0))
    return frame


def verify_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(['dataset', 'condition', 'delay', 'fast_budget'])[VERIFY_METRICS]
    return grouped.mean().join(grouped.std().add_suffix('_std')).join(grouped.size().rename('runs')).reset_index()


# verify-stage runs that are points of the trade-off curve, under their sweep names
TRADEOFF_ANCHORS = {'er+purity@d5_k16': 'er+purity@d5_k16_r4',
                    'er+representative@d5_k16': 'er+representative@d5_k16_c16',
                    'er+uniform@d5_k16': 'er+uniform@d5_k16', 'er+uniform_knn@d5_k16': 'er+uniform_knn@d5_k16',
                    'er+hierarchical@d5_k16': 'er+hierarchical@d5_k16', 'er@d5': 'er@d5', 'er@d0': 'er@d0'}


def pareto_front(summary: pd.DataFrame) -> pd.Series:
    """Per dataset: True when no other condition has both higher-or-equal F1
    and lower-or-equal FPR (means), with one of them strict."""
    out = pd.Series(False, index=summary.index)
    for _, part in summary.groupby('dataset'):
        f1, fpr = part.cumulative_macro_f1.to_numpy(), part.cumulative_benign_fpr.to_numpy()
        for index, a, b in zip(part.index, f1, fpr):
            out[index] = not ((f1 >= a) & (fpr <= b) & ((f1 > a) | (fpr < b))).any()
    return out


def analyze_tradeoff(output_root: Path, verify_root: Path, replicates: int = 10000,
                     allow_incomplete: bool = False) -> dict:
    """F1/FPR trade-off at d5, k16: purity over its certification size r
    against representative propagation over its number of clusters, joined
    with the matching verify-stage runs. Each representative point is paired
    with the default purity (r = 4) and with the purity point of highest mean
    F1 whose mean FPR is not above it, per dataset."""

    sweep = confusion_frame(output_root, _load_stage(output_root, 'tradeoff', allow_incomplete))
    anchors = confusion_frame(verify_root, _load_stage(verify_root, 'verify', allow_incomplete))
    anchors = anchors[anchors.condition.isin(TRADEOFF_ANCHORS) & anchors.dataset.isin(set(sweep.dataset))].copy()
    anchors['condition'] = anchors.condition.map(TRADEOFF_ANCHORS)
    frame = pd.concat([sweep, anchors], ignore_index=True)
    frame.to_csv(output_root / 'tradeoff_runs.csv', index=False)
    summary = verify_summary(frame)
    summary['pareto'] = pareto_front(summary)
    summary.to_csv(output_root / 'tradeoff_summary.csv', index=False)
    families = {}
    for dataset, part in summary.groupby('dataset'):
        purity = part[part.condition.str.startswith('er+purity@')]
        runs = frame[frame.dataset == dataset]
        pairs, matched = [], {}
        for _, rep in part[part.condition.str.startswith('er+representative@')].iterrows():
            pairs.append(('er+purity@d5_k16_r4', rep.condition))
            eligible = purity[purity.cumulative_benign_fpr <= rep.cumulative_benign_fpr]
            if len(eligible):
                best = eligible.loc[eligible.cumulative_macro_f1.idxmax(), 'condition']
                matched[rep.condition] = best
                if best != 'er+purity@d5_k16_r4':
                    pairs.append((best, rep.condition))
        families[dataset] = {'matched_purity_at_no_higher_fpr': matched,
                             'f1': run_level_family(runs, pairs, 'cumulative_macro_f1', replicates),
                             'fpr': run_level_family(runs, pairs, 'cumulative_benign_fpr', replicates)}
    result = {'stage': 'tradeoff', 'runs': len(frame), 'rope': ROPE,
              'positive_difference_means': 'representative better than purity', 'contrasts': families}
    (output_root / 'tradeoff_analysis.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    return result


BURST_MIN_SUPPORT = 10
BURST_DETECTED_RECALL = 0.5


def _pooled(windows: pd.DataFrame, prefix: str) -> dict[str, float]:
    if f'{prefix}attack_tp' not in windows:
        prefix = ''
    tp, fn, fp, tn = (float(windows[f'{prefix}attack_{k}'].sum()) for k in ('tp', 'fn', 'fp', 'tn'))
    attack_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    benign_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else np.nan
    return {'macro_f1': float(np.nanmean([attack_f1, benign_f1])),
            'benign_fpr': fp / (fp + tn) if fp + tn else np.nan,
            'attack_recall': tp / (tp + fn) if tp + fn else np.nan,
            'false_alarms_per_10k': 1e4 * fp / (tp + fn + fp + tn)}


def burst_metrics(windows: pd.DataFrame, prefix: str = 'uq_') -> dict[str, float]:
    """Attack bursts: maximal runs of consecutive scored windows in which a
    subtype has at least BURST_MIN_SUPPORT flows. A burst is detected in the
    first window whose subtype recall reaches BURST_DETECTED_RECALL; latency
    counts windows from the burst's first window (a missed burst counts its
    length). Scored on unqueried flows when the run logged them."""
    if not any(c.startswith(f'{prefix}subtype_support_') for c in windows):
        prefix = ''
    windows = windows.sort_values('window')
    bursts = []
    for column in [c for c in windows if c.startswith(f'{prefix}subtype_support_')]:
        slug = column[len(f'{prefix}subtype_support_'):]
        support = windows[column].fillna(0).to_numpy()
        recall = windows.get(f'{prefix}subtype_recall_{slug}', pd.Series(np.nan, index=windows.index)).to_numpy()
        index = windows.window.to_numpy()
        present = support >= BURST_MIN_SUPPORT
        start = None
        for i in range(len(present) + 1):
            inside = i < len(present) and present[i] and (start is None or index[i] == index[i - 1] + 1)
            if start is not None and not inside:
                part = recall[start:i]
                hits = np.flatnonzero(part >= BURST_DETECTED_RECALL)
                bursts.append({'length': i - start, 'detected': bool(len(hits)),
                               'latency': int(hits[0]) if len(hits) else i - start,
                               'onset_recall': float(part[0])})
                start = None
            if i < len(present) and present[i] and start is None:
                start = i
    if not bursts:
        return {'bursts': 0, 'burst_detection_rate': np.nan, 'burst_latency_windows': np.nan,
                'burst_onset_recall': np.nan}
    table = pd.DataFrame(bursts)
    return {'bursts': len(table), 'burst_detection_rate': float(table.detected.mean()),
            'burst_latency_windows': float(table.latency.mean()),
            'burst_onset_recall': float(table.onset_recall.mean())}


def verify2_run_metrics(root: Path, record: dict) -> dict[str, float]:
    windows = pd.read_csv(run_directory(root, record) / 'windows.csv')
    windows = windows[~windows.warmup.astype(bool)]
    out = {}
    for prefix, name in (('', 'all'), ('uq_', 'uq'), ('model_uq_', 'model')):
        out |= {f'{name}_{k}': v for k, v in _pooled(windows, prefix).items()}
    out |= burst_metrics(windows)
    out['fast_lane_seconds_per_window'] = float(windows['fast_lane_seconds'].mean()) if 'fast_lane_seconds' in windows else 0.0
    out['fast_groups_per_window'] = float(windows['fast_groups'].mean()) if 'fast_groups' in windows else 0.0
    return out


VERIFY2_BASELINES = ('er@d0', 'er@d1', 'er@d5', 'er_iwms@d1', 'er_iwms@d5', 'er_uncertainty@d1',
                     'er_uncertainty@d5', 'arf@d1', 'arf@d5')
VERIFY2_METRICS = ['uq_macro_f1', 'uq_benign_fpr', 'uq_attack_recall', 'uq_false_alarms_per_10k',
                   'all_macro_f1', 'model_macro_f1', 'burst_detection_rate', 'burst_latency_windows',
                   'burst_onset_recall', 'fast_lane_seconds_per_window', 'fast_groups_per_window']


def verify2_pairs(present: set[str]) -> dict[str, list[tuple[str, str]]]:
    """Contrast families; pairs are (reference, condition), kept when both ran."""
    families = {}
    for d in (1, 5):
        ref = f'er+purity@d{d}_k16'
        rivals = [f'er@d{d}', f'er_iwms@d{d}', f'er_uncertainty@d{d}', f'arf@d{d}', f'er+anomaly@d{d}_k16',
                  *[f'er+{lane}@d{d}_k16' for lane in ('uniform', 'representative', 'uniform_knn', 'hierarchical')]]
        families[f'd{d}_vs_purity'] = [(ref, c) for c in rivals]
    families['prospective'] = [('er@d5', 'er+purity@d5_k16_h1'), ('er@d5', 'er+representative@d5_k16_h1'),
                               ('er@d5', 'er+purity@d5_k16_h2'), ('er@d5', 'er+representative@d5_k16_h2'),
                               ('er@d1', 'er+purity@d1_k16_h1'), ('er@d1', 'er+representative@d1_k16_h1'),
                               ('er@d0', 'er+purity@d5_k16_h1'), ('er@d0', 'er+representative@d5_k16_h1'),
                               ('er+uniform@d5_k16', 'er+purity@d5_k16_h1'),
                               ('er+uniform@d5_k16', 'er+representative@d5_k16_h1')]
    families['learner_arf'] = [('arf+uniform@d5_k16', 'arf+purity@d5_k16'),
                               ('arf+uniform@d5_k16', 'arf+representative@d5_k16'),
                               ('arf@d5', 'arf+purity@d5_k16'), ('arf@d5', 'arf+purity@d5_k16_h1')]
    families['ablation'] = [('er+purity@d5_k16', 'er+purity_nosplit@d5_k16'),
                            ('er+purity@d5_k16', 'er+purity_all@d5_k16'),
                            ('er+purity@d5_k16', 'er+purity_pseudo@d5_k16'),
                            ('er+purity@d5_k16', 'er+purity@d5_k16_noise05'),
                            ('er+purity@d5_k16', 'er+purity@d5_k16_noise10'),
                            ('er+representative@d5_k16', 'er+representative@d5_k16_noise05'),
                            ('er+representative@d5_k16', 'er+representative@d5_k16_noise10')]
    families['budget'] = [(f'er+uniform@d5_k{k}', f'er+purity@d5_k{k}') for k in (4, 8, 16, 32)]
    families['delay'] = [(f'er@d{d}', f'er+{lane}@d{d}_k16') for d in (1, 2, 5, 10)
                         for lane in ('purity', 'representative')]
    return {name: [pair for pair in pairs if set(pair) <= present] for name, pairs in families.items()}


def stream_level(frame: pd.DataFrame, pairs: list[tuple[str, str]], column: str) -> list[dict]:
    """Model seeds averaged within each (dataset, stream): the honest
    independent unit. Reports how many streams favour the condition."""
    means = frame.groupby(['dataset', 'stream_seed', 'condition'])[column].mean().unstack('condition')
    out = []
    for reference, condition in pairs:
        diff = (means[condition] - means[reference]).dropna()
        out.append({'reference': reference, 'condition': condition, 'metric': column, 'streams': int(len(diff)),
                    'condition_better_streams': int((diff > 0).sum()), 'mean_difference': float(diff.mean()),
                    'min_difference': float(diff.min()), 'max_difference': float(diff.max()),
                    'per_dataset_mean': {d: float(part.mean()) for d, part in diff.groupby(level='dataset')}})
    return out


def analyze_verify2(output_root: Path, verify_root: Path, replicates: int = 10000,
                    allow_incomplete: bool = False) -> dict:
    """Review-driven evaluation: fast lanes scored on the flows nobody
    labelled, burst detection latency, the prospective lane, the label-free
    and ARF lanes and the ablations; learners without a fast lane come from
    the verify stage (their predictions never use labels of the scored
    window, so all-flow and unqueried scores coincide)."""

    records = _load_stage(output_root, 'verify2', allow_incomplete)
    base = [r for r in _load_stage(verify_root, 'verify', allow_incomplete) if r['condition'] in VERIFY2_BASELINES]
    rows = []
    for root, group in ((output_root, records), (verify_root, base)):
        for record in group:
            rows.append({key: record[key] for key in (*PAIR, 'condition')} | verify2_run_metrics(root, record))
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / 'verify2_runs.csv', index=False)
    grouped = frame.groupby(['dataset', 'condition'])[VERIFY2_METRICS]
    summary = grouped.mean().join(grouped.std()[['uq_macro_f1', 'uq_benign_fpr']].add_suffix('_std'))
    summary = summary.join(grouped.size().rename('runs')).reset_index()
    summary.to_csv(output_root / 'verify2_summary.csv', index=False)
    families = {}
    for name, pairs in verify2_pairs(set(frame.condition)).items():
        families[name] = {metric: {'run_level': run_level_family(frame, pairs, metric, replicates),
                                   'stream_level': stream_level(frame, pairs, metric)}
                          for metric in ('uq_macro_f1', 'uq_benign_fpr', 'burst_latency_windows')}
    result = {'stage': 'verify2', 'runs': len(frame), 'rope': ROPE,
              'burst_rule': {'min_support': BURST_MIN_SUPPORT, 'detected_recall': BURST_DETECTED_RECALL},
              'positive_difference_means': 'condition minus reference', 'contrasts': families}
    (output_root / 'verify2_analysis.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    return result


def analyze_verify(output_root: Path, replicates: int = 10000, allow_incomplete: bool = False) -> dict:
    """Same-window verification under label delay: the purity-verified fast
    lane against every other learner at the same label delay and total
    label budget, and against the uniform fast lane at each fast budget.
    Natural streams are scored by cumulative Macro-F1."""

    records = _load_stage(output_root, 'verify', allow_incomplete)
    frame = confusion_frame(output_root, records)
    frame.to_csv(output_root / 'verify_runs.csv', index=False)
    verify_summary(frame).to_csv(output_root / 'verify_summary.csv', index=False)
    present = set(frame.condition)
    families = {}
    for delay in sorted(frame.delay.unique()):
        reference = f'er+purity@d{delay}_k16'
        if reference not in present:
            continue
        rivals = sorted(c for c in present if c.endswith(f'@d{delay}_k16') or c.endswith(f'@d{delay}'))
        pairs = [(reference, c) for c in rivals if c != reference]
        reported = set(frame.loc[frame.cumulative_benign_fpr.notna(), 'condition'])
        fpr_pairs = [pair for pair in pairs if set(pair) <= reported]
        families[f'd{delay}_vs_purity'] = {
            'f1': run_level_family(frame, pairs, 'cumulative_macro_f1', replicates),
            'fpr': run_level_family(frame, fpr_pairs, 'cumulative_benign_fpr', replicates),
            'fpr_not_reported': sorted({c for _, c in pairs} - reported)}
    sweep = [(f'er+uniform@d5_k{k}', f'er+purity@d5_k{k}') for k in sorted(frame.fast_budget.unique())
             if {f'er+uniform@d5_k{k}', f'er+purity@d5_k{k}'} <= present]
    if sweep:
        families['purity_vs_uniform_by_fast_budget'] = run_level_family(frame, sweep, 'cumulative_macro_f1', replicates)
    if 'er@d0' in present:
        references = sorted(c for c in present if c.startswith('er+purity@') and c.endswith('_k16'))
        families['purity_vs_immediate_labels'] = run_level_family(
            frame, [(c, 'er@d0') for c in references], 'cumulative_macro_f1', replicates)
    result = {'stage': 'verify', 'runs': len(records), 'rope': ROPE,
              'positive_difference_means': 'condition better than reference', 'contrasts': families}
    (output_root / 'verify_analysis.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    return result


def analyze_shortcut(output_root: Path, replicates: int = 10000, allow_incomplete: bool = False) -> dict:
    """E4: how much each method leans on known shortcut features, and whether
    the method ranking survives their removal."""

    records = _load_stage(output_root, 'shortcut', allow_incomplete)
    frame = analysis_frame(records)
    frame.to_csv(output_root / 'shortcut_runs.csv', index=False)
    summary = frame.pivot_table(index=['track', 'dataset', 'method_name'], columns='setting', values='score',
                                aggfunc='mean')
    summary.to_csv(output_root / 'shortcut_summary.csv')
    contrasts = {}
    for track, part in frame.groupby('track'):
        column = 'cumulative_macro_f1' if track == 'natural' else 'mean_macro_f1'
        present = set(part.condition)
        drops = sorted({c.split('@', 1)[1] for c in present} - {'none'})
        families = {}
        for method in sorted(set(part.method_name)):
            pairs = [(f'{method}@none', f'{method}@{d}') for d in drops if f'{method}@{d}' in present]
            if pairs:
                families[f'{method}_drop_vs_none'] = paired_family(part, pairs, column, replicates,
                                                                   subset_to_condition=True)
        for rival in ('experience_replay', 'arf_adapted'):
            pairs = [(f'final@{d}', f'{rival}@{d}') for d in ['none', *drops]
                     if {f'final@{d}', f'{rival}@{d}'} <= present]
            if pairs:
                families[f'{rival}_vs_final_per_drop'] = paired_family(part, pairs, column, replicates)
        contrasts[track] = families
    result = {'stage': 'shortcut', 'runs': len(records), 'rope': ROPE,
              'positive_difference_means': 'condition better than reference', 'contrasts': contrasts}
    (output_root / 'shortcut_analysis.json').write_text(json.dumps(result, indent=2, default=float),
                                                        encoding='utf-8')
    return result


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


def dataset_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Per configuration and dataset: mean over cells (each cell = seed mean)."""

    cells = frame.groupby(['method', 'condition', 'overrides', 'dataset', 'scenario']).agg(
        f1=('mean_macro_f1', 'mean'), cost=('gradient_cost', 'mean')).reset_index()
    return cells.groupby(['method', 'condition', 'overrides', 'dataset']).agg(
        f1=('f1', 'mean'), cost=('cost', 'mean'), cells=('f1', 'size')).reset_index()


def select_on(means: pd.DataFrame, datasets: list[str]) -> dict[str, dict]:
    """Best configuration per method by the dataset-weighted mean over
    ``datasets`` (each dataset counts once); ties within TIE_MARGIN go to
    fewer gradient evaluations."""

    part = means[means.dataset.isin(datasets)]
    table = part.groupby(['method', 'condition', 'overrides']).agg(
        f1=('f1', 'mean'), cost=('cost', 'mean'), datasets=('dataset', 'nunique')).reset_index()
    if (table.datasets != len(datasets)).any():
        raise ValueError(f'configurations missing datasets among {datasets}')
    selection = {}
    for method, rows in table.groupby('method'):
        best = rows.f1.max()
        tied = rows[rows.f1 >= best - TIE_MARGIN]
        chosen = tied.sort_values(['cost', 'f1', 'condition'], ascending=[True, False, True]).iloc[0]
        selection[method] = {'condition': chosen.condition, 'overrides': json.loads(chosen.overrides),
                             'mean_macro_f1': float(chosen.f1), 'best_mean_macro_f1': float(best),
                             'tied_configurations': int(len(tied)), 'gradient_cost': float(chosen.cost),
                             'datasets': sorted(datasets)}
    return selection


def analyze_lodo(output_root: Path, reference: str = 'modular_dasp_v2', replicates: int = 10000,
                 allow_incomplete: bool = False) -> dict:
    """Leave-one-dataset-out: configurations chosen on four datasets, scored on
    the fifth; the scored runs of all folds form one comparison set."""

    plan = json.loads((output_root / 'lodo_plan.json').read_text(encoding='utf-8'))
    records = load_completed(output_root, 'lodo')
    if len(records) != plan['planned_runs'] and not allow_incomplete:
        raise ValueError(f"lodo stage incomplete: {len(records)}/{plan['planned_runs']}")
    frame = runs_frame(records)
    frame.to_csv(output_root / 'lodo_runs.csv', index=False)
    means = dataset_means(frame)
    datasets = sorted(frame.dataset.unique())
    folds, scored = {}, []
    for held in datasets:
        selection = select_on(means, [d for d in datasets if d != held])
        folds[held] = {method: {'condition': s['condition'], 'overrides': s['overrides'],
                                'train_mean_macro_f1': s['mean_macro_f1']} for method, s in selection.items()}
        for method, chosen in selection.items():
            rows = frame[(frame.condition == chosen['condition']) & (frame.dataset == held)].copy()
            rows['condition'] = method
            scored.append(rows)
    scored = pd.concat(scored, ignore_index=True)
    scored['track'] = scored.scenario.map(track_of)
    scored.to_csv(output_root / 'lodo_scored_runs.csv', index=False)
    cell = scored.groupby(['condition', 'dataset', 'scenario']).mean_macro_f1.mean().reset_index()
    by_dataset = cell.pivot_table(index='condition', columns='dataset', values='mean_macro_f1')
    by_dataset['dataset_weighted_mean'] = by_dataset.mean(axis=1)
    by_track = scored.groupby(['condition', 'track']).mean_macro_f1.mean().unstack()
    costs = scored.groupby('condition')[['gradient_cost', 'cumulative_benign_fpr']].mean()
    summary = by_dataset.join(by_track).join(costs).sort_values('dataset_weighted_mean', ascending=False)
    summary.to_csv(output_root / 'lodo_summary.csv')
    others = [method for method in summary.index if method != reference]
    contrasts = {'all_tracks': compare_family(scored, reference, others, None, replicates)}
    for track, part in scored.groupby('track'):
        contrasts[track] = compare_family(part, reference, others, None, replicates)
    final_selection = select_on(means, datasets)
    (output_root / 'lodo_selection.json').write_text(json.dumps(
        {'stage': 'lodo', 'selection': final_selection,
         'selection_rule': f'dataset-weighted mean Macro-F1 over all {len(datasets)} development datasets; '
                           f'ties within {TIE_MARGIN} -> fewer gradient evaluations',
         'note': 'configuration for the untouched test datasets'}, indent=2), encoding='utf-8')
    result = {'stage': 'lodo', 'complete': len(records) == plan['planned_runs'], 'runs': len(records),
              'reference': reference, 'positive_difference_means': 'condition better than reference',
              'folds': folds, 'summary': json.loads(summary.reset_index().to_json(orient='records')),
              'contrasts': contrasts,
              'note': 'every dataset is development data; the untouched test is a separate stage'}
    (output_root / 'lodo_analysis.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
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
BASELINES = ('online_mlp', 'experience_replay', 'agem', 'online_ewc_v2', 'ssf_budget', 'cdda_md_v2',
             'aoc_ids_adapted', 'cnd_ids_adapted', 'soul_adapted', 'arf_adapted', 'static')
ABLATIONS = ('final_no_realign', 'final_realign_random', 'final_realign_global', 'final_realign_noaccept',
             'final_strict_locality', 'final_route_random', 'final_route_head_only', 'final_route_all',
             'final_contiguous_modules', 'final_no_logit_adjustment', 'final_no_prior_correction')
GROUP_ORDER = ('volume', 'timing', 'tcp', 'protocol')


def track_of(scenario: str) -> str:
    if scenario == 'Natural-Chronological':
        return 'natural'
    if scenario.startswith(('FS-', 'FR-')):
        return 'feature_shift_moderate'
    if scenario.startswith(('XS-', 'XR-')):
        return 'feature_shift_severe'
    if scenario.startswith('CX-'):
        return 'compound'
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


REALIGN_CONDITIONS = ('final', 'final_realign_random', 'final_realign_noaccept', 'final_realign_global')


def realign_table(root: Path, frame: pd.DataFrame) -> list[dict]:
    """Where routed recalibration acted: did its first map hit the drifted
    group (feature-shift tracks), and how often did it act with no
    measurement drift (every other track)."""

    rows = []
    for record in frame[frame.condition.isin(REALIGN_CONDITIONS)].itertuples():
        windows = pd.read_csv(run_directory(root, {'stage': 'final', 'dataset': record.dataset,
                                                   'scenario': record.scenario, 'condition': record.condition,
                                                   'stream_seed': record.stream_seed,
                                                   'model_seed': record.model_seed}) / 'windows.csv')
        if 'realign_accepted' not in windows:
            continue
        acted = windows[windows.realign_accepted.fillna('').astype(str) != '']
        row = {'condition': record.condition, 'dataset': record.dataset, 'track': record.track,
               'interventions': len(acted)}
        if record.track.startswith('feature_shift'):
            target = f"module_{GROUP_ORDER.index(record.scenario.split('-')[1]) + 1}"
            first = acted.realign_accepted.iloc[0].split('|')[0] if len(acted) else None
            row |= {'acted': first is not None, 'first_hits_target': first == target}
        rows.append(row)
    if not rows:
        return []
    table = pd.DataFrame(rows)
    return json.loads(table.groupby(['condition', 'track']).mean(numeric_only=True).reset_index()
                      .to_json(orient='records'))


def final_contrasts(frame: pd.DataFrame, replicates: int = 10000) -> dict:
    contrasts = {}
    for (split, track), part in frame.groupby(['split', 'track']):
        metric = 'cumulative_macro_f1' if track == 'natural' else 'mean_macro_f1'
        present = set(part.condition)
        families = {'baselines': [c for c in BASELINES if c in present],
                    'ablations': [c for c in ABLATIONS if c in present]}
        contrasts[f'{split}/{track}'] = {
            name: compare_family(part, 'final', conditions, None, replicates, metric)
            for name, conditions in families.items() if conditions and 'final' in present}
    return contrasts


def analyze_rope(runs: Path, output: Path, replicates: int = 10000) -> dict:
    """E15: the final-stage contrasts recomputed from final_runs.csv, with the
    Bayesian signed-rank test and ROPE alongside the frequentist ones."""

    frame = pd.read_csv(runs)
    result = {'stage': 'rope', 'source': str(runs), 'rope': ROPE, 'reference': 'final',
              'positive_difference_means': 'condition better than final',
              'contrasts': final_contrasts(frame, replicates)}
    output.write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    return result


def rope_table(result: dict) -> pd.DataFrame:
    rows = []
    for key, families in result['contrasts'].items():
        for family, entries in families.items():
            for entry in entries:
                cells, bayes = entry['all_cells'], entry['all_cells'].get('bayes_signed_rank', {})
                rows.append({'split/track': key, 'family': family, 'condition': entry['condition'],
                             'mean_diff': cells['mean_difference'], 'holm_p': cells.get('holm_adjusted_p'),
                             'P(worse)': bayes.get('p_condition_worse'),
                             'P(rope)': bayes.get('p_practically_equivalent'),
                             'P(better)': bayes.get('p_condition_better')})
    return pd.DataFrame(rows)


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
    contrasts = final_contrasts(frame, replicates)
    locality = frame[frame.condition.isin(['final', 'final_strict_locality'])]
    result = {'stage': 'final', 'complete': len(records) == plan['planned_runs'], 'runs': len(records),
              'reference': 'final', 'positive_difference_means': 'condition better than final',
              'contrasts': contrasts, 'attribution_top1': attribution_table(output_root, frame),
              'realignment': realign_table(output_root, frame),
              'note': 'held-out datasets were not used for any design or tuning decision'}
    if 'optimizer_fraction' in locality:
        result['locality'] = locality.groupby('condition').optimizer_fraction.mean().to_dict()
    (output_root / 'final_analysis.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    return result



# audit stage: certificate validity and the remaining purity ablations
AUDIT_REFERENCE = 'er+purity@d5_k16'
AUDIT_BINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0000001)


def certificate_rows(root: Path, record: dict) -> pd.DataFrame:
    """Every verified cluster of a purity run: size, true benign fraction,
    draws, attack draws and outcome (cert_records in windows.csv)."""
    windows = pd.read_csv(run_directory(root, record) / 'windows.csv')
    windows = windows[~windows.warmup.astype(bool)]
    rows = []
    for window, cell in zip(windows.window, windows.get('cert_records', pd.Series(dtype=object))):
        if isinstance(cell, str):
            rows += [[window, *row] for row in json.loads(cell)]
    frame = pd.DataFrame(rows, columns=['window', 'size', 'benign', 'draws', 'attack_draws', 'outcome'])
    frame['benign_fraction'] = frame.benign / frame['size'].clip(lower=1)
    return frame


def certificate_audit(clusters: pd.DataFrame, epsilon: float, delta: float, r: int) -> dict:
    """Empirical check of the purity certificate. A cluster is tested when at
    least one of its members was drawn and it closed (pass, benign or mixed);
    clusters still open when the budget ran out are censored."""
    tested = clusters[clusters.outcome.isin(['pass', 'benign', 'mixed']) & (clusters.draws > 0)]
    passed = tested[tested.outcome == 'pass']
    impure = tested[tested.benign_fraction > epsilon]
    bins = pd.cut(tested.benign_fraction, AUDIT_BINS, right=False)
    calibration = []
    for interval, part in tested.groupby(bins, observed=True):
        calibration.append({'benign_fraction_bin': f'[{interval.left:.2f}, {min(interval.right, 1):.2f})',
                            'clusters': int(len(part)), 'passed': int((part.outcome == 'pass').sum()),
                            'pass_rate': float((part.outcome == 'pass').mean()),
                            'bound_with_replacement': float(((1 - part.benign_fraction) ** r).mean())})
    return {
        'epsilon': epsilon, 'delta': delta, 'r': r,
        'tested_clusters': int(len(tested)), 'censored_open': int((clusters.outcome == 'open').sum()),
        'passed': int(len(passed)),
        'impure_tested': int(len(impure)),
        'impure_passed': int((impure.outcome == 'pass').sum()),
        # P(pass | benign fraction > epsilon): the certificate promises <= (1 - epsilon)^r <= delta
        'pass_rate_given_impure': float((impure.outcome == 'pass').mean()) if len(impure) else np.nan,
        'promised_bound': float((1 - epsilon) ** r),
        # share of certificates that are wrong, and the benign share of propagated alarms
        'impure_share_of_passed': float((passed.benign_fraction > epsilon).mean()) if len(passed) else np.nan,
        'benign_share_of_certified_flows': float(passed.benign.sum() / max(1, passed['size'].sum())),
        'max_benign_fraction_passed': float(passed.benign_fraction.max()) if len(passed) else np.nan,
        'mean_certified_size': float(passed['size'].mean()) if len(passed) else np.nan,
        'calibration': calibration,
    }


def analyze_audit(output_root: Path, replicates: int = 10000, allow_incomplete: bool = False,
                  epsilon: float = 0.3, delta: float = 0.25, r: int = 4) -> dict:
    records = _load_stage(output_root, 'audit', allow_incomplete)
    rows, clusters = [], []
    for record in records:
        metrics = verify2_run_metrics(output_root, record)
        rows.append({key: record[key] for key in (*PAIR, 'condition')} | metrics)
        part = certificate_rows(output_root, record)
        part[['dataset', 'stream_seed', 'model_seed', 'condition']] = (
            record['dataset'], record['stream_seed'], record['model_seed'], record['condition'])
        clusters.append(part)
    frame = pd.DataFrame(rows)
    clusters = pd.concat(clusters, ignore_index=True)
    frame.to_csv(output_root / 'audit_runs.csv', index=False)
    clusters.to_csv(output_root / 'audit_clusters.csv.gz', index=False)
    grouped = frame.groupby(['dataset', 'condition'])[VERIFY2_METRICS]
    summary = grouped.mean().join(grouped.std()[['uq_macro_f1', 'uq_benign_fpr']].add_suffix('_std'))
    summary = summary.join(grouped.size().rename('runs')).reset_index()
    summary.to_csv(output_root / 'audit_summary.csv', index=False)
    audits = {}
    for (dataset, condition), part in clusters.groupby(['dataset', 'condition']):
        audits.setdefault(condition, {})[dataset] = certificate_audit(part, epsilon, delta, r)
    for condition, part in clusters.groupby('condition'):
        audits[condition]['pooled'] = certificate_audit(part, epsilon, delta, r)
    pairs = [(AUDIT_REFERENCE, c) for c in sorted(set(frame.condition) - {AUDIT_REFERENCE})]
    contrasts = {}
    if len(pairs):
        contrasts = {metric: {'run_level': run_level_family(frame, pairs, metric, replicates),
                              'stream_level': stream_level(frame, pairs, metric)}
                     for metric in ('uq_macro_f1', 'uq_benign_fpr')}
    result = {'stage': 'audit', 'runs': len(frame), 'rope': ROPE, 'reference': AUDIT_REFERENCE,
              'positive_difference_means': 'condition minus reference',
              'certificate': audits, 'contrasts': contrasts}
    (output_root / 'audit_analysis.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stage', choices=('dev', 'lodo', 'screen', 'final', 'sweep', 'shortcut', 'rope', 'verify', 'tradeoff', 'verify2', 'audit'),
                        default='dev')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='stage results; for --stage rope, where rope_analysis.json is written')
    parser.add_argument('--runs', type=Path, help='rope stage: final_runs.csv(.gz) to recompute contrasts from')
    parser.add_argument('--rope', type=float, default=ROPE, help='half-width of the practical-equivalence region')
    parser.add_argument('--prepared-root', type=Path, default=Path('prepared_streams_v2'))
    parser.add_argument('--reference', default='v2_linear')
    parser.add_argument('--verify-root', type=Path, help='tradeoff stage: verify-stage results to join')
    parser.add_argument('--allow-incomplete', action='store_true')
    arguments = parser.parse_args()
    ROPE = arguments.rope
    if arguments.stage == 'rope':
        arguments.output_root.mkdir(parents=True, exist_ok=True)
        report = analyze_rope(arguments.runs or arguments.output_root / 'final_runs.csv',
                              arguments.output_root / 'rope_analysis.json')
        table = rope_table(report)
        table.to_csv(arguments.output_root / 'rope_table.csv', index=False)
        print(table[table.family == 'baselines'].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    elif arguments.stage == 'sweep':
        analyze_sweep(arguments.output_root, allow_incomplete=arguments.allow_incomplete)
        print(pd.read_csv(arguments.output_root / 'sweep_summary.csv').round(4).to_string(index=False))
    elif arguments.stage == 'verify':
        analyze_verify(arguments.output_root, allow_incomplete=arguments.allow_incomplete)
        print(pd.read_csv(arguments.output_root / 'verify_summary.csv').round(4).to_string(index=False))
    elif arguments.stage == 'verify2':
        analyze_verify2(arguments.output_root, arguments.verify_root, allow_incomplete=arguments.allow_incomplete)
        print(pd.read_csv(arguments.output_root / 'verify2_summary.csv').round(4).to_string(index=False))
    elif arguments.stage == 'audit':
        analyze_audit(arguments.output_root, allow_incomplete=arguments.allow_incomplete)
        print(pd.read_csv(arguments.output_root / 'audit_summary.csv').round(4).to_string(index=False))
    elif arguments.stage == 'tradeoff':
        analyze_tradeoff(arguments.output_root, arguments.verify_root, allow_incomplete=arguments.allow_incomplete)
        print(pd.read_csv(arguments.output_root / 'tradeoff_summary.csv').round(4).to_string(index=False))
    elif arguments.stage == 'shortcut':
        analyze_shortcut(arguments.output_root, allow_incomplete=arguments.allow_incomplete)
        print(pd.read_csv(arguments.output_root / 'shortcut_summary.csv').round(4).to_string(index=False))
    elif arguments.stage == 'final':
        report = analyze_final(arguments.output_root, allow_incomplete=arguments.allow_incomplete)
        print(pd.read_csv(arguments.output_root / 'final_summary.csv').round(4).to_string(index=False))
    elif arguments.stage == 'lodo':
        analyze_lodo(arguments.output_root, allow_incomplete=arguments.allow_incomplete)
        print(pd.read_csv(arguments.output_root / 'lodo_summary.csv').round(4).to_string(index=False))
    elif arguments.stage == 'screen':
        report = analyze_screen(arguments.output_root, arguments.prepared_root, arguments.reference)
        print(pd.DataFrame(report['table']).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    else:
        print(json.dumps(analyze_dev(arguments.output_root, arguments.allow_incomplete), indent=2))
