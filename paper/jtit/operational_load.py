"""Operational load at real traffic volume and per-family recall (review R6, R3/W12).

    python paper/jtit/operational_load.py [--results results] [--prepared ../../prepared_streams_v2]

Each stream window is a 4,096-flow sample of a capture segment with a known
number of flows and (except CIC-IDS2017) a known time span. A window's false
positive rate on its unqueried benign flows is extrapolated to the benign
flows of its whole segment, which gives false alarms per hour of capture and
per 8-hour analyst shift, converted with the median window span because
captures contain idle gaps. Certified or representative clusters raised per
window are counted as incident-level alerts. CIC-IDS2017 has no timestamps
in its feature files; its capture covers five working days of about 8 hours,
so each of its 36 windows is taken to span 40/36 hours.

Writes paper/jtit/tables/{operations,families}.tex and
results_snapshot/revision_20261113/operational_load.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASETS = {'nf-unsw-nb15-v3': ('NF-UNSW', 2001), 'nf-cse-cic-ids2018-v3': ('NF-CSE', 3001),
            'nf-ton-iot-v3': ('NF-ToN', 3001), 'cic-ids2017': ('CIC-IDS2017', 4001), 'cidds-001': ('CIDDS-001', 5001)}
CIC_HOURS = 40.0
PRETTY = {'dos': 'DoS', 'ddos': 'DDoS', 'portscan': 'PortScan', 'ftp': 'FTP', 'ssh': 'SSH', 'xss': 'XSS',
          'goldeneye': 'GoldenEye', 'slowhttptest': 'Slowhttptest', 'attack': 'attack', 'brute': 'brute', 'force': 'force'}
SCENARIO = 'Natural-Chronological'
CONDITIONS = {'er@d5': 'ER, $d=5$', 'er@d0': 'ER, labels without delay',
              'er+uniform@d5_k16': 'Uniform fast labels', 'er+uniform_knn@d5_k16': 'Uniform + kNN',
              'er+hierarchical@d5_k16': 'Hierarchical sampling', 'er+representative@d5_k16': 'Representative',
              'er+purity@d5_k16': 'Purity-certified'}


def run_roots(results: Path, dataset: str, condition: str) -> list[Path]:
    """Stage folders that hold this condition (fast lanes: verify2; learners: verify / heldout)."""
    held = dataset in ('cic-ids2017', 'cidds-001')
    suffix = {'cic-ids2017': '', 'cidds-001': '_cidds'}.get(dataset, '')
    if '+' in condition:
        base = results / (f'verify2_heldout{suffix}' if held else 'verify2') / 'verify2'
    else:
        base = results / (f'heldout{suffix}' if held else 'verify') / 'verify'
    return [base / dataset / SCENARIO]


def segments(prepared: Path, dataset: str, seed: int) -> pd.DataFrame:
    manifest = json.loads((prepared / 'natural' / dataset / SCENARIO / f'stream_seed{seed}.manifest.json')
                          .read_text(encoding='utf-8'))
    rows = []
    for seg in manifest['segments']:
        flows = seg.get('segment_flows', seg.get('segment_rows'))
        hours = (seg['end_ms'] - seg['start_ms']) / 3.6e6 if 'end_ms' in seg else CIC_HOURS / len(manifest['segments'])
        rows.append({'window': seg['window'], 'flows': flows, 'benign': flows * (1 - seg['segment_attack_rate']),
                     'hours': hours})
    return pd.DataFrame(rows)


def load(results: Path, prepared: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, fam = [], []
    for dataset, (_, first_seed) in DATASETS.items():
        for condition in CONDITIONS:
            for root in run_roots(results, dataset, condition):
                for run in sorted(root.glob(f'{condition}_stream*_model*')):
                    if not (run / 'COMPLETE').is_file():
                        continue
                    seed = int(run.name.split('_stream')[1].split('_model')[0])
                    seg = segments(prepared, dataset, seed)
                    w = pd.read_csv(run / 'windows.csv')
                    w = w[~w.warmup.astype(bool)].merge(seg, on='window', how='left')
                    p = 'uq_' if 'uq_attack_fp' in w else ''
                    fpr_w = w[f'{p}attack_fp'] / (w[f'{p}attack_fp'] + w[f'{p}attack_tn']).clip(lower=1)
                    # captures have idle gaps (nights, days off), so a window's span can be
                    # far longer than its traffic; rates use the median window span
                    hours = float(w.hours.median())
                    groups = w.get('fast_groups', pd.Series(0, index=w.index))
                    rows.append({'dataset': dataset, 'condition': condition, 'run': run.name,
                                 'hours_per_window': hours, 'flows_per_window': float(w.flows.median()),
                                 'false_alarms_per_window': float((fpr_w * w.benign).mean()),
                                 'false_alarms_per_shift': float((fpr_w * w.benign).mean() / hours * 8),
                                 'incident_alerts_per_shift': float(groups.mean() / hours * 8),
                                 'verifications_per_hour': (16 / hours) if '+' in condition else 0.0})
                    for col in [c for c in w if c.startswith(f'{p}subtype_support_')]:
                        slug = col[len(f'{p}subtype_support_'):]
                        support = w[col].fillna(0)
                        recall = w.get(f'{p}subtype_recall_{slug}', pd.Series(np.nan, index=w.index)).fillna(0)
                        fam.append({'dataset': dataset, 'condition': condition, 'run': run.name, 'family': slug,
                                    'support': float(support.sum()), 'detected': float((support * recall).sum())})
    return pd.DataFrame(rows), pd.DataFrame(fam)


def fmt(v: float) -> str:
    return f'{v:,.0f}' if v >= 10 else f'{v:.1f}'


def write_tables(ops: pd.DataFrame, fam: pd.DataFrame) -> None:
    tab = HERE / 'tables'
    mean = ops.groupby(['dataset', 'condition']).mean(numeric_only=True)
    lines = []
    hours = ' & '.join(f"{mean.loc[(d, 'er+purity@d5_k16'), 'hours_per_window']:.2f}" for d in DATASETS)
    lines.append(f'Hours of capture per window & {hours}\\\\')
    ver = ' & '.join(f"{mean.loc[(d, 'er+purity@d5_k16'), 'verifications_per_hour']:.0f}" for d in DATASETS)
    lines.append(f'Fast verifications per hour & {ver}\\\\')
    gap = ' & '.join(f"{60 * mean.loc[(d, 'er+purity@d5_k16'), 'hours_per_window'] / 16:.1f}" for d in DATASETS)
    lines.append(f'Minutes per verification & {gap}\\\\')
    flows = ' & '.join(f"{mean.loc[(d, 'er+purity@d5_k16'), 'flows_per_window'] / 1e3:,.0f}k" for d in DATASETS)
    lines.append(f'Capture flows per window & {flows}\\\\')
    lines.append('\\midrule')
    lines.append('\\multicolumn{6}{l}{\\textit{False alarms per 8-hour shift at full capture volume}}\\\\')
    for condition, name in CONDITIONS.items():
        lines.append(f'{name} & ' + ' & '.join(fmt(mean.loc[(d, condition), 'false_alarms_per_shift'])
                                                if (d, condition) in mean.index else '--' for d in DATASETS) + '\\\\')
    lines.append('\\midrule')
    lines.append('\\multicolumn{6}{l}{\\textit{Cluster-level alerts per 8-hour shift}}\\\\')
    for condition in ('er+representative@d5_k16', 'er+purity@d5_k16'):
        lines.append(f'{CONDITIONS[condition]} & ' + ' & '.join(
            f"{mean.loc[(d, condition), 'incident_alerts_per_shift']:.1f}" for d in DATASETS) + '\\\\')
    (tab / 'operations.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    cic = fam[fam.dataset == 'cic-ids2017'].groupby(['condition', 'family'])[['support', 'detected']].sum()
    cic['recall'] = cic.detected / cic.support.clip(lower=1)
    support = cic.xs('er+purity@d5_k16', level='condition').support
    families = support[support > 0].sort_values(ascending=False).index
    cols = ['er@d5', 'er@d0', 'er+representative@d5_k16', 'er+purity@d5_k16']
    lines = []
    families = [f for f in families if support[f] / support.sum() >= 0.001]
    for f in families:
        words = [w for w in f.split('_') if w.isascii() and w.isalnum()]
        name = ' '.join(PRETTY.get(w, w.capitalize()) for w in words)
        cells = [f"{100 * cic.loc[(c, f), 'recall']:.0f}" if (c, f) in cic.index else '--' for c in cols]
        share = 100 * support[f] / support.sum()
        lines.append(f'{name} & {share:.1f} & ' + ' & '.join(cells) + '\\\\')
    (tab / 'families.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--prepared', type=Path, default=ROOT.parents[1] / 'prepared_streams_v2')
    args = parser.parse_args()
    ops, fam = load(args.results, args.prepared)
    out = ROOT / 'results_snapshot' / 'revision_20261113'
    out.mkdir(parents=True, exist_ok=True)
    ops.to_csv(out / 'operational_load.csv', index=False)
    fam.to_csv(out / 'family_recall.csv', index=False)
    write_tables(ops, fam)
    print(ops.groupby(['dataset', 'condition']).mean(numeric_only=True).round(1).to_string())
