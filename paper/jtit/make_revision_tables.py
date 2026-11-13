"""Tables and contrasts for the round-1 revision (stage `revision`).

    .venv-revision python paper/jtit/make_revision_tables.py [--results results]

Joins the revision runs with the verify2 run metrics (same scoring: flows not
queried by the fast lane), writes results_snapshot/revision_20261113/
{revision_runs.csv, revision_summary.csv, stream_contrasts.csv} and the
LaTeX rows paper/jtit/tables/{main_rev,noise_rev,controls}.tex.

Stream level is the unit of evidence: model seeds are averaged within each
(dataset, stream seed), so each dataset contributes three paired values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from tools.analyze_dasp_v2 import PAIR, VERIFY2_METRICS, load_completed, verify2_run_metrics  # noqa: E402

OUT = ROOT / 'results_snapshot' / 'revision_20261113'
TAB = HERE / 'tables'
DATASETS = ['nf-unsw-nb15-v3', 'nf-cse-cic-ids2018-v3', 'nf-ton-iot-v3', 'cic-ids2017', 'cidds-001']


def frames(results: Path) -> pd.DataFrame:
    rows = []
    for root in ('revision', 'revision_heldout'):
        for record in load_completed(results / root, 'revision'):
            rows.append({k: record[k] for k in (*PAIR, 'condition')} | verify2_run_metrics(results / root, record))
    rev = pd.DataFrame(rows)
    old = pd.concat([pd.read_csv(results / part / 'verify2_runs.csv')
                     for part in ('verify2', 'verify2_heldout', 'verify2_heldout_cidds')])
    return pd.concat([old[~old.condition.isin(set(rev.condition))], rev], ignore_index=True)


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby(['dataset', 'condition'])
    stream = frame.groupby(['dataset', 'condition', 'stream_seed'])[['uq_macro_f1', 'uq_benign_fpr']].mean()
    rng = stream.groupby(['dataset', 'condition']).agg(['min', 'max'])
    rng.columns = [f'{a}_stream_{b}' for a, b in rng.columns]
    return g[VERIFY2_METRICS].mean().join(rng).join(g.size().rename('runs')).reset_index()


def stream_contrasts(frame: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    means = frame.groupby(['dataset', 'stream_seed', 'condition'])[['uq_macro_f1', 'uq_benign_fpr']].mean()
    rows = []
    for ref, cond in pairs:
        for dataset in DATASETS:
            try:
                a = means.xs((dataset, ref), level=('dataset', 'condition'))
                b = means.xs((dataset, cond), level=('dataset', 'condition'))
            except KeyError:
                continue
            d = (b - a).dropna()
            if not len(d):
                continue
            rows.append({'reference': ref, 'condition': cond, 'dataset': dataset, 'streams': len(d),
                         'f1_diff_mean': d.uq_macro_f1.mean(), 'f1_diff_min': d.uq_macro_f1.min(),
                         'f1_diff_max': d.uq_macro_f1.max(), 'f1_better_streams': int((d.uq_macro_f1 > 0).sum()),
                         'fpr_diff_mean': d.uq_benign_fpr.mean(), 'fpr_lower_streams': int((d.uq_benign_fpr < 0).sum())})
    return pd.DataFrame(rows)


def cell(s: pd.DataFrame, dataset: str, condition: str, spread: bool = False) -> str:
    row = s[(s.dataset == dataset) & (s.condition == condition)]
    if not len(row):
        return '--'
    r = row.iloc[0]
    text = f'{r.uq_macro_f1:.3f} ({100 * r.uq_benign_fpr:.1f})'
    if spread and r.uq_macro_f1_stream_max - r.uq_macro_f1_stream_min >= 0.0005:
        text = f'{r.uq_macro_f1:.3f}$_{{\\pm{(r.uq_macro_f1_stream_max - r.uq_macro_f1_stream_min) / 2:.3f}}}$ ({100 * r.uq_benign_fpr:.1f})'
    return text


def write(lines: list[str], name: str) -> None:
    (TAB / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main_table(s: pd.DataFrame) -> None:
    lanes = [('er@d{d}', 'ER'), ('er_iwms@d{d}', 'ER + IWMS'), ('er_uncertainty@d{d}', 'ER + uncertainty AL'),
             ('arf@d{d}', 'ARF'), ('er+anomaly@d{d}_k16', 'Anomaly lane (label-free)'),
             ('er+uniform@d{d}_k16', 'Uniform fast labels'), ('er+self_train@d{d}_k16', 'Uniform + self-training'),
             ('er+ilab@d{d}_k16', 'ILAB-style queries'),
             ('er+uniform_knn@d{d}_k16', 'Uniform + kNN'), ('er+hierarchical@d{d}_k16', 'Hierarchical sampling'),
             ('er+representative@d{d}_k16', 'Representative (uncertified)'),
             ('er+purity@d{d}_k16', 'Purity-certified')]
    datasets = DATASETS
    lines = [f'ER, labels without delay$^\\dagger$ & ' + ' & '.join(cell(s, d, 'er@d0') for d in datasets) + '\\\\',
             f'\\quad + representative$^\\dagger$ & ' + ' & '.join(cell(s, d, 'er+representative@d0_k16') for d in datasets) + '\\\\',
             f'\\quad + purity-certified$^\\dagger$ & ' + ' & '.join(cell(s, d, 'er+purity@d0_k16') for d in datasets) + '\\\\',
             '\\midrule']
    for d in (5,):
        lines.append(f'\\multicolumn{{6}}{{l}}{{\\textit{{Label delay $d={d}$ windows}}}}\\\\')
        for t, n in lanes:
            c = t.format(d=d)
            lines.append(f'{n} & ' + ' & '.join(cell(s, ds, c, spread=(d == 5)) for ds in datasets) + '\\\\')
        lines.append('\\midrule')
    write(lines[:-1], 'main_rev.tex')


def noise_table(s: pd.DataFrame) -> None:
    lanes = [('uniform_knn', 'Uniform + kNN'), ('hierarchical', 'Hierarchical sampling'),
             ('representative', 'Representative'),
             ('purity', 'Purity-certified, $r=4$')]
    models = [('', 'clean'), ('_noise10', 'independent'), ('_group10', 'look-alike groups')]
    datasets = DATASETS[:4]
    lines = []
    for lane, name in lanes:
        for suffix, label in models:
            c = f'er+{lane}@d5_k16{suffix}'
            first = name if suffix == '' else ''
            lines.append(f'{first} & {label} & ' + ' & '.join(cell(s, d, c) for d in datasets) + '\\\\')
        lines.append('\\midrule')
    write(lines[:-1], 'noise_rev.tex')


def controls_table(s: pd.DataFrame) -> None:
    rows = [('er+purity_cap1@d5_k16', 'At most one certified cluster per window'),
            ('er+purity_r1cap1@d5_k16', '\\quad the same with $r=1$'),
            ('er+purity@d5_k4_K8', '$k=4$ fast labels, $K=8$'), ('er+purity@d5_k32_K8', '$k=32$ fast labels, $K=8$'),
            ('er+purity@d5_k16_c4', '$K=4$ initial clusters (held-out only)'), ('er+purity@d5_k16_c16', '$K=16$ initial clusters (held-out only)')]
    datasets = DATASETS
    lines = [f'{n} & ' + ' & '.join(cell(s, d, c) for d in datasets) + '\\\\' for c, n in rows]
    write(lines, 'controls.tex')


PAIRS = ([('er@d5', 'er+purity@d5_k16'), ('er@d0', 'er+purity@d5_k16'), ('er@d0', 'er+purity@d0_k16'),
          ('er+representative@d0_k16', 'er+purity@d0_k16'),
          ('er+representative@d5_k16', 'er+purity@d5_k16'), ('er+uniform_knn@d5_k16', 'er+purity@d5_k16'),
          ('er+hierarchical@d5_k16', 'er+purity@d5_k16'), ('er+uniform@d5_k16', 'er+self_train@d5_k16'),
          ('er+uniform@d5_k16', 'er+ilab@d5_k16'), ('er+ilab@d5_k16', 'er+purity@d5_k16'),
          ('er@d0', 'er+purity@d5_k16_h1'), ('er+purity_cap1@d5_k16', 'er+purity_r1cap1@d5_k16'),
          ('er+purity@d5_k16', 'er+purity_pseudo@d5_k16')]
         + [(f'er+representative@d5_k16{m}', f'er+purity@d5_k16{m}') for m in ('_noise10', '_group10', '_subtype10')]
         + [(f'er+purity_r2@d5_k16{m}', f'er+purity@d5_k16{m}') for m in ('', '_noise10', '_group10', '_subtype10')]
         + [(f'er+{lane}@d5_k16', f'er+{lane}@d5_k16{m}') for lane in ('purity', 'representative', 'uniform_knn', 'hierarchical', 'purity_r2')
            for m in ('_noise10', '_group10', '_subtype10')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = frames(args.results)
    frame.to_csv(OUT / 'revision_runs.csv', index=False)
    s = summary(frame)
    s.to_csv(OUT / 'revision_summary.csv', index=False)
    contrasts = stream_contrasts(frame, PAIRS)
    contrasts.to_csv(OUT / 'stream_contrasts.csv', index=False)
    main_table(s)
    noise_table(s)
    controls_table(s)
    print(contrasts.round(3).to_string(index=False))
