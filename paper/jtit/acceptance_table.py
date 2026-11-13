"""Acceptance-rule ablation on identical clusters (stage `acceptance`).

    .venv-revision python paper/jtit/acceptance_table.py [--results results]

Same partition, screening, splitting and budget as the certified lane; only the
rule that raises a cluster changes: one attack label (r=1), a strict majority of
four draws, or four of four (the certificate, the er+purity runs). Clean,
independently noisy (10%) and group-correlated (10%) analyst labels. Writes
paper/jtit/tables/acceptance.tex and results_snapshot/revision_20261113/
{acceptance_runs,acceptance_contrasts}.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'src'), str(ROOT)]
from make_revision_tables import cell, stream_contrasts, summary  # noqa: E402
from tools.analyze_dasp_v2 import PAIR, load_completed, verify2_run_metrics  # noqa: E402

OUT = ROOT / 'results_snapshot' / 'revision_20261113'
DATASETS = ['nf-cse-cic-ids2018-v3', 'nf-ton-iot-v3', 'cic-ids2017']
RULES = [('purity_one', 'one label ($r=1$)'), ('purity_majority', 'majority of four'),
         ('purity', 'all four (certificate)')]
ERRORS = [('', 'clean'), ('_noise10', 'independent'), ('_group10', 'look-alike groups')]


def frames(results: Path) -> pd.DataFrame:
    rows = []
    for root in ('acceptance', 'acceptance_heldout'):
        for record in load_completed(results / root, 'acceptance'):
            rows.append({k: record[k] for k in (*PAIR, 'condition')} | verify2_run_metrics(results / root, record))
    new = pd.DataFrame(rows)
    old = pd.concat([pd.read_csv(results / part / 'verify2_runs.csv') for part in ('verify2', 'verify2_heldout')])
    rev = pd.read_csv(OUT / 'revision_runs.csv')
    wanted = {f'er+purity@d5_k16{s}' for s, _ in ERRORS}
    ref = pd.concat([old, rev])
    ref = ref[ref.condition.isin(wanted) & ref.dataset.isin(DATASETS)].drop_duplicates([*PAIR, 'condition'])
    return pd.concat([ref, new[new.dataset.isin(DATASETS)]], ignore_index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    frame = frames(args.results)
    frame.to_csv(OUT / 'acceptance_runs.csv', index=False)
    s = summary(frame)
    lines = []
    for suffix, label in ERRORS:
        for i, (rule, name) in enumerate(RULES):
            first = label if i == 0 else ''
            lines.append(f'{first} & {name} & ' + ' & '.join(cell(s, d, f'er+{rule}@d5_k16{suffix}') for d in DATASETS) + '\\\\')
        lines.append('\\midrule')
    (HERE / 'tables').mkdir(exist_ok=True)
    (HERE / 'tables' / 'acceptance.tex').write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    pairs = [(f'er+{rule}@d5_k16{suffix}', f'er+purity@d5_k16{suffix}')
             for suffix, _ in ERRORS for rule in ('purity_one', 'purity_majority')]
    contrasts = stream_contrasts(frame, pairs)
    contrasts.to_csv(OUT / 'acceptance_contrasts.csv', index=False)
    print(s[s.condition.str.startswith('er+purity')][['dataset', 'condition', 'uq_macro_f1', 'uq_benign_fpr', 'runs']]
          .round(3).to_string(index=False))
    print(contrasts.round(3).to_string(index=False))
