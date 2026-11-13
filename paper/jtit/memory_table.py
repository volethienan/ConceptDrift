"""Certificate-memory ablation (stages pilot2, memory).

    .venv-revision python paper/jtit/memory_table.py [--results results]

Signatures of clusters raised in window t alarm matching model-benign flows of
later windows without new labels. The same mechanism is applied to the
certified and to the uncertified (representative) lane, without renewal for
two windows and with renewal by one uniform draw per window, on clean and on
10%-noisy analyst labels. Writes paper/jtit/tables/memory.tex and
results_snapshot/revision_20261113/{memory_runs,memory_contrasts}.csv.
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
ROWS = [('', 'none'), ('_m2', 'two windows'), ('_mc', 'renewed by one draw'),
        ('@noise', 'none, 10\\% analyst errors'), ('_m2@noise', 'two windows, 10\\% errors')]


def condition(lane: str, suffix: str) -> str:
    variant, noisy = (suffix.split('@') + [''])[:2]
    return f'er+{lane}{variant}@d5_k16' + ('_noise10' if noisy else '')


def frames(results: Path) -> pd.DataFrame:
    rows = []
    for root, stage in (('pilot2', 'pilot2'), ('memory', 'memory'), ('memory_heldout', 'memory')):
        for record in load_completed(results / root, stage):
            rows.append({k: record[k] for k in (*PAIR, 'condition')} | verify2_run_metrics(results / root, record))
    new = pd.DataFrame(rows)
    old = pd.concat([pd.read_csv(results / part / 'verify2_runs.csv') for part in ('verify2', 'verify2_heldout')])
    wanted = {condition(lane, s) for lane in ('purity', 'representative') for s, _ in ROWS}
    old = old[old.condition.isin(wanted - set(new.condition)) & old.dataset.isin(DATASETS)]
    return pd.concat([old, new[new.dataset.isin(DATASETS)]], ignore_index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    frame = frames(args.results)
    frame.to_csv(OUT / 'memory_runs.csv', index=False)
    s = summary(frame)
    lines = []
    for lane, name in (('purity', 'Purity-certified'), ('representative', 'Representative')):
        for i, (suffix, label) in enumerate(ROWS):
            first = name if i == 0 else ''
            lines.append(f'{first} & {label} & ' + ' & '.join(cell(s, d, condition(lane, suffix)) for d in DATASETS) + '\\\\')
        lines.append('\\midrule')
    (HERE / 'tables' / 'memory.tex').write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    pairs = [(condition(lane, base), condition(lane, var)) for lane in ('purity', 'representative')
             for base, var in (('', '_m2'), ('', '_mc'), ('@noise', '_m2@noise'))]
    pairs += [(condition('representative', v), condition('purity', v)) for v in ('', '_m2', '_mc', '@noise', '_m2@noise')]
    contrasts = stream_contrasts(frame, pairs)
    contrasts.to_csv(OUT / 'memory_contrasts.csv', index=False)
    print(s[s.condition.str.contains('purity|representative')][['dataset', 'condition', 'uq_macro_f1', 'uq_benign_fpr', 'runs']]
          .round(3).to_string(index=False))
    print(contrasts.round(3).to_string(index=False))
