"""Development-only comparison of purity-lane variants (stage `pilot`).

    .venv-revision python paper/jtit/pilot_summary.py [--results results] [--root pilot] [--stage pilot]

Same scoring as the main table (flows not queried by the fast lane), joined
with the verify2 development runs of the existing lanes. Stream level is the
unit: model seeds are averaged within each (dataset, stream seed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'src'), str(ROOT)]
from make_revision_tables import stream_contrasts  # noqa: E402
from tools.analyze_dasp_v2 import PAIR, load_completed, verify2_run_metrics  # noqa: E402

REFERENCE = ['er+purity@d5_k16', 'er+representative@d5_k16', 'er+uniform_knn@d5_k16', 'er+hierarchical@d5_k16']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--root', default='pilot')
    parser.add_argument('--stage', default='pilot')
    args = parser.parse_args()
    rows = [{k: r[k] for k in (*PAIR, 'condition')} | verify2_run_metrics(args.results / args.root, r)
            for r in load_completed(args.results / args.root, args.stage)]
    new = pd.DataFrame(rows)
    old = pd.read_csv(args.results / 'verify2' / 'verify2_runs.csv')
    old = old[old.condition.isin(REFERENCE) & old.dataset.isin(new.dataset.unique())]
    frame = pd.concat([old, new], ignore_index=True)
    table = frame.groupby(['dataset', 'condition'])[['uq_macro_f1', 'uq_benign_fpr']].mean().unstack(0)
    print(table.round(3).to_string())
    pairs = [(ref, cond) for cond in sorted(new.condition.unique()) for ref in REFERENCE]
    print(stream_contrasts(frame, pairs).round(3).to_string(index=False))
