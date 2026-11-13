"""Conflict-free NF-ToN-IoT check (review round 2, item R7).

    .venv-revision python paper/jtit/dedup_summary.py [--results results]

Summarises stage `dedup` (NF-ToN rebuilt with tools/build_natural_streams.py
--drop-conflicts) with the same scoring as the main table, next to the
original NF-ToN runs, and writes results_snapshot/revision_20261113/
{dedup_runs,dedup_summary,dedup_contrasts}.csv.
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
from tools.analyze_dasp_v2 import PAIR, VERIFY2_METRICS, load_completed, verify2_run_metrics  # noqa: E402

OUT = ROOT / 'results_snapshot' / 'revision_20261113'
LANES = ['er@d0', 'er@d5', 'er+uniform@d5_k16', 'er+uniform_knn@d5_k16', 'er+hierarchical@d5_k16',
         'er+representative@d5_k16', 'er+purity@d5_k16']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    rows = [{k: r[k] for k in (*PAIR, 'condition')} | verify2_run_metrics(args.results / 'dedup', r)
            for r in load_completed(args.results / 'dedup', 'dedup')]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'dedup_runs.csv', index=False)
    summary = frame.groupby('condition')[VERIFY2_METRICS].mean().reindex(LANES)
    summary['runs'] = frame.groupby('condition').size()
    original = pd.read_csv(OUT / 'revision_summary.csv')
    original = original[original.dataset == 'nf-ton-iot-v3'].set_index('condition')
    summary['original_uq_macro_f1'] = original.uq_macro_f1.reindex(LANES)
    summary['original_uq_benign_fpr'] = original.uq_benign_fpr.reindex(LANES)
    summary.to_csv(OUT / 'dedup_summary.csv')
    pairs = [('er@d5', 'er+purity@d5_k16'), ('er@d0', 'er+purity@d5_k16')] + [
        (lane, 'er+purity@d5_k16') for lane in LANES[2:-1]]
    contrasts = stream_contrasts(frame, pairs)
    contrasts.to_csv(OUT / 'dedup_contrasts.csv', index=False)
    print(summary[['uq_macro_f1', 'uq_benign_fpr', 'original_uq_macro_f1', 'original_uq_benign_fpr', 'runs']]
          .round(3).to_string())
    print(contrasts.round(3).to_string(index=False))
