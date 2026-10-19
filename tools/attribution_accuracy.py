"""Drift attribution accuracy per signal on feature-shift streams, and false alarms on label shift.

    python tools/attribution_accuracy.py outputs_v2_screen11_attribution outputs_v2_screen11_attribution_labelshift
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ('volume', 'timing', 'tcp', 'protocol')
ONSET = range(12, 16)  # first four shifted windows


def columns(frame, prefix):
    return [f'{prefix}module_{i}' for i in range(1, 5) if f'{prefix}module_{i}' in frame]


def main(feature_root: Path, label_root: Path | None) -> None:
    hits = collections.defaultdict(collections.Counter)
    for path in (feature_root / 'screen').glob('*/*/*/windows.csv'):
        condition = path.parts[-2].split('_stream')[0]
        if not condition.startswith(('attr_', 'blk_attr')):
            continue
        dataset, scenario = path.parts[-4][:7], path.parts[-3]
        target = GROUPS.index(scenario.split('-')[1])
        frame = pd.read_csv(path).set_index('window')
        repr_cols, cc_cols = columns(frame, 'z_'), columns(frame, 'attr_cc_z_')
        for window in ONSET:
            if window not in frame.index:
                continue
            row = frame.loc[window]
            signals = {'representation': row[repr_cols].astype(float).to_numpy()}
            if cc_cols:
                cc = np.nan_to_num(row[cc_cols].astype(float).to_numpy())
                signals['class_conditional'] = cc
                signals['combined'] = cc + signals['representation']
            scores = row.get('attribution_scores')
            if isinstance(scores, str) and scores:
                values = json.loads(scores)
                signals['routing_used'] = np.asarray([values[f'module_{i}'] for i in range(1, 5)])
            for signal, values in signals.items():
                key = (condition, signal, dataset)
                hits[key]['n'] += 1
                hits[key]['top1'] += int(np.argmax(values) == target)
    table = pd.DataFrame([{'condition': k[0], 'signal': k[1], 'dataset': k[2],
                           'top1': c['top1'] / c['n'], 'n': c['n']} for k, c in hits.items()])
    pd.set_option('display.width', 200)
    print(table.pivot_table(index=['condition', 'signal'], columns='dataset', values='top1').round(2).to_string())
    if label_root is None:
        return
    rows = []
    for path in (label_root / 'screen').glob('*/*/attr_pseudo_k1_stream*/windows.csv'):
        frame = pd.read_csv(path)
        cc = frame[columns(frame, 'attr_cc_z_')].astype(float)
        shifted = frame.window.between(12, 35)
        rows.append({'dataset': path.parts[-4][:7], 'max_cc_z_median': float(cc[shifted].max(axis=1).median()),
                     'max_cc_z_p90': float(cc[shifted].max(axis=1).quantile(.9))})
    print('label shift (no feature drift): class-conditional z during prior changes')
    print(pd.DataFrame(rows).groupby('dataset').mean().round(2).to_string())


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
