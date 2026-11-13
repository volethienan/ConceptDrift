"""Majority-label ceiling of NF-BoT-IoT-v3 on the model feature set (with ports).

    python -I paper/jtit/bot_ceiling.py [--csv ../../data/NF-BoT-IoT-v3/NF-BoT-IoT-v3.csv]

The exclusion argument quotes a ceiling computed with identifiers (addresses,
ports, timestamps) removed. The models see port numbers, so this script
recomputes it on exactly the model features: every flow is predicted with the
majority label of its feature vector, the best any behaviour-only classifier
can do. Writes results_snapshot/revision_20261113/bot_ceiling.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NON_FEATURES = {'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label', 'Attack'}


def ceiling(path: Path, chunksize: int = 500_000) -> dict:
    features = [c for c in pd.read_csv(path, nrows=0).columns if c not in NON_FEATURES and not c.startswith('__')]
    hashes, labels = [], []
    for chunk in pd.read_csv(path, usecols=[*features, 'Label'], chunksize=chunksize, low_memory=False):
        values = chunk[features].apply(pd.to_numeric, errors='coerce')
        hashes.append(pd.util.hash_pandas_object(values, index=False).to_numpy(np.uint64))
        labels.append(pd.to_numeric(chunk['Label'], errors='coerce').to_numpy())
    frame = pd.DataFrame({'h': np.concatenate(hashes), 'y': np.concatenate(labels)})
    frame = frame[frame.y.isin((0, 1))]
    share = frame.groupby('h')['y'].transform('mean')
    pred = (share > 0.5).astype(int)
    y = frame.y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    f1_attack = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_benign = 2 * tn / max(2 * tn + fn + fp, 1)
    mixed = share.between(0, 1, inclusive='neither')
    return {'features': len(features), 'flows': int(len(frame)), 'benign': int((y == 0).sum()),
            'macro_f1': (f1_attack + f1_benign) / 2, 'benign_fpr': fp / max(fp + tn, 1),
            'benign_with_attack_twin': float(mixed[y == 0].mean()), 'conflicting_flows': int(mixed.sum())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=ROOT.parents[1] / 'data' / 'NF-BoT-IoT-v3' / 'NF-BoT-IoT-v3.csv')
    args = parser.parse_args()
    result = ceiling(args.csv)
    out = ROOT / 'results_snapshot' / 'revision_20261113' / 'bot_ceiling.json'
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))
