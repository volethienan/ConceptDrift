"""Build compound drift streams: a new attack family and a measurement fault at once.

From window 12 the S1-Sudden stream switches attack family; in the same
window one semantic feature group receives the severe measurement fault of
build_feature_shift_streams.py (kept to the end). CX-<group> tests whether a
label-free recalibration of the faulty group also hides the new attack
family, the risk a measurement-only recalibration must not take.

    python tools/build_compound_shift_streams.py --prepared-root prepared_streams_v2 \\
        --datasets nf-bot-iot-v3,nf-ton-iot-v3 --seeds nf-bot-iot-v3=2001,2002 nf-ton-iot-v3=3001,3002
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from tools.build_feature_shift_streams import shifted  # noqa: E402

PROTOCOL_VERSION = 'compound-shift-1.0'
BASE_SCENARIO = 'S1-Sudden'
GROUPS = ('volume', 'timing', 'tcp', 'protocol')


def build(prepared_root: Path, dataset: str, seed: int, group: str, output_root: Path) -> Path:
    source = prepared_root / dataset / BASE_SCENARIO / f'stream_seed{seed}.npz'
    with np.load(source, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    base_manifest = json.loads(source.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    stream, detail = shifted(arrays, group, revert=False, severity='severe')
    scenario = f'CX-{group}'
    destination = output_root / dataset / scenario / f'stream_seed{seed}.npz'
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **stream)
    manifest = {'protocol_version': PROTOCOL_VERSION, 'builder': 'tools/build_compound_shift_streams.py',
                'dataset_key': dataset, 'scenario': scenario, 'stream_seed': seed, 'base_stream': str(source),
                'base_scenario': BASE_SCENARIO, 'base_protocol': base_manifest.get('protocol_version'),
                'measurement_fault': detail, 'labels_and_rows_identical_to_base': True,
                'preprocessing_fit_windows': base_manifest.get('preprocessing_fit_windows')}
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--prepared-root', type=Path, default=Path('prepared_streams_v2'))
    parser.add_argument('--output-root', type=Path, help='default: <prepared-root>/compound')
    parser.add_argument('--seeds', nargs='+', required=True, help='dataset=seed,seed ...')
    parser.add_argument('--groups', default=','.join(GROUPS))
    args = parser.parse_args()
    output_root = args.output_root or args.prepared_root / 'compound'
    for item in args.seeds:
        dataset, seeds = item.split('=')
        for seed in (int(value) for value in seeds.split(',')):
            for group in args.groups.split(','):
                print('wrote', build(args.prepared_root, dataset, seed, group, output_root), flush=True)


if __name__ == '__main__':
    main()
