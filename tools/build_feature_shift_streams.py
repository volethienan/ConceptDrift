"""Build feature-group-local drift streams from existing controlled S0 streams.

From window 12 one semantic feature group (feature_groups.py) is changed the
way a measurement or network change would change it, for benign and attack
flows alike; every other group and the attack family mixture stay as in S0.
FS-<group> keeps the change to the end, FR-<group> reverts at window 24.
Values are mapped back to raw units with the stream's stored warm-up scaler,
transformed, and re-standardised with the same scaler, so warm-up statistics
never see the change.

    python tools/build_feature_shift_streams.py --prepared-root prepared_streams_v2 \
        --datasets nf-bot-iot-v3,nf-unsw-nb15-v3 --stream-seeds 2001,2002,2003
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

from dasp_nids.feature_groups import GROUP_NAMES, SCHEMAS, schema_for  # noqa: E402

PROTOCOL_VERSION = 'feature-shift-1.0'
START, REVERT = 12, 24
# (operation, value, features) per schema and group; mul scales, add offsets.
TRANSFORMS = {
    'netflow_v3': {
        'volume': ('mul', 0.25, ('IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 'SRC_TO_DST_SECOND_BYTES',
                                 'DST_TO_SRC_SECOND_BYTES', 'SRC_TO_DST_AVG_THROUGHPUT',
                                 'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES',
                                 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES',
                                 'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES')),
        'timing': ('mul', 3.0, SCHEMAS['netflow_v3']['timing']),
        'tcp': ('mul', 2.0, ('TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'RETRANSMITTED_IN_BYTES',
                             'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS')),
        'protocol': ('add', -16.0, ('MIN_TTL', 'MAX_TTL')),
    },
    'cicflowmeter': {
        'volume': ('mul', 0.25, ('Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
                                 'Total Length of Bwd Packets', 'Flow Bytes/s', 'Flow Packets/s',
                                 'Fwd Packets/s', 'Bwd Packets/s', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
                                 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'act_data_pkt_fwd')),
        'timing': ('mul', 3.0, SCHEMAS['cicflowmeter']['timing']),
        'tcp': ('mul', 2.0, ('Init_Win_bytes_forward', 'Init_Win_bytes_backward')),
        'protocol': ('mul', 1.5, ('Fwd Header Length', 'Bwd Header Length', 'Fwd Header Length.1')),
    },
}


def shifted(arrays: dict[str, np.ndarray], group: str, revert: bool) -> tuple[dict[str, np.ndarray], dict]:
    names = arrays['feature_names'].astype(str).tolist()
    schema = schema_for(names)
    operation, value, features = TRANSFORMS[schema][group]
    if not set(features) <= set(SCHEMAS[schema][group]):
        raise ValueError(f'{group}: transform touches features outside the group')
    columns = [names.index(feature) for feature in features]
    mean, scale = arrays['scaler_mean'], arrays['scaler_scale']
    raw = arrays['x'].astype(np.float64) * scale + mean
    stop = REVERT if revert else raw.shape[0]
    block = raw[START:stop][:, :, columns]
    block = block * value if operation == 'mul' else np.maximum(block + value, 1.0)
    raw[START:stop][:, :, columns] = block
    result = dict(arrays)
    result['x'] = ((raw - mean) / scale).astype(np.float32)
    phases = arrays['phases'].astype(str).copy()
    phases[START:stop] = [f'{phase}|{group}_shift' for phase in phases[START:stop]]
    result['phases'] = phases.astype('<U48')
    detail = {'schema': schema, 'group': group, 'operation': operation, 'value': value,
              'features': list(features), 'start_window': START, 'stop_window': stop}
    return result, detail


def build(args) -> None:
    root = Path(args.prepared_root)
    seeds = [int(value) for value in args.stream_seeds.split(',') if value.strip()]
    for dataset in [value.strip() for value in args.datasets.split(',') if value.strip()]:
        for seed in seeds:
            source = root / dataset / 'S0' / f'stream_seed{seed}.npz'
            with np.load(source, allow_pickle=False) as loaded:
                arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
            base_manifest = json.loads(source.with_suffix('.manifest.json').read_text(encoding='utf-8'))
            for group in GROUP_NAMES:
                for prefix, revert in (('FS', False), ('FR', True)):
                    scenario = f'{prefix}-{group}'
                    stream, detail = shifted(arrays, group, revert)
                    destination = Path(args.output_dir) / dataset / scenario / f'stream_seed{seed}.npz'
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(destination, **stream)
                    manifest = {'protocol_version': PROTOCOL_VERSION,
                                'builder': 'tools/build_feature_shift_streams.py',
                                'dataset_key': dataset, 'scenario': scenario, 'stream_seed': seed,
                                'base_stream': str(source), 'base_protocol': base_manifest.get('protocol_version'),
                                'base_scenario': 'S0', 'feature_shift': detail,
                                'labels_and_rows_identical_to_base': True,
                                'preprocessing_fit_windows': base_manifest.get('preprocessing_fit_windows')}
                    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2),
                                                                        encoding='utf-8')
                    print(f'wrote {destination}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--prepared-root', default='prepared_streams_v2')
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--stream-seeds', default='2001,2002,2003')
    parser.add_argument('--output-dir', default='prepared_streams_v2/feature_shift')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
