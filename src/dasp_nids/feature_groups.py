"""Semantic feature groups for modular models, defined from feature documentation.

Each group is one physical aspect of a flow, so that a change in one part of
the measurement pipeline (sampling, path latency, TCP stack, addressing) is
confined to one module. Groups are assigned by feature name, never by column
position, and every feature must belong to exactly one group.
"""
from __future__ import annotations

GROUP_NAMES = ('volume', 'timing', 'tcp', 'protocol')

# NetFlow v3 (Luay et al., 2025) as exported in the NF-*-v3 datasets.
NETFLOW_V3 = {
    'volume': (
        'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
        'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
        'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES',
        'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES',
        'NUM_PKTS_1024_TO_1514_BYTES'),
    'timing': (
        'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'SRC_TO_DST_IAT_MIN',
        'SRC_TO_DST_IAT_MAX', 'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV', 'DST_TO_SRC_IAT_MIN',
        'DST_TO_SRC_IAT_MAX', 'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV'),
    'tcp': (
        'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
        'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES',
        'RETRANSMITTED_OUT_PKTS'),
    'protocol': (
        'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'MIN_TTL', 'MAX_TTL', 'ICMP_TYPE',
        'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE'),
}

# CICFlowMeter as exported in the CIC-IDS2017 MachineLearningCVE files.
CICFLOWMETER = {
    'volume': (
        'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets',
        'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
        'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
        'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'Down/Up Ratio',
        'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
        'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
        'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
        'Subflow Bwd Bytes', 'act_data_pkt_fwd'),
    'timing': (
        'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Active Mean', 'Active Std',
        'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'),
    'tcp': (
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'FIN Flag Count',
        'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWE Flag Count', 'ECE Flag Count', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward'),
    'protocol': (
        'Destination Port', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Header Length.1',
        'min_seg_size_forward'),
}
SCHEMAS = {'netflow_v3': NETFLOW_V3, 'cicflowmeter': CICFLOWMETER}


def schema_for(feature_names: list[str], dropped: list[str] | tuple[str, ...] = ()) -> str:
    """The documented schema whose features are exactly ``feature_names`` plus
    the explicitly ``dropped`` ones (shortcut ablation)."""

    names = set(feature_names)
    matches = [key for key, groups in SCHEMAS.items()
               if names <= (members := {feature for group in groups.values() for feature in group})
               and members <= names | set(dropped)]
    if len(matches) != 1:
        raise ValueError('feature names do not match exactly one documented schema')
    return matches[0]


def semantic_groups(feature_names: list[str], dropped: list[str] | tuple[str, ...] = ()) -> list[list[int]]:
    """Column indices of each group in GROUP_NAMES order, for this column order."""

    groups = SCHEMAS[schema_for(feature_names, dropped)]
    position = {name: index for index, name in enumerate(feature_names)}
    result = [sorted(position[name] for name in groups[group] if name in position) for group in GROUP_NAMES]
    if not all(result):
        raise ValueError('dropping features emptied a semantic group')
    return result
