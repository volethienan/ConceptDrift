"""LaTeX tables for the audit stage (certificate validity, clustering ablations, cost).

    python paper/jtit/make_audit_tables.py

Reads results_snapshot/audit_20261112/{audit,audit_heldout} and the verify2
snapshot (fast-lane cost). Scores are on the flows nobody labelled
(uq_macro_f1, uq_benign_fpr), as in the verify2 analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'results_snapshot'
AUDIT = ROOT / 'audit_20261112'
VERIFY2 = ROOT / 'verify2_20261110'
TAB = HERE / 'tables'
DEV = {'nf-unsw-nb15-v3': 'NF-UNSW', 'nf-cse-cic-ids2018-v3': 'NF-CSE', 'nf-ton-iot-v3': 'NF-ToN'}
HELD = {'cic-ids2017': 'CIC-IDS2017', 'cidds-001': 'CIDDS-001'}
REF = 'er+purity@d5_k16'


def cell(summary, dataset, condition):
    row = summary[(summary.dataset == dataset) & (summary.condition == condition)]
    if not len(row):
        return '--'
    r = row.iloc[0]
    return f'{r.uq_macro_f1:.3f} ({100 * r.uq_benign_fpr:.1f})'


def ablation_table(summary):
    rows = [(REF, 'Default: $k$-means, 8 initial clusters, signed-log space, $m=10$'),
            (f'{REF}_c2', 'Initial clusters 2'), (f'{REF}_c4', 'Initial clusters 4'),
            (f'{REF}_c16', 'Initial clusters 16'), (f'{REF}_c32', 'Initial clusters 32'),
            (f'{REF}_ward', 'Ward agglomeration'), (f'{REF}_random', 'Random partition (no feature structure)'),
            (f'{REF}_rawspace', 'Raw $z$-scored feature space'),
            (f'{REF}_min5', 'Smallest verifiable cluster $m=5$'), (f'{REF}_min20', 'Smallest verifiable cluster $m=20$'),
            (f'{REF}_min40', 'Smallest verifiable cluster $m=40$')]
    lines = [f'{name} & ' + ' & '.join(cell(summary, d, c) for d in DEV) + '\\\\' for c, name in rows]
    (TAB / 'ablation_cluster.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def certificate_table(analyses):
    names = {REF: 'clean analyst', f'{REF}_noise05': '5\\% label noise', f'{REF}_noise10': '10\\% label noise'}
    lines = []
    for condition, label in names.items():
        for datasets in (DEV, HELD):
            for dataset, short in datasets.items():
                audit = next((a['certificate'].get(condition, {}).get(dataset) for a in analyses
                              if dataset in a['certificate'].get(condition, {})), None)
                if not audit:
                    continue
                rate = audit['pass_rate_given_impure']
                rate = '--' if rate != rate else f'{rate:.3f}'
                lines.append(f"{short}, {label} & {audit['tested_clusters']} & {audit['passed']} & "
                             f"{audit['impure_tested']} & {audit['impure_passed']} & {rate} & "
                             f"{100 * audit['benign_share_of_certified_flows']:.1f} \\\\")
        lines.append('\\midrule')
    (TAB / 'certificate.tex').write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')


def cost_table():
    frames = [pd.read_csv(VERIFY2 / part / 'verify2_summary.csv')
              for part in ('verify2', 'verify2_heldout', 'verify2_heldout_cidds')
              if (VERIFY2 / part / 'verify2_summary.csv').is_file()]
    summary = pd.concat(frames)
    lanes = {'er+uniform@d5_k16': 'Uniform fast labels', 'er+uniform_knn@d5_k16': 'Uniform + kNN',
             'er+hierarchical@d5_k16': 'Hierarchical sampling',
             'er+representative@d5_k16': 'Representative', REF: 'Purity-certified'}
    datasets = {**DEV, **HELD}
    lines = []
    for condition, name in lanes.items():
        cells = []
        for dataset in datasets:
            row = summary[(summary.dataset == dataset) & (summary.condition == condition)]
            cells.append('--' if not len(row) else
                         f"{row.iloc[0].fast_lane_seconds_per_window:.2f} / {row.iloc[0].uq_false_alarms_per_10k:.0f}")
        lines.append(f'{name} & ' + ' & '.join(cells) + '\\\\')
    (TAB / 'cost.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def calibration_figure(r: int = 4, epsilon: float = 0.3):
    """Observed certification rate vs the true benign fraction of each tested
    cluster, against (1 - b)^r (draws with replacement)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False})
    bins = np.array([0, .05, .1, .2, .3, .4, .5, .75, 1.0000001])
    series = [('audit', REF, 'Development, clean analyst', '#2a78d6', 'o'),
              ('audit', f'{REF}_noise10', 'Development, 10% label noise', '#eb6834', 's'),
              ('audit_heldout', REF, 'Held-out, clean analyst', '#1baf7a', '^')]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    grid = np.linspace(0, 1, 101)
    ax.plot(grid, (1 - grid) ** r, color='#1f1f1e', lw=1, label=f'$(1-b)^{r}$')
    ax.axvline(epsilon, color='#6b6a64', lw=0.6, ls=':')
    ax.text(epsilon + 0.01, 0.45, f'$\\varepsilon={epsilon}$', color='#6b6a64')
    for part, condition, label, colour, marker in series:
        clusters = pd.read_csv(AUDIT / part / 'audit_clusters.csv.gz')
        tested = clusters[(clusters.condition == condition) & (clusters.draws > 0)
                          & clusters.outcome.isin(['pass', 'benign', 'mixed'])]
        groups = tested.groupby(pd.cut(tested.benign_fraction, bins, right=False), observed=True)
        x = groups.benign_fraction.mean()
        y = groups.outcome.apply(lambda o: (o == 'pass').mean())
        n = groups.size()
        keep = n >= 20
        ax.scatter(x[keep], y[keep], s=14, color=colour, marker=marker, label=label, zorder=3)
    ax.set_xlabel('True benign fraction $b$ of a tested cluster')
    ax.set_ylabel('Share certified')
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6.5, loc='upper right')
    fig.tight_layout()
    fig.savefig(HERE / 'figures' / 'certificate.pdf'); fig.savefig(HERE / 'figures' / 'certificate.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    TAB.mkdir(exist_ok=True)
    calibration_figure()
    dev = pd.read_csv(AUDIT / 'audit' / 'audit_summary.csv')
    analyses = [json.loads((AUDIT / part / 'audit_analysis.json').read_text(encoding='utf-8'))
                for part in ('audit', 'audit_heldout')]
    ablation_table(dev)
    certificate_table(analyses)
    cost_table()
    print('wrote', *(p.name for p in TAB.glob('*.tex')))
