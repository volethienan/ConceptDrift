"""Figures and LaTeX tables for the JTIT manuscript, from results_snapshot/verify_20261109.

    python paper/jtit/make_figures.py [--runs-root results]

The per-window figure needs the held-out windows.csv files (results/heldout);
everything else reads the committed snapshot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SNAP = HERE.parents[1] / 'results_snapshot' / 'verify_20261109'
FIG, TAB = HERE / 'figures', HERE / 'tables'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3de'
DATASETS = {'nf-unsw-nb15-v3': 'NF-UNSW', 'nf-cse-cic-ids2018-v3': 'NF-CSE', 'nf-ton-iot-v3': 'NF-ToN',
            'cic-ids2017': 'CIC-IDS2017'}

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
                     'legend.frameon': False})


def summaries():
    dev = pd.read_csv(SNAP / 'dev' / 'verify_summary.csv')
    held = pd.read_csv(SNAP / 'heldout' / 'verify_summary.csv')
    trade = pd.read_csv(SNAP / 'tradeoff' / 'tradeoff_summary.csv')
    return dev, held, trade


def tradeoff_figure(trade, held):
    others = {'er@d5': ('ER, delay 5', 'v'), 'er@d0': ('ER, no delay', '^'),
              'er+uniform@d5_k16': ('Uniform fast labels', 's'), 'er+uniform_knn@d5_k16': ('Uniform + kNN', 'D'),
              'er+hierarchical@d5_k16': ('Hierarchical (DH08)', 'P')}
    held = held[held.condition.str.contains('@d5_k16$|@d5$|@d0$', regex=True)].copy()
    held['condition'] = held.condition.replace({'er+purity@d5_k16': 'er+purity@d5_k16_r4',
                                                'er+representative@d5_k16': 'er+representative@d5_k16_c16'})
    panels = [(trade[trade.dataset == d], DATASETS[d] + ' (dev)') for d in ('nf-cse-cic-ids2018-v3', 'nf-ton-iot-v3')]
    panels.append((held, 'CIC-IDS2017 (held-out)'))
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8))
    for ax, (part, title) in zip(axes, panels):
        for prefix, colour, label, key in (('er+purity@d5_k16_r', BLUE, 'Purity-certified (r)', 'r'),
                                           ('er+representative@d5_k16_c', ORANGE, 'Representative (clusters)', 'c')):
            curve = part[part.condition.str.startswith(prefix)].copy()
            curve['knob'] = curve.condition.str.rsplit(key, n=1).str[-1].astype(int)
            curve = curve.sort_values('knob')
            ax.plot(100 * curve.cumulative_benign_fpr, curve.cumulative_macro_f1, '-o', color=colour, lw=2,
                    ms=4.5, mec='white', mew=1, label=label, zorder=3)
            if len(curve) > 1:  # label only the two ends of each curve, on opposite sides
                for (_, row), offset in zip(curve.iloc[[0, -1]].iterrows(), ((5, -10), (-22, 5)) if key == 'r' else ((5, 4), (5, 4))):
                    ax.annotate(f"{key}={row.knob}", (100 * row.cumulative_benign_fpr, row.cumulative_macro_f1),
                                textcoords='offset points', xytext=offset, fontsize=6, color=colour)
        for condition, (label, marker) in others.items():
            row = part[part.condition == condition]
            if len(row):
                ax.plot(100 * row.cumulative_benign_fpr, row.cumulative_macro_f1, marker, color=MUTED, ms=4.5,
                        mec='white', mew=0.6, zorder=2, label=label)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel('Benign FPR (%)')
    axes[0].set_ylabel('Cumulative Macro-F1')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / 'tradeoff.pdf'); fig.savefig(FIG / 'tradeoff.png', dpi=160)
    plt.close(fig)


def onset_figure(runs_root: Path):
    root = runs_root / 'heldout' / 'verify' / 'cic-ids2017' / 'Natural-Chronological'
    series = {'er@d5': ('ER, delay 5', MUTED), 'er@d0': ('ER, no delay', YELLOW),
              'er+representative@d5_k16': ('Representative, delay 5', ORANGE),
              'er+purity@d5_k16': ('Purity-certified, delay 5', BLUE)}
    fig, ax = plt.subplots(figsize=(7.1, 2.3))
    rate = None
    for condition, (label, colour) in series.items():
        frames = [pd.read_csv(root / f'{condition}_stream4001_model{m}' / 'windows.csv') for m in range(91, 96)]
        frames = [f[~f.warmup.astype(bool)] for f in frames]
        recall = pd.concat([f.set_index('window').attack_recall for f in frames], axis=1).mean(axis=1)
        rate = frames[0].set_index('window').attack_rate
        ax.plot(recall.index, recall.values, '-o', color=colour, lw=2, ms=3.5, mec='white', mew=0.8, label=label)
    ax2 = ax.twinx()
    ax2.bar(rate.index, rate.values, color=GRID, width=0.8, zorder=0)
    ax2.set_ylim(0, 4)
    ax2.set_yticks([])
    ax2.spines['right'].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel('Window (CIC-IDS2017 held-out stream 4001; bars: attack share of the window)')
    ax.set_ylabel('Attack recall')
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc='upper left', ncol=4, fontsize=6.5, bbox_to_anchor=(0, 1.18))
    fig.tight_layout()
    fig.savefig(FIG / 'onset.pdf'); fig.savefig(FIG / 'onset.png', dpi=160)
    plt.close(fig)


def cell(summary, dataset, condition):
    row = summary[(summary.dataset == dataset) & (summary.condition == condition)]
    if not len(row):
        return '--'
    r = row.iloc[0]
    return f"{r.cumulative_macro_f1:.3f} ({100 * r.cumulative_benign_fpr:.1f})"


def main_table(dev, held):
    allr = pd.concat([dev, held])
    rows = [('er@d0', 'ER, labels without delay$^\\dagger$'), ('er@d{d}', 'ER'), ('er_iwms@d{d}', 'ER + IWMS'),
            ('er_uncertainty@d{d}', 'ER + uncertainty AL'), ('arf@d{d}', 'ARF'),
            ('er+uniform@d{d}_k16', 'Uniform fast labels'), ('er+uniform_knn@d{d}_k16', 'Uniform + kNN'),
            ('er+hierarchical@d{d}_k16', 'Hierarchical sampling'),
            ('er+representative@d{d}_k16', 'Cluster verification, representative'),
            ('er+purity@d{d}_k16', 'Cluster verification, purity-certified')]
    lines = []
    for d in (1, 5):
        lines.append(f'\\multicolumn{{5}}{{l}}{{\\textit{{Label delay $d={d}$ windows}}}}\\\\')
        for template, name in rows:
            if template == 'er@d0' and d == 5:
                continue
            condition = template.format(d=d)
            cells = [cell(allr, ds, condition) for ds in DATASETS]
            lines.append(f"{name} & " + ' & '.join(cells) + '\\\\')
        lines.append('\\midrule')
    body = '\n'.join(lines[:-1])
    (TAB / 'main.tex').write_text(body + '\n', encoding='utf-8')


def ablation_table(dev, held, trade):
    allr = pd.concat([dev, held])
    lines = []
    for k in (4, 8, 16, 32):
        u, p = f'er+uniform@d5_k{k}', f'er+purity@d5_k{k}'
        lines.append(f'Fast labels $k={k}$, uniform & ' + ' & '.join(cell(allr, ds, u) for ds in DATASETS) + '\\\\')
        lines.append(f'Fast labels $k={k}$, purity-certified & ' + ' & '.join(cell(allr, ds, p) for ds in DATASETS) + '\\\\')
    lines.append('\\midrule')
    lines.append('Purity-certified + pseudo-label training & '
                 + ' & '.join(cell(allr, ds, 'er+purity_pseudo@d5_k16') for ds in DATASETS) + '\\\\')
    (TAB / 'ablation_k.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    lines = []
    for r in (1, 2, 3, 4, 6, 8):
        eps = {1: 0.75, 2: 0.50, 3: 0.37, 4: 0.30, 6: 0.21, 8: 0.16}[r]
        c = f'er+purity@d5_k16_r{r}'
        lines.append(f'Purity-certified, $r={r}$ ($\\varepsilon={eps:.2f}$) & '
                     + ' & '.join(cell(trade, ds, c) for ds in ('nf-cse-cic-ids2018-v3', 'nf-ton-iot-v3')) + '\\\\')
    lines.append('\\midrule')
    for c in (4, 8, 12, 16):
        name = f'er+representative@d5_k16_c{c}'
        lines.append(f'Representative, {c} clusters & '
                     + ' & '.join(cell(trade, ds, name) for ds in ('nf-cse-cic-ids2018-v3', 'nf-ton-iot-v3')) + '\\\\')
    (TAB / 'ablation_r.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-root', type=Path, default=HERE.parents[1] / 'results')
    args = parser.parse_args()
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    dev, held, trade = summaries()
    tradeoff_figure(trade, held)
    onset_figure(args.runs_root)
    main_table(dev, held)
    ablation_table(dev, held, trade)
