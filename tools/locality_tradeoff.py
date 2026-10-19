"""Locality-accuracy trade-off: share of parameters changed per window vs Macro-F1.

    python tools/locality_tradeoff.py outputs_v2_screen10_blockwise outputs_v2_screen10_blockwise_controlled
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def run_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in (root / 'screen').glob('*/*/*/windows.csv'):
        condition = path.parts[-2].split('_stream')[0]
        frame = pd.read_csv(path)
        evaluated = frame[~frame.warmup.astype(bool)]
        record = {'condition': condition, 'dataset': path.parts[-4], 'scenario': path.parts[-3],
                  'macro_f1': evaluated.macro_f1.mean()}
        if 'optimizer_included_parameter_fraction' in evaluated:
            updated = evaluated[evaluated.optimizer_included_parameter_fraction.notna()]
            record.update({
                'optimizer_fraction': updated.optimizer_included_parameter_fraction.mean(),
                'changed_parameters': updated.changed_parameter_count.mean(),
                'module_update_share': (updated.localized_repair_module_count.fillna(0) > 0).mean(),
            })
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    tables = []
    for argument in sys.argv[1:]:
        table = run_table(Path(argument))
        table['track'] = Path(argument).name
        tables.append(table)
    frame = pd.concat(tables)
    summary = frame.pivot_table(index='condition', columns='track', values='macro_f1', aggfunc='mean')
    cost = frame.groupby('condition')[['optimizer_fraction', 'changed_parameters', 'module_update_share']].mean()
    out = summary.join(cost).sort_values('optimizer_fraction')
    pd.set_option('display.width', 220)
    print(out.round(4).to_string())


if __name__ == '__main__':
    main()
