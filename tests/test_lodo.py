from __future__ import annotations

import json

import pytest

from tools.analyze_dasp_v2 import analyze_lodo, dataset_means, runs_frame, select_on
from tools.run_dasp_v2 import LODO_DATASETS, LODO_SEEDS, SUITES, jobs_for, parser

SCORES = {  # config -> dataset -> Macro-F1
    'A': {'a': 0.95, 'b': 0.95, 'c': 0.50},
    'B': {'a': 0.90, 'b': 0.90, 'c': 0.90},
}


def records():
    rows = []
    for method in ('ref', 'other'):
        for config, by_dataset in SCORES.items():
            for dataset, score in by_dataset.items():
                for scenario in ('S0', 'S1-Sudden', 'FS-volume'):
                    shift = 0.01 if method == 'ref' else 0.0
                    rows.append(dict(stage='lodo', dataset=dataset, scenario=scenario, stream_seed=1, model_seed=91,
                                     method=method, condition=f'{method}-{config}', overrides={'k': config},
                                     summary={'mean_macro_f1': score + shift, 'total_gradient_evaluations': 10}))
    return rows


def test_selection_never_sees_the_scored_dataset(tmp_path):
    frame = runs_frame(records())
    means = dataset_means(frame)
    # With c held out, A wins on a and b although it fails on c.
    assert select_on(means, ['a', 'b'])['ref']['overrides'] == {'k': 'A'}
    assert select_on(means, ['a', 'c'])['ref']['overrides'] == {'k': 'B'}
    plan = {'planned_runs': len(records())}
    (tmp_path / 'lodo_plan.json').write_text(json.dumps(plan), encoding='utf-8')
    (tmp_path / 'lodo_completed.json').write_text(json.dumps(records()), encoding='utf-8')
    result = analyze_lodo(tmp_path, reference='ref', replicates=200)
    assert result['folds']['c']['ref']['overrides'] == {'k': 'A'}
    summary = {row['condition']: row for row in result['summary']}
    assert summary['ref']['c'] == pytest.approx(0.51)  # A scored on c: the honest fold result
    assert summary['ref']['a'] == pytest.approx(0.91)  # with a held out, B wins on b and c
    final = json.loads((tmp_path / 'lodo_selection.json').read_text(encoding='utf-8'))['selection']
    assert final['ref']['overrides'] == {'k': 'B'}  # dataset-weighted over all three
    other = result['contrasts']['all_tracks'][0]
    assert other['condition'] == 'other' and other['all_cells']['mean_difference'] == pytest.approx(-0.01)


def test_lodo_plan_covers_five_datasets_with_their_seeds(tmp_path):
    jobs = jobs_for(parser().parse_args(['--stage', 'lodo', '--output-root', str(tmp_path),
                                         '--seeds-per-dataset', '2']))
    assert {job['dataset'] for job in jobs} == set(LODO_DATASETS)
    for dataset in LODO_DATASETS:
        assert {job['stream_seed'] for job in jobs if job['dataset'] == dataset} == set(LODO_SEEDS[dataset][:2])
    assert len(jobs) == len(SUITES['all']) * 24 * 5 * 21 * 2
    with pytest.raises(ValueError, match='leave-one-dataset-out'):
        jobs_for(parser().parse_args(['--stage', 'lodo', '--output-root', str(tmp_path), '--datasets', 'x']))
