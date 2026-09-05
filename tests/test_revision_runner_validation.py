from __future__ import annotations

from tools.run_revision_experiments import (
    CONFIRMATION_MODEL_SEEDS,
    CONFIRMATION_STREAM_SEEDS,
    METHODS,
    MODEL_SEED,
    PRIMARY_SCENARIOS,
    SMOKE_DATASET,
    SMOKE_SCENARIO,
    STREAM_SEED,
    _configure,
    _mode_spec,
)


def test_smoke_matrix_is_locked_to_one_cell_and_all_conditions() -> None:
    _, datasets, scenarios, stream_seeds, model_seeds, event_matched = _mode_spec(
        "smoke"
    )
    assert datasets == [SMOKE_DATASET]
    assert scenarios == [SMOKE_SCENARIO]
    assert stream_seeds == [STREAM_SEED] == [2001]
    assert model_seeds == [MODEL_SEED] == [91]
    assert len(METHODS) == 10
    assert event_matched


def test_reduced_pilot_is_locked_to_150_runs() -> None:
    _, datasets, scenarios, stream_seeds, model_seeds, event_matched = _mode_spec(
        "reduced_pilot"
    )
    assert len(datasets) == 5
    assert len(scenarios) == 3
    assert len(stream_seeds) == 1
    assert len(model_seeds) == 1
    assert (
        len(datasets)
        * len(scenarios)
        * len(stream_seeds)
        * len(model_seeds)
        * len(METHODS)
        == 150
    )
    assert not event_matched


def test_full_p0_is_locked_to_500_cells_and_5000_runs() -> None:
    _, datasets, scenarios, stream_seeds, model_seeds, event_matched = _mode_spec(
        "full_p0"
    )
    assert len(datasets) == 5
    assert scenarios == PRIMARY_SCENARIOS
    assert stream_seeds == CONFIRMATION_STREAM_SEEDS
    assert model_seeds == CONFIRMATION_MODEL_SEEDS
    assert len(datasets) * len(scenarios) * len(stream_seeds) * len(model_seeds) == 500
    assert (
        len(datasets)
        * len(scenarios)
        * len(stream_seeds)
        * len(model_seeds)
        * len(METHODS)
        == 5000
    )
    assert not event_matched


def test_revision_run_configuration_preserves_locked_contract() -> None:
    output_root, _, _, _, _, event_matched = _mode_spec("smoke")
    config = _configure(
        output_root,
        SMOKE_DATASET,
        SMOKE_SCENARIO,
        "revision_a1_combined",
        event_matched,
    )
    assert config.data.source == "prepared_npz"
    assert config.data.window_size == 4096
    assert config.data.warmup_windows == 4
    assert config.data.stream_seed == 2001
    assert config.training.model_seed == 91
    assert config.revision.label_budget == 512
    assert config.revision.repair_candidate_size == 384
    assert config.revision.gate_size == 128
    assert config.revision.optimization_size == 128
    assert config.revision.replay_size == 512
    assert config.revision.event_matched_diagnostics
