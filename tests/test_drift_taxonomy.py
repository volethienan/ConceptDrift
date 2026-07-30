from __future__ import annotations

from tools.build_controlled_nf_streams import (
    DATASETS,
    _allocate,
    _window_recipe,
)


SPEC = DATASETS["nf-unsw-nb15-v3"]


def test_sudden_replaces_a_with_b_at_w12() -> None:
    pre, _ = _window_recipe(SPEC, "S1-Sudden", 11, 100, 0.05)
    post, phase = _window_recipe(
        SPEC, "S1-Sudden", 12, 100, 0.05
    )
    assert pre == _allocate(100, SPEC.concept_a)
    assert post == _allocate(100, SPEC.concept_b)
    assert phase == "B"


def test_gradual_interleaves_pure_a_and_b_windows() -> None:
    a_window, a_phase = _window_recipe(
        SPEC, "S2-Gradual", 12, 100, 0.05
    )
    b_window, b_phase = _window_recipe(
        SPEC, "S2-Gradual", 14, 100, 0.05
    )
    assert a_window == _allocate(100, SPEC.concept_a)
    assert b_window == _allocate(100, SPEC.concept_b)
    assert a_phase == "gradual_A"
    assert b_phase == "gradual_B"


def test_incremental_uses_intermediate_compositions() -> None:
    first, first_phase = _window_recipe(
        SPEC, "S3-Incremental", 12, 120, 0.05
    )
    last, last_phase = _window_recipe(
        SPEC, "S3-Incremental", 23, 120, 0.05
    )
    assert sum(first.values()) == 120
    assert first != _allocate(120, SPEC.concept_a)
    assert first != _allocate(120, SPEC.concept_b)
    assert last == _allocate(120, SPEC.concept_b)
    assert first_phase == "incremental_01"
    assert last_phase == "incremental_12"


def test_recurrent_returns_to_new_a_flows() -> None:
    middle, _ = _window_recipe(
        SPEC, "S4-Recurrent", 12, 100, 0.05
    )
    returned, phase = _window_recipe(
        SPEC, "S4-Recurrent", 24, 100, 0.05
    )
    assert "generic" in middle
    assert returned == _allocate(100, SPEC.concept_a)
    assert phase == "A2"


def test_blip_last_two_windows_then_returns() -> None:
    blip, blip_phase = _window_recipe(
        SPEC, "S5-Blip", 13, 100, 0.05
    )
    returned, returned_phase = _window_recipe(
        SPEC, "S5-Blip", 14, 100, 0.05
    )
    assert "generic" in blip
    assert blip_phase == "B_blip"
    assert returned == _allocate(100, SPEC.concept_a)
    assert returned_phase == "A2"


def test_combined_contains_sudden_incremental_and_recurrence() -> None:
    sudden, sudden_phase = _window_recipe(
        SPEC, "S6-Combined", 12, 100, 0.05
    )
    intermediate, intermediate_phase = _window_recipe(
        SPEC, "S6-Combined", 16, 100, 0.05
    )
    recurrence, recurrence_phase = _window_recipe(
        SPEC, "S6-Combined", 35, 100, 0.05
    )
    assert "generic" in sudden
    assert sudden_phase == "B_sudden"
    assert sum(intermediate.values()) == 100
    assert intermediate_phase == "B_to_C_incremental_01"
    assert recurrence == _allocate(100, SPEC.concept_a)
    assert recurrence_phase == "C_to_A_recurrence_08"


def test_local_levels_report_attack_fraction() -> None:
    expected = {
        "S7-Local-Low": 5,
        "S7-Local-Medium": 10,
        "S7-Local-High": 20,
    }
    for scenario, local_count in expected.items():
        recipe, _ = _window_recipe(
            SPEC, scenario, 12, 100, 0.05
        )
        assert recipe["reconnaissance"] == local_count
