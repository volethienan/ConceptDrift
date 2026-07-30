from __future__ import annotations

from tools.validate_controlled_streams import _construct_classification


def test_construct_classifier_does_not_equate_mmd_with_real_drift() -> None:
    result = _construct_classification(
        "S1",
        attack_mmd=0.10,
        probe_drop=-0.01,
        mmd_threshold=0.005,
        boundary_impact_threshold=0.02,
    )
    assert result == "distribution_shift_only_or_boundary_inconclusive"


def test_construct_classifier_marks_material_probe_degradation() -> None:
    result = _construct_classification(
        "S1",
        attack_mmd=0.10,
        probe_drop=0.25,
        mmd_threshold=0.005,
        boundary_impact_threshold=0.02,
    )
    assert result == "boundary_impacting_proxy"
