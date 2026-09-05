from __future__ import annotations

import tools.run_revision_p1 as p1
from tools.run_revision_p1 import (
    COMPONENT_METHODS,
    NATURAL_METHODS,
    PERMUTATION_METHODS,
    _configure,
    _mode_spec,
)


def test_p1_mode_run_counts_are_locked() -> None:
    expected = {
        "component_smoke": (5, 5),
        "components": (225, 5),
        "permutation_smoke": (3, 3),
        "permutations": (405, 3),
        "natural_smoke": (4, 4),
        "natural": (12, 4),
    }
    for mode, (run_count, methods_per_group) in expected.items():
        _, specs, group_size = _mode_spec(mode)
        assert len(specs) == run_count
        assert group_size == methods_per_group


def test_p1_modes_contain_only_locked_methods() -> None:
    _, component_specs, _ = _mode_spec("components")
    _, permutation_specs, _ = _mode_spec("permutations")
    _, natural_specs, _ = _mode_spec("natural")
    assert {spec.method for spec in component_specs} == set(COMPONENT_METHODS)
    assert {spec.method for spec in permutation_specs} == set(PERMUTATION_METHODS)
    assert {spec.method for spec in natural_specs} == set(NATURAL_METHODS)


def test_b5_parameter_matched_dimensions_are_dataset_specific() -> None:
    output_root, specs, _ = _mode_spec("components")
    cic_spec = next(
        spec
        for spec in specs
        if spec.dataset == "cic-ids2017" and spec.method == "revision_b5_flat_global"
    )
    nf_spec = next(
        spec
        for spec in specs
        if spec.dataset == "nf-ton-iot-v3" and spec.method == "revision_b5_flat_global"
    )
    cic_config = _configure(output_root, cic_spec)
    nf_config = _configure(output_root, nf_spec)
    assert cic_config.model.architecture == "sequential"
    assert cic_config.model.hidden_dims == [184, 40]
    assert nf_config.model.architecture == "sequential"
    assert nf_config.model.hidden_dims == [136, 96]


def test_natural_mode_uses_common_label_exposure_contract(tmp_path, monkeypatch) -> None:
    natural_stream = tmp_path / "natural_chronological_v1.npz"
    natural_stream.touch()
    monkeypatch.setattr(p1, "NATURAL_STREAM", natural_stream)
    output_root, specs, _ = _mode_spec("natural")
    for spec in specs:
        config = _configure(output_root, spec)
        assert config.data.scenario == "Natural-Chronological"
        assert config.data.stream_seed == 4101
        assert config.revision.common_label_exposure_baselines
        assert config.revision.feature_permutation_seed is None
