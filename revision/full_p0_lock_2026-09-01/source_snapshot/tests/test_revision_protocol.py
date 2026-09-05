from __future__ import annotations

import copy

import torch

from dasp_nids.config import load_config
from dasp_nids.model import ModularOnlineMLP
from dasp_nids.revision import (
    REVISION_CONDITIONS,
    equal_label_pools,
    index_digest,
    locality_diagnostics,
    parameter_snapshot,
    select_routing,
    select_training_indices,
    stable_seed,
)
from dasp_nids.runner import StreamingExperiment, _train_dasp
from dasp_nids.memory import ReservoirMemory


def _pools():
    return equal_label_pools(
        4096,
        512,
        384,
        128,
        "NF-ToN-IoT-v3",
        "S2-Gradual",
        2001,
        7,
        "1.1",
    )


def test_equal_label_query_is_label_blind_deterministic_and_disjoint() -> None:
    first = _pools()
    second = _pools()
    assert torch.equal(first.query, second.query)
    assert len(first.query) == 512
    assert len(first.repair_candidates) == 384
    assert len(first.gate) == 128
    assert len(torch.unique(first.query)) == 512
    assert not set(first.repair_candidates.tolist()) & set(first.gate.tolist())
    assert index_digest(first.query) == index_digest(second.query)


def test_routing_variants_share_training_indices() -> None:
    pools = _pools()
    labels = torch.arange(4096) % 2
    digests = set()
    for method, condition in REVISION_CONDITIONS.items():
        if condition.identifier == "A10":
            continue
        selected = select_training_indices(
            labels,
            pools.repair_candidates,
            128,
            pools.selection_seed,
            label_balanced=True,
        )
        assert len(selected) == 128, method
        digests.add(index_digest(selected))
    assert len(digests) == 1


def test_routing_policies_have_locked_k_and_random_is_deterministic() -> None:
    layers = ["module_1", "module_2", "module_3", "module_4"]
    requirements = dict(zip(layers, [0.2, 0.8, 0.4, 0.1]))
    conflicts = dict(zip(layers, [0.0, 0.5, 0.25, 1.0]))
    seed = stable_seed("revision-routing-primary-v1", "cell", 7)
    for condition in REVISION_CONDITIONS.values():
        if condition.identifier == "A10":
            continue
        first = select_routing(
            condition.routing_policy,
            layers,
            requirements,
            conflicts,
            condition.routing_k,
            1.0,
            1.0,
            1.0,
            ["module_1", "module_2", "module_3"],
            random_seed=seed,
        )
        second = select_routing(
            condition.routing_policy,
            layers,
            requirements,
            conflicts,
            condition.routing_k,
            1.0,
            1.0,
            1.0,
            ["module_1", "module_2", "module_3"],
            random_seed=seed,
        )
        assert len(first.active_modules) == condition.routing_k
        assert first == second
    conflict_only = select_routing(
        "conflict",
        layers,
        requirements,
        conflicts,
        1,
        1.0,
        1.0,
        1.0,
        [],
        random_seed=seed,
    )
    assert conflict_only.active_modules == ("module_1",)


def test_locality_logging_detects_strict_freezing_and_classifier_only() -> None:
    config = load_config("configs/modular_dasp_smoke.yaml")
    model = ModularOnlineMLP(16, 2, [32, 16], n_modules=4, module_dim=8)
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(64, 16, generator=generator)
    y = (x[:, 0] > 0).long()
    memory = ReservoirMemory(64, 123)
    memory.add(x, y)
    before = parameter_snapshot(model)
    evaluations = _train_dasp(
        model,
        x,
        y,
        {
            "module_1": 0.0,
            "module_2": 0.0,
            "module_3": 0.0,
            "module_4": 0.0,
            "classifier": 1.0,
        },
        config,
        torch.device("cpu"),
        memory,
        strict_freeze=True,
        use_projection_memory=False,
    )
    diagnostics = locality_diagnostics(model, before, (), evaluations, 1e-12)
    assert diagnostics["active_representation_module_count"] == 0
    assert diagnostics["active_representation_parameter_count"] == 0
    assert diagnostics["inactive_module_max_abs_change"] == 0.0
    assert diagnostics["changed_parameter_count"] > 0
    assert diagnostics["parameter_gradient_element_evaluations"] > 0


def _tiny_revision_config(method: str):
    config = copy.deepcopy(load_config("configs/modular_dasp_smoke.yaml"))
    config.dasp.baseline = method
    config.data.n_windows = 5
    config.data.warmup_windows = 2
    config.data.window_size = 96
    config.drift.calibration_windows = 2
    config.dasp.memory_size = 128
    config.dasp.memory_batch_size = 16
    config.dasp.modular_replay_size = 16
    config.dasp.modular_repair_epochs = 1
    config.dasp.modular_emergency_epochs = 1
    config.dasp.v12_guard_min_macro_f1 = 1.1
    config.revision.enabled = True
    config.revision.dataset_key = "synthetic-smoke"
    config.revision.label_budget = 48
    config.revision.repair_candidate_size = 36
    config.revision.gate_size = 12
    config.revision.optimization_size = 12
    config.revision.replay_size = 16
    config.revision.event_matched_diagnostics = False
    config.output.directory = "tmp/revision_protocol_tests"
    config.output.run_name = method
    return config


def test_revision_a1_smoke_populates_label_and_locality_logs() -> None:
    result = StreamingExperiment(_tiny_revision_config("revision_a1_combined")).run()
    revised = result.windows[~result.windows["warmup"]]
    assert (revised["labels_exposed_total"] == 48).all()
    assert (revised["repair_candidate_labels_exposed"] == 36).all()
    assert (revised["gate_labels_exposed"] == 12).all()
    assert revised["query_index_digest"].notna().all()
    assert revised["training_index_digest"].notna().all()
    repaired = revised[revised["revision_repair_applied"]]
    assert not repaired.empty
    assert (repaired["active_representation_module_count"] == 3).all()
    assert (repaired["inactive_module_max_abs_change"] == 0.0).all()


def test_revision_classifier_only_smoke_keeps_representation_exactly_frozen() -> None:
    result = StreamingExperiment(_tiny_revision_config("revision_a9_classifier")).run()
    revised = result.windows[~result.windows["warmup"]]
    repaired = revised[revised["revision_repair_applied"]]
    assert not repaired.empty
    assert (repaired["active_representation_module_count"] == 0).all()
    assert (repaired["active_representation_parameter_count"] == 0).all()
    assert (repaired["inactive_module_max_abs_change"] == 0.0).all()


def test_revision_ssf_smoke_uses_common_exposure_contract() -> None:
    result = StreamingExperiment(_tiny_revision_config("revision_a10_ssf")).run()
    revised = result.windows[~result.windows["warmup"]]
    assert (revised["labels_exposed_total"] == 48).all()
    assert (revised["labels_used_for_gating"] == 0).all()
    assert (revised["pseudo_labels_used"] == 0).all()
    assert (revised["new_training_labels_used"] <= 12).all()
    assert revised["training_index_digest"].notna().all()


def test_event_matched_diagnostic_evaluates_discarded_control_branches() -> None:
    config = _tiny_revision_config("revision_a1_combined")
    config.data.n_windows = 6
    config.revision.event_matched_diagnostics = True
    result = StreamingExperiment(config).run()
    completed = result.windows[result.windows["event_matched_complete"] == True]
    assert not completed.empty
    for identifier in config.revision.event_matched_controls:
        assert completed[f"event_matched_{identifier}_next_macro_f1"].notna().all()
        assert completed[
            f"event_matched_{identifier}_changed_parameter_fraction"
        ].notna().all()


def test_end_to_end_routing_controls_share_label_and_replay_digests() -> None:
    combined = StreamingExperiment(
        _tiny_revision_config("revision_a1_combined")
    ).run().windows
    random_control = StreamingExperiment(
        _tiny_revision_config("revision_a2_random")
    ).run().windows
    combined = combined[~combined["warmup"]].set_index("window")
    random_control = random_control[~random_control["warmup"]].set_index("window")
    for column in (
        "query_index_digest",
        "repair_candidate_index_digest",
        "gate_index_digest",
        "training_index_digest",
        "replay_index_digest",
    ):
        assert combined[column].equals(random_control[column])
