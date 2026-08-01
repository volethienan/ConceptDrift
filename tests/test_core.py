from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from dasp_nids.config import load_config
from dasp_nids.data import (
    load_nf_v3_stream,
    load_prepared_stream,
    make_synthetic_stream,
)
from dasp_nids.baselines import build_gpm_subspaces, select_ssf_buffer
from dasp_nids.drift import DriftDecision, LayerDriftMonitor, multiscale_mmd
from dasp_nids.model import (
    AdapterOnlineMLP,
    LowRankResidualAdapter,
    ModularOnlineMLP,
    OnlineMLP,
)
from dasp_nids.runner import (
    PRIMARY_METHODS,
    StreamingExperiment,
    _clone_stateful_optimizer,
    _progressive_layer_candidates,
    _select_v9_candidate,
    _severity_weighted_value,
    _soft_reset_toward_reference,
    _surprise_class_balanced_weights,
    _counterfactual_utilities,
    _make_stateful_optimizer,
    _train_dasp,
    _label_blind_uniform_query_indices,
    _v18_split_query_train_guard,
    _update_v18_guard_history,
    _v18_label_query_indices,
    _v17_soft_commit,
    _v12_warning_eligible,
    _v13_class_balanced_score,
    _v16_class_balanced_score,
    _v16_oracle_event,
    set_deterministic_seed,
)
from dasp_nids.metrics import classification_metrics
from dasp_nids.memory import (
    ReservoirMemory,
    propose_layer_adaptive_memory,
    select_feature_similarity_replay,
    select_uncertainty_weighted_similarity_replay,
)


def test_deterministic_seed_configures_cublas_workspace(monkeypatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_deterministic_seed(7)
    assert torch.initial_seed() == 7
    assert __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpm_subspace_build_is_deterministic_on_cuda() -> None:
    set_deterministic_seed(17)
    device = torch.device("cuda")
    model = OnlineMLP(6, 2, [8, 4]).to(device)
    memory_x = torch.randn(32, 6, generator=torch.Generator().manual_seed(19))
    first = build_gpm_subspaces(model, memory_x, 0.95, device)
    second = build_gpm_subspaces(model, memory_x, 0.95, device)
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name])


def test_mmd_is_small_for_identity_and_larger_for_shift() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(128, 8, generator=generator)
    identical = multiscale_mmd(x, x)
    shifted = multiscale_mmd(x, x + 2.0)
    assert identical < 1e-7
    assert shifted > identical + 0.1


def test_emergency_drift_bypasses_persistence() -> None:
    monitor = LayerDriftMonitor(
        ["layer_1"],
        kappa=0.0,
        z_max=1.0,
        threshold=0.1,
        persistence=5,
        top_k=1,
        max_samples=64,
        kernel_scales=[1.0],
        emergency_threshold=0.5,
    )
    reference = {"layer_1": torch.zeros(64, 4)}
    calibration = [
        {"layer_1": torch.randn(64, 4, generator=torch.Generator().manual_seed(seed)) * 0.01}
        for seed in range(3)
    ]
    monitor.calibrate(reference, calibration)
    decision = monitor.evaluate(reference, {"layer_1": torch.ones(64, 4) * 3.0})
    assert decision.emergency
    assert decision.triggered


def test_model_exposes_nonoverlapping_layer_groups() -> None:
    model = OnlineMLP(10, 3, [16, 8])
    groups = model.adaptation_parameter_groups()
    identities = [id(parameter) for values in groups.values() for parameter in values]
    assert list(groups) == ["layer_1", "layer_2", "classifier"]
    assert len(identities) == len(set(identities))


def test_v17_adapter_training_cannot_change_frozen_backbone() -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.training.batch_size = 16
    model = AdapterOnlineMLP(6, 2, [8, 4], adapter_max_rank=4)
    model.set_adapter_rank(4, trainable_rank=2)
    with torch.no_grad():
        for adapter in model.adapters:
            adapter.up[:, 2:].normal_()
            adapter.down[2:].normal_()
    model.freeze_backbone()
    backbone_before = {
        name: value.detach().clone()
        for name, value in model.blocks.state_dict().items()
    }
    adaptation_before = [
        parameter.detach().clone()
        for values in model.adaptation_parameter_groups().values()
        for parameter in values
    ]
    inactive_before = [
        (adapter.down[2:].detach().clone(), adapter.up[:, 2:].detach().clone())
        for adapter in model.adapters
    ]
    generator = torch.Generator().manual_seed(1701)
    x = torch.randn(64, 6, generator=generator)
    y = (x[:, 0] > 0).long()
    memory = ReservoirMemory(64, 1701)
    memory.add(x, y)
    steps = _train_dasp(
        model,
        x,
        y,
        {"layer_1": 1.0, "layer_2": 1.0, "classifier": 1.0},
        config,
        torch.device("cpu"),
        memory,
        strict_freeze=True,
        epochs=1,
    )
    assert steps > 0
    assert all(
        torch.equal(backbone_before[name], value)
        for name, value in model.blocks.state_dict().items()
    )
    assert not any(parameter.requires_grad for parameter in model.blocks.parameters())
    assert all(
        torch.equal(down, adapter.down[2:])
        and torch.equal(up, adapter.up[:, 2:])
        for (down, up), adapter in zip(inactive_before, model.adapters)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            adaptation_before,
            [
                parameter.detach()
                for values in model.adaptation_parameter_groups().values()
                for parameter in values
            ],
        )
    )


def test_v17_adapter_rank_does_not_silently_reduce_residual_scale() -> None:
    adapter = LowRankResidualAdapter(3, 2, max_rank=2, scale=1.0)
    with torch.no_grad():
        adapter.down.fill_(1.0)
        adapter.up.fill_(1.0)
    x = torch.ones(1, 3)
    adapter.set_active_rank(1)
    rank_one = adapter(x)
    adapter.set_active_rank(2)
    rank_two = adapter(x)
    assert torch.allclose(rank_two, 2.0 * rank_one)


def test_v17_uncertainty_iwms_is_seeded_bounded_and_class_safe() -> None:
    generator = torch.Generator().manual_seed(1702)
    model = OnlineMLP(6, 2, [8, 4])
    memory_x = torch.randn(80, 6, generator=generator)
    memory_y = torch.tensor([0] * 56 + [1] * 24)
    current_x = torch.randn(32, 6, generator=generator) + 0.25
    kwargs = dict(
        model=model,
        memory_x=memory_x,
        memory_y=memory_y,
        current_unlabeled_x=current_x,
        active_layers=model.layer_names,
        replay_size=24,
        device=torch.device("cpu"),
        seed=1702,
        temperature=0.1,
        confidence_power=1.0,
        uniform_mix=0.25,
        batch_size=16,
    )
    first = select_uncertainty_weighted_similarity_replay(**kwargs)
    second = select_uncertainty_weighted_similarity_replay(**kwargs)
    assert first is not None and second is not None
    assert len(first.y) == 24
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.y, second.y)
    assert set(first.y.tolist()) == {0, 1}


def test_v18_modulation_changes_features_without_changing_backbone() -> None:
    model = AdapterOnlineMLP(6, 2, [8, 4], adapter_max_rank=4)
    model.enable_feature_modulation()
    model.freeze_backbone()
    x = torch.randn(16, 6, generator=torch.Generator().manual_seed(1801))
    backbone_before = {
        name: value.detach().clone()
        for name, value in model.blocks.state_dict().items()
    }
    _, before = model(x, return_representations=True)
    with torch.no_grad():
        model.modulators[0].gain.add_(0.5)
        model.modulators[0].bias.add_(0.25)
    _, after = model(x, return_representations=True)
    assert not torch.allclose(before["layer_1"], after["layer_1"])
    assert all(
        torch.equal(backbone_before[name], value)
        for name, value in model.blocks.state_dict().items()
    )
    assert not any(parameter.requires_grad for parameter in model.blocks.parameters())
    assert "modulation_layer_1" in model.adaptation_parameter_groups()


def test_v18_fused_inference_matches_adapter_model() -> None:
    generator = torch.Generator().manual_seed(18020)
    model = AdapterOnlineMLP(6, 2, [8, 4], adapter_max_rank=4)
    model.set_adapter_rank(4, trainable_rank=4)
    model.enable_input_adapter()
    model.enable_feature_modulation()
    with torch.no_grad():
        for group in [
            model.adapters.parameters(),
            model.modulators.parameters(),
            model.input_adapter.parameters(),
        ]:
            for parameter in group:
                parameter.copy_(
                    0.1
                    * torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    )
                )
    x = torch.randn(32, 6, generator=generator)
    model.eval()
    backbone_before = {
        name: value.detach().clone()
        for name, value in model.blocks.state_dict().items()
    }
    fused = model.to_fused_inference()
    assert torch.allclose(model(x), fused(x), atol=2e-5, rtol=2e-5)
    assert not any(parameter.requires_grad for parameter in fused.parameters())
    assert all(
        torch.equal(backbone_before[name], value)
        for name, value in model.blocks.state_dict().items()
    )
    with pytest.raises(RuntimeError, match="prediction-only"):
        fused(x, return_representations=True)


def test_v18_robust_iwms_increases_balance_under_prediction_collapse() -> None:
    generator = torch.Generator().manual_seed(1802)
    model = OnlineMLP(6, 2, [8, 4])
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([8.0, -8.0]))
    memory_x = torch.randn(80, 6, generator=generator)
    memory_y = torch.tensor([0] * 40 + [1] * 40)
    current_x = torch.randn(32, 6, generator=generator)
    replay = select_uncertainty_weighted_similarity_replay(
        model,
        memory_x,
        memory_y,
        current_x,
        model.layer_names,
        replay_size=24,
        device=torch.device("cpu"),
        seed=1802,
        uniform_mix=0.25,
        max_uniform_mix=0.75,
        js_threshold=0.10,
        batch_size=16,
    )
    assert replay is not None
    assert replay.predicted_per_class == (32, 0)
    assert replay.effective_uniform_mix > 0.70
    assert replay.selected_per_class[1] >= 9
    assert replay.prior_js_divergence > 0.10


def test_v18_label_query_is_bounded_unique_and_label_blind() -> None:
    generator = torch.Generator().manual_seed(1803)
    model = OnlineMLP(6, 2, [8, 4])
    memory = ReservoirMemory(64, 1803)
    memory_x = torch.randn(64, 6, generator=generator)
    memory_y = torch.tensor([0, 1] * 32)
    memory.add(memory_x, memory_y)
    current_x = torch.randn(96, 6, generator=generator) + 0.5
    first, diagnostics = _v18_label_query_indices(
        model,
        current_x,
        memory,
        24,
        torch.device("cpu"),
        1803,
        entropy_weight=0.5,
        novelty_weight=0.5,
        disagreement_weight=0.25,
        global_fraction=0.5,
        candidate_multiplier=4,
        batch_size=16,
    )
    second, _ = _v18_label_query_indices(
        model,
        current_x,
        memory,
        24,
        torch.device("cpu"),
        1803,
        entropy_weight=0.5,
        novelty_weight=0.5,
        disagreement_weight=0.25,
        global_fraction=0.5,
        candidate_multiplier=4,
        batch_size=16,
    )
    assert len(first) == 24
    assert len(torch.unique(first)) == 24
    assert torch.equal(first, second)
    assert int(first.min()) >= 0 and int(first.max()) < len(current_x)
    assert diagnostics["v18_query_size"] == 24


def test_v18_prototype_query_spreads_classes_when_classifier_collapses() -> None:
    model = OnlineMLP(2, 2, [2])
    with torch.no_grad():
        linear = model.blocks[0].net[0]
        linear.weight.copy_(torch.eye(2))
        linear.bias.zero_()
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([8.0, -8.0]))
    generator = torch.Generator().manual_seed(18030)
    class_zero = torch.randn(40, 2, generator=generator) * 0.05
    class_zero[:, 0] -= 2.0
    class_one = torch.randn(40, 2, generator=generator) * 0.05
    class_one[:, 0] += 2.0
    memory = ReservoirMemory(80, 18030)
    memory.add(
        torch.cat([class_zero, class_one]),
        torch.tensor([0] * 40 + [1] * 40),
    )
    current = torch.cat([class_zero + 0.02, class_one - 0.02])
    selected, diagnostics = _v18_label_query_indices(
        model,
        current,
        memory,
        20,
        torch.device("cpu"),
        18030,
        entropy_weight=0.5,
        novelty_weight=0.5,
        disagreement_weight=0.25,
        global_fraction=0.25,
        candidate_multiplier=4,
        batch_size=16,
    )
    assert diagnostics["v18_query_predicted_per_class"] == "80|0"
    prototype_counts = [
        int(value)
        for value in str(
            diagnostics["v18_query_selected_prototype_per_class"]
        ).split("|")
    ]
    assert len(selected) == 20
    assert min(prototype_counts) >= 5


def test_label_blind_uniform_query_is_deterministic_and_bounded() -> None:
    first = _label_blind_uniform_query_indices(100, 24, 18031)
    second = _label_blind_uniform_query_indices(100, 24, 18031)
    assert torch.equal(first, second)
    assert len(first) == 24
    assert len(torch.unique(first)) == 24
    assert int(first.min()) >= 0 and int(first.max()) < 100


def test_v18_current_guard_is_disjoint_and_class_spread() -> None:
    generator = torch.Generator().manual_seed(18032)
    x = torch.randn(40, 4, generator=generator)
    y = torch.tensor([0] * 24 + [1] * 10 + [2] * 6)
    train_x, train_y, guard_x, guard_y = _v18_split_query_train_guard(
        x, y, 0.25, 2, 18032
    )
    assert len(train_y) + len(guard_y) == len(y)
    assert len(guard_y) == 10
    assert set(guard_y.tolist()) == {0, 1, 2}
    assert set(train_y.tolist()) == {0, 1, 2}
    combined = torch.cat([train_x, guard_x])
    assert len(torch.unique(combined, dim=0)) == len(y)


def test_v18_guard_history_is_bounded_and_class_balanced() -> None:
    generator = torch.Generator().manual_seed(1804)
    x = torch.randn(100, 4, generator=generator)
    y = torch.tensor([0] * 80 + [1] * 20)
    history = _update_v18_guard_history(None, x, y, 32, 1804)
    assert history is not None
    assert len(history[1]) == 32
    counts = torch.bincount(history[1], minlength=2)
    assert counts.tolist() == [16, 16]


def test_v18_tiny_runner_preserves_budget_and_no_global_update(
    tmp_path,
) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.dasp.baseline = "dasp_v18"
    config.data.n_windows = 6
    config.data.warmup_windows = 3
    config.data.window_size = 64
    config.training.batch_size = 16
    config.dasp.memory_size = 128
    config.dasp.memory_batch_size = 16
    config.baselines.update_label_budget = 16
    config.dasp.v17_adapter_max_rank = 4
    config.dasp.v17_fast_rank = 2
    config.dasp.v17_strong_rank = 4
    config.dasp.v17_fast_replay_size = 16
    config.dasp.v17_strong_replay_size = 32
    config.dasp.v17_refresh_size = 16
    config.dasp.v17_iwms_query_size = 32
    config.dasp.v18_guard_history_size = 32
    config.output.directory = str(tmp_path)
    config.output.run_name = "v18-technical-smoke"
    result = StreamingExperiment(config).run()
    assert result.summary["global_update_count"] == 0
    evaluated = result.windows[~result.windows["warmup"]]
    assert evaluated["ground_truth_update_labels"].max() <= 16
    assert set(evaluated["v18_guard_source"].dropna()) == {
        "current_label_holdout"
    }


def test_fast_dasp_stationary_gate_skips_gradients_and_uses_small_budget(
    tmp_path,
) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.dasp.baseline = "dasp"
    config.dasp.fast_warning_severity = 1.1
    config.dasp.fast_stable_label_budget = 8
    config.dasp.fast_guard_history_size = 16
    config.dasp.v16_performance_min_history = 100
    config.output.directory = str(tmp_path)
    config.output.run_name = "fast-stationary"
    result = StreamingExperiment(config).run()
    evaluated = config.data.n_windows - config.data.warmup_windows
    assert result.summary["fast_revision"] == "event-gated-fused-v1"
    assert result.summary["fast_stable_skips"] == evaluated
    assert result.summary["fast_layer_detector_calls"] == 0
    assert result.summary["fast_adaptation_events"] == 0
    assert result.summary["total_gradient_evaluations"] == 0
    assert (
        result.summary["total_ground_truth_update_labels"]
        == 8 * evaluated
    )
    assert result.summary["global_update_count"] == 0
    assert set(
        result.windows.loc[
            ~result.windows["warmup"].astype(bool),
            "selected_candidate",
        ]
    ) == {"stable_memory_refresh"}


def test_fast_dasp_warning_path_retains_selective_adaptation(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.dasp.baseline = "dasp"
    config.dasp.fast_warning_severity = 0.0
    config.dasp.fast_input_gate_samples = 16
    config.dasp.fast_fast_replay_size = 8
    config.dasp.fast_strong_replay_size = 16
    config.dasp.fast_fast_epochs = 1
    config.dasp.fast_strong_epochs = 1
    config.dasp.v17_iwms_query_size = 16
    config.dasp.memory_batch_size = 8
    config.baselines.update_label_budget = 16
    config.output.directory = str(tmp_path)
    config.output.run_name = "fast-warning"
    result = StreamingExperiment(config).run()
    evaluated = config.data.n_windows - config.data.warmup_windows
    assert result.summary["fast_layer_detector_calls"] == evaluated
    assert result.summary["fast_adaptation_events"] > 0
    assert result.summary["total_gradient_evaluations"] > 0
    assert result.summary["global_update_count"] == 0
    assert not result.windows["global_update"].astype(bool).any()


def test_v17_soft_commit_selects_largest_safe_partial_step(monkeypatch) -> None:
    fallback = AdapterOnlineMLP(4, 2, [6, 3], adapter_max_rank=4)
    fallback.set_adapter_rank(2, trainable_rank=2)
    candidate = copy.deepcopy(fallback)
    with torch.no_grad():
        candidate.classifier.bias.add_(1.0)
    base_bias = fallback.classifier.bias.detach().clone()

    def fake_diagnostics(_fallback, blended, *_args, **_kwargs):
        magnitude = float(
            torch.max(torch.abs(blended.classifier.bias - base_bias)).item()
        )
        candidate_macro = 0.70 if magnitude > 0.60 else 0.82
        return {
            "guard_recent_lcb": 0.0,
            "guard_recent_mean_delta": 0.0,
            "guard_memory_lcb": 0.0,
            "guard_memory_mean_delta": 0.0,
            "guard_combined_lcb": -1.0,
            "guard_attack_lcb": 0.0,
            "guard_attack_mean_delta": 0.0,
            "guard_memory_weight": 1.0,
            "guard_candidate_macro_f1": candidate_macro,
            "guard_fallback_macro_f1": 0.80,
            "guard_candidate_attack_recall": 0.90,
            "guard_fallback_attack_recall": 0.90,
            "guard_candidate_min_per_class_recall": 0.80,
            "guard_fallback_min_per_class_recall": 0.80,
        }

    monkeypatch.setattr(
        "dasp_nids.runner._class_guard_diagnostics", fake_diagnostics
    )
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    chosen, alpha, diagnostics = _v17_soft_commit(
        fallback,
        candidate,
        (torch.zeros(4, 4), torch.zeros(4, dtype=torch.long)),
        None,
        config,
        torch.device("cpu"),
    )
    assert alpha == 0.5
    assert diagnostics["v17_soft_commit_accepted"]
    assert torch.allclose(
        chosen.classifier.bias, base_bias + 0.5, atol=1e-7
    )


def test_synthetic_stream_is_seed_reproducible() -> None:
    left = make_synthetic_stream("S1", 6, 64, 10, 5)
    right = make_synthetic_stream("S1", 6, 64, 10, 5)
    assert torch.equal(left.windows[4].x, right.windows[4].x)
    assert torch.equal(left.windows[4].y, right.windows[4].y)


def test_smoke_run_writes_expected_outputs(tmp_path) -> None:
    config = load_config("configs/smoke.yaml")
    config = copy.deepcopy(config)
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-smoke"
    result = StreamingExperiment(config).run()
    assert (result.output_dir / "windows.csv").exists()
    assert (result.output_dir / "summary.json").exists()
    assert np.isfinite(result.summary["mean_macro_f1"])
    assert len(result.windows) == config.data.n_windows
    assert result.summary["p95_forward_seconds"] >= 0.0
    assert result.summary["prediction_decision_throughput_flows_per_second"] > 0


def test_prepared_stream_preserves_order_and_source_ids(tmp_path) -> None:
    x = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2)
    y = np.asarray([[0, 1, 0, 1]] * 3, dtype=np.int64)
    source_ids = np.arange(12, dtype=np.int64).reshape(3, 4)
    path = tmp_path / "stream.npz"
    np.savez(
        path,
        x=x,
        y=y,
        phases=np.asarray(["warmup", "A", "B"]),
        feature_names=np.asarray(["f0", "f1"]),
        class_names=np.asarray(["Benign", "Attack"]),
        source_ids=source_ids,
    )
    stream = load_prepared_stream(path)
    assert len(stream.windows) == 3
    assert torch.equal(stream.windows[2].x, torch.tensor(x[2].tolist()))
    assert np.array_equal(stream.windows[1].source_ids, source_ids[1])
    assert stream.windows[2].phase == "B"


def test_v16_oracle_schedule_uses_labeled_drift_windows() -> None:
    assert not _v16_oracle_event("S0", 12)
    assert _v16_oracle_event("S1", 12)
    assert _v16_oracle_event("S3", 12)
    assert _v16_oracle_event("S3", 24)
    assert not _v16_oracle_event("S3", 23)
    assert _v16_oracle_event("S5", 12)


def test_v16_selective_only_smoke_has_no_global_update(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.dasp.baseline = "dasp_v16"
    config.output.directory = str(tmp_path)
    config.output.run_name = "v16-selective"
    config.dasp.memory_size = 128
    config.dasp.memory_batch_size = 32
    config.dasp.v7_replay_size = 64
    config.dasp.v14_light_current_size = 32
    config.dasp.v15_light_replay_size = 32
    config.dasp.v13_refresh_size = 8
    config.baselines.update_label_budget = 16
    result = StreamingExperiment(config).run()
    assert result.summary["global_update_count"] == 0
    assert not result.windows["global_update"].any()
    assert result.windows["ground_truth_update_labels"].max() <= 16


def test_attack_recall_is_undefined_without_attacks() -> None:
    metrics = classification_metrics(np.zeros(8, dtype=int), np.zeros(8, dtype=int), 2)
    assert np.isnan(metrics["attack_recall"])
    assert metrics["attack_fn"] == 0


def test_counterfactual_probe_returns_bounded_layer_plasticities() -> None:
    config = load_config("configs/smoke.yaml")
    model = OnlineMLP(6, 2, [8, 4])
    generator = torch.Generator().manual_seed(13)
    train_x = torch.randn(32, 6, generator=generator)
    train_y = (train_x[:, 0] > 0).long()
    guard_x = torch.randn(16, 6, generator=generator)
    guard_y = (guard_x[:, 0] > 0).long()
    memory = (torch.randn(16, 6, generator=generator), torch.zeros(16, dtype=torch.long))
    gains, costs, plasticities = _counterfactual_utilities(
        model, train_x, train_y, guard_x, guard_y, memory, config, torch.device("cpu")
    )
    assert set(gains) == set(model.layer_names)
    assert all(value >= 0 for value in costs.values())
    assert all(0.0 <= value <= 1.0 for value in plasticities.values())


def test_v9_preselection_skips_harmful_and_routes_uniform_utility() -> None:
    plasticities = {"layer_1": 1.0, "layer_2": 0.9, "classifier": 0.8}
    selected, selected_plasticities, diagnostics = _select_v9_candidate(
        gains={"layer_1": 1.0, "layer_2": 0.9, "classifier": 0.8},
        costs={"layer_1": 0.0, "layer_2": 0.0, "classifier": 0.0},
        plasticities=plasticities,
        forgetting_weight=1.0,
        minimum_utility=0.0,
        global_utility_ratio=0.75,
    )
    assert selected == "global"
    assert len(set(selected_plasticities.values())) == 1
    assert diagnostics["v9_positive_utility_layers"] == 3

    selected, selected_plasticities, _ = _select_v9_candidate(
        gains={"layer_1": 1.0, "layer_2": 0.1, "classifier": -0.1},
        costs={"layer_1": 0.0, "layer_2": 0.0, "classifier": 0.0},
        plasticities={"layer_1": 1.0, "layer_2": 0.1, "classifier": 0.0},
        forgetting_weight=1.0,
        minimum_utility=0.0,
        global_utility_ratio=0.75,
    )
    assert selected == "layer"
    assert selected_plasticities["classifier"] == 0.0

    selected, selected_plasticities, _ = _select_v9_candidate(
        gains={"layer_1": -0.1, "layer_2": 0.0, "classifier": -0.2},
        costs={"layer_1": 0.0, "layer_2": 0.0, "classifier": 0.0},
        plasticities={"layer_1": 0.0, "layer_2": 0.0, "classifier": 0.0},
        forgetting_weight=1.0,
        minimum_utility=0.0,
        global_utility_ratio=0.75,
    )
    assert selected is None
    assert selected_plasticities == {}


def test_v10_progressive_candidates_are_nested_and_strictly_layerwise() -> None:
    candidates, diagnostics = _progressive_layer_candidates(
        gains={"layer_1": 0.8, "layer_2": 0.3, "layer_3": -0.1},
        costs={"layer_1": 0.1, "layer_2": 0.1, "layer_3": 0.0},
        plasticities={"layer_1": 1.0, "layer_2": 0.3, "layer_3": 0.0},
        all_groups=["layer_1", "layer_2", "layer_3", "classifier"],
        forgetting_weight=1.0,
        minimum_utility=0.0,
        max_adaptive_layers=2,
        minimum_active_plasticity=0.25,
        classifier_plasticity=1.0,
    )
    assert [name for name, _ in candidates] == ["layer_top1", "layer_top2"]
    top1 = candidates[0][1]
    top2 = candidates[1][1]
    assert top1 == {
        "layer_1": 1.0,
        "layer_2": 0.0,
        "layer_3": 0.0,
        "classifier": 1.0,
    }
    assert top2["layer_1"] > 0.0 and top2["layer_2"] >= 0.25
    assert top2["layer_3"] == 0.0
    assert diagnostics["v10_ranked_layers"] == "layer_1,layer_2"


def test_v10_1_progressive_candidates_can_expand_to_three_layers() -> None:
    candidates, _ = _progressive_layer_candidates(
        gains={"layer_1": 0.8, "layer_2": 0.6, "layer_3": 0.4},
        costs={"layer_1": 0.1, "layer_2": 0.1, "layer_3": 0.1},
        plasticities={"layer_1": 1.0, "layer_2": 0.7, "layer_3": 0.4},
        all_groups=["layer_1", "layer_2", "layer_3", "classifier"],
        forgetting_weight=1.0,
        minimum_utility=0.0,
        max_adaptive_layers=3,
        minimum_active_plasticity=0.25,
        classifier_plasticity=1.0,
    )
    assert [name for name, _ in candidates] == [
        "layer_top1",
        "layer_top2",
        "layer_top3",
    ]
    top3 = candidates[-1][1]
    assert all(top3[name] > 0.0 for name in ["layer_1", "layer_2", "layer_3"])
    assert len({top3[name] for name in ["layer_1", "layer_2", "layer_3"]}) == 3
    assert top3["classifier"] == 1.0


def test_v10_3_surprise_weights_upweight_minority_class_with_equal_loss() -> None:
    model = OnlineMLP(4, 2, [6, 3])
    for parameter in model.parameters():
        parameter.data.zero_()
    x = torch.zeros(10, 4)
    y = torch.tensor([0] * 8 + [1] * 2)
    weights = _surprise_class_balanced_weights(
        model,
        x,
        y,
        base_weight=3.0,
        maximum_weight=6.0,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(weights).all()
    assert float(weights.min()) >= 1.0
    assert float(weights.max()) <= 6.0
    assert float(weights[y == 1].mean()) > float(weights[y == 0].mean())


def test_v10_strict_freeze_leaves_unselected_layer_unchanged() -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    model = OnlineMLP(6, 2, [8, 4])
    memory = ReservoirMemory(32, seed=19)
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(32, 6, generator=generator)
    y = (x[:, 0] > 0).long()
    frozen_before = [
        parameter.detach().clone()
        for parameter in model.adaptation_parameter_groups()["layer_2"]
    ]
    _train_dasp(
        model,
        x,
        y,
        {"layer_1": 1.0, "layer_2": 0.0, "classifier": 1.0},
        config,
        torch.device("cpu"),
        memory,
        strict_freeze=True,
    )
    frozen_after = model.adaptation_parameter_groups()["layer_2"]
    assert all(
        torch.equal(before, after)
        for before, after in zip(frozen_before, frozen_after)
    )


def test_stateful_optimizer_clone_preserves_state_and_rebinds_parameters() -> None:
    config = load_config("configs/smoke.yaml")
    model = OnlineMLP(6, 2, [8, 4])
    optimizer = _make_stateful_optimizer(model, config)
    x = torch.randn(16, 6)
    y = (x[:, 0] > 0).long()
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(x), y).backward()
    optimizer.step()
    cloned_model = copy.deepcopy(model)
    cloned_optimizer = _clone_stateful_optimizer(optimizer, cloned_model, config)
    assert len(cloned_optimizer.state) == len(optimizer.state)
    original_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    cloned_ids = {id(p) for group in cloned_optimizer.param_groups for p in group["params"]}
    assert original_ids.isdisjoint(cloned_ids)


def test_modular_mlp_exposes_disjoint_feature_modules() -> None:
    model = ModularOnlineMLP(16, 3, [32, 16], n_modules=4, module_dim=8)
    x = torch.randn(5, 16)
    logits, representations = model(x, return_representations=True)
    assert logits.shape == (5, 3)
    assert list(representations) == ["module_1", "module_2", "module_3", "module_4"]
    groups = model.adaptation_parameter_groups()
    identities = [id(p) for values in groups.values() for p in values]
    assert len(identities) == len(set(identities))


def test_v5_portfolio_smoke_run_records_candidate_selection(tmp_path) -> None:
    config = load_config("configs/smoke.yaml")
    config = copy.deepcopy(config)
    config.data.scenario = "S3"
    config.data.n_windows = 9
    config.data.warmup_windows = 3
    config.drift.persistence = 1
    config.drift.severity_threshold = 0.0
    config.dasp.baseline = "utility_portfolio_v5"
    config.dasp.lcb_z = 1.96
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v5-portfolio"
    result = StreamingExperiment(config).run()
    assert "selected_candidate" in result.windows.columns
    assert set(result.windows["selected_candidate"]) <= {
        "none", "online", "global", "layer"
    }
    assert result.summary["total_gradient_evaluations"] > 0


def test_severity_weighted_stability_reaches_configured_floor() -> None:
    assert _severity_weighted_value(2.0, 0.25, 0.0) == pytest.approx(2.0)
    assert _severity_weighted_value(2.0, 0.25, 1.0) == pytest.approx(0.5)


def test_soft_reset_interpolates_parameters_without_mutating_inputs() -> None:
    model = OnlineMLP(6, 2, [8, 4])
    reference = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(2.0)
    original = copy.deepcopy(model.state_dict())
    candidate = _soft_reset_toward_reference(model, reference, 0.25)
    for name, value in candidate.state_dict().items():
        if value.is_floating_point():
            expected = original[name] * 0.75 + reference.state_dict()[name] * 0.25
            assert torch.allclose(value, expected)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_v6_reactive_dasp_smoke_run_records_recurrence(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.data.scenario = "S3"
    config.data.n_windows = 9
    config.data.warmup_windows = 3
    config.drift.persistence = 1
    config.drift.severity_threshold = 0.0
    config.drift.emergency_severity_threshold = 0.0
    config.dasp.recurrence_departure_threshold = 0.0
    config.dasp.recurrence_return_threshold = 1.0
    config.dasp.baseline = "reactive_dasp_v6"
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v6-reactive"
    result = StreamingExperiment(config).run()
    assert result.windows["recurrence_detected"].any()
    assert set(result.windows["selected_candidate"]) <= {
        "none", "online", "global", "layer", "anchor"
    }
    assert result.summary["total_gradient_evaluations"] > 0


def test_v9_records_emergency_recurrence_and_uses_bounded_portfolio(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.data.scenario = "S3"
    config.data.n_windows = 9
    config.data.warmup_windows = 3
    config.drift.persistence = 5
    config.drift.severity_threshold = 0.9
    config.drift.emergency_severity_threshold = 0.0
    config.dasp.recurrence_departure_threshold = 0.0
    config.dasp.recurrence_return_threshold = 1.0
    config.dasp.baseline = "dasp_v9"
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v9"
    result = StreamingExperiment(config).run()
    assert result.windows["emergency_trigger"].any()
    assert result.windows["recurrence_detected"].any()
    assert set(result.windows["selected_candidate"]) <= {
        "none", "replay_fallback", "global", "layer", "anchor"
    }
    assert result.summary["total_gradient_evaluations"] > 0


def test_v9_uses_less_gradient_compute_than_v7_under_forced_drift(tmp_path) -> None:
    base = copy.deepcopy(load_config("configs/smoke.yaml"))
    base.data.scenario = "S3"
    base.data.n_windows = 9
    base.data.warmup_windows = 3
    base.drift.persistence = 1
    base.drift.severity_threshold = -1.0
    base.drift.emergency_severity_threshold = 2.0
    base.dasp.recurrence_departure_threshold = 2.0
    results = {}
    for method in ["dasp_v7", "dasp_v9", "dasp_v9_1", "dasp_v9_2"]:
        config = copy.deepcopy(base)
        config.dasp.baseline = method
        config.output.directory = str(tmp_path)
        config.output.run_name = f"compute-{method}"
        results[method] = StreamingExperiment(config).run()
    assert (
        results["dasp_v9"].summary["total_gradient_evaluations"]
        < results["dasp_v7"].summary["total_gradient_evaluations"]
    )
    assert (
        results["dasp_v9_1"].summary["total_gradient_evaluations"]
        < results["dasp_v7"].summary["total_gradient_evaluations"]
    )
    assert (
        results["dasp_v9_2"].summary["total_gradient_evaluations"]
        < results["dasp_v7"].summary["total_gradient_evaluations"]
    )


def test_sampling_does_not_change_reservoir_replacement() -> None:
    left = ReservoirMemory(32, seed=17)
    right = ReservoirMemory(32, seed=17)
    x1 = torch.arange(80, dtype=torch.float32).reshape(40, 2)
    y1 = torch.arange(40) % 3
    left.add(x1, y1)
    right.add(x1, y1)
    right.sample(8, torch.device("cpu"))
    right.sample(12, torch.device("cpu"))
    x2 = torch.arange(80, 160, dtype=torch.float32).reshape(40, 2)
    y2 = torch.arange(40, 80) % 3
    left.add(x2, y2)
    right.add(x2, y2)
    assert torch.equal(left.x, right.x)
    assert torch.equal(left.y, right.y)


@pytest.mark.parametrize(
    "method",
    ["static", "replay", "online_ewc", "agem", "river_logreg", "river_hat", "river_arf"],
)
def test_external_baseline_smoke_runs(method, tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.dasp.baseline = method
    config.output.directory = str(tmp_path)
    config.output.run_name = f"test-{method}"
    result = StreamingExperiment(config).run()
    assert np.isfinite(result.summary["mean_macro_f1"])
    assert len(result.windows) == config.data.n_windows
    assert result.summary["evaluated_windows"] == (
        config.data.n_windows - config.data.warmup_windows
    )
    if method.startswith("river_"):
        assert result.summary["model_parameter_bytes"] > 0
        assert "peak_rss_increment_bytes" in result.summary
        assert "peak_cuda_reserved_bytes" in result.summary


def test_primary_benchmark_exposes_the_eight_declared_methods() -> None:
    assert PRIMARY_METHODS == [
        "online_mlp",
        "experience_replay",
        "online_ewc",
        "agem",
        "ssf",
        "repshield",
        "cdda_md",
        "modular_dasp",
    ]


@pytest.mark.parametrize(
    "method",
    [
        "ssf",
        "repshield",
        "cdda_md",
        "dasp",
        "dasp_v9_1",
        "dasp_v9_2",
        "dasp_v10",
        "dasp_v10_1",
        "dasp_v10_2",
        "dasp_v10_3",
        "dasp_v11_novelty",
        "dasp_v11",
        "dasp_v11_1",
        "dasp_v12",
        "dasp_v13_iwms",
        "dasp_v13_refresh",
        "dasp_v13",
        "dasp_v14_light",
        "dasp_v14",
        "dasp_v15_light",
        "dasp_v15",
    ],
)
def test_new_primary_method_smoke_runs(method, tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.dasp.baseline = method
    config.output.directory = str(tmp_path)
    config.output.run_name = f"test-primary-{method}"
    result = StreamingExperiment(config).run()
    assert np.isfinite(result.summary["mean_macro_f1"])
    assert "window_stability_macro_f1" in result.summary
    assert "mean_online_forgetting" in result.summary
    assert len(result.windows) == config.data.n_windows


def test_historical_dasp_fallbacks_match_without_drift(tmp_path) -> None:
    base = copy.deepcopy(load_config("configs/smoke.yaml"))
    base.drift.severity_threshold = 2.0
    base.drift.emergency_severity_threshold = 2.0
    base.dasp.v7_balanced_replay = False
    base.dasp.v7_replay_size = 2 * base.dasp.memory_batch_size
    base.dasp.recurrence_departure_threshold = 2.0
    results = {}
    for method in [
        "dasp_v7",
        "dasp_v9",
        "dasp_v9_1",
        "dasp_v9_2",
        "dasp_v10",
        "dasp_v10_1",
        "dasp_v10_2",
        "dasp_v10_3",
        "dasp_v11_novelty",
        "dasp_v11",
        "dasp_v11_1",
        "dasp_v12",
    ]:
        config = copy.deepcopy(base)
        config.dasp.baseline = method
        config.output.directory = str(tmp_path)
        config.output.run_name = f"fallback-{method}"
        results[method] = StreamingExperiment(config).run().windows
    reference = results["dasp_v7"]["macro_f1"].to_numpy()
    for method, frame in results.items():
        assert np.array_equal(reference, frame["macro_f1"].to_numpy()), method


def test_v12_warning_requires_shift_bad_guard_and_no_confirmed_alarm() -> None:
    warning = DriftDecision({}, {}, {}, severity=0.4, triggered=False)
    alarm = DriftDecision({}, {}, {}, severity=0.4, triggered=True)
    kwargs = {
        "severity_threshold": 0.15,
        "guard_macro_f1": 0.79,
        "guard_attack_recall": 0.90,
        "minimum_macro_f1": 0.80,
        "minimum_attack_recall": 0.80,
    }
    assert _v12_warning_eligible(warning, **kwargs)
    assert not _v12_warning_eligible(alarm, **kwargs)
    assert not _v12_warning_eligible(
        DriftDecision({}, {}, {}, severity=0.10, triggered=False), **kwargs
    )
    assert not _v12_warning_eligible(
        warning,
        **{
            **kwargs,
            "guard_macro_f1": 0.90,
            "guard_attack_recall": 0.90,
        },
    )


def test_ssf_budget_mode_never_spends_extra_ground_truth_labels() -> None:
    old_x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    old_y = torch.tensor([0, 1] * 5)
    new_x = old_x + 100
    new_y = torch.tensor([1, 0] * 5)
    old_scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0])
    new_scores = torch.linspace(0.0, 1.0, 10)
    result = select_ssf_buffer(
        old_x,
        old_y,
        new_x,
        new_y,
        old_scores,
        new_scores,
        capacity=10,
        label_budget=2,
        drift=True,
        allow_pseudo_labels=False,
    )
    _, _, _, forgotten, inserted, ground_truth, pseudo = result
    assert forgotten == inserted == ground_truth == 2
    assert pseudo == 0


def test_ssf_official_mode_uses_pseudo_labels_beyond_budget() -> None:
    old_x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    old_y = torch.tensor([0, 1] * 5)
    new_x = old_x + 100
    new_y = torch.tensor([1, 0] * 5)
    pseudo_y = torch.zeros(10, dtype=torch.long)
    old_scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0])
    new_scores = torch.linspace(0.0, 1.0, 10)
    result = select_ssf_buffer(
        old_x,
        old_y,
        new_x,
        new_y,
        old_scores,
        new_scores,
        capacity=10,
        label_budget=2,
        drift=True,
        allow_pseudo_labels=True,
        pseudo_y=pseudo_y,
    )
    _, _, _, forgotten, inserted, ground_truth, pseudo = result
    assert forgotten == inserted == 5
    assert ground_truth == 2
    assert pseudo == 3


def test_s4_keeps_two_attack_classes_identifiable_during_dual_drift() -> None:
    stream = make_synthetic_stream("S4", 4, 2048, 32, 2026)
    for window, expected in [
        (stream.windows[0], ("hulk", "ftp")),
        (stream.windows[3], ("ddos", "ssh")),
    ]:
        labels = np.asarray(window.y.tolist())
        subtype_one = set(window.subtypes[labels == 1].tolist())
        subtype_two = set(window.subtypes[labels == 2].tolist())
        assert subtype_one == {expected[0]}
        assert subtype_two == {expected[1]}
        center_one = window.x[window.y == 1].mean(dim=0)
        center_two = window.x[window.y == 2].mean(dim=0)
        assert torch.linalg.vector_norm(center_one - center_two).item() > 1.0


def test_s5_synthetic_fallback_moves_attacks_toward_benign() -> None:
    stream = make_synthetic_stream("S5", 4, 4096, 32, 20260729)

    def attack_to_benign_distance(window) -> float:
        benign = window.x[window.y == 0].mean(dim=0)
        attack = window.x[window.y != 0].mean(dim=0)
        return float(torch.linalg.vector_norm(attack - benign))

    pre_distance = attack_to_benign_distance(stream.windows[1])
    post_distance = attack_to_benign_distance(stream.windows[3])
    assert stream.windows[3].phase == "B_evasive"
    assert post_distance < 0.4 * pre_distance


def test_feature_similarity_replay_is_deterministic_and_bounded() -> None:
    generator = torch.Generator().manual_seed(91)
    model = OnlineMLP(6, 3, hidden_dims=[8, 4])
    memory_x = torch.randn(90, 6, generator=generator)
    memory_y = torch.tensor([0] * 30 + [1] * 30 + [2] * 30)
    current_x = torch.randn(24, 6, generator=generator) + 0.25
    first = select_feature_similarity_replay(
        model,
        memory_x,
        memory_y,
        current_x,
        ["layer_1"],
        replay_size=30,
        device=torch.device("cpu"),
        batch_size=16,
    )
    second = select_feature_similarity_replay(
        model,
        memory_x,
        memory_y,
        current_x,
        ["layer_1"],
        replay_size=30,
        device=torch.device("cpu"),
        batch_size=16,
    )
    assert first is not None and second is not None
    assert len(first.y) == 30
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.y, second.y)
    assert len(torch.unique(first.y)) == 3


def test_v13_guard_rejects_a_hidden_per_class_recall_drop() -> None:
    diagnostics = {
        "guard_candidate_macro_f1": 0.91,
        "guard_fallback_macro_f1": 0.90,
        "guard_candidate_recall_class_0": 0.96,
        "guard_fallback_recall_class_0": 0.90,
        "guard_candidate_recall_class_1": 0.80,
        "guard_fallback_recall_class_1": 0.85,
        "guard_candidate_recall_class_2": 0.97,
        "guard_fallback_recall_class_2": 0.95,
        "guard_combined_lcb": 0.01,
    }
    assert (
        _v13_class_balanced_score(
            diagnostics,
            macro_f1_tolerance=0.0,
            per_class_recall_tolerance=0.02,
        )
        is None
    )


def test_v16_guard_accepts_balanced_minority_recovery() -> None:
    diagnostics = {
        "guard_candidate_macro_f1": 0.82,
        "guard_fallback_macro_f1": 0.44,
        "guard_candidate_attack_recall": 0.79,
        "guard_fallback_attack_recall": 0.08,
        "guard_candidate_min_per_class_recall": 0.79,
        "guard_fallback_min_per_class_recall": 0.08,
        "guard_combined_lcb": 1.0,
    }
    assert (
        _v16_class_balanced_score(
            diagnostics,
            macro_f1_tolerance=0.0,
            attack_recall_tolerance=0.01,
            min_per_class_recall_tolerance=0.01,
        )
        is not None
    )


def test_v16_guard_rejects_attack_or_weakest_class_regression() -> None:
    diagnostics = {
        "guard_candidate_macro_f1": 0.91,
        "guard_fallback_macro_f1": 0.90,
        "guard_candidate_attack_recall": 0.83,
        "guard_fallback_attack_recall": 0.85,
        "guard_candidate_min_per_class_recall": 0.83,
        "guard_fallback_min_per_class_recall": 0.85,
        "guard_combined_lcb": 0.01,
    }
    assert (
        _v16_class_balanced_score(
            diagnostics,
            macro_f1_tolerance=0.0,
            attack_recall_tolerance=0.01,
            min_per_class_recall_tolerance=0.01,
        )
        is None
    )


def test_v13_non_drift_windows_refresh_memory_without_model_gradients(
    tmp_path,
) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.model.hidden_dims = [16, 8, 4]
    config.data.n_windows = 8
    config.data.warmup_windows = 2
    config.drift.persistence = 99
    config.drift.severity_threshold = 1e9
    config.dasp.v13_refresh_size = 8
    config.dasp.v11_min_per_class = 2
    config.dasp.baseline = "dasp_v13"
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v13-frozen"
    result = StreamingExperiment(config).run()
    evaluated = result.windows[result.windows["window"] >= config.data.warmup_windows]
    assert (evaluated["gradient_evaluations"].fillna(0) == 0).all()
    assert set(evaluated["selected_candidate"]) == {"frozen_fallback"}
    assert (evaluated["v13_refresh_inserted"].fillna(0) == 8).all()


def test_v13_forced_drift_is_layer_only_and_uses_balanced_guard(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.model.hidden_dims = [16, 8, 4]
    config.data.scenario = "S4"
    config.data.n_windows = 8
    config.data.warmup_windows = 2
    config.drift.persistence = 1
    config.drift.severity_threshold = -1.0
    config.dasp.v10_min_layer_utility = -1e9
    config.dasp.v13_refresh_size = 8
    config.dasp.v11_min_per_class = 2
    config.dasp.v13_similarity_replay_size = 32
    config.dasp.baseline = "dasp_v13"
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v13-drift"
    result = StreamingExperiment(config).run()
    selected = set(result.windows["selected_candidate"])
    assert "global" not in selected
    assert selected <= {
        "none",
        "frozen_fallback",
        "layer_top1",
        "layer_top2",
        "layer_top3",
    }
    assert "layer_top1_guard_candidate_macro_f1" in result.windows
    assert "layer_top1_guard_candidate_recall_class_1" in result.windows


@pytest.mark.parametrize("method", ["dasp_v14_light", "dasp_v14"])
def test_v14_non_drift_update_is_light_layer_only(method, tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.model.hidden_dims = [16, 8, 4]
    config.data.n_windows = 8
    config.data.warmup_windows = 2
    config.drift.persistence = 99
    config.drift.severity_threshold = 1e9
    config.dasp.v14_light_current_size = 24
    config.dasp.v14_light_replay_size = 32
    config.dasp.v13_refresh_size = 8
    config.dasp.v11_min_per_class = 2
    config.dasp.baseline = method
    config.output.directory = str(tmp_path)
    config.output.run_name = f"test-{method}"
    result = StreamingExperiment(config).run()
    evaluated = result.windows[result.windows["window"] >= config.data.warmup_windows]
    attempted = evaluated["v14_light_attempted"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    assert attempted.all()
    assert (evaluated["gradient_evaluations"].fillna(0) > 0).all()
    selected = set(evaluated["selected_candidate"])
    assert "global" not in selected
    assert "replay_fallback" not in selected
    assert all(
        name == "frozen_fallback" or name.startswith("light_layer_")
        for name in selected
    )


def test_v15_uses_current_unlabeled_alarm_with_only_delayed_labels(
    tmp_path,
) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.model.hidden_dims = [16, 8, 4]
    config.data.scenario = "S3"
    config.data.n_windows = 7
    config.data.warmup_windows = 2
    config.drift.persistence = 2
    config.drift.severity_threshold = -1.0
    config.dasp.v10_min_layer_utility = -1e9
    config.dasp.v14_light_current_size = 24
    config.dasp.v15_light_replay_size = 24
    config.dasp.v13_refresh_size = 8
    config.dasp.v11_min_per_class = 2
    config.dasp.baseline = "dasp_v15"
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v15-alarm-timeline"
    result = StreamingExperiment(config).run()
    context_triggered = result.windows["v15_event_context_triggered"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    delayed_triggered = result.windows["triggered"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    context_alarm = result.windows[context_triggered & ~delayed_triggered]
    assert len(context_alarm) > 0
    assert context_alarm["layer_top1_guard_candidate_macro_f1"].notna().any()


def test_strict_layer_candidate_keeps_unselected_layer_exactly_frozen() -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    model = OnlineMLP(10, 2, hidden_dims=[16, 8, 4])
    memory = ReservoirMemory(128, 77)
    generator = torch.Generator().manual_seed(77)
    x = torch.randn(96, 10, generator=generator)
    y = (x[:, 0] > 0).long()
    memory.add(x[:64], y[:64])
    before_frozen = {
        name: value.detach().clone()
        for name, value in model.blocks[1].state_dict().items()
    }
    before_active = {
        name: value.detach().clone()
        for name, value in model.blocks[0].state_dict().items()
    }
    _train_dasp(
        model,
        x,
        y,
        {
            "layer_1": 1.0,
            "layer_2": 0.0,
            "layer_3": 0.0,
            "classifier": 1.0,
        },
        config,
        torch.device("cpu"),
        memory,
        strict_freeze=True,
        epochs=1,
    )
    assert all(
        torch.equal(value, model.blocks[1].state_dict()[name])
        for name, value in before_frozen.items()
    )
    assert any(
        not torch.equal(value, model.blocks[0].state_dict()[name])
        for name, value in before_active.items()
    )


def test_v12_forced_warning_is_layer_only_and_logs_psla_fields(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.model.hidden_dims = [16, 8, 4]
    config.data.scenario = "S4"
    config.data.n_windows = 8
    config.data.warmup_windows = 2
    config.drift.persistence = 99
    config.drift.severity_threshold = -1.0
    config.dasp.v10_min_layer_utility = -1e9
    config.dasp.v12_guard_min_macro_f1 = 1.01
    config.dasp.v12_guard_min_attack_recall = 1.01
    config.dasp.baseline = "dasp_v12"
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v12-warning"
    result = StreamingExperiment(config).run()
    attempted = result.windows["v12_warning_attempted"].astype(bool)
    assert attempted.any()
    assert "global" not in set(result.windows["selected_candidate"])
    assert set(result.windows["selected_candidate"]) <= {
        "none",
        "replay_fallback",
        "layer_top1",
        "layer_top2",
        "layer_top3",
    }
    assert "layer_top3_guard_recent_lcb" in result.windows.columns


def test_v12_rejected_warning_does_not_commit_memory(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.model.hidden_dims = [16, 8, 4]
    config.data.scenario = "S4"
    config.data.n_windows = 9
    config.data.warmup_windows = 3
    config.drift.persistence = 99
    config.drift.severity_threshold = 0.0
    config.dasp.v10_min_layer_utility = -1e9
    config.dasp.v10_2_guard_lcb_budget = -100.0
    config.dasp.v12_guard_min_macro_f1 = 1.01
    config.dasp.v12_guard_min_attack_recall = 1.01
    config.dasp.baseline = "dasp_v12"
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v12-atomic-reject"
    result = StreamingExperiment(config).run()
    attempted = result.windows[result.windows["v12_warning_attempted"].astype(bool)]
    assert len(attempted) > 0
    assert not attempted["v12_warning_committed"].astype(bool).any()


def test_v11_memory_proposal_is_pure_deterministic_and_bounded() -> None:
    generator = torch.Generator().manual_seed(2026)
    model = OnlineMLP(6, 2, hidden_dims=[8, 4])
    old_x = torch.randn(64, 6, generator=generator)
    old_y = torch.tensor([0] * 48 + [1] * 16)
    new_x = torch.randn(32, 6, generator=generator) + 0.5
    new_y = torch.tensor([0] * 8 + [1] * 24)
    old_x_before, old_y_before = old_x.clone(), old_y.clone()
    kwargs = dict(
        model=model,
        old_x=old_x,
        old_y=old_y,
        new_x=new_x,
        new_y=new_y,
        plasticities={"layer_1": 1.0, "layer_2": 0.5, "classifier": 1.0},
        capacity=64,
        label_budget=16,
        novelty_weight=1.0,
        interference_weight=0.5,
        balance_weight=0.5,
        redundancy_weight=0.25,
        min_per_class=8,
        device=torch.device("cpu"),
        batch_size=16,
    )
    first = propose_layer_adaptive_memory(**kwargs)
    second = propose_layer_adaptive_memory(**kwargs)
    assert torch.equal(old_x, old_x_before)
    assert torch.equal(old_y, old_y_before)
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.y, second.y)
    assert len(first.y) == 64
    assert first.inserted == 16
    assert first.evicted == 16
    counts = torch.bincount(first.y, minlength=2)
    assert torch.all(counts >= 8)


def test_v11_rejected_candidate_does_not_commit_memory(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.data.scenario = "S4"
    config.data.n_windows = 7
    config.data.warmup_windows = 2
    config.drift.persistence = 1
    config.drift.severity_threshold = -1.0
    config.drift.emergency_severity_threshold = 2.0
    config.dasp.v10_2_guard_lcb_budget = -100.0
    config.dasp.baseline = "dasp_v11"
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v11-atomic-reject"
    result = StreamingExperiment(config).run()
    attempted = result.windows[
        result.windows["selected_candidate"] == "replay_fallback"
    ]
    assert len(attempted) > 0
    committed = attempted["v11_memory_committed"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )
    assert not committed.any()


def test_dasp_v10_forced_drift_never_selects_global_candidate(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.data.scenario = "S4"
    config.data.n_windows = 9
    config.data.warmup_windows = 3
    config.drift.persistence = 1
    config.drift.severity_threshold = -1.0
    config.drift.emergency_severity_threshold = 2.0
    config.dasp.baseline = "dasp_v10"
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v10-layer-selective"
    result = StreamingExperiment(config).run()
    selected = set(result.windows["selected_candidate"])
    assert "global" not in selected
    assert selected <= {"none", "replay_fallback", "layer_top1", "layer_top2"}


def test_dasp_v10_1_forced_drift_never_selects_global_candidate(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.data.scenario = "S4"
    config.data.n_windows = 9
    config.data.warmup_windows = 3
    config.drift.persistence = 1
    config.drift.severity_threshold = -1.0
    config.drift.emergency_severity_threshold = 2.0
    config.dasp.baseline = "dasp_v10_1"
    config.dasp.v10_1_full_expansion_severity = -1.0
    config.output.directory = str(tmp_path)
    config.output.run_name = "test-v10-1-layer-selective"
    result = StreamingExperiment(config).run()
    selected = set(result.windows["selected_candidate"])
    assert "global" not in selected
    assert selected <= {
        "none",
        "replay_fallback",
        "layer_top1",
        "layer_top2",
        "layer_top3",
    }


def test_nf_full_stream_sorts_time_and_keeps_partial_window(tmp_path) -> None:
    frame = pd.DataFrame({
        "FLOW_START_MILLISECONDS": [30, 10, 20, 50, 40, 60, 90, 80, 70, 100],
        "FLOW_END_MILLISECONDS": [31, 11, 21, 51, 41, 61, 91, 81, 71, 101],
        "IPV4_SRC_ADDR": ["1.1.1.1"] * 10,
        "IPV4_DST_ADDR": ["2.2.2.2"] * 10,
        "feature": list(range(10)),
        "Label": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        "Attack": ["Benign", "Attack", "Benign", "Benign", "Attack",
                   "Benign", "Attack", "Benign", "Benign", "Attack"],
    })
    path = tmp_path / "tiny_nf.csv"
    frame.to_csv(path, index=False)
    stream = load_nf_v3_stream(
        path,
        window_size=4,
        n_windows=0,
        warmup_windows=1,
        sort_by_time=True,
        include_partial_window=True,
    )
    assert [len(window.x) for window in stream.windows] == [4, 4, 2]
    assert [window.start_time for window in stream.windows] == [10, 50, 90]
    assert [window.end_time for window in stream.windows] == [40, 80, 100]


def test_nf_bounded_stream_sorts_globally_before_start_row(tmp_path) -> None:
    frame = pd.DataFrame({
        "FLOW_START_MILLISECONDS": [90, 10, 70, 30, 50, 20, 80, 40, 60, 100],
        "FLOW_END_MILLISECONDS": [91, 11, 71, 31, 51, 21, 81, 41, 61, 101],
        "IPV4_SRC_ADDR": ["1.1.1.1"] * 10,
        "IPV4_DST_ADDR": ["2.2.2.2"] * 10,
        "feature": list(range(10)),
        "Label": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        "Attack": ["Benign", "Attack", "Benign", "Benign", "Attack",
                   "Benign", "Attack", "Benign", "Benign", "Attack"],
    })
    path = tmp_path / "bounded_nf.csv"
    frame.to_csv(path, index=False)
    stream = load_nf_v3_stream(
        path,
        window_size=2,
        n_windows=2,
        warmup_windows=1,
        start_row=2,
        sort_by_time=True,
    )
    assert [window.start_time for window in stream.windows] == [30, 50]
    assert [window.end_time for window in stream.windows] == [40, 60]


def test_modular_dasp_strict_freeze_updates_only_routed_module() -> None:
    config = load_config("configs/modular_dasp_smoke.yaml")
    model = ModularOnlineMLP(16, 2, [32, 16], n_modules=4, module_dim=8)
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(64, 16, generator=generator)
    y = (x[:, 0] > 0).long()
    memory = ReservoirMemory(64, 123)
    memory.add(x, y)
    before_module_2 = {
        name: value.detach().clone()
        for name, value in model.modules_list[1].state_dict().items()
    }
    before_module_1 = {
        name: value.detach().clone()
        for name, value in model.modules_list[0].state_dict().items()
    }
    steps = _train_dasp(
        model,
        x,
        y,
        {"module_1": 1.0, "module_2": 0.0, "module_3": 0.0, "module_4": 0.0, "classifier": 1.0},
        config,
        torch.device("cpu"),
        memory,
        strict_freeze=True,
        use_projection_memory=False,
    )
    assert steps > 0
    assert any(
        not torch.equal(value, before_module_1[name])
        for name, value in model.modules_list[0].state_dict().items()
    )
    assert all(
        torch.equal(value, before_module_2[name])
        for name, value in model.modules_list[1].state_dict().items()
    )
