from __future__ import annotations

import copy

import numpy as np
import torch

from dasp_nids.config import load_config
from dasp_nids.memory import ReservoirMemory
from dasp_nids.model import AdapterOnlineMLP
from dasp_nids.runner import (
    StreamingExperiment,
    _bounded_label_history_update,
    _causal_logit_adjustment,
    _has_class_support,
    _informative_true_label_memory_update,
    _merge_safe_label_buffer,
    _restore_v3_checkpoint,
    _safe_soft_commit,
    _true_label_memory_update,
    _v18_label_query_indices,
)


def test_soft_class_query_is_label_blind_and_survives_argmax_collapse() -> None:
    generator = torch.Generator().manual_seed(9101)
    model = AdapterOnlineMLP(6, 2, [10, 5], adapter_max_rank=4)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([8.0, -8.0]))
    memory = ReservoirMemory(64, 9101)
    memory.add(
        torch.randn(64, 6, generator=generator),
        torch.tensor([0] * 32 + [1] * 32),
    )
    current_x = torch.randn(80, 6, generator=generator)
    first, first_diagnostics = _v18_label_query_indices(
        model,
        current_x,
        memory,
        24,
        torch.device("cpu"),
        9101,
        entropy_weight=0.5,
        novelty_weight=0.5,
        disagreement_weight=0.25,
        global_fraction=0.5,
        candidate_multiplier=4,
        batch_size=16,
        soft_class_fraction=0.5,
    )
    second, second_diagnostics = _v18_label_query_indices(
        model,
        current_x,
        memory,
        24,
        torch.device("cpu"),
        9101,
        entropy_weight=0.5,
        novelty_weight=0.5,
        disagreement_weight=0.25,
        global_fraction=0.5,
        candidate_multiplier=4,
        batch_size=16,
        soft_class_fraction=0.5,
    )
    assert torch.equal(first, second)
    assert len(first) == len(torch.unique(first)) == 24
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["v18_query_predicted_per_class"] == "80|0"
    assert first_diagnostics["safe_query_soft_class_reserved"] == 12


def test_safe_commit_rejects_one_class_guard_with_exact_rollback() -> None:
    fallback = AdapterOnlineMLP(4, 2, [8, 4], adapter_max_rank=4)
    fallback.set_adapter_rank(2, trainable_rank=2)
    candidate = copy.deepcopy(fallback)
    with torch.no_grad():
        candidate.classifier.bias.add_(2.0)
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.dasp.safe_guard_min_per_class = 1
    recent = (torch.randn(8, 4), torch.zeros(8, dtype=torch.long))
    memory = (
        torch.randn(16, 4),
        torch.tensor([0] * 8 + [1] * 8),
    )
    chosen, alpha, diagnostics = _safe_soft_commit(
        fallback,
        candidate,
        recent,
        memory,
        config,
        torch.device("cpu"),
    )
    assert alpha == 0.0
    assert diagnostics["safe_gate_reason"] == "recent_guard_class_support"
    assert diagnostics["safe_rollback_exact"] is True
    for name, value in fallback.state_dict().items():
        assert torch.equal(value, chosen.state_dict()[name])


def test_class_support_requires_every_model_class() -> None:
    assert _has_class_support(torch.tensor([0, 0, 1, 1]), 2, 2)
    assert not _has_class_support(torch.tensor([0, 0, 0, 0]), 2, 1)
    assert not _has_class_support(torch.tensor([0, 1, 2]), 3, 2)


def test_safe_accumulator_keeps_rare_revealed_class_without_replacement() -> None:
    generator = torch.Generator().manual_seed(9151)
    first = _merge_safe_label_buffer(
        None,
        torch.randn(32, 4, generator=generator),
        torch.ones(32, dtype=torch.long),
        32,
        9151,
    )
    assert first is not None
    second = _merge_safe_label_buffer(
        first,
        torch.randn(32, 4, generator=generator),
        torch.tensor([0, 0, 0, 0] + [1] * 28),
        32,
        9152,
    )
    assert second is not None
    counts = torch.bincount(second[1], minlength=2)
    assert counts.tolist() == [4, 28]
    assert len(second[1]) == 32


def test_v3_true_label_memory_preserves_available_minority_quota() -> None:
    generator = torch.Generator().manual_seed(9161)
    history = (
        torch.randn(20, 4, generator=generator),
        torch.tensor([0] * 18 + [1] * 2),
    )
    updated = _true_label_memory_update(
        history,
        torch.randn(20, 4, generator=generator),
        torch.zeros(20, dtype=torch.long),
        capacity=20,
        n_classes=2,
        minimum_class_quota=6,
        seed=9161,
    )
    counts = torch.bincount(updated[1], minlength=2)
    assert counts.tolist() == [18, 2]
    assert len(updated[1]) == len(torch.unique(updated[0], dim=0)) == 20


def test_v3_true_label_memory_is_equal_when_classes_are_available() -> None:
    generator = torch.Generator().manual_seed(9162)
    updated = _true_label_memory_update(
        None,
        torch.randn(200, 4, generator=generator),
        torch.tensor([0] * 100 + [1] * 100),
        capacity=80,
        n_classes=2,
        minimum_class_quota=16,
        seed=9162,
    )
    assert torch.bincount(updated[1], minlength=2).tolist() == [40, 40]


def test_v3_checkpoint_restore_uses_only_eligible_checkpoint(
    monkeypatch,
) -> None:
    current = AdapterOnlineMLP(4, 2, [8, 4], adapter_max_rank=4)
    current.set_adapter_rank(2, trainable_rank=2)
    checkpoint = copy.deepcopy(current)
    with torch.no_grad():
        current.classifier.bias.add_(3.0)
    expected = {
        name: value.detach().clone()
        for name, value in checkpoint.state_dict().items()
    }

    def safe_diagnostics(*_args, **_kwargs):
        return {
            "guard_candidate_macro_f1": 0.90,
            "guard_fallback_macro_f1": 0.50,
            "guard_candidate_attack_recall": 0.90,
            "guard_fallback_attack_recall": 0.50,
            "guard_candidate_min_per_class_recall": 0.90,
            "guard_fallback_min_per_class_recall": 0.50,
            "guard_combined_lcb": 1.0,
        }

    monkeypatch.setattr(
        "dasp_nids.runner._class_guard_diagnostics",
        safe_diagnostics,
    )
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.dasp.safe_guard_min_per_class = 1
    recent = (
        torch.randn(8, 4),
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    restored, diagnostics = _restore_v3_checkpoint(
        current,
        [(0.90, 5, checkpoint)],
        recent=recent,
        memory=recent,
        cfg=config,
        device=torch.device("cpu"),
        require_current_guard=True,
    )
    assert restored
    assert diagnostics["v3_checkpoint_window"] == 5
    for name, value in current.state_dict().items():
        assert torch.equal(value, expected[name])


def test_safe_pipeline_mock_is_bounded_finite_and_noncollapsed(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.data.source = "synthetic"
    config.data.scenario = "S1"
    config.data.n_windows = 10
    config.data.warmup_windows = 3
    config.data.window_size = 96
    config.data.n_features = 12
    config.data.random_seed = 9201
    config.training.model_seed = 51
    config.training.device = "cpu"
    config.training.batch_size = 24
    config.training.label_delay = 1
    config.dasp.baseline = "dasp_safe"
    config.dasp.memory_size = 192
    config.dasp.memory_batch_size = 24
    config.dasp.fast_warning_severity = 0.0
    config.dasp.fast_stable_label_budget = 24
    config.dasp.fast_guard_history_size = 48
    config.dasp.fast_fast_replay_size = 24
    config.dasp.fast_strong_replay_size = 48
    config.dasp.fast_fast_epochs = 1
    config.dasp.fast_strong_epochs = 1
    config.dasp.v17_iwms_query_size = 24
    config.dasp.v17_refresh_size = 24
    config.dasp.v18_current_guard_min_per_class = 1
    config.dasp.safe_guard_min_per_class = 1
    config.baselines.update_label_budget = 24
    config.output.directory = str(tmp_path)
    config.output.run_name = "safe-pipeline"
    result = StreamingExperiment(config).run()
    evaluated = result.windows.loc[~result.windows["warmup"].astype(bool)]
    assert evaluated["ground_truth_update_labels"].max() <= 24
    assert not evaluated["global_update"].astype(bool).any()
    assert np.isfinite(evaluated["macro_f1"]).all()
    assert np.isfinite(evaluated["mcc"]).all()
    rollback = evaluated["safe_rollback_exact"].dropna()
    assert not (rollback.astype(str).str.lower() == "false").any()
    assert result.summary["cumulative_benign_recall"] > 0.05
    assert result.summary["cumulative_attack_recall"] > 0.05


def test_safe_v2_pipeline_balances_warmup_memory_and_stays_bounded(
    tmp_path,
) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.data.source = "synthetic"
    config.data.scenario = "S1"
    config.data.n_windows = 8
    config.data.warmup_windows = 3
    config.data.window_size = 96
    config.data.n_features = 12
    config.data.random_seed = 9251
    config.training.model_seed = 61
    config.training.device = "cpu"
    config.training.batch_size = 24
    config.training.label_delay = 1
    config.dasp.baseline = "dasp_safe_v2"
    config.dasp.memory_size = 192
    config.dasp.memory_batch_size = 24
    config.dasp.fast_warning_severity = 0.0
    config.dasp.fast_guard_history_size = 48
    config.dasp.fast_fast_replay_size = 24
    config.dasp.fast_strong_replay_size = 48
    config.dasp.fast_fast_epochs = 1
    config.dasp.fast_strong_epochs = 1
    config.dasp.v17_iwms_query_size = 24
    config.dasp.v17_refresh_size = 24
    config.dasp.v18_current_guard_min_per_class = 1
    config.dasp.safe_guard_min_per_class = 1
    config.dasp.safe_accumulator_size = 24
    config.dasp.safe_accumulator_min_per_class = 1
    config.baselines.update_label_budget = 24
    config.output.directory = str(tmp_path)
    config.output.run_name = "safe-v2-pipeline"
    result = StreamingExperiment(config).run()
    evaluated = result.windows.loc[~result.windows["warmup"].astype(bool)]
    assert evaluated["ground_truth_update_labels"].max() <= 24
    assert not evaluated["global_update"].astype(bool).any()
    assert np.isfinite(evaluated["macro_f1"]).all()
    rollback = evaluated["safe_rollback_exact"].dropna()
    assert not (rollback.astype(str).str.lower() == "false").any()


def test_dasp_v3_pipeline_is_causal_bounded_and_selective(tmp_path) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.data.source = "synthetic"
    config.data.scenario = "S1"
    config.data.n_windows = 8
    config.data.warmup_windows = 3
    config.data.window_size = 96
    config.data.n_features = 12
    config.data.random_seed = 9351
    config.training.model_seed = 71
    config.training.device = "cpu"
    config.training.batch_size = 24
    config.training.label_delay = 1
    config.dasp.baseline = "dasp_v3"
    config.dasp.memory_size = 192
    config.dasp.memory_batch_size = 24
    config.dasp.fast_warning_severity = 0.0
    config.dasp.fast_guard_history_size = 48
    config.dasp.fast_fast_replay_size = 24
    config.dasp.fast_strong_replay_size = 48
    config.dasp.fast_fast_epochs = 1
    config.dasp.fast_strong_epochs = 1
    config.dasp.v17_iwms_query_size = 24
    config.dasp.v17_refresh_size = 24
    config.dasp.v18_current_guard_min_per_class = 1
    config.dasp.safe_guard_min_per_class = 1
    config.dasp.safe_accumulator_size = 24
    config.dasp.safe_accumulator_min_per_class = 1
    config.dasp.v3_memory_min_class_quota = 8
    config.dasp.v3_fallback_min_per_class = 1
    config.baselines.update_label_budget = 24
    config.output.directory = str(tmp_path)
    config.output.run_name = "dasp-v3-pipeline"
    result = StreamingExperiment(config).run()
    evaluated = result.windows.loc[~result.windows["warmup"].astype(bool)]
    assert evaluated["ground_truth_update_labels"].max() <= 24
    assert not evaluated["global_update"].astype(bool).any()
    assert np.isfinite(
        evaluated[["macro_f1", "mcc", "balanced_accuracy"]]
    ).all().all()
    assert set(evaluated["window"]) == set(range(3, 8))


def test_v4_informative_memory_is_balanced_bounded_and_deterministic() -> None:
    generator = torch.Generator().manual_seed(9451)
    model = AdapterOnlineMLP(6, 2, [10, 5], adapter_max_rank=4)
    history_x = torch.randn(80, 6, generator=generator)
    history_y = torch.tensor([0] * 60 + [1] * 20)
    new_x = torch.randn(24, 6, generator=generator) + 5.0
    new_y = torch.tensor([0] * 12 + [1] * 12)
    kwargs = {
        "capacity": 64,
        "n_classes": 2,
        "minimum_class_quota": 16,
        "recent_fraction": 0.25,
        "boundary_fraction": 0.50,
        "diversity_fraction": 0.20,
        "diversity_cap": 8,
        "device": torch.device("cpu"),
        "seed": 9451,
    }
    first = _informative_true_label_memory_update(
        model,
        (history_x, history_y),
        new_x,
        new_y,
        **kwargs,
    )
    second = _informative_true_label_memory_update(
        model,
        (history_x, history_y),
        new_x,
        new_y,
        **kwargs,
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert len(first[1]) == 64
    assert torch.equal(
        torch.bincount(first[1], minlength=2),
        torch.tensor([32, 32]),
    )
    assert (first[0].mean(dim=1) > 3.0).any()


def test_v4_logit_adjustment_uses_only_bounded_revealed_history() -> None:
    history = _bounded_label_history_update(
        None, torch.tensor([0] * 90 + [1] * 10), 64
    )
    assert history is not None and len(history) == 64
    adjustment = _causal_logit_adjustment(
        history,
        n_classes=2,
        tau=0.5,
        smoothing=1.0,
        prior_floor=0.05,
        device=torch.device("cpu"),
    )
    assert adjustment is not None
    assert torch.isfinite(adjustment).all()
    assert adjustment[0] > adjustment[1]


def test_dasp_v4_pipeline_is_delayed_bounded_and_projection_light(
    tmp_path,
) -> None:
    config = copy.deepcopy(load_config("configs/smoke.yaml"))
    config.data.source = "synthetic"
    config.data.scenario = "S5"
    config.data.n_windows = 8
    config.data.warmup_windows = 3
    config.data.window_size = 96
    config.data.n_features = 12
    config.data.random_seed = 9452
    config.training.model_seed = 72
    config.training.device = "cpu"
    config.training.batch_size = 24
    config.training.label_delay = 1
    config.dasp.baseline = "dasp_v4"
    config.dasp.memory_size = 192
    config.dasp.memory_batch_size = 24
    config.dasp.fast_warning_severity = 0.0
    config.dasp.fast_guard_history_size = 48
    config.dasp.fast_fast_replay_size = 24
    config.dasp.fast_strong_replay_size = 48
    config.dasp.fast_fast_epochs = 1
    config.dasp.fast_strong_epochs = 1
    config.dasp.v17_iwms_query_size = 24
    config.dasp.v17_refresh_size = 24
    config.dasp.v18_current_guard_min_per_class = 1
    config.dasp.safe_guard_min_per_class = 1
    config.dasp.safe_accumulator_size = 24
    config.dasp.safe_accumulator_min_per_class = 1
    config.dasp.v3_memory_min_class_quota = 8
    config.baselines.update_label_budget = 24
    config.output.directory = str(tmp_path)
    config.output.run_name = "dasp-v4-pipeline"
    result = StreamingExperiment(config).run()
    evaluated = result.windows.loc[~result.windows["warmup"].astype(bool)]
    assert evaluated["ground_truth_update_labels"].max() <= 24
    assert not evaluated["global_update"].astype(bool).any()
    assert np.isfinite(
        evaluated[["macro_f1", "mcc", "balanced_accuracy"]]
    ).all().all()
    assert (
        evaluated["gradient_evaluations"].fillna(0)
        <= config.dasp.fast_strong_epochs
        * int(np.ceil(
            (
                config.baselines.update_label_budget
                + config.dasp.fast_strong_replay_size
            )
            / config.training.batch_size
        ))
    ).all()
