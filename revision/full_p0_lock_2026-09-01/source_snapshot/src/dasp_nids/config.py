from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.0
    architecture: str = "sequential"
    n_modules: int = 4
    module_dim: int = 64


@dataclass
class DataConfig:
    source: str = "synthetic"
    path: str | None = None
    secondary_path: str | None = None
    scenario: str = "S1"
    window_size: int = 4096
    n_windows: int = 12
    warmup_windows: int = 3
    n_features: int = 32
    label_column: str = "Label"
    subtype_column: str = "Attack"
    task: str = "binary"
    start_row: int = 0
    chunksize: int = 65536
    sort_by_time: bool = False
    time_column: str = "FLOW_START_MILLISECONDS"
    include_partial_window: bool = False
    random_seed: int = 42
    stream_seed: int | None = None


@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_window: int = 1
    label_delay: int = 1
    device: str = "cpu"
    num_threads: int = 1
    model_seed: int | None = None


@dataclass
class DriftConfig:
    mmd_samples: int = 256
    calibration_windows: int = 3
    kappa: float = 2.5
    z_max: float = 8.0
    severity_threshold: float = 0.25
    emergency_severity_threshold: float = 0.45
    persistence: int = 2
    top_k: int = 2
    kernel_scales: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    cooldown_windows: int = 0


@dataclass
class DASPConfig:
    baseline: str = "full_dasp"
    eta_floor: float = 0.1
    eta_ceiling: float = 2.0
    conflict_weight: float = 1.0
    memory_size: int = 5000
    memory_batch_size: int = 256
    guard_ratio: float = 0.2
    lcb_z: float = 1.645
    refresh_reference_after_commit: bool = True
    utility_forgetting_weight: float = 1.0
    severity_forgetting_floor: float = 0.25
    utility_probe_samples: int = 256
    utility_probe_lr_scale: float = 1.0
    memory_guard_weight: float = 1.0
    severity_memory_guard_floor: float = 0.25
    recent_guard_tolerance: float = 0.02
    recurrence_departure_threshold: float = 0.25
    recurrence_return_threshold: float = 0.05
    recurrence_anchor_strength: float = 0.75
    recurrence_inference_anchor: bool = True
    compute_match_epochs: int = 3
    ewc_lambda: float = 10.0
    ewc_decay: float = 0.9
    v7_balanced_replay: bool = False
    v7_replay_size: int = 512
    v7_rehearsal_candidates: bool = True
    v8_balanced_guard: bool = True
    v9_candidate_replay_size: int = 128
    v9_min_candidate_utility: float = 0.0
    v9_global_utility_ratio: float = 0.75
    v9_one_shot_recurrence: bool = True
    v9_recurrence_min_windows: int = 2
    v9_attempt_cooldown: int = 2
    v9_lazy_accept_lcb: float = 0.01
    v10_max_adaptive_layers: int = 2
    v10_min_layer_utility: float = 0.0
    v10_min_active_plasticity: float = 0.25
    v10_classifier_plasticity: float = 1.0
    v10_lazy_accept_lcb: float = 0.01
    v10_class_balanced_guard: bool = True
    v10_attack_guard_tolerance: float = 0.02
    v10_1_max_adaptive_layers: int = 3
    v10_1_new_sample_weight: float = 3.0
    v10_1_full_expansion_severity: float = 0.60
    v10_1_guard_lcb_budget: float = 0.02
    v10_1_attack_guard_extra_budget: float = 0.01
    v10_2_full_expansion_severity: float = 0.25
    v10_2_min_active_plasticity: float = 0.75
    v10_2_adaptation_epochs: int = 3
    v10_2_guard_lcb_budget: float = 0.03
    v10_2_attack_guard_extra_budget: float = 0.02
    v10_3_max_new_sample_weight: float = 6.0
    v11_label_budget: int = 256
    v11_candidate_replay_size: int = 1024
    v11_novelty_weight: float = 1.0
    v11_interference_weight: float = 0.5
    v11_balance_weight: float = 0.5
    v11_redundancy_weight: float = 0.25
    v11_min_per_class: int = 64
    v11_recovery_epochs: int = 6
    v11_focal_gamma: float = 2.0
    v11_effective_number_beta: float = 0.999
    v11_recovery_severity: float = 0.75
    v11_min_guard_attack_recall: float = 0.80
    v12_guard_min_macro_f1: float = 0.80
    v12_guard_min_attack_recall: float = 0.80
    v12_max_adaptive_layers: int = 3
    v13_similarity_replay_size: int = 1024
    v13_refresh_size: int = 64
    v13_refresh_layers: int = 1
    v13_macro_f1_tolerance: float = 0.0
    v13_per_class_recall_tolerance: float = 0.02
    modular_replay_size: int = 512
    modular_new_sample_weight: float = 3.0
    modular_background_plasticity: float = 0.25
    modular_repair_epochs: int = 2
    modular_emergency_epochs: int = 3
    v14_light_current_size: int = 256
    v14_light_replay_size: int = 512
    v14_light_layers: int = 1
    v14_light_epochs: int = 1
    v15_light_replay_size: int = 256
    v16_performance_reference_windows: int = 5
    v16_performance_min_history: int = 4
    v16_macro_f1_drop: float = 0.08
    v16_attack_recall_drop: float = 0.08
    v16_performance_cooldown: int = 2
    v16_max_adaptive_layers: int = 2
    v16_attack_recall_tolerance: float = 0.01
    v16_min_per_class_recall_tolerance: float = 0.01
    v17_adapter_max_rank: int = 8
    v17_adapter_scale: float = 1.0
    v17_input_adapter_dim: int = 16
    v17_fast_input_adapter_plasticity: float = 0.0
    v17_strong_input_adapter_plasticity: float = 1.0
    v17_fast_rank: int = 2
    v17_strong_rank: int = 8
    v17_fast_epochs: int = 1
    v17_strong_epochs: int = 3
    v17_strong_recovery_windows: int = 1
    v17_strong_base_layers: int = 0
    v17_fast_replay_size: int = 256
    v17_strong_replay_size: int = 1024
    v17_refresh_size: int = 256
    v17_new_sample_weight: float = 3.0
    v17_strong_new_sample_weight: float = 3.0
    v17_iwms_temperature: float = 0.10
    v17_iwms_confidence_power: float = 1.0
    v17_iwms_uniform_mix: float = 0.25
    v17_iwms_query_size: int = 512
    v17_macro_f1_tolerance: float = 0.0
    v17_attack_recall_tolerance: float = 0.01
    v17_min_per_class_recall_tolerance: float = 0.01
    v17_recent_mean_tolerance: float = 0.02
    v17_memory_mean_tolerance: float = 0.02
    v17_soft_commit_alphas: tuple[float, ...] = (
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
    )
    # v18 keeps the backbone structurally frozen while addressing v17.5's
    # representation, replay self-confirmation, and small-guard bottlenecks.
    v18_modulation_scale: float = 0.5
    v18_fast_modulation_plasticity: float = 0.5
    v18_strong_modulation_plasticity: float = 1.0
    v18_replay_max_uniform_mix: float = 0.75
    v18_replay_js_threshold: float = 0.10
    v18_guard_history_size: int = 512
    v18_query_entropy_weight: float = 0.5
    v18_query_novelty_weight: float = 0.5
    v18_query_disagreement_weight: float = 0.25
    v18_query_global_fraction: float = 0.5
    v18_query_candidate_multiplier: int = 4
    v18_current_guard_fraction: float = 0.25
    v18_current_guard_min_per_class: int = 8
    v18_class_weight_multiplier: float = 2.0
    # Safety revision used only by the explicitly named development
    # ablations.  The published/current ``dasp`` path remains unchanged.
    safe_query_soft_class_fraction: float = 0.5
    safe_replay_uniform_mix: float = 1.0
    safe_guard_min_per_class: int = 4
    safe_accumulator_size: int = 256
    safe_accumulator_min_per_class: int = 8
    safe_strong_base_layers: int = 1
    v3_memory_min_class_quota: int = 256
    v3_query_rare_class_threshold: float = 0.05
    v3_query_rare_soft_fraction: float = 0.75
    v3_query_raw_diversity_fraction: float = 0.25
    v3_backbone_max_layers: int = 2
    v3_backbone_plasticity: float = 0.25
    v3_checkpoint_horizon: int = 5
    v3_checkpoint_min_gain: float = 0.0
    v3_rollback_class_recall_floor: float = 0.10
    v3_fallback_min_per_class: int = 8
    # DASP-v4 development path.  These values are global across datasets:
    # labels enter only after the configured delay, memory remains bounded,
    # and the informative selector operates inside true-label class quotas.
    v4_memory_recent_fraction: float = 0.20
    v4_memory_boundary_fraction: float = 0.50
    v4_memory_diversity_fraction: float = 0.20
    v4_memory_diversity_cap: int = 64
    v4_prior_history_size: int = 512
    v4_logit_adjust_tau: float = 0.50
    v4_logit_prior_smoothing: float = 1.0
    v4_logit_prior_floor: float = 0.05
    # Main DASP fast-path revision. Keeping this value in the resolved config
    # prevents legacy and upgraded runs from being resumed into one matrix.
    fast_revision: str = "event-gated-fused-v1"
    fast_warning_severity: float = 0.075
    fast_input_gate_samples: int = 64
    fast_stable_label_budget: int = 64
    fast_guard_history_size: int = 128
    fast_fast_replay_size: int = 128
    fast_strong_replay_size: int = 256
    fast_fast_epochs: int = 1
    fast_strong_epochs: int = 2
    fast_commit_alphas: tuple[float, ...] = (1.0, 0.5)


@dataclass
class BaselineConfig:
    """Hyperparameters for protocol-aligned external baselines."""

    update_label_budget: int = 256
    ssf_label_budget: int = 256
    ssf_new_sample_weight: float = 3.0
    ssf_lwf_lambda: float = 0.5
    ssf_drift_p_value: float = 0.05
    ssf_mask_steps: int = 25
    ssf_mask_bins: int = 10
    ssf_allow_pseudo_labels: bool = False
    repshield_pca_threshold: float = 0.15
    repshield_subspace_energy: float = 0.95
    repshield_replay_ratio: float = 0.5
    cdda_hidden_dim: int = 32
    cdda_attention_heads: int = 4
    cdda_error_z: float = 2.0
    cdda_warning_patience: int = 3
    cdda_min_error_increase: float = 0.01
    periodic_ft_interval: int = 4

@dataclass
class RevisionConfig:
    enabled: bool = False
    protocol_version: str = "1.1"
    dataset_key: str | None = None
    label_budget: int = 512
    repair_candidate_size: int = 384
    gate_size: int = 128
    optimization_size: int = 128
    replay_size: int = 512
    fixed_modules: list[str] = field(
        default_factory=lambda: ["module_1", "module_2", "module_3"]
    )
    routing_namespace: str = "revision-routing-primary-v1"
    representation_plasticity: float = 1.0
    classifier_plasticity: float = 1.0
    change_tolerance: float = 1e-12
    performance_gate_enabled: bool = True
    event_matched_diagnostics: bool = False
    event_matched_controls: list[str] = field(
        default_factory=lambda: ["A2", "A3", "A4", "A5", "A6", "A9"]
    )


@dataclass
class OutputConfig:
    directory: str = "outputs"
    save_checkpoints: bool = False
    run_name: str | None = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    dasp: DASPConfig = field(default_factory=DASPConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    revision: RevisionConfig = field(default_factory=RevisionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type, raw: dict[str, Any] | None):
    return cls(**(raw or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ExperimentConfig(
        model=_construct(ModelConfig, raw.get("model")),
        data=_construct(DataConfig, raw.get("data")),
        training=_construct(TrainingConfig, raw.get("training")),
        drift=_construct(DriftConfig, raw.get("drift")),
        dasp=_construct(DASPConfig, raw.get("dasp")),
        baselines=_construct(BaselineConfig, raw.get("baselines")),
        revision=_construct(RevisionConfig, raw.get("revision")),
        output=_construct(OutputConfig, raw.get("output")),
    )
