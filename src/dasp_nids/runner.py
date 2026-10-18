from __future__ import annotations

import copy
import gc
import json
import os
import pickle
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import ks_2samp

try:
    from river import forest, linear_model, optim, tree
except ImportError:
    forest = linear_model = optim = tree = None

from .baselines import (
    ErrorRateDriftDetector,
    build_gpm_subspaces,
    clone_teacher,
    normal_probabilities,
    pca_variance_distance,
    select_ssf_buffer,
    ssf_masks,
    strategic_coreset,
    train_agem,
    train_cdda,
    train_repshield,
    train_ssf,
)
from .config import ExperimentConfig
from . import original_review
from .data import (
    PreparedStream,
    StreamWindow,
    load_luflow_stream,
    load_nf_v3_stream,
    load_prepared_stream,
    make_synthetic_stream,
)
from .drift import DriftDecision, LayerDriftMonitor
from .memory import (
    MemoryProposal,
    ReservoirMemory,
    propose_layer_adaptive_memory,
    select_feature_similarity_replay,
    select_uncertainty_weighted_similarity_replay,
)
from .metrics import classification_metrics, summarize_windows
from .model import (
    AdapterOnlineMLP,
    CDDAAttentionNet,
    FusedAdapterInferenceMLP,
    ModularOnlineMLP,
    OnlineMLP,
)
from .revision import (
    REVISION_CONDITIONS,
    REVISION_METHODS,
    equal_label_pools,
    index_digest,
    locality_diagnostics,
    parameter_snapshot,
    routing_jaccard,
    select_routing,
    select_training_indices,
    stable_seed,
)


METHODS = {
    "static",
    "periodic_ft",
    "online",
    "online_mlp",
    "replay",
    "experience_replay",
    "global_protection",
    "global_dasp",
    "layer_dasp",
    "full_dasp",
    "modular_dasp",
    "shift_only",
    "conflict_only",
    "utility_global",
    "utility_dasp",
    "utility_full",
    "stateful_online",
    "compute_online_v3",
    "utility_global_v3",
    "utility_layer_v3",
    "utility_global_v5",
    "utility_layer_v5",
    "utility_portfolio_v5",
    "reactive_dasp_v6",
    "online_ewc",
    "agem",
    "ssf",
    "ssf_budget",
    "ssf_faithful",
    "repshield",
    "cdda_md",
    "dasp",
    "dasp_v7",
    "dasp_v8",
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
    "dasp_v13_iwms",
    "dasp_v13_refresh",
    "dasp_v13",
    "dasp_v14_light",
    "dasp_v14",
    "dasp_v15_light",
    "dasp_v15",
    "dasp_v16_no_detector",
    "dasp_v16_oracle",
    "dasp_v16",
    "dasp_v17",
    "dasp_v18",
    "dasp_fast",
    "dasp_safe_query",
    "dasp_safe_replay",
    "dasp_safe_oneclass",
    "dasp_safe",
    "dasp_safe_v2",
    "dasp_v3_memory",
    "dasp_v3_query",
    "dasp_v3_backbone",
    "dasp_v3_checkpoint",
    "dasp_v3",
    "dasp_v4_replay",
    "dasp_v4_prior",
    "dasp_v4",
    "river_logreg",
    "river_hat",
    "river_arf",
}
METHODS.update(REVISION_METHODS)

REVISION_ROUTING_METHODS = REVISION_METHODS - {"revision_a10_ssf"}

V3_METHODS = {
    "stateful_online",
    "compute_online_v3",
    "utility_global_v3",
    "utility_layer_v3",
}

V5_METHODS = {
    "utility_global_v5",
    "utility_layer_v5",
    "utility_portfolio_v5",
}

V6_METHODS = {"reactive_dasp_v6"}

V7_METHODS = {"dasp_v7"}

V8_METHODS = {"dasp_v8"}

V9_METHODS = {"dasp_v9"}

V91_METHODS = {"dasp_v9_1"}

V92_METHODS = {"dasp_v9_2"}

V12_METHODS = {"dasp_v12"}

V13_METHODS = {"dasp_v13_iwms", "dasp_v13_refresh", "dasp_v13"}

V13_IWMS_METHODS = {"dasp_v13_iwms", "dasp_v13"}

V13_REFRESH_METHODS = {"dasp_v13_refresh", "dasp_v13"}

V14_METHODS = {"dasp_v14_light", "dasp_v14"}

V15_METHODS = {"dasp_v15_light", "dasp_v15"}

V16_METHODS = {
    "dasp_v16_no_detector",
    "dasp_v16_oracle",
    "dasp_v16",
}

V17_METHODS = {"dasp_v17"}

SAFE_QUERY_METHODS = {
    "dasp_safe_query",
    "dasp_safe_replay",
    "dasp_safe_oneclass",
    "dasp_safe",
    "dasp_safe_v2",
}

SAFE_REPLAY_METHODS = {
    "dasp_safe_replay",
    "dasp_safe_oneclass",
    "dasp_safe",
    "dasp_safe_v2",
}

SAFE_ONECLASS_METHODS = {
    "dasp_safe_oneclass",
    "dasp_safe",
    "dasp_safe_v2",
}

SAFE_GATE_METHODS = {"dasp_safe", "dasp_safe_v2"}

SAFE_V2_METHODS = {"dasp_safe_v2"}

V3_MEMORY_METHODS = {
    "dasp_v3_memory",
    "dasp_v3_query",
    "dasp_v3_backbone",
    "dasp_v3_checkpoint",
    "dasp_v3",
}

V3_QUERY_METHODS = {
    "dasp_v3_query",
    "dasp_v3_backbone",
    "dasp_v3_checkpoint",
    "dasp_v3",
}

V3_BACKBONE_METHODS = {
    "dasp_v3_backbone",
    "dasp_v3_checkpoint",
    "dasp_v3",
}

V3_CHECKPOINT_METHODS = {"dasp_v3_checkpoint", "dasp_v3"}

V3_FINAL_METHODS = {"dasp_v3"}

V4_INFO_MEMORY_METHODS = {
    "dasp_v4_replay",
    "dasp_v4_prior",
    "dasp_v4",
}

V4_PRIOR_METHODS = {"dasp_v4_prior", "dasp_v4"}

V4_BACKBONE_METHODS = {"dasp_v4"}

V4_METHODS = V4_INFO_MEMORY_METHODS

V3_MEMORY_METHODS |= V4_METHODS
V3_QUERY_METHODS |= V4_METHODS
V3_BACKBONE_METHODS |= V4_BACKBONE_METHODS

SAFE_QUERY_METHODS |= V3_MEMORY_METHODS
SAFE_REPLAY_METHODS |= V3_MEMORY_METHODS
SAFE_ONECLASS_METHODS |= V3_MEMORY_METHODS
SAFE_GATE_METHODS |= V3_MEMORY_METHODS
SAFE_V2_METHODS |= V3_MEMORY_METHODS

FAST_DASP_METHODS = {"dasp", "dasp_fast", *SAFE_QUERY_METHODS}

V18_METHODS = {"dasp_v18", *FAST_DASP_METHODS}

ADAPTER_METHODS = V17_METHODS | V18_METHODS

LIGHT_UPDATE_METHODS = V14_METHODS | V15_METHODS | V16_METHODS

NO_FULL_UPDATE_METHODS = V13_METHODS | LIGHT_UPDATE_METHODS

SIMILARITY_REPLAY_METHODS = V13_IWMS_METHODS | LIGHT_UPDATE_METHODS

LIGHT_REFRESH_METHODS = V13_REFRESH_METHODS | {
    "dasp_v14",
    "dasp_v15",
    *V16_METHODS,
}

BUDGET_MATCHED_BASELINES = {
    "periodic_ft",
    "online",
    "online_mlp",
    "replay",
    "experience_replay",
    "online_ewc",
    "agem",
    "ssf_budget",
    "repshield",
    "cdda_md",
}

V11_METHODS = {"dasp_v11_novelty", "dasp_v11", "dasp_v11_1", *V12_METHODS}

V11_FULL_METHODS = {"dasp_v11"}

V11_RECOVERY_METHODS = {"dasp_v11_1"}

V10_METHODS = {
    "dasp_v10",
    "dasp_v10_1",
    "dasp_v10_2",
    "dasp_v10_3",
    *V11_METHODS,
    *V13_METHODS,
    *V14_METHODS,
    *V15_METHODS,
    *V16_METHODS,
}

V101_METHODS = {"dasp_v10_1"}

V102_METHODS = {
    "dasp_v10_2",
    "dasp_v10_3",
    *V11_METHODS,
    *V13_METHODS,
    *V14_METHODS,
    *V15_METHODS,
    *V16_METHODS,
}

V103_METHODS = {"dasp_v10_3"}

RIVER_METHODS = {"river_logreg", "river_hat", "river_arf"}

LAYER_DETECTOR_FREE_METHODS = {
    "static",
    "periodic_ft",
    "online",
    "online_mlp",
    "replay",
    "experience_replay",
    "online_ewc",
    "agem",
    "ssf",
    "ssf_budget",
    "ssf_faithful",
    "repshield",
    "cdda_md",
    "dasp_v16_no_detector",
}

PRIMARY_METHODS = [
    "online_mlp",
    "experience_replay",
    "online_ewc",
    "agem",
    "ssf",
    "repshield",
    "cdda_md",
    "modular_dasp",
]


@dataclass
class ExperimentResult:
    output_dir: Path
    summary: dict[str, object]
    windows: pd.DataFrame


@dataclass
class PendingUpdate:
    window: StreamWindow
    decision: DriftDecision
    recurrence: bool = False
    warning: bool = False
    current_unlabeled_x: torch.Tensor | None = None
    current_decision: DriftDecision | None = None
    current_warning: bool | None = None
    prediction_snapshot: ModelSnapshot | None = None
    prediction_labels: np.ndarray | None = None


@dataclass
class ModelSnapshot:
    """CPU checkpoint captured at prediction time without copying a module."""

    state: dict[str, torch.Tensor]
    adapter_rank: int
    adapter_trainable_rank: int
    input_adapter_enabled: bool
    feature_modulation_enabled: bool


def set_deterministic_seed(seed: int, num_threads: int = 1) -> None:
    # Required by deterministic CuBLAS operations on CUDA >= 10.2. This must
    # be configured before the first CUDA operation in the process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, num_threads))
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def _latency_summary(
    rows: list[dict[str, object]], warmup_windows: int
) -> dict[str, float]:
    evaluated = [
        row for row in rows if int(row["window"]) >= int(warmup_windows)
    ]
    result: dict[str, float] = {}
    for name in [
        "preprocessing_seconds",
        "forward_seconds",
        "metric_seconds",
        "drift_scoring_seconds",
        "prediction_decision_seconds",
        "update_seconds",
    ]:
        values = np.asarray(
            [float(row.get(name, 0.0)) for row in evaluated], dtype=float
        )
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        result[f"median_{name}"] = float(np.quantile(values, 0.50))
        result[f"p95_{name}"] = float(np.quantile(values, 0.95))
        result[f"p99_{name}"] = float(np.quantile(values, 0.99))
        result[f"total_{name}"] = float(values.sum())
    decision_total = sum(
        float(row.get("prediction_decision_seconds", 0.0))
        for row in evaluated
    )
    samples = sum(int(row.get("n_samples", 0)) for row in evaluated)
    result["prediction_decision_throughput_flows_per_second"] = (
        float(samples / decision_total) if decision_total > 0 else np.nan
    )
    return result


def _representations(model: OnlineMLP, x: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        _, values = model(x.to(device), return_representations=True)
    return {name: value.detach().cpu() for name, value in values.items()}


def _tensor_to_numpy(tensor: torch.Tensor, dtype: np.dtype | type | None = None) -> np.ndarray:
    values = np.asarray(tensor.detach().cpu().tolist())
    return values.astype(dtype, copy=False) if dtype is not None else values


def _balanced_tensor_sample(
    x: torch.Tensor,
    y: torch.Tensor,
    n: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Class-balanced tensor sample without touching the reservoir RNG."""

    if not len(x) or n <= 0:
        return None
    x, y = x.detach().cpu(), y.detach().cpu()
    count = min(int(n), len(y))
    labels = torch.unique(y).tolist()
    quota, remainder = divmod(count, max(1, len(labels)))
    generator = torch.Generator().manual_seed(int(seed))
    selected: list[int] = []
    for position, label in enumerate(labels):
        candidates = torch.nonzero(y == label, as_tuple=False).flatten()
        take = min(len(candidates), quota + int(position < remainder))
        if take:
            order = torch.randperm(len(candidates), generator=generator)[:take]
            selected.extend(candidates[order].tolist())
    if len(selected) < count:
        selected_set = set(selected)
        remaining = torch.tensor(
            [index for index in range(len(y)) if index not in selected_set],
            dtype=torch.long,
        )
        take = min(count - len(selected), len(remaining))
        if take:
            order = torch.randperm(len(remaining), generator=generator)[:take]
            selected.extend(remaining[order].tolist())
    indices = torch.tensor(selected, dtype=torch.long)
    if len(indices) > 1:
        indices = indices[torch.randperm(len(indices), generator=generator)]
    return x[indices].to(device), y[indices].to(device)


def _label_blind_uniform_query_indices(
    n_samples: int,
    budget: int,
    seed: int,
) -> torch.Tensor:
    """Choose a reproducible update-label query without reading labels."""

    count = min(max(0, int(budget)), max(0, int(n_samples)))
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(int(n_samples), generator=generator)[:count]


def _v18_split_query_train_guard(
    x: torch.Tensor,
    y: torch.Tensor,
    guard_fraction: float,
    min_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split already-acquired labels into train and current-concept guard sets.

    The query itself remains label-blind. Labels are used only after acquisition
    to create a class-spread holdout. At least one sample of every represented
    class is retained for candidate training whenever possible.
    """

    if len(y) < 2:
        raise ValueError("v18 current guard requires at least two queried labels")
    target = max(
        1,
        min(
            len(y) - 1,
            int(round(len(y) * float(np.clip(guard_fraction, 0.0, 0.9)))),
        ),
    )
    labels = torch.unique(y.detach().cpu().long()).tolist()
    generator = torch.Generator().manual_seed(int(seed))
    shuffled: dict[int, list[int]] = {}
    for label in labels:
        candidates = torch.nonzero(
            y.detach().cpu().long() == label, as_tuple=False
        ).flatten()
        order = torch.randperm(len(candidates), generator=generator)
        shuffled[int(label)] = candidates[order].tolist()

    chosen: list[int] = []
    positions = {int(label): 0 for label in labels}

    def take_one(label: int) -> bool:
        candidates = shuffled[label]
        position = positions[label]
        # Preserve one example of a represented class for training.
        if position >= max(0, len(candidates) - 1):
            return False
        chosen.append(int(candidates[position]))
        positions[label] += 1
        return True

    for _ in range(max(0, int(min_per_class))):
        for label in labels:
            if len(chosen) >= target:
                break
            take_one(int(label))
    while len(chosen) < target:
        progressed = False
        for label in labels:
            if len(chosen) >= target:
                break
            progressed = take_one(int(label)) or progressed
        if not progressed:
            break
    if not chosen:
        raise RuntimeError("v18 could not create a non-empty current guard")

    guard_indices = torch.tensor(chosen, dtype=torch.long)
    guard_set = set(chosen)
    train_indices = torch.tensor(
        [index for index in range(len(y)) if index not in guard_set],
        dtype=torch.long,
    )
    if len(train_indices) > 1:
        train_indices = train_indices[
            torch.randperm(len(train_indices), generator=generator)
        ]
    if len(guard_indices) > 1:
        guard_indices = guard_indices[
            torch.randperm(len(guard_indices), generator=generator)
        ]
    return (
        x[train_indices],
        y[train_indices],
        x[guard_indices],
        y[guard_indices],
    )


def _v18_label_query_indices(
    model: OnlineMLP,
    x: torch.Tensor,
    memory: ReservoirMemory,
    n: int,
    device: torch.device,
    seed: int,
    *,
    entropy_weight: float,
    novelty_weight: float,
    disagreement_weight: float,
    global_fraction: float,
    candidate_multiplier: int,
    batch_size: int,
    soft_class_fraction: float = 0.0,
    class_quota_weights: torch.Tensor | None = None,
    raw_diversity_fraction: float = 0.0,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Select label queries without reading labels from the current window.

    A global uncertainty/novelty pool catches confidently misclassified novel
    regions even when pseudo-classes collapse. The remaining quota is spread
    across available pseudo-classes. Selection is deterministic and labels are
    accessed by the caller only after the indices have been fixed.
    """

    count = min(max(0, int(n)), len(x))
    if count <= 0:
        return torch.empty(0, dtype=torch.long), {
            "v18_query_size": 0,
            "v18_query_mean_entropy": np.nan,
            "v18_query_mean_novelty": np.nan,
            "v18_query_predicted_per_class": "",
            "v18_query_prototype_per_class": "",
            "v18_query_selected_predicted_per_class": "",
            "v18_query_selected_prototype_per_class": "",
        }
    model.eval()
    logits_parts: list[torch.Tensor] = []
    feature_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(x), max(1, int(batch_size))):
            logits, representations = model(
                x[start : start + batch_size].to(device),
                return_representations=True,
            )
            logits_parts.append(logits.detach().cpu())
            feature_parts.append(
                list(representations.values())[-1].detach().cpu()
            )
    logits = torch.cat(logits_parts)
    features = F.normalize(torch.cat(feature_parts).float(), dim=1)
    probabilities = logits.float().softmax(dim=1)
    predicted = probabilities.argmax(dim=1)
    if probabilities.shape[1] > 1:
        entropy = -torch.sum(
            probabilities * probabilities.clamp_min(1e-12).log(), dim=1
        ) / float(np.log(probabilities.shape[1]))
    else:
        entropy = torch.zeros(len(probabilities))

    novelty = torch.ones(len(x), dtype=torch.float32)
    prototype_predicted = predicted.clone()
    historical = memory.tensors(device)
    if historical is not None and len(historical[0]):
        memory_features = _representations(
            model, historical[0], device
        )
        memory_features_last = F.normalize(
            list(memory_features.values())[-1].float(), dim=1
        )
        memory_y = historical[1].detach().cpu().long()
        centroids: list[torch.Tensor] = []
        centroid_labels: list[int] = []
        for label in torch.unique(memory_y).tolist():
            class_features = memory_features_last[memory_y == label]
            if len(class_features):
                centroids.append(
                    F.normalize(
                        class_features.mean(dim=0, keepdim=True), dim=1
                    )[0]
                )
                centroid_labels.append(int(label))
        if centroids:
            centroid_matrix = torch.stack(centroids)
            similarities = features @ centroid_matrix.T
            maximum_similarity, prototype_positions = similarities.max(dim=1)
            novelty = ((1.0 - maximum_similarity) * 0.5).clamp(0.0, 1.0)
            label_lookup = torch.tensor(centroid_labels, dtype=torch.long)
            prototype_predicted = label_lookup[prototype_positions]

    entropy_weight = max(0.0, float(entropy_weight))
    novelty_weight = max(0.0, float(novelty_weight))
    disagreement_weight = max(0.0, float(disagreement_weight))
    disagreement = (prototype_predicted != predicted).float()
    weight_sum = max(
        entropy_weight + novelty_weight + disagreement_weight, 1e-12
    )
    score = (
        entropy_weight * entropy
        + novelty_weight * novelty
        + disagreement_weight * disagreement
    ) / weight_sum
    soft_class_reserved = 0
    raw_diversity_reserved = 0
    if count == len(x):
        selected = torch.arange(len(x), dtype=torch.long)
    else:
        pool_size = min(
            len(x),
            max(count, int(candidate_multiplier) * count),
        )
        pool = torch.argsort(score, descending=True, stable=True)[:pool_size]
        global_count = min(
            count,
            max(1, int(round(count * float(np.clip(global_fraction, 0.0, 1.0))))),
        )
        chosen: list[int] = []
        chosen_set: set[int] = set()

        # Raw-feature k-center coverage is independent of a potentially
        # collapsed representation and remains label-blind.
        raw_count = min(
            count,
            max(
                0,
                int(
                    round(
                        count
                        * float(
                            np.clip(raw_diversity_fraction, 0.0, 1.0)
                        )
                    )
                ),
            ),
        )
        if raw_count:
            raw = x.detach().cpu().float()
            raw = (raw - raw.mean(dim=0, keepdim=True)) / raw.std(
                dim=0, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            raw = F.normalize(raw, dim=1)
            center = F.normalize(raw.mean(dim=0, keepdim=True), dim=1)[0]
            first = int(
                torch.argmin(raw @ center).item()
            )
            raw_selected = [first]
            minimum_distance = (
                1.0 - raw @ raw[first]
            ).clamp_min(0.0)
            for _ in range(1, raw_count):
                minimum_distance[
                    torch.tensor(raw_selected, dtype=torch.long)
                ] = -torch.inf
                index = int(torch.argmax(minimum_distance).item())
                raw_selected.append(index)
                distance = (1.0 - raw @ raw[index]).clamp_min(0.0)
                minimum_distance = torch.minimum(
                    minimum_distance, distance
                )
            chosen.extend(raw_selected)
            chosen_set.update(raw_selected)
            raw_diversity_reserved = len(raw_selected)

        # A label-blind soft-class reserve takes the highest relative
        # probability for every output class.  It still works when argmax
        # predictions collapse to one class, because the non-winning class
        # retains an ordered probability tail.  No current label is read.
        reserve_count = min(
            count - len(chosen),
            max(
                0,
                int(
                    round(
                        count
                        * float(np.clip(soft_class_fraction, 0.0, 1.0))
                    )
                ),
            ),
        )
        if reserve_count and probabilities.shape[1] > 1:
            if (
                class_quota_weights is not None
                and len(class_quota_weights) == probabilities.shape[1]
            ):
                mass = class_quota_weights.detach().cpu().float().clamp_min(0)
                mass = mass / mass.sum().clamp_min(1e-12)
            else:
                mass = torch.full(
                    (probabilities.shape[1],),
                    1.0 / probabilities.shape[1],
                )
            raw_quota = mass * reserve_count
            quotas = torch.floor(raw_quota).long()
            remainder = reserve_count - int(quotas.sum().item())
            if remainder:
                order = torch.argsort(
                    raw_quota - quotas.float(),
                    descending=True,
                    stable=True,
                )
                quotas[order[:remainder]] += 1
            for label in range(int(probabilities.shape[1])):
                take = int(quotas[label].item())
                if not take:
                    continue
                ordered = torch.argsort(
                    probabilities[:, label],
                    descending=True,
                    stable=True,
                )
                selected_for_class: list[int] = []
                for index in ordered.tolist():
                    if int(index) in chosen_set:
                        continue
                    selected_for_class.append(int(index))
                    if len(selected_for_class) >= take:
                        break
                chosen.extend(selected_for_class)
                chosen_set.update(selected_for_class)
            soft_class_reserved = len(chosen)

        # Diversity-aware greedy selection inside the high-value global pool.
        if not chosen:
            chosen = [int(pool[0].item())]
            chosen_set = set(chosen)
        chosen_pool_positions: list[int] = [
            position
            for position, index in enumerate(pool.tolist())
            if int(index) in chosen_set
        ]
        pool_features = features[pool]
        selected_feature_matrix = features[
            torch.tensor(chosen, dtype=torch.long)
        ]
        minimum_distance = (
            1.0 - pool_features @ selected_feature_matrix.T
        ).clamp_min(0.0).min(dim=1).values
        score_in_pool = score[pool]
        global_target = min(count, max(global_count, len(chosen)))
        for _ in range(len(chosen), global_target):
            utility = 0.5 * score_in_pool + 0.5 * minimum_distance
            utility[chosen_pool_positions] = -torch.inf
            local = int(torch.argmax(utility).item())
            index = int(pool[local].item())
            if index in chosen_set:
                break
            chosen.append(index)
            chosen_set.add(index)
            chosen_pool_positions.append(local)
            distance = (1.0 - pool_features @ features[index]).clamp_min(0.0)
            minimum_distance = torch.minimum(minimum_distance, distance)

        remaining_count = count - len(chosen)
        # Prototype strata remain informative when the classifier collapses
        # to the majority class after drift.
        classes = torch.unique(prototype_predicted).tolist()
        if remaining_count > 0 and classes:
            quota, remainder = divmod(remaining_count, len(classes))
            for position, label in enumerate(classes):
                candidates = torch.nonzero(
                    prototype_predicted == label, as_tuple=False
                ).flatten()
                candidates = torch.tensor(
                    [
                        int(index)
                        for index in candidates.tolist()
                        if int(index) not in chosen_set
                    ],
                    dtype=torch.long,
                )
                take = min(
                    len(candidates), quota + int(position < remainder)
                )
                if take:
                    ordered = candidates[
                        torch.argsort(
                            score[candidates], descending=True, stable=True
                        )[:take]
                    ].tolist()
                    chosen.extend(ordered)
                    chosen_set.update(ordered)
        if len(chosen) < count:
            candidates = torch.tensor(
                [
                    index
                    for index in range(len(x))
                    if index not in chosen_set
                ],
                dtype=torch.long,
            )
            ordered = candidates[
                torch.argsort(score[candidates], descending=True, stable=True)[
                    : count - len(chosen)
                ]
            ].tolist()
            chosen.extend(ordered)
        generator = torch.Generator().manual_seed(int(seed))
        selected = torch.tensor(chosen, dtype=torch.long)
        selected = selected[
            torch.randperm(len(selected), generator=generator)
        ]

    predicted_counts = torch.bincount(
        predicted, minlength=probabilities.shape[1]
    )
    prototype_counts = torch.bincount(
        prototype_predicted, minlength=probabilities.shape[1]
    )
    selected_predicted_counts = torch.bincount(
        predicted[selected], minlength=probabilities.shape[1]
    )
    selected_prototype_counts = torch.bincount(
        prototype_predicted[selected], minlength=probabilities.shape[1]
    )
    diagnostics: dict[str, object] = {
        "v18_query_size": int(len(selected)),
        "v18_query_mean_entropy": float(entropy[selected].mean().item()),
        "v18_query_mean_novelty": float(novelty[selected].mean().item()),
        "v18_query_predicted_per_class": "|".join(
            str(int(value)) for value in predicted_counts.tolist()
        ),
        "v18_query_prototype_per_class": "|".join(
            str(int(value)) for value in prototype_counts.tolist()
        ),
        "v18_query_selected_predicted_per_class": "|".join(
            str(int(value)) for value in selected_predicted_counts.tolist()
        ),
        "v18_query_selected_prototype_per_class": "|".join(
            str(int(value)) for value in selected_prototype_counts.tolist()
        ),
        "safe_query_soft_class_reserved": int(soft_class_reserved),
        "v3_query_raw_diversity_reserved": int(raw_diversity_reserved),
    }
    return selected, diagnostics


def _has_class_support(
    y: torch.Tensor,
    n_classes: int,
    minimum_per_class: int = 1,
) -> bool:
    """Return whether every model class has enough already revealed labels."""

    if not len(y) or n_classes <= 0:
        return False
    counts = torch.bincount(
        y.detach().cpu().long(), minlength=int(n_classes)
    )
    return bool(
        len(counts) >= int(n_classes)
        and torch.all(counts[: int(n_classes)] >= int(minimum_per_class))
    )


def _merge_safe_label_buffer(
    history: tuple[torch.Tensor, torch.Tensor] | None,
    new_x: torch.Tensor,
    new_y: torch.Tensor,
    capacity: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Accumulate revealed labels without training on a one-class batch."""

    if capacity <= 0 or not len(new_y):
        return history
    if history is None:
        combined_x = new_x.detach().cpu()
        combined_y = new_y.detach().cpu()
    else:
        combined_x = torch.cat(
            [history[0].detach().cpu(), new_x.detach().cpu()]
        )
        combined_y = torch.cat(
            [history[1].detach().cpu(), new_y.detach().cpu()]
        )
    return _balanced_tensor_sample(
        combined_x,
        combined_y,
        min(int(capacity), len(combined_y)),
        torch.device("cpu"),
        seed,
    )


def _true_label_memory_update(
    history: tuple[torch.Tensor, torch.Tensor] | None,
    new_x: torch.Tensor,
    new_y: torch.Tensor,
    *,
    capacity: int,
    n_classes: int,
    minimum_class_quota: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-quota reservoir using only labels already revealed."""

    if history is None:
        combined_x = new_x.detach().cpu()
        combined_y = new_y.detach().cpu().long()
    else:
        combined_x = torch.cat(
            [history[0].detach().cpu(), new_x.detach().cpu()]
        )
        combined_y = torch.cat(
            [history[1].detach().cpu().long(), new_y.detach().cpu().long()]
        )
    count = min(max(1, int(capacity)), len(combined_y))
    generator = torch.Generator().manual_seed(int(seed))
    class_count = max(1, int(n_classes))
    equal_quota = count // class_count
    per_class_floor = min(
        max(0, int(minimum_class_quota)),
        equal_quota,
    )
    available = torch.bincount(
        combined_y, minlength=class_count
    )[:class_count].tolist()
    allocations = [
        min(
            int(available[label]),
            max(
                per_class_floor,
                equal_quota + (1 if label < count % class_count else 0),
            ),
        )
        for label in range(class_count)
    ]
    remaining = count - sum(allocations)
    while remaining > 0:
        eligible = [
            label
            for label in range(class_count)
            if allocations[label] < int(available[label])
        ]
        if not eligible:
            break
        for label in eligible:
            if remaining <= 0:
                break
            allocations[label] += 1
            remaining -= 1
    selected: list[int] = []
    for label in range(class_count):
        candidates = torch.nonzero(
            combined_y == label, as_tuple=False
        ).flatten()
        take = allocations[label]
        if take:
            order = torch.randperm(len(candidates), generator=generator)
            chosen = candidates[order[:take]].tolist()
            selected.extend(chosen)
    indices = torch.tensor(selected, dtype=torch.long)
    if len(indices) > 1:
        indices = indices[
            torch.randperm(len(indices), generator=generator)
        ]
    return combined_x[indices], combined_y[indices]


def _bounded_label_history_update(
    history: torch.Tensor | None,
    new_y: torch.Tensor,
    capacity: int,
) -> torch.Tensor | None:
    """Append newly revealed labels to a causal, bounded FIFO history."""

    if capacity <= 0 or not len(new_y):
        return history
    labels = new_y.detach().cpu().long()
    combined = labels if history is None else torch.cat([history.cpu(), labels])
    return combined[-min(int(capacity), len(combined)) :].clone()


def _causal_logit_adjustment(
    label_history: torch.Tensor | None,
    *,
    n_classes: int,
    tau: float,
    smoothing: float,
    prior_floor: float,
    device: torch.device,
) -> torch.Tensor | None:
    """Return LAS-style offsets from labels revealed before this update.

    The offset is used only in the training loss.  Inference keeps raw logits,
    so a majority-class positive offset during training induces a compensating
    minority margin without reading labels from the prediction window.
    """

    if label_history is None or not _has_class_support(
        label_history, n_classes, 1
    ):
        return None
    counts = torch.bincount(
        label_history.detach().cpu().long(), minlength=int(n_classes)
    ).float()
    counts = counts[: int(n_classes)] + max(0.0, float(smoothing))
    prior = counts / counts.sum().clamp_min(1e-12)
    floor = float(np.clip(prior_floor, 0.0, 1.0 / max(1, n_classes)))
    if floor > 0.0:
        prior = prior.clamp_min(floor)
        prior = prior / prior.sum()
    return (float(tau) * torch.log(prior.clamp_min(1e-8))).to(device)


def _informative_true_label_memory_update(
    model: OnlineMLP,
    history: tuple[torch.Tensor, torch.Tensor] | None,
    new_x: torch.Tensor,
    new_y: torch.Tensor,
    *,
    capacity: int,
    n_classes: int,
    minimum_class_quota: int,
    recent_fraction: float,
    boundary_fraction: float,
    diversity_fraction: float,
    diversity_cap: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal class-quota memory with boundary, diversity and recency strata."""

    if history is None:
        historical_count = 0
        combined_x = new_x.detach().cpu()
        combined_y = new_y.detach().cpu().long()
    else:
        historical_count = len(history[1])
        combined_x = torch.cat(
            [history[0].detach().cpu(), new_x.detach().cpu()]
        )
        combined_y = torch.cat(
            [history[1].detach().cpu().long(), new_y.detach().cpu().long()]
        )
    count = min(max(1, int(capacity)), len(combined_y))
    class_count = max(1, int(n_classes))
    equal_quota = count // class_count
    per_class_floor = min(
        max(0, int(minimum_class_quota)),
        equal_quota,
    )
    available = torch.bincount(
        combined_y, minlength=class_count
    )[:class_count].tolist()
    allocations = [
        min(
            int(available[label]),
            max(
                per_class_floor,
                equal_quota + (1 if label < count % class_count else 0),
            ),
        )
        for label in range(class_count)
    ]
    remaining = count - sum(allocations)
    while remaining > 0:
        eligible = [
            label
            for label in range(class_count)
            if allocations[label] < int(available[label])
        ]
        if not eligible:
            break
        for label in eligible:
            if remaining <= 0:
                break
            allocations[label] += 1
            remaining -= 1

    model.eval()
    logits_parts: list[torch.Tensor] = []
    representation_parts: list[torch.Tensor] = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(combined_x), batch_size):
            batch = combined_x[start : start + batch_size].to(device)
            logits, representations = model(
                batch, return_representations=True
            )
            logits_parts.append(logits.detach().cpu())
            representation_parts.append(
                representations[model.layer_names[-1]].detach().cpu()
            )
    logits = torch.cat(logits_parts)
    features = F.normalize(torch.cat(representation_parts).float(), dim=1)
    probabilities = F.softmax(logits, dim=1)
    top_values = torch.topk(
        probabilities, k=min(2, probabilities.shape[1]), dim=1
    ).values
    margin = (
        top_values[:, 0] - top_values[:, 1]
        if probabilities.shape[1] > 1
        else top_values[:, 0]
    )
    losses = F.cross_entropy(logits, combined_y, reduction="none")
    loss_scale = (losses - losses.min()) / (
        losses.max() - losses.min()
    ).clamp_min(1e-12)
    information = 0.5 * (1.0 - margin) + 0.5 * loss_scale

    recent_share = float(np.clip(recent_fraction, 0.0, 1.0))
    boundary_share = float(np.clip(boundary_fraction, 0.0, 1.0))
    diversity_share = float(np.clip(diversity_fraction, 0.0, 1.0))
    generator = torch.Generator().manual_seed(int(seed))
    selected: list[int] = []

    for label, allocation in enumerate(allocations):
        candidates = torch.nonzero(
            combined_y == label, as_tuple=False
        ).flatten()
        if allocation <= 0 or not len(candidates):
            continue
        chosen: list[int] = []
        chosen_set: set[int] = set()

        recent_candidates = candidates[candidates >= historical_count]
        recent_take = min(
            len(recent_candidates),
            int(round(allocation * recent_share)),
        )
        if recent_take:
            recent_order = recent_candidates[
                torch.argsort(
                    information[recent_candidates],
                    descending=True,
                    stable=True,
                )[:recent_take]
            ].tolist()
            chosen.extend(int(index) for index in recent_order)
            chosen_set.update(int(index) for index in recent_order)

        boundary_take = min(
            allocation - len(chosen),
            int(round(allocation * boundary_share)),
        )
        if boundary_take:
            eligible = torch.tensor(
                [
                    int(index)
                    for index in candidates.tolist()
                    if int(index) not in chosen_set
                ],
                dtype=torch.long,
            )
            boundary = eligible[
                torch.argsort(
                    information[eligible],
                    descending=True,
                    stable=True,
                )[:boundary_take]
            ].tolist()
            chosen.extend(int(index) for index in boundary)
            chosen_set.update(int(index) for index in boundary)

        diversity_take = min(
            allocation - len(chosen),
            int(round(allocation * diversity_share)),
            max(0, int(diversity_cap)),
        )
        if diversity_take:
            eligible = torch.tensor(
                [
                    int(index)
                    for index in candidates.tolist()
                    if int(index) not in chosen_set
                ],
                dtype=torch.long,
            )
            pool_limit = min(
                len(eligible),
                max(256, 4 * diversity_take),
            )
            if len(eligible) > pool_limit:
                order = torch.randperm(len(eligible), generator=generator)
                eligible = eligible[order[:pool_limit]]
            pool_features = features[eligible]
            centroid = F.normalize(
                pool_features.mean(dim=0, keepdim=True), dim=1
            )[0]
            first = int(torch.argmin(pool_features @ centroid).item())
            local_selected = [first]
            minimum_distance = (
                1.0 - pool_features @ pool_features[first]
            ).clamp_min(0.0)
            for _ in range(1, min(diversity_take, len(eligible))):
                minimum_distance[
                    torch.tensor(local_selected, dtype=torch.long)
                ] = -torch.inf
                local = int(torch.argmax(minimum_distance).item())
                local_selected.append(local)
                distance = (
                    1.0 - pool_features @ pool_features[local]
                ).clamp_min(0.0)
                minimum_distance = torch.minimum(
                    minimum_distance, distance
                )
            diversity = eligible[
                torch.tensor(local_selected, dtype=torch.long)
            ].tolist()
            chosen.extend(int(index) for index in diversity)
            chosen_set.update(int(index) for index in diversity)

        fill = allocation - len(chosen)
        if fill:
            eligible = torch.tensor(
                [
                    int(index)
                    for index in candidates.tolist()
                    if int(index) not in chosen_set
                ],
                dtype=torch.long,
            )
            order = torch.randperm(len(eligible), generator=generator)
            chosen.extend(
                int(index) for index in eligible[order[:fill]].tolist()
            )
        selected.extend(chosen[:allocation])

    indices = torch.tensor(selected, dtype=torch.long)
    if len(indices) > 1:
        indices = indices[
            torch.randperm(len(indices), generator=generator)
        ]
    return combined_x[indices], combined_y[indices]


def _method_true_label_memory_update(
    method: str,
    model: OnlineMLP,
    history: tuple[torch.Tensor, torch.Tensor] | None,
    new_x: torch.Tensor,
    new_y: torch.Tensor,
    *,
    cfg: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method in V4_INFO_MEMORY_METHODS:
        return _informative_true_label_memory_update(
            model,
            history,
            new_x,
            new_y,
            capacity=cfg.dasp.memory_size,
            n_classes=int(model.classifier.out_features),
            minimum_class_quota=cfg.dasp.v3_memory_min_class_quota,
            recent_fraction=cfg.dasp.v4_memory_recent_fraction,
            boundary_fraction=cfg.dasp.v4_memory_boundary_fraction,
            diversity_fraction=cfg.dasp.v4_memory_diversity_fraction,
            diversity_cap=cfg.dasp.v4_memory_diversity_cap,
            device=device,
            seed=seed,
        )
    return _true_label_memory_update(
        history,
        new_x,
        new_y,
        capacity=cfg.dasp.memory_size,
        n_classes=int(model.classifier.out_features),
        minimum_class_quota=cfg.dasp.v3_memory_min_class_quota,
        seed=seed,
    )


def _update_v18_guard_history(
    history: tuple[torch.Tensor, torch.Tensor] | None,
    new_x: torch.Tensor,
    new_y: torch.Tensor,
    capacity: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Keep a bounded class-balanced prequential guard for future updates."""

    if capacity <= 0 or not len(new_y):
        return history
    if history is None:
        combined_x, combined_y = new_x.detach().cpu(), new_y.detach().cpu()
    else:
        combined_x = torch.cat([history[0].cpu(), new_x.detach().cpu()])
        combined_y = torch.cat([history[1].cpu(), new_y.detach().cpu()])
    return _balanced_tensor_sample(
        combined_x,
        combined_y,
        min(int(capacity), len(combined_y)),
        torch.device("cpu"),
        seed,
    )


def _split_train_guard(
    window: StreamWindow, guard_ratio: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(window.x)
    guard_n = max(1, min(n - 1, int(round(n * guard_ratio))))
    generator = torch.Generator().manual_seed(seed + window.index * 104729)
    order = torch.randperm(n, generator=generator)
    train_idx, guard_idx = order[:-guard_n], order[-guard_n:]
    return window.x[train_idx], window.y[train_idx], window.x[guard_idx], window.y[guard_idx]


def _train_online(
    model: OnlineMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
    replay: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> int:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    if replay is not None:
        x = torch.cat([x.to(device), replay[0]], dim=0)
        y = torch.cat([y.to(device), replay[1]], dim=0)
    else:
        x, y = x.to(device), y.to(device)
    gradient_evaluations = 0
    for epoch in range(cfg.training.epochs_per_window):
        generator = torch.Generator(device="cpu").manual_seed(
            cfg.data.random_seed + epoch + len(x)
        )
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.training.batch_size):
            idx = order[start : start + cfg.training.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            gradient_evaluations += 1
    return gradient_evaluations


def _make_stateful_optimizer(
    model: OnlineMLP, cfg: ExperimentConfig
) -> torch.optim.AdamW:
    groups = [
        {
            "params": params,
            "lr": cfg.training.learning_rate,
            "name": name,
        }
        for name, params in model.adaptation_parameter_groups().items()
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )


def _clone_stateful_optimizer(
    optimizer: torch.optim.AdamW,
    model: OnlineMLP,
    cfg: ExperimentConfig,
) -> torch.optim.AdamW:
    cloned = _make_stateful_optimizer(model, cfg)
    cloned.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return cloned


def _set_optimizer_plasticities(
    optimizer: torch.optim.AdamW,
    plasticities: dict[str, float] | None,
    cfg: ExperimentConfig,
) -> None:
    for group in optimizer.param_groups:
        name = str(group.get("name", "classifier"))
        if plasticities is None:
            multiplier = 1.0
        else:
            p = float(np.clip(plasticities.get(name, 1.0), 0.0, 1.0))
            multiplier = cfg.dasp.eta_floor + (1.0 - cfg.dasp.eta_floor) * p
        group["lr"] = cfg.training.learning_rate * multiplier


def _train_stateful_online(
    model: OnlineMLP,
    optimizer: torch.optim.AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
    seed_offset: int,
    epochs_multiplier: int = 1,
) -> int:
    model.train()
    _set_optimizer_plasticities(optimizer, None, cfg)
    x, y = x.to(device), y.to(device)
    gradient_evaluations = 0
    total_epochs = cfg.training.epochs_per_window * max(1, epochs_multiplier)
    for epoch in range(total_epochs):
        generator = torch.Generator(device="cpu").manual_seed(
            cfg.data.random_seed + seed_offset * 104729 + epoch
        )
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.training.batch_size):
            idx = order[start : start + cfg.training.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            optimizer.step()
            gradient_evaluations += 1
    return gradient_evaluations


def _estimate_fisher(
    model: OnlineMLP,
    batch: tuple[torch.Tensor, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if batch is None or not len(batch[0]):
        return {
            name: torch.zeros_like(parameter, device=device)
            for name, parameter in model.named_parameters()
        }
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(batch[0].to(device)), batch[1].to(device))
    loss.backward()
    return {
        name: (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach().square().clone()
        )
        for name, parameter in model.named_parameters()
    }


def _train_online_ewc(
    model: OnlineMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    reference: dict[str, torch.Tensor],
    fisher: dict[str, torch.Tensor],
    cfg: ExperimentConfig,
    device: torch.device,
    seed_offset: int,
) -> int:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    x, y = x.to(device), y.to(device)
    gradient_evaluations = 0
    for epoch in range(cfg.training.epochs_per_window):
        generator = torch.Generator(device="cpu").manual_seed(
            cfg.data.random_seed + seed_offset * 104729 + epoch
        )
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.training.batch_size):
            idx = order[start : start + cfg.training.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            penalty = torch.zeros((), device=device)
            for name, parameter in model.named_parameters():
                penalty = penalty + torch.sum(
                    fisher[name] * (parameter - reference[name]).square()
                )
            (loss + 0.5 * cfg.dasp.ewc_lambda * penalty).backward()
            optimizer.step()
            gradient_evaluations += 1
    return gradient_evaluations


def _spend_probe_compute_budget(
    model: OnlineMLP,
    optimizer: torch.optim.AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> int:
    """Give the compute baseline the same number of probe backpropagations."""
    _set_optimizer_plasticities(optimizer, None, cfg)
    model.train()
    take = min(cfg.dasp.utility_probe_samples, len(x))
    probe_x, probe_y = x[:take].to(device), y[:take].to(device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(probe_x), probe_y).backward()
        optimizer.step()
    return steps


def _all_group_gradients(
    model: OnlineMLP,
    loss: torch.Tensor,
    groups: dict[str, list[torch.nn.Parameter]] | None = None,
) -> dict[str, list[torch.Tensor]]:
    groups = model.adaptation_parameter_groups() if groups is None else groups
    params = [parameter for values in groups.values() for parameter in values]
    original_review.count_gradient_request(params)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    result: dict[str, list[torch.Tensor]] = {}
    offset = 0
    for name, values in groups.items():
        group: list[torch.Tensor] = []
        for parameter, grad in zip(values, grads[offset : offset + len(values)]):
            group.append(torch.zeros_like(parameter) if grad is None else grad.detach())
        result[name] = group
        offset += len(values)
    return result


def _gradient_conflicts(
    model: OnlineMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    memory_batch: tuple[torch.Tensor, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, float]:
    if memory_batch is None:
        return {name: 0.0 for name in model.layer_names}
    model.train()
    new_loss = F.cross_entropy(model(x.to(device)), y.to(device))
    new_grads = _all_group_gradients(model, new_loss)
    memory_loss = F.cross_entropy(model(memory_batch[0]), memory_batch[1])
    memory_grads = _all_group_gradients(model, memory_loss)
    conflicts: dict[str, float] = {}
    for name in model.layer_names:
        dot = sum(torch.sum(a * b) for a, b in zip(new_grads[name], memory_grads[name]))
        norm_a = torch.sqrt(sum(torch.sum(a.square()) for a in new_grads[name]))
        norm_b = torch.sqrt(sum(torch.sum(b.square()) for b in memory_grads[name]))
        cosine = dot / (norm_a * norm_b + 1e-12)
        conflicts[name] = float(torch.clamp(-cosine, min=0.0, max=1.0).item())
    return conflicts


def _plasticities(
    method: str,
    requirements: dict[str, float],
    conflicts: dict[str, float],
    conflict_weight: float,
) -> dict[str, float]:
    if method == "shift_only":
        values = dict(requirements)
    elif method == "conflict_only":
        values = {name: 1.0 / (1.0 + conflict_weight * conflicts[name]) for name in requirements}
    else:
        values = {
            name: requirements[name] / (1.0 + conflict_weight * conflicts[name])
            for name in requirements
        }
    if method == "global_dasp":
        scalar = float(np.mean(list(values.values())))
        return {name: scalar for name in values}
    if method == "global_protection":
        return {name: 0.0 for name in values}
    return values


def _train_dasp(
    model: OnlineMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    plasticities: dict[str, float],
    cfg: ExperimentConfig,
    device: torch.device,
    memory: ReservoirMemory,
    projection_memory: tuple[torch.Tensor, torch.Tensor] | None = None,
    eta_ceiling: float = 1.0,
    strict_freeze: bool = False,
    sample_weights: torch.Tensor | None = None,
    epochs: int | None = None,
    focal_gamma: float = 0.0,
    use_projection_memory: bool = True,
    parameter_groups: dict[str, list[torch.nn.Parameter]] | None = None,
    logit_adjustment: torch.Tensor | None = None,
) -> int:
    model.train()
    x, y = x.to(device), y.to(device)
    weights = sample_weights.to(device) if sample_weights is not None else None
    if weights is not None and len(weights) != len(x):
        raise ValueError("sample_weights must have the same length as x")
    groups = (
        model.adaptation_parameter_groups()
        if parameter_groups is None
        else parameter_groups
    )
    if cfg.dasp.baseline == "modular_dasp" and cfg.original_review.enabled and strict_freeze:
        # Original DASP correction: exclude frozen groups from differentiation
        # as well as AdamW. Other methods retain their historical behavior.
        groups = {name: params for name, params in groups.items()
                  if float(plasticities.get(name, 1.0)) > 0.0}
    optimizer_groups = []
    for name, params in groups.items():
        p = float(np.clip(plasticities.get(name, 1.0), 0.0, 1.0))
        if strict_freeze and p <= 0.0:
            continue
        lr = cfg.training.learning_rate * (
            cfg.dasp.eta_floor + (eta_ceiling - cfg.dasp.eta_floor) * p
        )
        optimizer_group: dict[str, object] = {
            "params": params,
            "lr": lr,
            "name": name,
        }
        if isinstance(model, AdapterOnlineMLP) and (
            name in model.layer_names
            or name == "input_adapter"
            or name.startswith("modulation_")
        ):
            # Rank masking detaches inactive columns.  AdamW would still alter
            # them through decoupled weight decay, violating the fast-rank
            # invariant, so low-rank adapter groups use no weight decay.
            optimizer_group["weight_decay"] = 0.0
        optimizer_groups.append(optimizer_group)
    if not optimizer_groups:
        return 0
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    gradient_evaluations = 0
    effective_epochs = (
        cfg.training.epochs_per_window if epochs is None else max(1, int(epochs))
    )
    for epoch in range(effective_epochs):
        generator = torch.Generator(device="cpu").manual_seed(
            cfg.data.random_seed + epoch + len(x) + 17
        )
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.training.batch_size):
            idx = order[start : start + cfg.training.batch_size].to(device)
            batch_x, batch_y = x[idx], y[idx]
            memory_batch = (
                (
                    projection_memory
                    if projection_memory is not None
                    else memory.sample(cfg.dasp.memory_batch_size, device)
                )
                if use_projection_memory
                else None
            )
            batch_logits = model(batch_x)
            if logit_adjustment is not None:
                batch_logits = batch_logits + logit_adjustment.to(device)
            per_sample_loss = F.cross_entropy(
                batch_logits,
                batch_y,
                reduction="none",
            )
            if focal_gamma > 0.0:
                per_sample_loss = (
                    1.0 - torch.exp(-per_sample_loss)
                ).pow(float(focal_gamma)) * per_sample_loss
            if weights is None:
                new_loss = per_sample_loss.mean()
            else:
                batch_weights = weights[idx]
                new_loss = torch.sum(per_sample_loss * batch_weights) / torch.clamp(
                    batch_weights.sum(),
                    min=1e-12,
                )
            new_grads = _all_group_gradients(model, new_loss, groups)
            gradient_evaluations += 1
            if memory_batch is None:
                memory_grads = {
                    name: [torch.zeros_like(value) for value in values]
                    for name, values in new_grads.items()
                }
            else:
                memory_logits = model(memory_batch[0])
                if logit_adjustment is not None:
                    memory_logits = (
                        memory_logits + logit_adjustment.to(device)
                    )
                memory_loss = F.cross_entropy(
                    memory_logits, memory_batch[1]
                )
                memory_grads = _all_group_gradients(
                    model, memory_loss, groups
                )
                gradient_evaluations += 1
            optimizer.zero_grad(set_to_none=True)
            for name, params in groups.items():
                p = float(np.clip(plasticities.get(name, 1.0), 0.0, 1.0))
                if strict_freeze and p <= 0.0:
                    continue
                dot = sum(
                    torch.sum(a * b) for a, b in zip(new_grads[name], memory_grads[name])
                )
                norm_sq = sum(torch.sum(value.square()) for value in memory_grads[name])
                coefficient = (1.0 - p) * dot / (norm_sq + 1e-12) if dot < 0 else 0.0
                for parameter, grad_new, grad_memory in zip(
                    params, new_grads[name], memory_grads[name]
                ):
                    protected = grad_new - coefficient * grad_memory
                    parameter.grad = protected.clone()
            optimizer.step()
    return gradient_evaluations


def _surprise_class_balanced_weights(
    model: OnlineMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    base_weight: float,
    maximum_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Weight delayed-label samples by surprise and inverse class frequency."""
    if len(x) == 0:
        return torch.empty(0, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        losses = F.cross_entropy(
            model(x.to(device)),
            y.to(device),
            reduction="none",
        ).detach().cpu()
    order = torch.argsort(losses, stable=True)
    ranks = torch.empty_like(losses)
    ranks[order] = torch.arange(len(losses), dtype=losses.dtype)
    ranks = ranks / max(1, len(losses) - 1)
    surprise = 0.5 + ranks
    labels = y.detach().cpu().long()
    classes = max(1, int(labels.max().item()) + 1)
    counts = torch.bincount(labels, minlength=classes).float().clamp_min(1.0)
    class_factor = len(labels) / (classes * counts[labels])
    class_factor = class_factor / class_factor.mean().clamp_min(1e-12)
    weights = 1.0 + (max(1.0, base_weight) - 1.0) * surprise * class_factor
    return weights.clamp(min=1.0, max=max(1.0, maximum_weight))


def _localized_rehearsal_batch(
    current_x: torch.Tensor,
    current_y: torch.Tensor,
    memory: ReservoirMemory,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    replay = memory.sample_balanced(cfg.dasp.modular_replay_size, device)
    current_x = current_x.to(device)
    current_y = current_y.to(device)
    if replay is None:
        weights = torch.full(
            (len(current_y),),
            float(cfg.dasp.modular_new_sample_weight),
            dtype=torch.float32,
            device=device,
        )
        return current_x, current_y, weights
    replay_x, replay_y = replay
    rehearsal_x = torch.cat([current_x, replay_x], dim=0)
    rehearsal_y = torch.cat([current_y, replay_y], dim=0)
    weights = torch.cat(
        [
            torch.full(
                (len(current_y),),
                float(cfg.dasp.modular_new_sample_weight),
                dtype=torch.float32,
                device=device,
            ),
            torch.ones(len(replay_y), dtype=torch.float32, device=device),
        ],
        dim=0,
    )
    return rehearsal_x, rehearsal_y, weights


def _train_dasp_stateful(
    model: OnlineMLP,
    optimizer: torch.optim.AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    plasticities: dict[str, float],
    cfg: ExperimentConfig,
    device: torch.device,
    projection_memory: tuple[torch.Tensor, torch.Tensor] | None,
    seed_offset: int,
) -> int:
    model.train()
    x, y = x.to(device), y.to(device)
    groups = model.adaptation_parameter_groups()
    _set_optimizer_plasticities(optimizer, plasticities, cfg)
    gradient_evaluations = 0
    for epoch in range(cfg.training.epochs_per_window):
        generator = torch.Generator(device="cpu").manual_seed(
            cfg.data.random_seed + seed_offset * 104729 + epoch + 17
        )
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.training.batch_size):
            idx = order[start : start + cfg.training.batch_size].to(device)
            batch_x, batch_y = x[idx], y[idx]
            new_loss = F.cross_entropy(model(batch_x), batch_y)
            new_grads = _all_group_gradients(model, new_loss)
            gradient_evaluations += 1
            if projection_memory is None:
                memory_grads = {
                    name: [torch.zeros_like(value) for value in values]
                    for name, values in new_grads.items()
                }
            else:
                memory_loss = F.cross_entropy(
                    model(projection_memory[0]), projection_memory[1]
                )
                memory_grads = _all_group_gradients(model, memory_loss)
                gradient_evaluations += 1
            optimizer.zero_grad(set_to_none=True)
            for name, params in groups.items():
                p = float(np.clip(plasticities.get(name, 1.0), 0.0, 1.0))
                dot = sum(
                    torch.sum(a * b)
                    for a, b in zip(new_grads[name], memory_grads[name])
                )
                norm_sq = sum(torch.sum(value.square()) for value in memory_grads[name])
                coefficient = (1.0 - p) * dot / (norm_sq + 1e-12) if dot < 0 else 0.0
                for parameter, grad_new, grad_memory in zip(
                    params, new_grads[name], memory_grads[name]
                ):
                    parameter.grad = (grad_new - coefficient * grad_memory).clone()
            optimizer.step()
    return gradient_evaluations


def _guard_lcb(
    online: OnlineMLP,
    dasp: OnlineMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    z_value: float,
    device: torch.device,
) -> tuple[float, float]:
    online.eval()
    dasp.eval()
    with torch.no_grad():
        target = y.to(device)
        online_losses = F.cross_entropy(online(x.to(device)), target, reduction="none")
        dasp_losses = F.cross_entropy(dasp(x.to(device)), target, reduction="none")
        delta = online_losses - dasp_losses
        mean = float(delta.mean().item())
        standard_error = float(delta.std(unbiased=True).item() / np.sqrt(max(1, len(delta))))
    return mean - z_value * standard_error, mean


def _loss_vector(
    model: OnlineMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return F.cross_entropy(
            model(x.to(device)), y.to(device), reduction="none"
        ).detach()


def _guard_classification(
    model: OnlineMLP,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(batch[0].to(device))
        predictions = _tensor_to_numpy(logits.argmax(dim=1), np.int64)
    metrics = classification_metrics(
        _tensor_to_numpy(batch[1], np.int64),
        predictions,
        int(logits.shape[1]),
    )
    target = _tensor_to_numpy(batch[1], np.int64)
    per_class_recall: list[float] = []
    result = {
        "macro_f1": float(metrics["macro_f1"]),
        "attack_recall": float(metrics["attack_recall"]),
    }
    for label in range(int(logits.shape[1])):
        mask = target == label
        recall = (
            float(np.mean(predictions[mask] == label))
            if np.any(mask)
            else np.nan
        )
        result[f"recall_class_{label}"] = recall
        if np.isfinite(recall):
            per_class_recall.append(recall)
    result["min_per_class_recall"] = (
        float(min(per_class_recall)) if per_class_recall else np.nan
    )
    return result


def _mean_loss(
    model: OnlineMLP,
    batch: tuple[torch.Tensor, torch.Tensor] | None,
    device: torch.device,
) -> float:
    if batch is None or not len(batch[0]):
        return np.nan
    return float(_loss_vector(model, batch[0], batch[1], device).mean().item())


def _counterfactual_utilities(
    model: OnlineMLP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    guard_x: torch.Tensor,
    guard_y: torch.Tensor,
    memory_batch: tuple[torch.Tensor, torch.Tensor] | None,
    cfg: ExperimentConfig,
    device: torch.device,
    forgetting_weight: float | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Estimate layer utility with an isolated, one-step counterfactual update.

    Adaptation gain is measured on a held-out slice of the current window and
    forgetting cost on historical memory.  MMD is deliberately not used here:
    it remains an alarm, while supervised evidence decides *where* to adapt.
    """
    penalty = (
        cfg.dasp.utility_forgetting_weight
        if forgetting_weight is None
        else float(forgetting_weight)
    )
    probe_n = min(cfg.dasp.utility_probe_samples, len(train_x))
    probe_x = train_x[:probe_n].to(device)
    probe_y = train_y[:probe_n].to(device)
    recent_batch = (guard_x, guard_y)
    base_recent = _mean_loss(model, recent_batch, device)
    base_memory = _mean_loss(model, memory_batch, device)
    gains: dict[str, float] = {}
    costs: dict[str, float] = {}
    raw_utilities: dict[str, float] = {}
    for name in model.layer_names:
        candidate = copy.deepcopy(model).to(device)
        params = candidate.adaptation_parameter_groups()[name]
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.training.learning_rate * cfg.dasp.utility_probe_lr_scale,
            weight_decay=cfg.training.weight_decay,
        )
        candidate.train()
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(candidate(probe_x), probe_y).backward()
        optimizer.step()
        gain = base_recent - _mean_loss(candidate, recent_batch, device)
        memory_after = _mean_loss(candidate, memory_batch, device)
        cost = (
            max(0.0, memory_after - base_memory)
            if np.isfinite(base_memory) and np.isfinite(memory_after)
            else 0.0
        )
        gains[name] = float(gain)
        costs[name] = float(cost)
        raw_utilities[name] = float(gain - penalty * cost)
    positive_max = max(0.0, max(raw_utilities.values(), default=0.0))
    if positive_max <= 1e-12:
        plasticities = {name: 0.0 for name in model.layer_names}
    else:
        plasticities = {
            name: float(np.clip(value / positive_max, 0.0, 1.0))
            for name, value in raw_utilities.items()
        }
    return gains, costs, plasticities


def _select_v9_candidate(
    gains: dict[str, float],
    costs: dict[str, float],
    plasticities: dict[str, float],
    forgetting_weight: float,
    minimum_utility: float,
    global_utility_ratio: float,
) -> tuple[str | None, dict[str, float], dict[str, float]]:
    """Preselect one V9 candidate from supervised counterfactual utilities.

    A candidate is not trained when no layer has positive utility. Global
    adaptation is used only when every layer is beneficial and the weakest
    utility is close to the strongest; otherwise heterogeneous layer-wise
    plasticity is retained.
    """
    utilities = {
        name: float(gains[name] - forgetting_weight * costs[name])
        for name in plasticities
    }
    maximum = max(utilities.values(), default=float("-inf"))
    diagnostics = {
        "v9_max_utility": float(maximum),
        "v9_min_utility": float(min(utilities.values(), default=np.nan)),
        "v9_positive_utility_layers": float(
            sum(value > minimum_utility for value in utilities.values())
        ),
    }
    if not np.isfinite(maximum) or maximum <= minimum_utility:
        diagnostics["v9_global_utility_ratio_observed"] = 0.0
        return None, {}, diagnostics

    minimum = min(utilities.values())
    observed_ratio = float(minimum / maximum) if maximum > 0.0 else 0.0
    diagnostics["v9_global_utility_ratio_observed"] = observed_ratio
    if minimum > minimum_utility and observed_ratio >= global_utility_ratio:
        scalar = float(np.mean(list(plasticities.values())))
        return (
            "global",
            {name: scalar for name in plasticities},
            diagnostics,
        )
    return "layer", dict(plasticities), diagnostics


def _progressive_layer_candidates(
    gains: dict[str, float],
    costs: dict[str, float],
    plasticities: dict[str, float],
    all_groups: list[str],
    forgetting_weight: float,
    minimum_utility: float,
    max_adaptive_layers: int,
    minimum_active_plasticity: float,
    classifier_plasticity: float,
) -> tuple[list[tuple[str, dict[str, float]]], dict[str, float | str]]:
    """Build nested, strictly layer-selective candidates for DASP-v10.

    Only hidden representation layers with positive supervised utility are
    eligible.  The classifier remains an explicit adaptation group, while all
    unselected hidden layers receive exactly zero plasticity and are frozen by
    ``_train_dasp(..., strict_freeze=True)``.
    """
    utilities = {
        name: float(gains[name] - forgetting_weight * costs[name])
        for name in plasticities
    }
    ranked = sorted(
        (
            (name, utility)
            for name, utility in utilities.items()
            if np.isfinite(utility) and utility > minimum_utility
        ),
        key=lambda item: (-item[1], item[0]),
    )
    limit = min(max(0, int(max_adaptive_layers)), len(ranked))
    diagnostics: dict[str, float | str] = {
        "v10_positive_utility_layers": float(len(ranked)),
        "v10_max_utility": float(
            max(utilities.values(), default=float("-inf"))
        ),
        "v10_ranked_layers": ",".join(name for name, _ in ranked),
    }
    if limit == 0:
        return [], diagnostics

    candidates: list[tuple[str, dict[str, float]]] = []
    active_floor = float(np.clip(minimum_active_plasticity, 0.0, 1.0))
    classifier_p = float(np.clip(classifier_plasticity, 0.0, 1.0))
    for width in range(1, limit + 1):
        selected = {name for name, _ in ranked[:width]}
        mask = {name: 0.0 for name in all_groups}
        for name in selected:
            mask[name] = max(
                active_floor,
                float(np.clip(plasticities.get(name, 0.0), 0.0, 1.0)),
            )
        if "classifier" in mask:
            mask["classifier"] = classifier_p
        candidates.append((f"layer_top{width}", mask))
    return candidates, diagnostics


def _split_memory_for_probe_and_guard(
    memory: ReservoirMemory,
    n: int,
    device: torch.device,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor] | None,
    tuple[torch.Tensor, torch.Tensor] | None,
]:
    sampled = memory.sample(2 * n, device)
    if sampled is None:
        return None, None
    x, y = sampled
    midpoint = max(1, len(x) // 2)
    probe = (x[:midpoint], y[:midpoint])
    guard = (x[midpoint:], y[midpoint:]) if midpoint < len(x) else None
    return probe, guard


def _dual_guard(
    online: OnlineMLP,
    candidate: OnlineMLP,
    recent: tuple[torch.Tensor, torch.Tensor],
    memory: tuple[torch.Tensor, torch.Tensor] | None,
    cfg: ExperimentConfig,
    device: torch.device,
    memory_weight: float | None = None,
    class_balanced: bool = False,
    attack_tolerance: float | None = None,
) -> tuple[bool, dict[str, float]]:
    def bounds(batch: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[float, float]:
        if batch is None or len(batch[0]) < 2:
            return np.nan, np.nan
        delta = _loss_vector(online, batch[0], batch[1], device) - _loss_vector(
            candidate, batch[0], batch[1], device
        )
        if class_balanced:
            target = batch[1].to(device)
            _, inverse, counts = torch.unique(
                target, return_inverse=True, return_counts=True
            )
            weights = counts[inverse].float().reciprocal()
            weights = weights / weights.mean().clamp_min(1e-12)
            delta = delta * weights
        mean = float(delta.mean().item())
        se = float(delta.std(unbiased=True).item() / np.sqrt(len(delta)))
        return mean - cfg.dasp.lcb_z * se, mean

    recent_lcb, recent_mean = bounds(recent)
    memory_lcb, memory_mean = bounds(memory)
    recent_target = recent[1].to(device)
    attack_mask = recent_target != 0
    attack_batch = (
        (
            recent[0][attack_mask.to(recent[0].device)],
            recent[1][attack_mask.to(recent[1].device)],
        )
        if int(attack_mask.sum().item()) >= 2
        else None
    )
    attack_lcb, attack_mean = bounds(attack_batch)
    effective_memory_weight = (
        cfg.dasp.memory_guard_weight
        if memory_weight is None
        else float(memory_weight)
    )
    if np.isfinite(memory_lcb):
        combined_lcb = recent_lcb + effective_memory_weight * memory_lcb
    else:
        combined_lcb = recent_lcb
    accepted = bool(
        recent_lcb >= -cfg.dasp.recent_guard_tolerance and combined_lcb > 0.0
        and (
            attack_tolerance is None
            or not np.isfinite(attack_lcb)
            or attack_lcb >= -float(attack_tolerance)
        )
    )
    return accepted, {
        "guard_recent_lcb": recent_lcb,
        "guard_recent_mean_delta": recent_mean,
        "guard_memory_lcb": memory_lcb,
        "guard_memory_mean_delta": memory_mean,
        "guard_combined_lcb": combined_lcb,
        "guard_attack_lcb": attack_lcb,
        "guard_attack_mean_delta": attack_mean,
        "guard_memory_weight": effective_memory_weight,
    }


def _severity_weighted_value(base: float, floor: float, pressure: float) -> float:
    """Relax a stability term as evidence for non-stationarity increases."""
    bounded_floor = float(np.clip(floor, 0.0, 1.0))
    bounded_pressure = float(np.clip(pressure, 0.0, 1.0))
    return float(base) * (bounded_floor + (1.0 - bounded_floor) * (1.0 - bounded_pressure))


def _v12_warning_eligible(
    decision: DriftDecision,
    *,
    severity_threshold: float,
    guard_macro_f1: float,
    guard_attack_recall: float,
    minimum_macro_f1: float,
    minimum_attack_recall: float,
) -> bool:
    """Return whether a delayed-labeled pre-confirmation PSLA event is allowed.

    This function is called only from ``_update``, after the configured label
    delay. It must never be used in the prediction path.
    """

    return bool(
        not decision.triggered
        and decision.severity > severity_threshold
        and (
            guard_macro_f1 < minimum_macro_f1
            or guard_attack_recall < minimum_attack_recall
        )
    )


def _v13_class_balanced_score(
    diagnostics: dict[str, float],
    *,
    macro_f1_tolerance: float,
    per_class_recall_tolerance: float,
) -> tuple[float, float, float] | None:
    """Rank a candidate by Macro-F1 while enforcing every observed class."""

    candidate_macro = diagnostics.get("guard_candidate_macro_f1", np.nan)
    fallback_macro = diagnostics.get("guard_fallback_macro_f1", np.nan)
    recall_gains: list[float] = []
    for key, candidate_value in diagnostics.items():
        prefix = "guard_candidate_recall_class_"
        if not key.startswith(prefix):
            continue
        label = key.removeprefix(prefix)
        fallback_value = diagnostics.get(
            f"guard_fallback_recall_class_{label}", np.nan
        )
        if np.isfinite(candidate_value) and np.isfinite(fallback_value):
            recall_gains.append(float(candidate_value - fallback_value))
    macro_gain = float(candidate_macro - fallback_macro)
    minimum_recall_gain = min(recall_gains) if recall_gains else -np.inf
    if (
        not np.isfinite(macro_gain)
        or macro_gain < -float(macro_f1_tolerance)
        or minimum_recall_gain < -float(per_class_recall_tolerance)
    ):
        return None
    return (
        macro_gain,
        minimum_recall_gain,
        float(diagnostics["guard_combined_lcb"]),
    )


def _v16_class_balanced_score(
    diagnostics: dict[str, float],
    *,
    macro_f1_tolerance: float,
    attack_recall_tolerance: float,
    min_per_class_recall_tolerance: float,
) -> tuple[float, float, float, float] | None:
    """Pareto-style guard for binary IDS adaptation.

    A candidate must preserve Macro-F1, attack recall, and the recall of the
    currently weakest class within their preregistered non-inferiority
    margins.  Unlike the v13 guard, it does not reject a candidate merely
    because the already-strong majority class gives up recall while both
    Macro-F1 and the weakest-class recall improve.
    """

    candidate_macro = diagnostics.get("guard_candidate_macro_f1", np.nan)
    fallback_macro = diagnostics.get("guard_fallback_macro_f1", np.nan)
    candidate_attack = diagnostics.get(
        "guard_candidate_attack_recall", np.nan
    )
    fallback_attack = diagnostics.get(
        "guard_fallback_attack_recall", np.nan
    )
    candidate_min = diagnostics.get(
        "guard_candidate_min_per_class_recall", np.nan
    )
    fallback_min = diagnostics.get(
        "guard_fallback_min_per_class_recall", np.nan
    )
    values = (
        candidate_macro,
        fallback_macro,
        candidate_attack,
        fallback_attack,
        candidate_min,
        fallback_min,
    )
    if not all(np.isfinite(value) for value in values):
        return None
    macro_gain = float(candidate_macro - fallback_macro)
    attack_gain = float(candidate_attack - fallback_attack)
    minimum_recall_gain = float(candidate_min - fallback_min)
    if (
        macro_gain < -float(macro_f1_tolerance)
        or attack_gain < -float(attack_recall_tolerance)
        or minimum_recall_gain
        < -float(min_per_class_recall_tolerance)
    ):
        return None
    return (
        macro_gain,
        minimum_recall_gain,
        attack_gain,
        float(diagnostics["guard_combined_lcb"]),
    )


def _class_guard_diagnostics(
    fallback: OnlineMLP,
    candidate: OnlineMLP,
    recent: tuple[torch.Tensor, torch.Tensor],
    memory: tuple[torch.Tensor, torch.Tensor] | None,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    """Return loss and class-balanced diagnostics for a guarded commit."""

    _, diagnostics = _dual_guard(
        fallback,
        candidate,
        recent,
        memory,
        cfg,
        device,
        class_balanced=True,
    )
    fallback_metrics = _guard_classification(fallback, recent, device)
    candidate_metrics = _guard_classification(candidate, recent, device)
    diagnostics.update(
        {
            "guard_candidate_macro_f1": candidate_metrics["macro_f1"],
            "guard_fallback_macro_f1": fallback_metrics["macro_f1"],
            "guard_candidate_attack_recall": candidate_metrics[
                "attack_recall"
            ],
            "guard_fallback_attack_recall": fallback_metrics["attack_recall"],
            "guard_candidate_min_per_class_recall": candidate_metrics[
                "min_per_class_recall"
            ],
            "guard_fallback_min_per_class_recall": fallback_metrics[
                "min_per_class_recall"
            ],
        }
    )
    for key, value in candidate_metrics.items():
        if key.startswith("recall_class_"):
            diagnostics[f"guard_candidate_{key}"] = value
            diagnostics[f"guard_fallback_{key}"] = fallback_metrics[key]
    return diagnostics


def _interpolate_adaptation_state(
    fallback: AdapterOnlineMLP,
    candidate: AdapterOnlineMLP,
    alpha: float,
    extra_parameter_names: set[str] | None = None,
) -> AdapterOnlineMLP:
    """Interpolate only adapters and classifier, retaining a frozen backbone."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = copy.deepcopy(candidate)
    trainable_ids = {
        id(parameter)
        for values in fallback.adaptation_parameter_groups().values()
        for parameter in values
    }
    trainable_names = {
        name
        for name, parameter in fallback.named_parameters()
        if id(parameter) in trainable_ids
    }
    trainable_names.update(extra_parameter_names or set())
    fallback_state = fallback.state_dict()
    candidate_state = candidate.state_dict()
    blended_state = {
        name: (
            fallback_state[name] * (1.0 - alpha) + value * alpha
            if name in trainable_names and value.is_floating_point()
            else value.clone()
        )
        for name, value in candidate_state.items()
    }
    blended.load_state_dict(blended_state)
    return blended


def _v17_soft_commit(
    fallback: AdapterOnlineMLP,
    candidate: AdapterOnlineMLP,
    recent: tuple[torch.Tensor, torch.Tensor],
    memory: tuple[torch.Tensor, torch.Tensor] | None,
    cfg: ExperimentConfig,
    device: torch.device,
    extra_parameter_names: set[str] | None = None,
    alphas: tuple[float, ...] | list[float] | None = None,
) -> tuple[AdapterOnlineMLP, float, dict[str, float]]:
    """Choose the largest class-safe line-search step without an LCB veto."""

    last_diagnostics: dict[str, float] = {}
    ordered_alphas = sorted(
        {
            float(np.clip(value, 0.0, 1.0))
            for value in (
                cfg.dasp.v17_soft_commit_alphas
                if alphas is None
                else alphas
            )
            if float(value) > 0.0
        },
        reverse=True,
    )
    for alpha in ordered_alphas:
        blended = _interpolate_adaptation_state(
            fallback,
            candidate,
            alpha,
            extra_parameter_names=extra_parameter_names,
        )
        diagnostics = _class_guard_diagnostics(
            fallback, blended, recent, memory, cfg, device
        )
        diagnostics["v17_soft_commit_alpha"] = alpha
        last_diagnostics = diagnostics
        class_safe = _v16_class_balanced_score(
            diagnostics,
            macro_f1_tolerance=cfg.dasp.v17_macro_f1_tolerance,
            attack_recall_tolerance=cfg.dasp.v17_attack_recall_tolerance,
            min_per_class_recall_tolerance=(
                cfg.dasp.v17_min_per_class_recall_tolerance
            ),
        )
        recent_safe = bool(
            np.isfinite(diagnostics["guard_recent_mean_delta"])
            and diagnostics["guard_recent_mean_delta"]
            >= -cfg.dasp.v17_recent_mean_tolerance
        )
        memory_delta = diagnostics["guard_memory_mean_delta"]
        memory_safe = bool(
            not np.isfinite(memory_delta)
            or memory_delta >= -cfg.dasp.v17_memory_mean_tolerance
        )
        if class_safe is not None and recent_safe and memory_safe:
            diagnostics["v17_soft_commit_accepted"] = True
            return blended, alpha, diagnostics
    if not last_diagnostics:
        last_diagnostics = _class_guard_diagnostics(
            fallback, fallback, recent, memory, cfg, device
        )
    last_diagnostics["v17_soft_commit_alpha"] = 0.0
    last_diagnostics["v17_soft_commit_accepted"] = False
    return copy.deepcopy(fallback), 0.0, last_diagnostics


def _safe_soft_commit(
    fallback: AdapterOnlineMLP,
    candidate: AdapterOnlineMLP,
    recent: tuple[torch.Tensor, torch.Tensor],
    memory: tuple[torch.Tensor, torch.Tensor] | None,
    cfg: ExperimentConfig,
    device: torch.device,
    extra_parameter_names: set[str] | None = None,
    alphas: tuple[float, ...] | list[float] | None = None,
) -> tuple[AdapterOnlineMLP, float, dict[str, object]]:
    """Atomic class-supported commit with an exact rollback invariant."""

    n_classes = int(fallback.classifier.out_features)
    minimum = max(1, int(cfg.dasp.safe_guard_min_per_class))
    recent_supported = _has_class_support(
        recent[1], n_classes, minimum
    )
    memory_supported = bool(
        memory is not None
        and _has_class_support(memory[1], n_classes, minimum)
    )
    if not recent_supported or not memory_supported:
        reason = (
            "recent_guard_class_support"
            if not recent_supported
            else "memory_guard_class_support"
        )
        return copy.deepcopy(fallback), 0.0, {
            "v17_soft_commit_alpha": 0.0,
            "v17_soft_commit_accepted": False,
            "safe_gate_passed": False,
            "safe_gate_reason": reason,
            "safe_rollback_exact": True,
        }

    chosen, alpha, diagnostics = _v17_soft_commit(
        fallback,
        candidate,
        recent,
        memory,
        cfg,
        device,
        extra_parameter_names=extra_parameter_names,
        alphas=alphas,
    )
    diagnostics = dict(diagnostics)
    diagnostics["safe_gate_passed"] = bool(alpha > 0.0)
    diagnostics["safe_gate_reason"] = (
        "accepted" if alpha > 0.0 else "noninferiority_rejected"
    )
    if alpha <= 0.0:
        fallback_state = fallback.state_dict()
        chosen_state = chosen.state_dict()
        rollback_exact = all(
            torch.equal(
                fallback_state[name].detach().cpu(),
                chosen_state[name].detach().cpu(),
            )
            for name in fallback_state
        )
        diagnostics["safe_rollback_exact"] = bool(rollback_exact)
        if not rollback_exact:
            raise RuntimeError("DASP safety rollback changed model state")
    else:
        diagnostics["safe_rollback_exact"] = np.nan
    return chosen, alpha, diagnostics


def _snapshot_adapter_model(model: AdapterOnlineMLP) -> ModelSnapshot:
    return ModelSnapshot(
        state={
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        adapter_rank=int(model.adapter_rank),
        adapter_trainable_rank=int(model.adapter_trainable_rank),
        input_adapter_enabled=bool(model.input_adapter_enabled),
        feature_modulation_enabled=bool(model.feature_modulation_enabled),
    )


def _materialize_snapshot(
    template: AdapterOnlineMLP,
    snapshot: ModelSnapshot,
    device: torch.device,
) -> AdapterOnlineMLP:
    candidate = copy.deepcopy(template).to(device)
    candidate.load_state_dict(snapshot.state)
    candidate.set_adapter_rank(
        snapshot.adapter_rank,
        trainable_rank=snapshot.adapter_trainable_rank,
    )
    candidate.enable_input_adapter(snapshot.input_adapter_enabled)
    candidate.enable_feature_modulation(snapshot.feature_modulation_enabled)
    candidate.freeze_backbone()
    return candidate.eval()


def _v3_checkpoint_score(
    labels: torch.Tensor,
    predictions: np.ndarray,
    n_classes: int,
) -> float | None:
    if not _has_class_support(
        labels, n_classes, 1
    ):
        return None
    metrics = classification_metrics(
        _tensor_to_numpy(labels, np.int64),
        np.asarray(predictions, dtype=np.int64),
        n_classes,
    )
    recalls: list[float] = []
    truth = _tensor_to_numpy(labels, np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    for label in range(n_classes):
        mask = truth == label
        recalls.append(
            float(np.mean(predicted[mask] == label))
            if np.any(mask)
            else float("nan")
        )
    metrics["min_per_class_recall"] = float(np.nanmin(recalls))
    values = (
        metrics["macro_f1"],
        metrics["attack_recall"],
        metrics["min_per_class_recall"],
    )
    if not all(np.isfinite(value) for value in values):
        return None
    return float(
        0.5 * metrics["macro_f1"]
        + 0.5
        * min(
            metrics["attack_recall"],
            metrics["min_per_class_recall"],
        )
    )


def _v3_prior_calibration(
    model: AdapterOnlineMLP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    """Apply a bounded, revealed-label-only log-prior correction."""

    n_classes = int(model.classifier.out_features)
    if (
        model.classifier.bias is None
        or not _has_class_support(train_y, n_classes, 1)
    ):
        return {
            "v3_calibration_applied": False,
            "v3_calibration_max_delta": 0.0,
        }
    model.eval()
    with torch.no_grad():
        probabilities = F.softmax(model(train_x.to(device)), dim=1)
        predicted_prior = probabilities.mean(dim=0).clamp_min(1e-4)
        target_counts = torch.bincount(
            train_y.detach().cpu().long(), minlength=n_classes
        ).to(device=device, dtype=torch.float32)
        target_prior = (target_counts / target_counts.sum()).clamp_min(1e-4)
        delta = torch.log(target_prior) - torch.log(predicted_prior)
        delta = (delta - delta.mean()).clamp(-1.0, 1.0)
        model.classifier.bias.add_(delta)
    return {
        "v3_calibration_applied": True,
        "v3_calibration_max_delta": float(delta.abs().max().item()),
    }


def _restore_v3_checkpoint(
    model: AdapterOnlineMLP,
    checkpoints: list[
        tuple[float, int, ModelSnapshot | AdapterOnlineMLP]
    ],
    *,
    recent: tuple[torch.Tensor, torch.Tensor] | None,
    memory: tuple[torch.Tensor, torch.Tensor] | None,
    cfg: ExperimentConfig,
    device: torch.device,
    require_current_guard: bool,
) -> tuple[bool, dict[str, object]]:
    """Restore a causally eligible recent checkpoint, never a future one."""

    if not checkpoints:
        return False, {
            "v3_checkpoint_restored": False,
            "v3_checkpoint_reason": "no_eligible_checkpoint",
        }
    n_classes = int(model.classifier.out_features)
    if require_current_guard and (
        recent is None
        or not _has_class_support(
            recent[1],
            n_classes,
            max(1, cfg.dasp.safe_guard_min_per_class),
        )
    ):
        return False, {
            "v3_checkpoint_restored": False,
            "v3_checkpoint_reason": "guard_class_support",
        }
    if require_current_guard and recent is not None:
        fallback_metrics = _guard_classification(
            model, recent, device
        )
        if (
            float(fallback_metrics["min_per_class_recall"])
            > float(cfg.dasp.v3_rollback_class_recall_floor)
        ):
            return False, {
                "v3_checkpoint_restored": False,
                "v3_checkpoint_reason": "current_model_not_collapsed",
            }
    for historical_score, checkpoint_window, stored in sorted(
        checkpoints, key=lambda item: (item[0], item[1]), reverse=True
    ):
        candidate = (
            _materialize_snapshot(model, stored, device)
            if isinstance(stored, ModelSnapshot)
            else copy.deepcopy(stored).to(device).eval()
        )
        if require_current_guard and recent is not None:
            diagnostics = _class_guard_diagnostics(
                model, candidate, recent, memory, cfg, device
            )
            class_safe = _v16_class_balanced_score(
                diagnostics,
                macro_f1_tolerance=cfg.dasp.v17_macro_f1_tolerance,
                attack_recall_tolerance=(
                    cfg.dasp.v17_attack_recall_tolerance
                ),
                min_per_class_recall_tolerance=(
                    cfg.dasp.v17_min_per_class_recall_tolerance
                ),
            )
            macro_gain = float(
                diagnostics["guard_candidate_macro_f1"]
                - diagnostics["guard_fallback_macro_f1"]
            )
            if (
                class_safe is None
                or macro_gain < float(cfg.dasp.v3_checkpoint_min_gain)
            ):
                continue
        model.load_state_dict(candidate.state_dict())
        model.set_adapter_rank(
            candidate.adapter_rank,
            trainable_rank=candidate.adapter_trainable_rank,
        )
        model.freeze_backbone()
        return True, {
            "v3_checkpoint_restored": True,
            "v3_checkpoint_window": int(checkpoint_window),
            "v3_checkpoint_score": float(historical_score),
            "v3_checkpoint_reason": (
                "guarded_restore"
                if require_current_guard
                else "insufficient_class_fallback"
            ),
        }
    return False, {
        "v3_checkpoint_restored": False,
        "v3_checkpoint_reason": "no_safe_checkpoint",
    }


def _v16_oracle_event(scenario: str, labeled_window: int) -> bool:
    """Return the preregistered labeled-window oracle schedule.

    The oracle fires when labels for the first affected window are available,
    not when the corresponding unlabeled window first arrives.  Gradual drift
    receives onset/midpoint/end checkpoints because no single abrupt boundary
    represents the transition.
    """

    schedule = {
        "S0": set(),
        "S1": {12},
        "S2": {12, 18, 23},
        "S3": {12, 24},
        "S4": {12},
        "S5": {12},
    }
    return int(labeled_window) in schedule.get(str(scenario), set())


def _soft_reset_toward_reference(
    model: OnlineMLP,
    reference: OnlineMLP,
    strength: float,
) -> OnlineMLP:
    """Return a parameter anchor without storing or replaying historical samples."""
    candidate = copy.deepcopy(model)
    alpha = float(np.clip(strength, 0.0, 1.0))
    current_state = candidate.state_dict()
    reference_state = reference.state_dict()
    blended: dict[str, torch.Tensor] = {}
    for name, current in current_state.items():
        anchor = reference_state[name].to(current.device)
        if current.is_floating_point():
            blended[name] = current * (1.0 - alpha) + anchor * alpha
        else:
            blended[name] = current.clone()
    candidate.load_state_dict(blended)
    return candidate


class StreamingExperiment:
    def __init__(self, config: ExperimentConfig):
        if config.dasp.baseline not in METHODS:
            raise ValueError(f"Unknown method {config.dasp.baseline!r}; choose from {sorted(METHODS)}")
        if config.dasp.baseline in REVISION_METHODS:
            required_architecture = REVISION_CONDITIONS[
                config.dasp.baseline
            ].required_architecture
            if config.model.architecture != required_architecture:
                raise ValueError(
                    f"{config.dasp.baseline} requires "
                    f"model.architecture={required_architecture!r}"
                )
        self.config = config
        self.device = torch.device(config.training.device)
        self._stateful_optimizer: torch.optim.AdamW | None = None
        self._ewc_reference: dict[str, torch.Tensor] | None = None
        self._ewc_fisher: dict[str, torch.Tensor] | None = None
        self._anchor_model: OnlineMLP | None = None
        self._ssf_teacher: torch.nn.Module | None = None
        self._cdda_detector: ErrorRateDriftDetector | None = None
        self._cdda_warning_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._v16_performance_history: list[tuple[float, float]] = []
        self._v16_performance_cooldown = 0
        self._v17_strong_remaining = 0
        self._v17_recovery_base_layers: list[str] = []
        self._v18_guard_history: tuple[
            torch.Tensor, torch.Tensor
        ] | None = None
        self._safe_label_buffer: tuple[
            torch.Tensor, torch.Tensor
        ] | None = None
        self._v3_checkpoints: list[
            tuple[float, int, ModelSnapshot]
        ] = []
        self._v4_label_history: torch.Tensor | None = None
        self._fast_inference_model: FusedAdapterInferenceMLP | None = None
        self._last_preprocessing_seconds = 0.0
        self._last_forward_seconds = 0.0
        self._last_metric_seconds = 0.0
        self._last_predictions = np.empty(0, dtype=np.int64)
        self._revision_previous_route: tuple[str, ...] = ()
        self._revision_event_pending: list[dict[str, object]] = []
        self._revision_feature_permutation_digest: str | None = None
        self._process = psutil.Process()
        self._run_start_rss_bytes = int(self._process.memory_info().rss)

    def _load_stream(self) -> PreparedStream:
        cfg = self.config.data
        if cfg.source == "synthetic":
            return make_synthetic_stream(
                cfg.scenario,
                cfg.n_windows,
                cfg.window_size,
                cfg.n_features,
                cfg.random_seed,
                cfg.task,
            )
        if cfg.path is None:
            raise ValueError(f"data.path is required for source={cfg.source!r}")
        if cfg.source == "prepared_npz":
            return load_prepared_stream(cfg.path)
        if cfg.source in {"nf_v3", "nf-unsw", "nf-cicids"}:
            return load_nf_v3_stream(
                cfg.path,
                cfg.window_size,
                cfg.n_windows,
                cfg.warmup_windows,
                cfg.label_column,
                cfg.subtype_column,
                cfg.task,
                cfg.start_row,
                cfg.chunksize,
                cfg.sort_by_time,
                cfg.time_column,
                cfg.include_partial_window,
            )
        if cfg.source == "luflow":
            return load_luflow_stream(
                cfg.path,
                cfg.window_size,
                cfg.n_windows,
                cfg.warmup_windows,
                cfg.start_row,
                cfg.chunksize,
            )
        raise ValueError(f"Unknown data source: {cfg.source}")

    def _apply_revision_feature_permutation(
        self,
        stream: PreparedStream,
    ) -> PreparedStream:
        permutation_seed = self.config.revision.feature_permutation_seed
        if permutation_seed is None:
            order = torch.arange(len(stream.feature_names), dtype=torch.long)
        else:
            generator = torch.Generator().manual_seed(
                stable_seed(
                    self.config.revision.feature_permutation_namespace,
                    self._revision_dataset_key(),
                    int(permutation_seed),
                )
            )
            order = torch.randperm(len(stream.feature_names), generator=generator)
        self._revision_feature_permutation_digest = index_digest(order)
        if permutation_seed is None:
            return stream
        indices = order.tolist()
        windows = [
            StreamWindow(
                x=window.x[:, order],
                y=window.y,
                index=window.index,
                phase=window.phase,
                subtypes=window.subtypes,
                source_ids=window.source_ids,
                start_time=window.start_time,
                end_time=window.end_time,
            )
            for window in stream.windows
        ]
        return PreparedStream(
            windows,
            [stream.feature_names[index] for index in indices],
            list(stream.class_names),
        )

    def _drift_representations(
        self,
        model: OnlineMLP | ModularOnlineMLP,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representations = _representations(model, x, self.device)
        condition = REVISION_CONDITIONS.get(self.config.dasp.baseline)
        if condition is not None and condition.final_representation_only:
            final_name = model.layer_names[-1]
            return {final_name: representations[final_name]}
        return representations

    def _build_model(
        self, input_dim: int, n_classes: int
    ) -> OnlineMLP | AdapterOnlineMLP | ModularOnlineMLP | CDDAAttentionNet:
        cfg = self.config.model
        if self.config.dasp.baseline in ADAPTER_METHODS:
            if cfg.architecture != "sequential":
                raise ValueError(
                    "adapter DASP requires model.architecture='sequential'"
                )
            model = AdapterOnlineMLP(
                input_dim,
                n_classes,
                cfg.hidden_dims,
                cfg.dropout,
                self.config.dasp.v17_adapter_max_rank,
                self.config.dasp.v17_adapter_scale,
                self.config.dasp.v17_input_adapter_dim,
                self.config.dasp.v18_modulation_scale,
            )
            return model
        if cfg.architecture == "sequential":
            return OnlineMLP(input_dim, n_classes, cfg.hidden_dims, cfg.dropout)
        if cfg.architecture == "modular":
            return ModularOnlineMLP(
                input_dim,
                n_classes,
                cfg.hidden_dims,
                cfg.dropout,
                cfg.n_modules,
                cfg.module_dim,
            )
        raise ValueError("model.architecture must be 'sequential' or 'modular'")

    def _refresh_fast_inference_model(
        self, model: AdapterOnlineMLP
    ) -> None:
        self._fast_inference_model = model.to_fused_inference().to(
            self.device
        )

    def _prediction_model(
        self, model: OnlineMLP
    ) -> OnlineMLP | FusedAdapterInferenceMLP:
        if self.config.dasp.baseline in FAST_DASP_METHODS:
            if self._fast_inference_model is None:
                assert isinstance(model, AdapterOnlineMLP)
                self._refresh_fast_inference_model(model)
            assert self._fast_inference_model is not None
            return self._fast_inference_model
        return model

    def run(self) -> ExperimentResult:
        cfg = self.config
        if cfg.dasp.baseline == "modular_dasp" and cfg.original_review.enabled:
            run_name = cfg.output.run_name or f"{cfg.data.source}_{cfg.data.scenario}_{cfg.dasp.baseline}_seed{cfg.data.random_seed}"
            if (Path(cfg.output.directory) / run_name / "summary.json").exists():
                raise FileExistsError("Use the review runner --resume or choose a new output run_name; archived runs are never overwritten")
        model_seed = (
            cfg.data.random_seed
            if cfg.training.model_seed is None
            else int(cfg.training.model_seed)
        )
        # Configure deterministic execution before the first CUDA runtime call.
        set_deterministic_seed(model_seed, cfg.training.num_threads)
        if cfg.dasp.baseline == "modular_dasp" and cfg.original_review.enabled:
            self._original_provenance = original_review.identity(cfg)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        self._run_start_rss_bytes = int(self._process.memory_info().rss)
        stream = self._apply_revision_feature_permutation(self._load_stream())
        if cfg.dasp.baseline in RIVER_METHODS:
            return self._run_river_stream(stream)
        model = self._build_model(len(stream.feature_names), len(stream.class_names)).to(
            self.device
        )
        if cfg.dasp.baseline in V3_METHODS:
            self._stateful_optimizer = _make_stateful_optimizer(model, cfg)
        memory = ReservoirMemory(cfg.dasp.memory_size, cfg.data.random_seed)
        rows: list[dict[str, object]] = []
        row_lookup: dict[int, dict[str, object]] = {}

        # Declared warm-up is training/calibration only and excluded from headline metrics.
        for window in stream.windows[: cfg.data.warmup_windows]:
            metrics = self._evaluate(model, window, len(stream.class_names))
            row = self._base_row(window, metrics, warmup=True)
            rows.append(row)
            row_lookup[window.index] = row
            train_x, train_y, _, _ = _split_train_guard(
                window, cfg.dasp.guard_ratio, cfg.data.random_seed
            )
            if cfg.dasp.baseline == "cdda_md":
                train_cdda(
                    model, train_x, train_y, cfg, self.device, window.index
                )
            elif cfg.dasp.baseline in V3_METHODS:
                assert self._stateful_optimizer is not None
                _train_stateful_online(
                    model,
                    self._stateful_optimizer,
                    train_x,
                    train_y,
                    cfg,
                    self.device,
                    window.index,
                )
            else:
                _train_online(model, train_x, train_y, cfg, self.device)
            memory.add(window.x, window.y)

        if cfg.dasp.baseline in SAFE_V2_METHODS:
            warmup_x = torch.cat(
                [window.x for window in stream.windows[: cfg.data.warmup_windows]]
            )
            warmup_y = torch.cat(
                [window.y for window in stream.windows[: cfg.data.warmup_windows]]
            )
            if cfg.dasp.baseline in V4_METHODS:
                self._v4_label_history = _bounded_label_history_update(
                    None,
                    warmup_y,
                    cfg.dasp.v4_prior_history_size,
                )
            if cfg.dasp.baseline in V3_MEMORY_METHODS:
                balanced_memory = _method_true_label_memory_update(
                    cfg.dasp.baseline,
                    model,
                    None,
                    warmup_x,
                    warmup_y,
                    cfg=cfg,
                    device=self.device,
                    seed=cfg.data.random_seed + 7001,
                )
            else:
                balanced_memory = _balanced_tensor_sample(
                    warmup_x,
                    warmup_y,
                    min(cfg.dasp.memory_size, len(warmup_y)),
                    torch.device("cpu"),
                    cfg.data.random_seed + 7001,
                )
                assert balanced_memory is not None
            memory.replace(*balanced_memory)
            seeded_guard = _balanced_tensor_sample(
                warmup_x,
                warmup_y,
                min(cfg.dasp.fast_guard_history_size, len(warmup_y)),
                torch.device("cpu"),
                cfg.data.random_seed + 7003,
            )
            self._v18_guard_history = seeded_guard

        if cfg.dasp.baseline in ADAPTER_METHODS:
            assert isinstance(model, AdapterOnlineMLP)
            model.set_adapter_rank(
                cfg.dasp.v17_fast_rank,
                trainable_rank=cfg.dasp.v17_fast_rank,
            )
            model.enable_input_adapter()
            if cfg.dasp.baseline in V18_METHODS:
                model.enable_feature_modulation()
            model.freeze_backbone()
            if cfg.dasp.baseline in FAST_DASP_METHODS:
                self._refresh_fast_inference_model(model)

        if cfg.dasp.baseline in {
            "ssf",
            "ssf_budget",
            "ssf_faithful",
            "revision_a10_ssf",
        }:
            self._ssf_teacher = clone_teacher(model)
        if cfg.dasp.baseline == "cdda_md":
            warmup_errors: list[float] = []
            for calibration_window in stream.windows[: cfg.data.warmup_windows]:
                _, _, calibration_x, calibration_y = _split_train_guard(
                    calibration_window,
                    cfg.dasp.guard_ratio,
                    cfg.data.random_seed,
                )
                model.eval()
                with torch.no_grad():
                    prediction = model(calibration_x.to(self.device)).argmax(dim=1).cpu()
                warmup_errors.append(
                    float((prediction != calibration_y).float().mean().item())
                )
            self._cdda_detector = ErrorRateDriftDetector.from_warmup(
                warmup_errors,
                cfg.baselines.cdda_error_z,
                cfg.baselines.cdda_warning_patience,
                cfg.baselines.cdda_min_error_increase,
            )

        if cfg.dasp.baseline == "online_ewc":
            fisher_batch = memory.sample(cfg.dasp.memory_batch_size, self.device)
            self._ewc_reference = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
            self._ewc_fisher = _estimate_fisher(model, fisher_batch, self.device)

        reference_model = copy.deepcopy(model).eval()
        self._anchor_model = copy.deepcopy(model).eval()
        warmup = stream.windows[: cfg.data.warmup_windows]
        reference_repr = self._drift_representations(reference_model, warmup[-1].x)
        calibration_windows = warmup[-max(1, cfg.drift.calibration_windows) :]
        calibration_repr = [
            self._drift_representations(reference_model, window.x)
            for window in calibration_windows
        ]
        monitor = LayerDriftMonitor(
            list(reference_repr),
            cfg.drift.kappa,
            cfg.drift.z_max,
            cfg.drift.severity_threshold,
            cfg.drift.persistence,
            cfg.drift.top_k,
            cfg.drift.mmd_samples,
            cfg.drift.kernel_scales,
            (
                cfg.drift.emergency_severity_threshold
                if cfg.dasp.baseline
                in V6_METHODS | V9_METHODS | V91_METHODS | V92_METHODS
                else None
            ),
        )
        monitor.calibrate(reference_repr, calibration_repr)
        anchor_reference_repr: dict[str, torch.Tensor] | None = None
        anchor_monitor: LayerDriftMonitor | None = None
        if cfg.dasp.baseline in V9_METHODS | V91_METHODS | V92_METHODS | ADAPTER_METHODS:
            assert self._anchor_model is not None
            anchor_reference_repr = _representations(
                self._anchor_model, warmup[-1].x, self.device
            )
            anchor_calibration_repr = [
                _representations(self._anchor_model, window.x, self.device)
                for window in calibration_windows
            ]
            anchor_monitor = LayerDriftMonitor(
                model.layer_names,
                cfg.drift.kappa,
                cfg.drift.z_max,
                cfg.drift.severity_threshold,
                cfg.drift.persistence,
                cfg.drift.top_k,
                cfg.drift.mmd_samples,
                cfg.drift.kernel_scales,
            )
            anchor_monitor.calibrate(
                anchor_reference_repr, anchor_calibration_repr
            )

        fast_input_reference: dict[str, torch.Tensor] | None = None
        fast_input_monitor: LayerDriftMonitor | None = None
        if cfg.dasp.baseline in FAST_DASP_METHODS:
            fast_input_reference = {
                "input": warmup[-1].x.to(self.device)
            }
            fast_input_monitor = LayerDriftMonitor(
                ["input"],
                cfg.drift.kappa,
                cfg.drift.z_max,
                cfg.dasp.fast_warning_severity,
                1,
                1,
                max(8, cfg.dasp.fast_input_gate_samples),
                cfg.drift.kernel_scales,
            )
            fast_input_monitor.calibrate(
                fast_input_reference,
                [
                    {"input": item.x.to(self.device)}
                    for item in calibration_windows
                ],
            )

        pending: list[PendingUpdate] = []
        cooldown_remaining = 0
        departed_from_anchor = False
        anchor_departure_index: int | None = None
        for window in stream.windows[cfg.data.warmup_windows :]:
            # A common measurement boundary; baseline learning code is unchanged.
            original_review.synchronize(self.device)
            original_cycle_started = time.perf_counter()
            drift_started = time.perf_counter()
            fast_gate_severity = np.nan
            fast_gate_warning = True
            layer_detector_evaluated = False
            if cfg.dasp.baseline in LAYER_DETECTOR_FREE_METHODS:
                decision = DriftDecision({}, {}, {}, 0.0, False, False)
            elif cfg.dasp.baseline in FAST_DASP_METHODS:
                assert (
                    fast_input_reference is not None
                    and fast_input_monitor is not None
                )
                fast_decision = fast_input_monitor.evaluate(
                    fast_input_reference,
                    {"input": window.x.to(self.device)},
                )
                fast_gate_severity = fast_decision.severity
                fast_gate_warning = bool(
                    fast_decision.triggered
                    or fast_decision.severity
                    >= cfg.dasp.fast_warning_severity
                )
                if fast_gate_warning:
                    current_repr = self._drift_representations(
                        reference_model, window.x
                    )
                    decision = monitor.evaluate(
                        reference_repr, current_repr
                    )
                    layer_detector_evaluated = True
                else:
                    monitor.reset_episode()
                    decision = DriftDecision(
                        fast_decision.distances,
                        fast_decision.z_scores,
                        {},
                        fast_decision.severity,
                        False,
                        False,
                    )
            else:
                current_repr = self._drift_representations(
                    reference_model, window.x
                )
                decision = monitor.evaluate(reference_repr, current_repr)
            if cooldown_remaining > 0:
                decision = DriftDecision(
                    decision.distances,
                    decision.z_scores,
                    decision.requirements,
                    decision.severity,
                    False,
                    False,
                )
                monitor.reset_episode()
                cooldown_remaining -= 1
            if (
                cfg.dasp.baseline not in V9_METHODS | V91_METHODS | V92_METHODS
                and decision.severity >= cfg.dasp.recurrence_departure_threshold
            ):
                departed_from_anchor = True
            anchor_severity = decision.severity
            evaluate_anchor = bool(
                cfg.dasp.baseline
                in V9_METHODS | V91_METHODS | V92_METHODS | ADAPTER_METHODS
                and (
                    cfg.dasp.baseline not in FAST_DASP_METHODS
                    or fast_gate_warning
                )
            )
            if evaluate_anchor:
                assert (
                    self._anchor_model is not None
                    and anchor_reference_repr is not None
                    and anchor_monitor is not None
                )
                current_anchor_repr = _representations(
                    self._anchor_model, window.x, self.device
                )
                anchor_decision = anchor_monitor.evaluate(
                    anchor_reference_repr, current_anchor_repr
                )
                anchor_severity = anchor_decision.severity
                if cfg.dasp.baseline in V91_METHODS | V92_METHODS | ADAPTER_METHODS:
                    if (
                        anchor_decision.triggered
                        and anchor_severity
                        >= cfg.dasp.recurrence_departure_threshold
                    ):
                        if not departed_from_anchor:
                            anchor_departure_index = window.index
                        departed_from_anchor = True
                    recurrence = bool(
                        departed_from_anchor
                        and anchor_departure_index is not None
                        and window.index - anchor_departure_index
                        >= cfg.dasp.v9_recurrence_min_windows
                        and anchor_severity
                        <= cfg.dasp.recurrence_return_threshold
                    )
                else:
                    if anchor_severity >= cfg.dasp.recurrence_departure_threshold:
                        departed_from_anchor = True
                    recurrence = bool(
                        departed_from_anchor
                        and anchor_severity
                        <= cfg.dasp.recurrence_return_threshold
                    )
                if recurrence and cfg.dasp.v9_one_shot_recurrence:
                    departed_from_anchor = False
                    anchor_departure_index = None
            else:
                if (
                    cfg.dasp.baseline in FAST_DASP_METHODS
                    and anchor_monitor is not None
                ):
                    anchor_monitor.reset_episode()
                recurrence = bool(
                    cfg.dasp.baseline in V6_METHODS
                    and departed_from_anchor
                    and decision.severity <= cfg.dasp.recurrence_return_threshold
                )
            inference_anchor_weight = (
                cfg.dasp.recurrence_anchor_strength
                if (
                    cfg.dasp.baseline in V6_METHODS
                    and recurrence
                    and cfg.dasp.recurrence_inference_anchor
                )
                else 0.0
            )
            drift_scoring_seconds = time.perf_counter() - drift_started
            if inference_anchor_weight > 0.0:
                assert self._anchor_model is not None
                metrics = self._evaluate_blended(
                    model,
                    self._anchor_model,
                    window,
                    len(stream.class_names),
                    inference_anchor_weight,
                )
            else:
                metrics = self._evaluate(
                    self._prediction_model(model),
                    window,
                    len(stream.class_names),
                )
            row = self._base_row(window, metrics, warmup=False)
            row["drift_scoring_seconds"] = drift_scoring_seconds
            row["prediction_decision_seconds"] = (
                float(row["preprocessing_seconds"])
                + float(row["forward_seconds"])
                + drift_scoring_seconds
            )
            self._record_decision(row, decision)
            if cfg.dasp.baseline in FAST_DASP_METHODS:
                row["fast_gate_severity"] = fast_gate_severity
                row["fast_gate_warning"] = fast_gate_warning
                row["fast_layer_detector_evaluated"] = (
                    layer_detector_evaluated
                )
            row["anchor_severity"] = anchor_severity
            row["recurrence_detected"] = recurrence
            row["inference_anchor_weight"] = inference_anchor_weight
            rows.append(row)
            row_lookup[window.index] = row
            pending.append(
                PendingUpdate(
                    window,
                    decision,
                    recurrence,
                    warning=fast_gate_warning,
                    prediction_snapshot=(
                        _snapshot_adapter_model(model)
                        if cfg.dasp.baseline
                        in V3_CHECKPOINT_METHODS
                        else None
                    ),
                    prediction_labels=(
                        self._last_predictions.copy()
                        if cfg.dasp.baseline
                        in V3_CHECKPOINT_METHODS
                        else None
                    ),
                )
            )
            if len(pending) > cfg.training.label_delay:
                item = pending.pop(0)
                # At update time, labels belong only to ``item.window``.  The
                # current window is passed as features only for label-delay
                # aware replay and never exposes its labels.
                item.current_unlabeled_x = window.x
                item.current_decision = decision
                item.current_warning = fast_gate_warning
                accepted = self._update(model, memory, item, row_lookup[item.window.index])
                if (
                    accepted == "dasp"
                    and cfg.dasp.refresh_reference_after_commit
                    and cfg.dasp.baseline not in V6_METHODS
                ):
                    reference_model = copy.deepcopy(model).eval()
                    reference_repr = self._drift_representations(
                        reference_model, item.window.x
                    )
                    start = max(0, item.window.index - cfg.drift.calibration_windows + 1)
                    recent = stream.windows[start : item.window.index + 1]
                    monitor.calibrate(
                        reference_repr,
                        [
                            self._drift_representations(
                                reference_model, recent_window.x
                            )
                            for recent_window in recent
                        ],
                    )
                    monitor.reset_episode()
                    cooldown_remaining = max(0, cfg.drift.cooldown_windows)
                    row_lookup[item.window.index]["reference_reset"] = True
                if (
                    cfg.dasp.baseline in FAST_DASP_METHODS
                    and accepted in {"light", "dasp"}
                ):
                    assert fast_input_monitor is not None
                    fast_input_reference = {
                        "input": item.window.x.to(self.device)
                    }
                    start = max(
                        0,
                        item.window.index
                        - cfg.drift.calibration_windows
                        + 1,
                    )
                    recent = stream.windows[
                        start : item.window.index + 1
                    ]
                    fast_input_monitor.calibrate(
                        fast_input_reference,
                        [
                            {"input": value.x.to(self.device)}
                            for value in recent
                        ],
                    )
                    fast_input_monitor.reset_episode()
                if (
                    cfg.dasp.baseline in V91_METHODS | V92_METHODS
                    and (item.decision.triggered or item.recurrence)
                ):
                    monitor.reset_episode()
                    cooldown_remaining = max(
                        cooldown_remaining,
                        max(1, cfg.dasp.v9_attempt_cooldown),
                    )

            original_review.synchronize(self.device)
            row["full_cycle_seconds"] = time.perf_counter() - original_cycle_started
            if cfg.dasp.baseline == "modular_dasp" and cfg.original_review.enabled:
                row["original_full_cycle_seconds"] = time.perf_counter() - original_cycle_started

        # Labels eventually arrive even when no later test window remains.
        for item in pending:
            original_review.synchronize(self.device)
            terminal_started = time.perf_counter()
            if cfg.dasp.baseline == "modular_dasp" and cfg.original_review.enabled:
                row_lookup[item.window.index]["original_terminal_update"] = True
            self._update(model, memory, item, row_lookup[item.window.index])
            original_review.synchronize(self.device)
            row_lookup[item.window.index]["terminal_update_seconds"] = time.perf_counter() - terminal_started

        summary = summarize_windows(rows, cfg.data.warmup_windows)
        summary.update(_latency_summary(rows, cfg.data.warmup_windows))
        summary.update({
            "method": cfg.dasp.baseline,
            "source": cfg.data.source,
            "scenario": cfg.data.scenario,
            "seed": cfg.data.random_seed,
            "stream_seed": (
                cfg.data.random_seed
                if cfg.data.stream_seed is None
                else cfg.data.stream_seed
            ),
            "model_seed": model_seed,
            "n_features": len(stream.feature_names),
            "n_classes": len(stream.class_names),
            "feature_permutation_seed": cfg.revision.feature_permutation_seed,
            "feature_permutation_digest": self._revision_feature_permutation_digest,
            "total_update_seconds": float(sum(float(row.get("update_seconds", 0.0)) for row in rows)),
            "total_gradient_evaluations": int(sum(int(row.get("gradient_evaluations", 0)) for row in rows)),
            "total_parameter_gradient_element_evaluations": int(
                sum(
                    int(row.get("parameter_gradient_element_evaluations", 0))
                    for row in rows
                )
            ),
            "reference_resets": int(sum(bool(row.get("reference_reset", False)) for row in rows)),
            "total_sample_updates": int(sum(int(row.get("sample_updates", 0)) for row in rows)),
            "total_ground_truth_update_labels": int(
                sum(
                    int(row.get("ground_truth_update_labels", 0))
                    for row in rows
                )
            ),
            "total_labels_exposed": int(
                sum(int(row.get("labels_exposed_total", 0)) for row in rows)
            ),
            "total_new_training_labels_used": int(
                sum(int(row.get("new_training_labels_used", 0)) for row in rows)
            ),
            "revision_repair_count": int(
                sum(bool(row.get("revision_repair_applied", False)) for row in rows)
            ),
            "global_update_count": int(
                sum(bool(row.get("global_update", False)) for row in rows)
            ),
            "fast_revision": (
                cfg.dasp.fast_revision
                if cfg.dasp.baseline in FAST_DASP_METHODS
                else None
            ),
            "fast_stable_skips": int(
                sum(bool(row.get("fast_stable_skip", False)) for row in rows)
            ),
            "fast_gate_warnings": int(
                sum(bool(row.get("fast_gate_warning", False)) for row in rows)
            ),
            "fast_layer_detector_calls": int(
                sum(
                    bool(
                        row.get(
                            "fast_layer_detector_evaluated",
                            False,
                        )
                    )
                    for row in rows
                )
            ),
            "fast_adaptation_events": int(
                sum(
                    int(row.get("gradient_evaluations", 0)) > 0
                    for row in rows
                )
            ),
            "peak_rss_bytes": int(
                max(int(row.get("rss_bytes", 0)) for row in rows)
            ),
            "run_start_rss_bytes": self._run_start_rss_bytes,
            "peak_rss_increment_bytes": max(
                0,
                int(max(int(row.get("rss_bytes", 0)) for row in rows))
                - self._run_start_rss_bytes,
            ),
            "model_parameter_bytes": int(
                sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in model.parameters()
                )
            ),
            "model_parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "memory_payload_bytes": int(
                len(memory)
                * (
                    len(stream.feature_names)
                    * torch.tensor([], dtype=torch.float32).element_size()
                    + torch.tensor([], dtype=torch.int64).element_size()
                )
            ),
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            "peak_cuda_reserved_bytes": int(
                torch.cuda.max_memory_reserved(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "scipy": scipy.__version__,
                "pyyaml": yaml.__version__,
                "psutil": psutil.__version__,
                "device": str(self.device),
                "torch_threads": torch.get_num_threads(),
            },
        })
        if cfg.dasp.baseline == "modular_dasp" and cfg.original_review.enabled:
            self._original_checkpoint = {
                "format_version": 1, "completed_run": True,
                "automatic_midstream_resume_supported": False,
                "model": model, "model_state_dict": model.state_dict(),
                "memory": memory, "reference_model": reference_model, "monitor": monitor,
                "config": cfg.to_dict(), "provenance": self._original_provenance,
                "optimizer_policy": "AdamW recreated per repair; no persistent optimizer state",
                "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            } if cfg.output.save_checkpoints else None
        output_dir = self._write_outputs(rows, summary)
        return ExperimentResult(output_dir, summary, pd.DataFrame(rows).sort_values("window"))

    def _build_river_model(self):
        if any(module is None for module in (forest, linear_model, optim, tree)):
            raise RuntimeError(
                "River baselines require the optional dependency set: "
                "pip install -e .[river]"
            )
        method = self.config.dasp.baseline
        model_seed = (
            self.config.data.random_seed
            if self.config.training.model_seed is None
            else int(self.config.training.model_seed)
        )
        if method == "river_logreg":
            return linear_model.SoftmaxRegression(
                optimizer=optim.Adam(lr=self.config.training.learning_rate),
                l2=self.config.training.weight_decay,
            )
        if method == "river_hat":
            return tree.HoeffdingAdaptiveTreeClassifier(
                seed=model_seed
            )
        if method == "river_arf":
            return forest.ARFClassifier(
                n_models=10,
                seed=model_seed,
            )
        raise ValueError(f"Unknown River method {method!r}")

    @staticmethod
    def _river_records(
        x: torch.Tensor, feature_names: list[str]
    ) -> list[dict[str, float]]:
        values = _tensor_to_numpy(x)
        return [
            {name: float(value) for name, value in zip(feature_names, row)}
            for row in values
        ]

    def _evaluate_river(
        self,
        model,
        window: StreamWindow,
        feature_names: list[str],
        n_classes: int,
    ) -> dict[str, float | int]:
        preprocessing_started = time.perf_counter()
        records = self._river_records(window.x, feature_names)
        self._last_preprocessing_seconds = (
            time.perf_counter() - preprocessing_started
        )
        forward_started = time.perf_counter()
        predictions = []
        for record in records:
            predicted = model.predict_one(record)
            predictions.append(0 if predicted is None else int(predicted))
        self._last_forward_seconds = time.perf_counter() - forward_started
        metric_started = time.perf_counter()
        result = classification_metrics(
            np.asarray(window.y.tolist(), dtype=np.int64),
            np.asarray(predictions, dtype=np.int64),
            n_classes,
        )
        self._last_metric_seconds = time.perf_counter() - metric_started
        return result

    def _update_river(
        self,
        model,
        window: StreamWindow,
        feature_names: list[str],
        row: dict[str, object],
    ) -> None:
        started = time.perf_counter()
        train_x, train_y, _, _ = _split_train_guard(
            window, self.config.dasp.guard_ratio, self.config.data.random_seed
        )
        budgeted = _balanced_tensor_sample(
            train_x,
            train_y,
            self.config.baselines.update_label_budget,
            torch.device("cpu"),
            self.config.data.random_seed + window.index * 104729 + 17,
        )
        assert budgeted is not None
        train_x, train_y = budgeted
        row["ground_truth_update_labels"] = len(train_y)
        records = self._river_records(train_x, feature_names)
        for record, target in zip(records, train_y.tolist()):
            model.learn_one(record, int(target))
        row["accepted"] = "online"
        row["selected_candidate"] = self.config.dasp.baseline
        row["sample_updates"] = len(records)
        row["rss_bytes"] = max(
            int(row.get("rss_bytes", 0)),
            int(self._process.memory_info().rss),
        )
        row["update_seconds"] = time.perf_counter() - started

    def _run_river_stream(self, stream: PreparedStream) -> ExperimentResult:
        cfg = self.config
        model = self._build_river_model()
        rows: list[dict[str, object]] = []
        row_lookup: dict[int, dict[str, object]] = {}
        for window in stream.windows[: cfg.data.warmup_windows]:
            metrics = self._evaluate_river(
                model, window, stream.feature_names, len(stream.class_names)
            )
            row = self._base_row(window, metrics, warmup=True)
            rows.append(row)
            row_lookup[window.index] = row
            self._update_river(model, window, stream.feature_names, row)

        pending: list[StreamWindow] = []
        for window in stream.windows[cfg.data.warmup_windows :]:
            metrics = self._evaluate_river(
                model, window, stream.feature_names, len(stream.class_names)
            )
            row = self._base_row(window, metrics, warmup=False)
            rows.append(row)
            row_lookup[window.index] = row
            pending.append(window)
            if len(pending) > cfg.training.label_delay:
                labeled = pending.pop(0)
                self._update_river(
                    model,
                    labeled,
                    stream.feature_names,
                    row_lookup[labeled.index],
                )
        for labeled in pending:
            self._update_river(
                model,
                labeled,
                stream.feature_names,
                row_lookup[labeled.index],
            )

        summary = summarize_windows(rows, cfg.data.warmup_windows)
        summary.update(_latency_summary(rows, cfg.data.warmup_windows))
        def counter(name: str) -> int:
            value = getattr(model, name, 0)
            return int(value() if callable(value) else value)

        summary.update(
            {
                "method": cfg.dasp.baseline,
                "source": cfg.data.source,
                "scenario": cfg.data.scenario,
                "seed": cfg.data.random_seed,
                "stream_seed": (
                    cfg.data.random_seed
                    if cfg.data.stream_seed is None
                    else cfg.data.stream_seed
                ),
                "model_seed": (
                    cfg.data.random_seed
                    if cfg.training.model_seed is None
                    else cfg.training.model_seed
                ),
                "n_features": len(stream.feature_names),
                "n_classes": len(stream.class_names),
                "total_update_seconds": float(
                    sum(float(row.get("update_seconds", 0.0)) for row in rows)
                ),
                "total_gradient_evaluations": 0,
                "total_sample_updates": int(
                    sum(int(row.get("sample_updates", 0)) for row in rows)
                ),
                "total_ground_truth_update_labels": int(
                    sum(
                        int(row.get("ground_truth_update_labels", 0))
                        for row in rows
                    )
                ),
                "reference_resets": 0,
                "global_update_count": int(
                    sum(bool(row.get("global_update", False)) for row in rows)
                ),
                "peak_rss_bytes": int(
                    max(int(row.get("rss_bytes", 0)) for row in rows)
                ),
                "run_start_rss_bytes": self._run_start_rss_bytes,
                "peak_rss_increment_bytes": max(
                    0,
                    int(max(int(row.get("rss_bytes", 0)) for row in rows))
                    - self._run_start_rss_bytes,
                ),
                "model_parameter_bytes": len(
                    pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                ),
                "memory_payload_bytes": 0,
                "peak_cuda_allocated_bytes": 0,
                "peak_cuda_reserved_bytes": 0,
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "logical_cpu_count": psutil.cpu_count(logical=True),
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "device": str(self.device),
                    "torch_threads": torch.get_num_threads(),
                },
                "internal_drifts": counter("n_drifts_detected"),
                "internal_warnings": counter("n_warnings_detected"),
            }
        )
        output_dir = self._write_outputs(rows, summary)
        return ExperimentResult(
            output_dir, summary, pd.DataFrame(rows).sort_values("window")
        )

    def _evaluate(
        self, model: OnlineMLP, window: StreamWindow, n_classes: int
    ) -> dict[str, float | int]:
        model.eval()
        preprocessing_started = time.perf_counter()
        device_x = window.x.to(self.device)
        self._last_preprocessing_seconds = (
            time.perf_counter() - preprocessing_started
        )
        forward_started = time.perf_counter()
        with torch.no_grad():
            probability_tensor = F.softmax(model(device_x), dim=1)
            predicted_tensor = probability_tensor.argmax(dim=1).cpu()
            attack_score = _tensor_to_numpy(1.0 - probability_tensor[:, 0])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._last_forward_seconds = time.perf_counter() - forward_started
        metric_started = time.perf_counter()
        predicted = np.asarray(predicted_tensor.tolist(), dtype=np.int64)
        self._last_predictions = predicted.copy()
        result = classification_metrics(
            np.asarray(window.y.tolist(), dtype=np.int64),
            predicted,
            n_classes,
            y_score=attack_score,
        )
        result.update(self._subtype_recall_metrics(window, predicted))
        self._last_metric_seconds = time.perf_counter() - metric_started
        return result

    def _evaluate_blended(
        self,
        model: OnlineMLP,
        anchor: OnlineMLP,
        window: StreamWindow,
        n_classes: int,
        anchor_weight: float,
    ) -> dict[str, float | int]:
        model.eval()
        anchor.eval()
        alpha = float(np.clip(anchor_weight, 0.0, 1.0))
        preprocessing_started = time.perf_counter()
        x = window.x.to(self.device)
        self._last_preprocessing_seconds = (
            time.perf_counter() - preprocessing_started
        )
        forward_started = time.perf_counter()
        with torch.no_grad():
            active_probability = F.softmax(model(x), dim=1)
            anchor_probability = F.softmax(anchor(x), dim=1)
            probability = (1.0 - alpha) * active_probability + alpha * anchor_probability
            predicted = np.asarray(
                probability.argmax(dim=1).cpu().tolist(), dtype=np.int64
            )
        self._last_predictions = predicted.copy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._last_forward_seconds = time.perf_counter() - forward_started
        metric_started = time.perf_counter()
        result = classification_metrics(
            np.asarray(window.y.tolist(), dtype=np.int64),
            predicted,
            n_classes,
            y_score=np.asarray(
                (1.0 - probability[:, 0]).cpu().tolist(),
                dtype=np.float64,
            ),
        )
        result.update(self._subtype_recall_metrics(window, predicted))
        self._last_metric_seconds = time.perf_counter() - metric_started
        return result

    @staticmethod
    def _subtype_recall_metrics(
        window: StreamWindow,
        predicted: np.ndarray,
    ) -> dict[str, float]:
        """Report attack-subtype recall without exposing subtype to the model."""
        if window.subtypes is None:
            return {}
        values = np.asarray(window.subtypes).astype(str)
        attack_mask = np.asarray(window.y.tolist(), dtype=np.int64) != 0
        recalls: list[float] = []
        result: dict[str, float] = {}
        for subtype in sorted(set(values[attack_mask].tolist())):
            mask = attack_mask & (values == subtype)
            if not np.any(mask):
                continue
            recall = float(np.mean(predicted[mask] != 0))
            slug = "".join(
                character if character.isalnum() else "_"
                for character in subtype.strip().casefold()
            ).strip("_")
            result[f"subtype_recall_{slug}"] = recall
            result[f"subtype_support_{slug}"] = float(np.sum(mask))
            recalls.append(recall)
        if recalls:
            result["minimum_attack_subtype_recall"] = float(min(recalls))
            result["macro_attack_subtype_recall"] = float(np.mean(recalls))
        return result

    def _base_row(
        self,
        window: StreamWindow,
        metrics: dict[str, float],
        warmup: bool,
    ) -> dict[str, object]:
        attack_rate = float((window.y != 0).float().mean().item())
        return {
            "window": window.index,
            "phase": window.phase,
            "warmup": warmup,
            "attack_rate": attack_rate,
            "n_samples": len(window.x),
            "feature_permutation_seed": self.config.revision.feature_permutation_seed,
            "feature_permutation_digest": self._revision_feature_permutation_digest,
            "rss_bytes": int(self._process.memory_info().rss),
            "preprocessing_seconds": self._last_preprocessing_seconds,
            "forward_seconds": self._last_forward_seconds,
            "metric_seconds": self._last_metric_seconds,
            "drift_scoring_seconds": 0.0,
            "prediction_decision_seconds": (
                self._last_preprocessing_seconds
                + self._last_forward_seconds
            ),
            "start_time": window.start_time,
            "end_time": window.end_time,
            **metrics,
            "severity": 0.0,
            "triggered": False,
            "emergency_trigger": False,
            "recurrence_detected": False,
            "anchor_severity": 0.0,
            "inference_anchor_weight": 0.0,
            "accepted": "none",
            "guard_lcb": np.nan,
            "guard_mean_delta": np.nan,
            "guard_recent_lcb": np.nan,
            "guard_recent_mean_delta": np.nan,
            "guard_memory_lcb": np.nan,
            "guard_memory_mean_delta": np.nan,
            "guard_combined_lcb": np.nan,
            "guard_memory_weight": np.nan,
            "gradient_evaluations": 0,
            "update_seconds": 0.0,
            "reference_reset": False,
            "selected_candidate": "none",
            "sample_updates": 0,
            "ground_truth_update_labels": 0,
            "global_update": False,
            "v12_warning_eligible": False,
            "v12_warning_attempted": False,
            "v12_warning_committed": False,
            "v12_alarm_committed": False,
            "v12_base_guard_macro_f1": np.nan,
            "v12_base_guard_attack_recall": np.nan,
            "v12_selected_width": 0,
        }

    @staticmethod
    def _record_decision(row: dict[str, object], decision: DriftDecision) -> None:
        row["severity"] = decision.severity
        row["triggered"] = decision.triggered
        row["emergency_trigger"] = decision.emergency
        for name, value in decision.distances.items():
            row[f"mmd_{name}"] = value
        for name, value in decision.z_scores.items():
            row[f"z_{name}"] = value
        for name, value in decision.requirements.items():
            row[f"requirement_{name}"] = value

    def _revision_dataset_key(self) -> str:
        cfg = self.config
        if cfg.revision.dataset_key:
            return cfg.revision.dataset_key
        if cfg.data.path:
            return Path(cfg.data.path).stem
        return cfg.data.source

    def _complete_revision_event_diagnostics(self, item: PendingUpdate) -> None:
        remaining: list[dict[str, object]] = []
        for event in self._revision_event_pending:
            if int(event["target_window"]) != item.window.index:
                remaining.append(event)
                continue
            source_row = event["source_row"]
            assert isinstance(source_row, dict)
            branch_metrics: dict[str, dict[str, float]] = {}
            branches = event["branches"]
            assert isinstance(branches, dict)
            for identifier, branch in branches.items():
                assert isinstance(branch, torch.nn.Module)
                metrics = _guard_classification(
                    branch,
                    (item.window.x, item.window.y),
                    self.device,
                )
                branch.eval()
                with torch.no_grad():
                    loss = float(
                        F.cross_entropy(
                            branch(item.window.x.to(self.device)),
                            item.window.y.to(self.device),
                        ).item()
                    )
                branch_metrics[str(identifier)] = {
                    "macro_f1": float(metrics["macro_f1"]),
                    "attack_recall": float(metrics["attack_recall"]),
                    "loss": loss,
                }
            reference = branch_metrics["A1"]
            for identifier, metrics in branch_metrics.items():
                prefix = f"event_matched_{identifier}"
                for metric_name, value in metrics.items():
                    source_row[f"{prefix}_next_{metric_name}"] = value
                    source_row[f"{prefix}_delta_{metric_name}"] = (
                        value - reference[metric_name]
                    )
            source_row["event_matched_next_window"] = item.window.index
            source_row["event_matched_complete"] = True
        self._revision_event_pending = remaining

    def _queue_revision_event_diagnostics(
        self,
        model_before: OnlineMLP,
        model_after: OnlineMLP,
        item: PendingUpdate,
        row: dict[str, object],
        requirements: dict[str, float],
        conflicts: dict[str, float],
        training_x: torch.Tensor,
        training_y: torch.Tensor,
        training_weights: torch.Tensor,
        epochs: int,
        random_seed: int,
    ) -> None:
        cfg = self.config
        if not cfg.revision.event_matched_diagnostics:
            return
        branches: dict[str, torch.nn.Module] = {"A1": copy.deepcopy(model_after)}
        condition_by_id = {
            condition.identifier: condition
            for condition in REVISION_CONDITIONS.values()
        }
        for identifier in cfg.revision.event_matched_controls:
            condition = condition_by_id[identifier]
            branch = copy.deepcopy(model_before).to(self.device)
            selection = select_routing(
                condition.routing_policy,
                list(branch.layer_names),
                requirements,
                conflicts,
                condition.routing_k,
                cfg.dasp.conflict_weight,
                cfg.revision.representation_plasticity,
                cfg.revision.classifier_plasticity,
                cfg.revision.fixed_modules,
                random_seed=random_seed,
            )
            before = parameter_snapshot(branch)
            branch_started = time.perf_counter()
            evaluations = _train_dasp(
                branch,
                training_x,
                training_y,
                selection.plasticities,
                cfg,
                self.device,
                ReservoirMemory(1, 0),
                strict_freeze=True,
                sample_weights=training_weights,
                epochs=epochs,
                use_projection_memory=False,
            )
            elapsed = time.perf_counter() - branch_started
            locality = locality_diagnostics(
                branch,
                before,
                selection.active_modules,
                evaluations,
                cfg.revision.change_tolerance,
            )
            row[f"event_matched_{identifier}_update_seconds"] = elapsed
            row[f"event_matched_{identifier}_changed_parameter_fraction"] = locality[
                "changed_parameter_fraction"
            ]
            branches[identifier] = branch
        row["event_matched_A1_update_seconds"] = row["revision_training_seconds"]
        row["event_matched_A1_changed_parameter_fraction"] = row[
            "changed_parameter_fraction"
        ]
        row["event_matched_complete"] = False
        self._revision_event_pending.append(
            {
                "target_window": item.window.index + 1,
                "source_row": row,
                "branches": branches,
            }
        )

    def _update_revision(
        self,
        model: OnlineMLP,
        memory: ReservoirMemory,
        item: PendingUpdate,
        row: dict[str, object],
        started: float,
    ) -> str:
        cfg = self.config
        if not cfg.revision.enabled:
            raise RuntimeError("revision methods require revision.enabled=true")
        condition = REVISION_CONDITIONS[cfg.dasp.baseline]
        self._complete_revision_event_diagnostics(item)
        pools = equal_label_pools(
            len(item.window.x),
            cfg.revision.label_budget,
            cfg.revision.repair_candidate_size,
            cfg.revision.gate_size,
            self._revision_dataset_key(),
            cfg.data.scenario,
            cfg.data.random_seed if cfg.data.stream_seed is None else cfg.data.stream_seed,
            item.window.index,
            cfg.revision.protocol_version,
        )
        if len(pools.query) != min(len(item.window.x), cfg.revision.label_budget):
            raise RuntimeError("equal-label query did not satisfy the exposure budget")
        if set(pools.repair_candidates.tolist()) & set(pools.gate.tolist()):
            raise RuntimeError("repair and gate pools must be disjoint")
        performance_gate_enabled = (
            cfg.revision.performance_gate_enabled
            if condition.performance_gate_enabled is None
            else condition.performance_gate_enabled
        )
        effective_new_sample_weight = (
            cfg.dasp.modular_new_sample_weight
            if condition.new_sample_weight is None
            else condition.new_sample_weight
        )
        row.update(
            {
                "revision_condition": condition.identifier,
                "label_contract_version": cfg.revision.protocol_version,
                "labels_exposed_total": len(pools.query),
                "repair_candidate_labels_exposed": len(pools.repair_candidates),
                "gate_labels_exposed": len(pools.gate),
                "labels_used_for_selection": len(pools.repair_candidates),
                "labels_used_for_gating": (
                    len(pools.gate)
                    if condition.identifier != "A10" and performance_gate_enabled
                    else 0
                ),
                "pseudo_labels_used": 0,
                "query_index_digest": index_digest(pools.query),
                "repair_candidate_index_digest": index_digest(pools.repair_candidates),
                "gate_index_digest": index_digest(pools.gate),
                "label_query_seed": pools.query_seed,
                "training_selection_seed": pools.selection_seed,
                "ground_truth_update_labels": len(pools.query),
                "revision_selection_policy": (
                    "class_balanced"
                    if condition.label_balanced_selection
                    else "uniform_label_blind"
                ),
                "revision_replay_enabled": condition.replay_enabled,
                "revision_new_sample_weight": effective_new_sample_weight,
                "revision_performance_gate_enabled": performance_gate_enabled,
            }
        )
        candidate_x = item.window.x[pools.repair_candidates]
        candidate_y = item.window.y[pools.repair_candidates]
        gate_x = item.window.x[pools.gate]
        gate_y = item.window.y[pools.gate]

        if condition.identifier == "A10":
            assert self._ssf_teacher is not None
            historical = memory.tensors(self.device)
            if historical is None:
                raise RuntimeError("revision SSF requires warm-up memory")
            old_x, old_y = historical
            old_probability = normal_probabilities(model, old_x, self.device)
            new_probability = normal_probabilities(model, candidate_x, self.device)
            statistic, p_value = ks_2samp(
                _tensor_to_numpy(old_probability),
                _tensor_to_numpy(new_probability),
            )
            drift = bool(p_value < cfg.baselines.ssf_drift_p_value)
            old_scores, new_scores = ssf_masks(
                old_probability,
                new_probability,
                cfg.baselines.ssf_mask_steps,
                cfg.baselines.ssf_mask_bins,
            )
            selected_count = min(
                cfg.revision.optimization_size,
                len(candidate_x),
                cfg.dasp.memory_size,
                len(old_x),
            )
            selected_positions = torch.argsort(new_scores, descending=True)[:selected_count]
            selected_global = pools.repair_candidates[selected_positions.detach().cpu()]
            row["training_index_digest"] = index_digest(selected_global)
            (
                buffer_x,
                buffer_y,
                new_mask,
                forgotten,
                inserted,
                ground_truth_inserted,
                pseudo_inserted,
            ) = select_ssf_buffer(
                old_x,
                old_y,
                candidate_x.to(self.device),
                candidate_y.to(self.device),
                old_scores,
                new_scores,
                cfg.dasp.memory_size,
                cfg.revision.optimization_size,
                drift,
                allow_pseudo_labels=False,
            )
            evaluations = train_ssf(
                model,
                self._ssf_teacher,
                buffer_x,
                buffer_y,
                new_mask,
                drift,
                cfg,
                self.device,
                item.window.index,
            )
            self._ssf_teacher = clone_teacher(model)
            memory.replace(buffer_x, buffer_y)
            row.update(
                {
                    "new_training_labels_used": ground_truth_inserted,
                    "gradient_evaluations": evaluations,
                    "parameter_gradient_element_evaluations": evaluations
                    * sum(parameter.numel() for parameter in model.parameters()),
                    "triggered": drift,
                    "ssf_ks_statistic": float(statistic),
                    "ssf_ks_p_value": float(p_value),
                    "buffer_forgotten": forgotten,
                    "buffer_inserted": inserted,
                    "ssf_ground_truth_inserted": ground_truth_inserted,
                    "ssf_pseudo_inserted": pseudo_inserted,
                    "buffer_size": len(memory),
                    "selected_candidate": "revision_a10_ssf",
                    "accepted": "online",
                    "update_seconds": time.perf_counter() - started,
                }
            )
            return "online"

        training_indices = select_training_indices(
            item.window.y,
            pools.repair_candidates,
            cfg.revision.optimization_size,
            pools.selection_seed,
            label_balanced=condition.label_balanced_selection,
        )
        train_x = item.window.x[training_indices]
        train_y = item.window.y[training_indices]
        row["training_index_digest"] = index_digest(training_indices)
        row["new_training_labels_used"] = len(training_indices)
        if performance_gate_enabled:
            gate_metrics = _guard_classification(
                model,
                (gate_x, gate_y),
                self.device,
            )
            row["revision_gate_macro_f1"] = gate_metrics["macro_f1"]
            row["revision_gate_attack_recall"] = gate_metrics["attack_recall"]
            gate_repair = bool(
                gate_metrics["macro_f1"] < cfg.dasp.v12_guard_min_macro_f1
            )
        else:
            row["revision_gate_macro_f1"] = np.nan
            row["revision_gate_attack_recall"] = np.nan
            gate_repair = False
        repair = bool(item.decision.triggered or gate_repair)
        row["revision_gate_repair"] = gate_repair
        row["revision_repair_applied"] = repair
        if not repair:
            memory.add(train_x, train_y)
            row.update(
                {
                    "gradient_evaluations": 0,
                    "parameter_gradient_element_evaluations": 0,
                    "selected_candidate": "none",
                    "accepted": "none",
                    "update_seconds": time.perf_counter() - started,
                }
            )
            return "none"

        historical = memory.tensors(torch.device("cpu"))
        replay_indices = torch.empty(0, dtype=torch.long)
        replay_x = torch.empty((0, train_x.shape[1]), dtype=train_x.dtype)
        replay_y = torch.empty(0, dtype=train_y.dtype)
        if (
            condition.replay_enabled
            and historical is not None
            and cfg.revision.replay_size > 0
        ):
            replay_seed = stable_seed(
                "revision-replay",
                self._revision_dataset_key(),
                cfg.data.scenario,
                cfg.data.random_seed if cfg.data.stream_seed is None else cfg.data.stream_seed,
                item.window.index,
                cfg.revision.protocol_version,
            )
            replay_indices = select_training_indices(
                historical[1],
                torch.arange(len(historical[1])),
                cfg.revision.replay_size,
                replay_seed,
                label_balanced=True,
            )
            replay_x = historical[0][replay_indices]
            replay_y = historical[1][replay_indices]
        row["replay_index_digest"] = index_digest(replay_indices)
        row["revision_replay_size"] = len(replay_indices)
        conflict_memory = (
            replay_x[: cfg.dasp.memory_batch_size].to(self.device),
            replay_y[: cfg.dasp.memory_batch_size].to(self.device),
        ) if len(replay_y) else None
        conflicts = _gradient_conflicts(
            model,
            train_x[: cfg.dasp.memory_batch_size],
            train_y[: cfg.dasp.memory_batch_size],
            conflict_memory,
            self.device,
        )
        routing_seed = stable_seed(
            cfg.revision.routing_namespace,
            self._revision_dataset_key(),
            cfg.data.scenario,
            cfg.data.random_seed if cfg.data.stream_seed is None else cfg.data.stream_seed,
            item.window.index,
            cfg.revision.protocol_version,
        )
        selection = select_routing(
            condition.routing_policy,
            list(model.layer_names),
            item.decision.requirements,
            conflicts,
            condition.routing_k,
            cfg.dasp.conflict_weight,
            cfg.revision.representation_plasticity,
            cfg.revision.classifier_plasticity,
            cfg.revision.fixed_modules,
            random_seed=routing_seed,
        )
        for name, value in conflicts.items():
            row[f"conflict_{name}"] = value
        for name, value in selection.scores.items():
            row[f"routing_score_{name}"] = value
        row["routing_policy"] = condition.routing_policy
        row["routing_k"] = condition.routing_k
        row["routing_seed"] = routing_seed
        row["routing_jaccard_previous"] = routing_jaccard(
            self._revision_previous_route,
            selection.active_modules,
        )
        self._revision_previous_route = selection.active_modules
        training_x = torch.cat([train_x, replay_x], dim=0).to(self.device)
        training_y = torch.cat([train_y, replay_y], dim=0).to(self.device)
        training_weights = torch.cat(
            [
                torch.full(
                    (len(train_y),),
                    float(effective_new_sample_weight),
                    dtype=torch.float32,
                ),
                torch.ones(len(replay_y), dtype=torch.float32),
            ]
        ).to(self.device)
        epochs = (
            cfg.dasp.modular_emergency_epochs
            if gate_repair and not item.decision.triggered
            else cfg.dasp.modular_repair_epochs
        )
        before_model = copy.deepcopy(model)
        before = parameter_snapshot(model)
        training_started = time.perf_counter()
        evaluations = _train_dasp(
            model,
            training_x,
            training_y,
            selection.plasticities,
            cfg,
            self.device,
            memory,
            strict_freeze=True,
            sample_weights=training_weights,
            epochs=epochs,
            use_projection_memory=False,
        )
        training_seconds = time.perf_counter() - training_started
        locality = locality_diagnostics(
            model,
            before,
            selection.active_modules,
            evaluations,
            cfg.revision.change_tolerance,
        )
        row.update(locality)
        if float(locality["inactive_module_max_abs_change"]) != 0.0:
            raise RuntimeError("revision strict-freeze invariant was violated")
        row.update(
            {
                "gradient_evaluations": evaluations,
                "revision_training_seconds": training_seconds,
                "selected_candidate": condition.identifier,
                "accepted": "dasp",
                "localized_repair_modules": "|".join(selection.active_modules),
                "localized_repair_module_count": len(selection.active_modules),
                "update_seconds": time.perf_counter() - started,
            }
        )
        memory.add(train_x, train_y)
        if condition.identifier == "A1":
            self._queue_revision_event_diagnostics(
                before_model,
                model,
                item,
                row,
                item.decision.requirements,
                conflicts,
                training_x,
                training_y,
                training_weights,
                epochs,
                routing_seed,
            )
        return "dasp"

    def _update(
        self,
        model: OnlineMLP,
        memory: ReservoirMemory,
        item: PendingUpdate,
        row: dict[str, object],
    ) -> str:
        cfg = self.config
        method = cfg.dasp.baseline
        if method == "modular_dasp" and cfg.original_review.enabled:
            return original_review.update_original(self, model, memory, item, row)
        started = time.perf_counter()
        if method == "static":
            row["ground_truth_update_labels"] = 0
            row["update_seconds"] = time.perf_counter() - started
            return "none"
        if method in REVISION_METHODS:
            return self._update_revision(model, memory, item, row, started)
        if method in FAST_DASP_METHODS:
            event_decision = item.current_decision or item.decision
            warning_event = bool(
                item.current_warning
                if item.current_warning is not None
                else item.warning
            )
            warning_event = bool(
                warning_event
                or event_decision.triggered
                or item.recurrence
            )
            row["fast_update_warning"] = warning_event
            if not warning_event:
                stable_budget = min(
                    cfg.dasp.fast_stable_label_budget,
                    cfg.baselines.update_label_budget,
                )
                adaptive_stable_query = False
                if method in V3_QUERY_METHODS:
                    stable_history = memory.tensors(
                        torch.device("cpu")
                    )
                    stable_weights: torch.Tensor | None = None
                    stable_soft_fraction = (
                        cfg.dasp.safe_query_soft_class_fraction
                    )
                    if stable_history is not None:
                        stable_counts = torch.bincount(
                            stable_history[1].detach().cpu().long(),
                            minlength=int(model.classifier.out_features),
                        ).float()
                        stable_weights = torch.rsqrt(stable_counts + 1.0)
                        stable_prevalence = stable_counts / (
                            stable_counts.sum().clamp_min(1.0)
                        )
                        if float(stable_prevalence.min().item()) < float(
                            cfg.dasp.v3_query_rare_class_threshold
                        ):
                            adaptive_stable_query = True
                            stable_soft_fraction = max(
                                stable_soft_fraction,
                                cfg.dasp.v3_query_rare_soft_fraction,
                            )
                if adaptive_stable_query:
                    (
                        stable_indices,
                        stable_query_diagnostics,
                    ) = _v18_label_query_indices(
                        model,
                        item.window.x,
                        memory,
                        stable_budget,
                        self.device,
                        cfg.data.random_seed
                        + item.window.index * 104729
                        + 13,
                        entropy_weight=cfg.dasp.v18_query_entropy_weight,
                        novelty_weight=cfg.dasp.v18_query_novelty_weight,
                        disagreement_weight=(
                            cfg.dasp.v18_query_disagreement_weight
                        ),
                        global_fraction=(
                            cfg.dasp.v18_query_global_fraction
                        ),
                        candidate_multiplier=(
                            cfg.dasp.v18_query_candidate_multiplier
                        ),
                        batch_size=max(1, cfg.training.batch_size),
                        soft_class_fraction=stable_soft_fraction,
                        class_quota_weights=stable_weights,
                        raw_diversity_fraction=(
                            cfg.dasp.v3_query_raw_diversity_fraction
                        ),
                    )
                    row.update(stable_query_diagnostics)
                    row["label_query_policy"] = (
                        "adaptive_label_blind_stable"
                    )
                else:
                    stable_indices = _label_blind_uniform_query_indices(
                        len(item.window.x),
                        stable_budget,
                        cfg.data.random_seed
                        + item.window.index * 104729
                        + 13,
                    )
                stable_x = item.window.x[stable_indices]
                stable_y = item.window.y[stable_indices]
                if method in V4_METHODS:
                    self._v4_label_history = (
                        _bounded_label_history_update(
                            self._v4_label_history,
                            stable_y,
                            cfg.dasp.v4_prior_history_size,
                        )
                    )
                gate_model = (
                    self._fast_inference_model
                    if self._fast_inference_model is not None
                    else model
                )
                stable_performance = _guard_classification(
                    gate_model,
                    (stable_x, stable_y),
                    self.device,
                )
                history = self._v16_performance_history[
                    -max(
                        1,
                        cfg.dasp.v16_performance_reference_windows,
                    ) :
                ]
                reference_macro = (
                    float(np.median([value[0] for value in history]))
                    if history
                    else stable_performance["macro_f1"]
                )
                historical_recalls = [
                    value[1]
                    for value in history
                    if np.isfinite(value[1])
                ]
                reference_recall = (
                    float(np.median(historical_recalls))
                    if historical_recalls
                    else stable_performance["attack_recall"]
                )
                macro_drop = (
                    reference_macro
                    - stable_performance["macro_f1"]
                )
                recall_drop = (
                    reference_recall
                    - stable_performance["attack_recall"]
                    if np.isfinite(reference_recall)
                    and np.isfinite(
                        stable_performance["attack_recall"]
                    )
                    else np.nan
                )
                performance_event = bool(
                    len(history)
                    >= (
                        1
                        if method in V3_FINAL_METHODS
                        else cfg.dasp.v16_performance_min_history
                    )
                    and (
                        macro_drop >= cfg.dasp.v16_macro_f1_drop
                        or (
                            np.isfinite(recall_drop)
                            and recall_drop
                            >= cfg.dasp.v16_attack_recall_drop
                        )
                    )
                )
                row.update(
                    {
                        "fast_stable_precheck": True,
                        "fast_precheck_performance_triggered": (
                            performance_event
                        ),
                        "fast_precheck_macro_f1": stable_performance[
                            "macro_f1"
                        ],
                        "fast_precheck_attack_recall": (
                            stable_performance["attack_recall"]
                        ),
                        "fast_precheck_macro_f1_drop": macro_drop,
                        "fast_precheck_attack_recall_drop": recall_drop,
                    }
                )
                if not performance_event:
                    stable_newly_revealed_count = len(stable_y)
                    if (
                        method in V3_CHECKPOINT_METHODS
                        and item.prediction_snapshot is not None
                        and item.prediction_labels is not None
                    ):
                        checkpoint_score = _v3_checkpoint_score(
                            stable_y,
                            item.prediction_labels[stable_indices],
                            int(model.classifier.out_features),
                        )
                        row["v3_checkpoint_observed_score"] = (
                            checkpoint_score
                            if checkpoint_score is not None
                            else np.nan
                        )
                        if checkpoint_score is not None:
                            self._v3_checkpoints.append(
                                (
                                    checkpoint_score,
                                    item.window.index,
                                    item.prediction_snapshot,
                                )
                            )
                            self._v3_checkpoints = self._v3_checkpoints[
                                -max(
                                    1,
                                    cfg.dasp.v3_checkpoint_horizon,
                                ) :
                            ]
                    stable_has_support = _has_class_support(
                        stable_y,
                        int(model.classifier.out_features),
                        1,
                    )
                    if method in SAFE_V2_METHODS:
                        self._safe_label_buffer = (
                            _merge_safe_label_buffer(
                                self._safe_label_buffer,
                                stable_x,
                                stable_y,
                                min(
                                    cfg.dasp.safe_accumulator_size,
                                    cfg.baselines.update_label_budget,
                                ),
                                cfg.data.random_seed
                                + item.window.index * 104729
                                + 23,
                            )
                        )
                        assert self._safe_label_buffer is not None
                        stable_has_support = _has_class_support(
                            self._safe_label_buffer[1],
                            int(model.classifier.out_features),
                            max(
                                2,
                                (
                                    cfg.dasp.v3_fallback_min_per_class
                                    if method in V3_FINAL_METHODS
                                    else cfg.dasp.safe_accumulator_min_per_class
                                ),
                            ),
                        )
                    if (
                        method in SAFE_ONECLASS_METHODS
                        and not stable_has_support
                    ):
                        if method in V3_MEMORY_METHODS:
                            updated_memory = _method_true_label_memory_update(
                                method,
                                model,
                                memory.tensors(torch.device("cpu")),
                                stable_x,
                                stable_y,
                                cfg=cfg,
                                device=self.device,
                                seed=(
                                    cfg.data.random_seed
                                    + item.window.index * 104729
                                    + 67
                                ),
                            )
                            memory.replace(*updated_memory)
                        checkpoint_diagnostics: dict[str, object] = {}
                        checkpoint_restored = False
                        if method in V3_FINAL_METHODS:
                            checkpoint_restored, checkpoint_diagnostics = (
                                _restore_v3_checkpoint(
                                    model,
                                    self._v3_checkpoints,
                                    recent=None,
                                    memory=memory.tensors(self.device),
                                    cfg=cfg,
                                    device=self.device,
                                    require_current_guard=False,
                                )
                            )
                            if checkpoint_restored:
                                self._refresh_fast_inference_model(model)
                        row.update(
                            {
                                "accepted": "none",
                                "selected_candidate": (
                                    "v3_checkpoint_stable_fallback"
                                    if checkpoint_restored
                                    else
                                    "safe_accumulator_stable_wait"
                                    if method in SAFE_V2_METHODS
                                    else "safe_oneclass_stable_skip"
                                ),
                                "ground_truth_update_labels": (
                                    stable_newly_revealed_count
                                ),
                                "label_query_policy": (
                                    "adaptive_label_blind_stable_oneclass_skip"
                                    if adaptive_stable_query
                                    else
                                    "uniform_label_blind_stable_oneclass_skip"
                                ),
                                "gradient_evaluations": 0,
                                "global_update": False,
                                "buffer_size": len(memory),
                                "fast_stable_skip": True,
                                "safe_oneclass_revealed": True,
                                "safe_gate_passed": False,
                                "safe_gate_reason": (
                                    "stable_accumulator_class_support"
                                    if method in SAFE_V2_METHODS
                                    else "stable_queried_batch_one_class"
                                ),
                                "safe_rollback_exact": True,
                            }
                        )
                        row.update(checkpoint_diagnostics)
                        row["rss_bytes"] = max(
                            int(row.get("rss_bytes", 0)),
                            int(self._process.memory_info().rss),
                        )
                        row["update_seconds"] = (
                            time.perf_counter() - started
                        )
                        return "none"
                    if method in SAFE_V2_METHODS:
                        stable_x, stable_y = self._safe_label_buffer
                        self._safe_label_buffer = None
                    self._v16_performance_history.append(
                        (
                            stable_performance["macro_f1"],
                            stable_performance["attack_recall"],
                        )
                    )
                    if method in V3_MEMORY_METHODS:
                        balanced_memory = _method_true_label_memory_update(
                            method,
                            model,
                            memory.tensors(torch.device("cpu")),
                            stable_x,
                            stable_y,
                            cfg=cfg,
                            device=self.device,
                            seed=(
                                cfg.data.random_seed
                                + item.window.index * 104729
                                + 67
                            ),
                        )
                        memory.replace(*balanced_memory)
                    elif method in SAFE_V2_METHODS:
                        balanced_memory = _merge_safe_label_buffer(
                            memory.tensors(torch.device("cpu")),
                            stable_x,
                            stable_y,
                            cfg.dasp.memory_size,
                            cfg.data.random_seed
                            + item.window.index * 104729
                            + 67,
                        )
                        assert balanced_memory is not None
                        memory.replace(*balanced_memory)
                    else:
                        memory.add(stable_x, stable_y)
                    self._v18_guard_history = (
                        _update_v18_guard_history(
                            self._v18_guard_history,
                            stable_x,
                            stable_y,
                            cfg.dasp.fast_guard_history_size,
                            cfg.data.random_seed
                            + item.window.index * 104729
                            + 71,
                        )
                    )
                    row.update(
                        {
                            "accepted": "none",
                            "selected_candidate": (
                                "stable_memory_refresh"
                            ),
                            "ground_truth_update_labels": (
                                stable_newly_revealed_count
                            ),
                            "label_query_policy": (
                                "adaptive_label_blind_stable"
                                if adaptive_stable_query
                                else "uniform_label_blind_stable"
                            ),
                            "gradient_evaluations": 0,
                            "global_update": False,
                            "buffer_size": len(memory),
                            "fast_stable_skip": True,
                            "v17_path": "stable",
                            "v18_guard_history_size": len(
                                self._v18_guard_history[1]
                            ),
                        }
                    )
                    if any(
                        parameter.requires_grad
                        for parameter in model.blocks.parameters()
                    ):
                        raise RuntimeError(
                            f"{method} backbone was unfrozen after "
                            "warm-up"
                        )
                    row["rss_bytes"] = max(
                        int(row.get("rss_bytes", 0)),
                        int(self._process.memory_info().rss),
                    )
                    row["update_seconds"] = (
                        time.perf_counter() - started
                    )
                    return "none"
        memory_probe, memory_guard = _split_memory_for_probe_and_guard(
            memory, cfg.dasp.memory_batch_size, self.device
        )
        if method in V18_METHODS:
            v3_quota_weights: torch.Tensor | None = None
            v3_soft_fraction = cfg.dasp.safe_query_soft_class_fraction
            v3_raw_fraction = 0.0
            if method in V3_QUERY_METHODS:
                labeled_history = memory.tensors(torch.device("cpu"))
                if labeled_history is not None:
                    history_counts = torch.bincount(
                        labeled_history[1].detach().cpu().long(),
                        minlength=int(model.classifier.out_features),
                    ).float()
                    v3_quota_weights = torch.rsqrt(
                        history_counts + 1.0
                    )
                    prevalence = history_counts / history_counts.sum().clamp_min(
                        1.0
                    )
                    if float(prevalence.min().item()) < float(
                        cfg.dasp.v3_query_rare_class_threshold
                    ):
                        v3_soft_fraction = max(
                            v3_soft_fraction,
                            cfg.dasp.v3_query_rare_soft_fraction,
                        )
                v3_raw_fraction = (
                    cfg.dasp.v3_query_raw_diversity_fraction
                )
            query_indices, query_diagnostics = _v18_label_query_indices(
                model,
                item.window.x,
                memory,
                cfg.baselines.update_label_budget,
                self.device,
                cfg.data.random_seed
                + item.window.index * 104729
                + 17,
                entropy_weight=cfg.dasp.v18_query_entropy_weight,
                novelty_weight=cfg.dasp.v18_query_novelty_weight,
                disagreement_weight=(
                    cfg.dasp.v18_query_disagreement_weight
                ),
                global_fraction=cfg.dasp.v18_query_global_fraction,
                candidate_multiplier=cfg.dasp.v18_query_candidate_multiplier,
                batch_size=max(1, cfg.training.batch_size),
                soft_class_fraction=(
                    v3_soft_fraction
                    if method in SAFE_QUERY_METHODS
                    else 0.0
                ),
                class_quota_weights=v3_quota_weights,
                raw_diversity_fraction=v3_raw_fraction,
            )
            if not len(query_indices):
                raise RuntimeError("dasp_v18 selected no label queries")
            # Labels are dereferenced only after model-only query selection.
            queried_x = item.window.x[query_indices]
            queried_y = item.window.y[query_indices]
            if method in V4_METHODS:
                self._v4_label_history = _bounded_label_history_update(
                    self._v4_label_history,
                    queried_y,
                    cfg.dasp.v4_prior_history_size,
                )
            newly_revealed_count = len(queried_y)
            true_counts = torch.bincount(
                queried_y.detach().cpu().long(),
                minlength=model.classifier.out_features,
            )
            row.update(query_diagnostics)
            row["v18_query_selected_true_per_class"] = "|".join(
                str(int(value)) for value in true_counts.tolist()
            )
            row["ground_truth_update_labels"] = newly_revealed_count
            row["safe_oneclass_revealed"] = not _has_class_support(
                queried_y,
                int(model.classifier.out_features),
                1,
            )
            if (
                method in V3_CHECKPOINT_METHODS
                and item.prediction_snapshot is not None
                and item.prediction_labels is not None
            ):
                checkpoint_score = _v3_checkpoint_score(
                    queried_y,
                    item.prediction_labels[query_indices],
                    int(model.classifier.out_features),
                )
                row["v3_checkpoint_observed_score"] = (
                    checkpoint_score
                    if checkpoint_score is not None
                    else np.nan
                )
                if checkpoint_score is not None:
                    self._v3_checkpoints.append(
                        (
                            checkpoint_score,
                            item.window.index,
                            item.prediction_snapshot,
                        )
                    )
                    self._v3_checkpoints = self._v3_checkpoints[
                        -max(1, cfg.dasp.v3_checkpoint_horizon) :
                    ]
            if method in SAFE_V2_METHODS:
                self._safe_label_buffer = _merge_safe_label_buffer(
                    self._safe_label_buffer,
                    queried_x,
                    queried_y,
                    min(
                        cfg.dasp.safe_accumulator_size,
                        cfg.baselines.update_label_budget,
                    ),
                    cfg.data.random_seed
                    + item.window.index * 104729
                    + 23,
                )
                assert self._safe_label_buffer is not None
                queried_x, queried_y = self._safe_label_buffer
                accumulated_counts = torch.bincount(
                    queried_y.detach().cpu().long(),
                    minlength=model.classifier.out_features,
                )
                row["safe_accumulated_train_size"] = len(queried_y)
                row["safe_accumulated_per_class"] = "|".join(
                    str(int(value))
                    for value in accumulated_counts.tolist()
                )
                accumulated_supported = _has_class_support(
                    queried_y,
                    int(model.classifier.out_features),
                    max(
                        2,
                        (
                            cfg.dasp.v3_fallback_min_per_class
                            if method in V3_FINAL_METHODS
                            else cfg.dasp.safe_accumulator_min_per_class
                        ),
                    ),
                )
            else:
                accumulated_supported = not row[
                    "safe_oneclass_revealed"
                ]
            if (
                method in SAFE_ONECLASS_METHODS
                and not accumulated_supported
            ):
                if method in V3_MEMORY_METHODS:
                    updated_memory = _method_true_label_memory_update(
                        method,
                        model,
                        memory.tensors(torch.device("cpu")),
                        item.window.x[query_indices],
                        item.window.y[query_indices],
                        cfg=cfg,
                        device=self.device,
                        seed=(
                            cfg.data.random_seed
                            + item.window.index * 104729
                            + 67
                        ),
                    )
                    memory.replace(*updated_memory)
                checkpoint_diagnostics: dict[str, object] = {}
                checkpoint_restored = False
                if method in V3_FINAL_METHODS:
                    checkpoint_restored, checkpoint_diagnostics = (
                        _restore_v3_checkpoint(
                            model,
                            self._v3_checkpoints,
                            recent=None,
                            memory=memory_guard or memory_probe,
                            cfg=cfg,
                            device=self.device,
                            require_current_guard=False,
                        )
                    )
                    if checkpoint_restored:
                        self._refresh_fast_inference_model(model)
                # The label query was fixed before labels became available.
                # A one-class reveal is recorded but cannot update the model,
                # guard, replay memory, detector reference, or fused cache.
                row.update(
                    {
                        "accepted": "none",
                        "selected_candidate": (
                            "v3_checkpoint_fallback"
                            if checkpoint_restored
                            else
                            "safe_accumulator_wait"
                            if method in SAFE_V2_METHODS
                            else "safe_oneclass_skip"
                        ),
                        "label_query_policy": (
                            "soft_class_label_blind_oneclass_skip"
                        ),
                        "gradient_evaluations": 0,
                        "global_update": False,
                        "buffer_size": len(memory),
                        "safe_gate_passed": False,
                        "safe_gate_reason": (
                            "accumulator_class_support"
                            if method in SAFE_V2_METHODS
                            else "queried_batch_one_class"
                        ),
                        "safe_rollback_exact": True,
                    }
                )
                row.update(checkpoint_diagnostics)
                if any(
                    parameter.requires_grad
                    for parameter in model.blocks.parameters()
                ):
                    raise RuntimeError(
                        f"{method} backbone was unfrozen after warm-up"
                    )
                row["rss_bytes"] = max(
                    int(row.get("rss_bytes", 0)),
                    int(self._process.memory_info().rss),
                )
                row["update_seconds"] = time.perf_counter() - started
                return "none"
            train_x, train_y, guard_x, guard_y = (
                _v18_split_query_train_guard(
                    queried_x,
                    queried_y,
                    cfg.dasp.v18_current_guard_fraction,
                    cfg.dasp.v18_current_guard_min_per_class,
                    cfg.data.random_seed
                    + item.window.index * 104729
                    + 29,
                )
            )
            historical_guard = self._v18_guard_history
            memory_guard = historical_guard or memory_guard or memory_probe
            row["v18_guard_source"] = "current_label_holdout"
            row["v18_guard_size"] = len(guard_y)
            if (
                method in V3_FINAL_METHODS
                and not _has_class_support(
                    guard_y,
                    int(model.classifier.out_features),
                    max(1, cfg.dasp.safe_guard_min_per_class),
                )
                and memory_guard is not None
                and _has_class_support(
                    memory_guard[1],
                    int(model.classifier.out_features),
                    max(1, cfg.dasp.safe_guard_min_per_class),
                )
            ):
                # The current holdout is too sparse for a valid gate. Use all
                # newly revealed labels for training and an older, disjoint
                # labeled guard. No current or future label enters prediction.
                train_x, train_y = queried_x, queried_y
                guard_x, guard_y = memory_guard
                row["v18_guard_source"] = (
                    "historical_class_supported_fallback"
                )
                row["v18_guard_size"] = len(guard_y)
        elif (
            cfg.revision.enabled
            and cfg.revision.common_label_exposure_baselines
            and method in {"online_mlp", "experience_replay"}
        ):
            pools = equal_label_pools(
                len(item.window.x),
                cfg.revision.label_budget,
                cfg.revision.repair_candidate_size,
                cfg.revision.gate_size,
                self._revision_dataset_key(),
                cfg.data.scenario,
                cfg.data.random_seed
                if cfg.data.stream_seed is None
                else cfg.data.stream_seed,
                item.window.index,
                cfg.revision.protocol_version,
            )
            training_indices = select_training_indices(
                item.window.y,
                pools.repair_candidates,
                cfg.revision.optimization_size,
                pools.selection_seed,
                label_balanced=False,
            )
            train_x = item.window.x[training_indices]
            train_y = item.window.y[training_indices]
            guard_pair = memory_guard or memory_probe
            if guard_pair is None:
                guard_x, guard_y = train_x, train_y
            else:
                guard_x, guard_y = guard_pair
            row.update(
                {
                    "label_contract_version": cfg.revision.protocol_version,
                    "labels_exposed_total": len(pools.query),
                    "repair_candidate_labels_exposed": len(
                        pools.repair_candidates
                    ),
                    "gate_labels_exposed": len(pools.gate),
                    "labels_used_for_selection": 0,
                    "labels_used_for_gating": 0,
                    "pseudo_labels_used": 0,
                    "query_index_digest": index_digest(pools.query),
                    "repair_candidate_index_digest": index_digest(
                        pools.repair_candidates
                    ),
                    "gate_index_digest": index_digest(pools.gate),
                    "training_index_digest": index_digest(training_indices),
                    "label_query_seed": pools.query_seed,
                    "training_selection_seed": pools.selection_seed,
                    "ground_truth_update_labels": len(pools.query),
                    "new_training_labels_used": len(training_indices),
                    "label_query_policy": "common_512_uniform_128",
                }
            )
        elif method in BUDGET_MATCHED_BASELINES:
            query_indices = _label_blind_uniform_query_indices(
                len(item.window.x),
                cfg.baselines.update_label_budget,
                cfg.data.random_seed
                + item.window.index * 104729
                + 17,
            )
            if not len(query_indices):
                raise RuntimeError(
                    f"{method} selected no budget-matched label queries"
                )
            train_x = item.window.x[query_indices]
            train_y = item.window.y[query_indices]
            guard_pair = memory_guard or memory_probe
            if guard_pair is None:
                guard_x, guard_y = train_x, train_y
            else:
                guard_x, guard_y = guard_pair
            row["ground_truth_update_labels"] = len(train_y)
            row["label_query_policy"] = "uniform_label_blind"
        else:
            train_x, train_y, guard_x, guard_y = _split_train_guard(
                item.window, cfg.dasp.guard_ratio, cfg.data.random_seed
            )
            guard_budget = (
                max(
                    2,
                    min(
                        cfg.baselines.update_label_budget - 2,
                        int(
                            round(
                                cfg.baselines.update_label_budget
                                * cfg.dasp.guard_ratio
                            )
                        ),
                    ),
                )
                if method in V17_METHODS
                and cfg.baselines.update_label_budget >= 4
                else 0
            )
            train_budget = (
                cfg.baselines.update_label_budget - guard_budget
                if method in V17_METHODS
                else cfg.baselines.update_label_budget
            )
            budgeted = _balanced_tensor_sample(
                train_x,
                train_y,
                train_budget,
                torch.device("cpu"),
                cfg.data.random_seed
                + item.window.index * 104729
                + 17,
            )
            assert budgeted is not None
            train_x, train_y = budgeted
            if guard_budget:
                budgeted_guard = _balanced_tensor_sample(
                    guard_x,
                    guard_y,
                    guard_budget,
                    torch.device("cpu"),
                    cfg.data.random_seed
                    + item.window.index * 104729
                    + 29,
                )
                assert budgeted_guard is not None
                guard_x, guard_y = budgeted_guard
            row["ground_truth_update_labels"] = len(train_y) + (
                len(guard_y) if method in V17_METHODS else 0
            )
        conflicts: dict[str, float] = {}
        if method not in {
            "online",
            "online_mlp",
            "replay",
            "experience_replay",
            "online_ewc",
            "ssf",
            "ssf_budget",
            "ssf_faithful",
            "repshield",
            "cdda_md",
            *ADAPTER_METHODS,
        }:
            conflicts = _gradient_conflicts(
                model,
                train_x[: cfg.dasp.memory_batch_size],
                train_y[: cfg.dasp.memory_batch_size],
                memory_probe,
                self.device,
            )
        for name, value in conflicts.items():
            row[f"conflict_{name}"] = value

        memory_managed = False

        if method in {"ssf", "ssf_budget", "ssf_faithful"}:
            assert self._ssf_teacher is not None
            historical = memory.tensors(self.device)
            assert historical is not None
            old_x, old_y = historical
            old_probability = normal_probabilities(model, old_x, self.device)
            new_probability = normal_probabilities(model, train_x, self.device)
            statistic, p_value = ks_2samp(
                _tensor_to_numpy(old_probability), _tensor_to_numpy(new_probability)
            )
            drift = bool(p_value < cfg.baselines.ssf_drift_p_value)
            old_scores, new_scores = ssf_masks(
                old_probability,
                new_probability,
                cfg.baselines.ssf_mask_steps,
                cfg.baselines.ssf_mask_bins,
            )
            with torch.no_grad():
                pseudo_y = model(train_x.to(self.device)).argmax(dim=1)
            (
                buffer_x,
                buffer_y,
                new_mask,
                forgotten,
                inserted,
                ground_truth_inserted,
                pseudo_inserted,
            ) = select_ssf_buffer(
                old_x,
                old_y,
                train_x.to(self.device),
                train_y.to(self.device),
                old_scores,
                new_scores,
                cfg.dasp.memory_size,
                cfg.baselines.ssf_label_budget,
                drift,
                allow_pseudo_labels=(
                    method == "ssf_faithful"
                    or (
                        method == "ssf"
                        and cfg.baselines.ssf_allow_pseudo_labels
                    )
                ),
                pseudo_y=pseudo_y,
            )
            row["gradient_evaluations"] = train_ssf(
                model,
                self._ssf_teacher,
                buffer_x,
                buffer_y,
                new_mask,
                drift,
                cfg,
                self.device,
                item.window.index,
            )
            self._ssf_teacher = clone_teacher(model)
            memory.replace(buffer_x, buffer_y)
            memory_managed = True
            row.update({
                "triggered": drift,
                "ssf_ks_statistic": float(statistic),
                "ssf_ks_p_value": float(p_value),
                "buffer_forgotten": forgotten,
                "buffer_inserted": inserted,
                "ssf_ground_truth_inserted": ground_truth_inserted,
                "ssf_pseudo_inserted": pseudo_inserted,
                "buffer_size": len(memory),
                "selected_candidate": method,
            })
            accepted = "online"
        elif method == "repshield":
            historical = memory.tensors(self.device)
            assert historical is not None
            old_x, old_y = historical
            pca_score = pca_variance_distance(old_x, train_x.to(self.device))
            drift = bool(pca_score >= cfg.baselines.repshield_pca_threshold)
            gpm_count = min(len(old_x), cfg.dasp.memory_batch_size)
            subspaces = build_gpm_subspaces(
                model,
                old_x[:gpm_count],
                cfg.baselines.repshield_subspace_energy,
                self.device,
            )
            replay_count = int(round(len(train_x) * cfg.baselines.repshield_replay_ratio))
            replay_batch = memory.sample(replay_count, self.device)
            update_x = train_x.to(self.device)
            update_y = train_y.to(self.device)
            if replay_batch is not None:
                update_x = torch.cat([update_x, replay_batch[0]])
                update_y = torch.cat([update_y, replay_batch[1]])
            row["gradient_evaluations"] = train_repshield(
                model,
                update_x,
                update_y,
                subspaces,
                cfg,
                self.device,
                item.window.index,
            )
            buffer_x, buffer_y = strategic_coreset(
                model,
                old_x,
                old_y,
                train_x.to(self.device),
                train_y.to(self.device),
                cfg.dasp.memory_size,
                self.device,
            )
            memory.replace(buffer_x, buffer_y)
            memory_managed = True
            row.update({
                "triggered": drift,
                "repshield_pca_distance": pca_score,
                "gpm_subspace_rank": int(sum(value.shape[1] for value in subspaces.values())),
                "buffer_size": len(memory),
                "selected_candidate": "repshield",
            })
            accepted = "online"
        elif method == "cdda_md":
            assert self._cdda_detector is not None
            error_rate = 1.0 - float(row["accuracy"])
            warning, drift, severity = self._cdda_detector.update(error_rate)
            if warning:
                self._cdda_warning_data.append((train_x, train_y))
            else:
                self._cdda_warning_data.clear()
            row.update({
                "triggered": drift,
                "severity": severity,
                "cdda_error_rate": error_rate,
                "cdda_error_threshold": self._cdda_detector.threshold,
                "cdda_warning": warning,
                "cdda_consecutive_warnings": self._cdda_detector.consecutive_warnings,
            })
            if drift:
                update_x = torch.cat([value[0] for value in self._cdda_warning_data])
                update_y = torch.cat([value[1] for value in self._cdda_warning_data])
                row["gradient_evaluations"] = train_cdda(
                    model, update_x, update_y, cfg, self.device, item.window.index
                )
                self._cdda_warning_data.clear()
                row["selected_candidate"] = "cdda_incremental"
                accepted = "online"
            else:
                row["selected_candidate"] = "no_update"
                accepted = "none"
        elif method in ADAPTER_METHODS:
            assert isinstance(model, AdapterOnlineMLP)
            event_decision = item.current_decision or item.decision
            performance = _guard_classification(
                model, (guard_x, guard_y), self.device
            )
            history = self._v16_performance_history[
                -max(1, cfg.dasp.v16_performance_reference_windows) :
            ]
            if history:
                reference_macro = float(
                    np.median([value[0] for value in history])
                )
                historical_recalls = [
                    value[1] for value in history if np.isfinite(value[1])
                ]
                reference_recall = (
                    float(np.median(historical_recalls))
                    if historical_recalls
                    else np.nan
                )
            else:
                reference_macro = performance["macro_f1"]
                reference_recall = performance["attack_recall"]
            macro_drop = reference_macro - performance["macro_f1"]
            recall_drop = (
                reference_recall - performance["attack_recall"]
                if np.isfinite(reference_recall)
                and np.isfinite(performance["attack_recall"])
                else np.nan
            )
            enough_history = (
                len(history)
                >= (
                    1
                    if method in V3_FINAL_METHODS
                    else cfg.dasp.v16_performance_min_history
                )
            )
            raw_performance_decline = bool(
                enough_history
                and (
                    macro_drop >= cfg.dasp.v16_macro_f1_drop
                    or (
                        np.isfinite(recall_drop)
                        and recall_drop >= cfg.dasp.v16_attack_recall_drop
                    )
                )
            )
            precheck_performance_decline = bool(
                row.get(
                    "fast_precheck_performance_triggered",
                    False,
                )
            )
            combined_performance_decline = bool(
                raw_performance_decline
                or precheck_performance_decline
            )
            if self._v16_performance_cooldown > 0:
                self._v16_performance_cooldown -= 1
            performance_event = bool(
                combined_performance_decline
                and self._v16_performance_cooldown == 0
            )
            if performance_event:
                self._v16_performance_cooldown = max(
                    0, cfg.dasp.v16_performance_cooldown
                )
            if not combined_performance_decline:
                self._v16_performance_history.append(
                    (performance["macro_f1"], performance["attack_recall"])
                )
            raw_strong_event = bool(
                event_decision.triggered
                or performance_event
                or item.recurrence
            )
            v3_covariate_event = bool(
                event_decision.triggered
                and event_decision.severity
                >= cfg.drift.severity_threshold
            )
            strong_base_layer_count = max(
                int(cfg.dasp.v17_strong_base_layers),
                (
                    int(cfg.dasp.v3_backbone_max_layers)
                    if method in V3_BACKBONE_METHODS
                    else int(cfg.dasp.safe_strong_base_layers)
                    if method in SAFE_V2_METHODS
                    else 0
                ),
            )
            if (
                method in V3_BACKBONE_METHODS
                and not v3_covariate_event
            ):
                strong_base_layer_count = 0
            if raw_strong_event:
                self._v17_strong_remaining = max(
                    self._v17_strong_remaining,
                    max(1, cfg.dasp.v17_strong_recovery_windows),
                )
                if strong_base_layer_count > 0:
                    ranked_event_layers = sorted(
                        model.layer_names,
                        key=lambda name: float(
                            event_decision.requirements.get(name, 0.0)
                        ),
                        reverse=True,
                    )
                    maximum_requirement = max(
                        (
                            float(
                                event_decision.requirements.get(name, 0.0)
                            )
                            for name in ranked_event_layers
                        ),
                        default=0.0,
                    )
                    if maximum_requirement > 0.0:
                        self._v17_recovery_base_layers = (
                            ranked_event_layers[
                                : min(
                                    len(ranked_event_layers),
                                    strong_base_layer_count,
                                )
                            ]
                        )
                    elif not self._v17_recovery_base_layers:
                        self._v17_recovery_base_layers = list(
                            model.layer_names[
                                -strong_base_layer_count:
                            ]
                        )
            strong_event = self._v17_strong_remaining > 0
            if strong_event:
                self._v17_strong_remaining -= 1
            fallback = copy.deepcopy(model)
            candidate: AdapterOnlineMLP
            if item.recurrence:
                assert isinstance(self._anchor_model, AdapterOnlineMLP)
                candidate = _soft_reset_toward_reference(
                    model,
                    self._anchor_model,
                    cfg.dasp.recurrence_anchor_strength,
                )
                assert isinstance(candidate, AdapterOnlineMLP)
            else:
                candidate = copy.deepcopy(model)

            trainable_rank = (
                cfg.dasp.v17_strong_rank
                if strong_event
                else cfg.dasp.v17_fast_rank
            )
            active_rank = max(candidate.adapter_rank, trainable_rank)
            candidate.set_adapter_rank(
                active_rank, trainable_rank=trainable_rank
            )
            plasticities = {
                name: (
                    max(
                        0.75,
                        float(event_decision.requirements.get(name, 0.0)),
                    )
                    if strong_event
                    else 1.0
                )
                for name in model.layer_names
            }
            plasticities["classifier"] = 1.0
            plasticities["input_adapter"] = (
                cfg.dasp.v17_strong_input_adapter_plasticity
                if strong_event
                else cfg.dasp.v17_fast_input_adapter_plasticity
            )
            if method in V18_METHODS:
                modulation_plasticity = (
                    cfg.dasp.v18_strong_modulation_plasticity
                    if strong_event
                    else cfg.dasp.v18_fast_modulation_plasticity
                )
                for layer_name in model.layer_names:
                    plasticities[
                        f"modulation_{layer_name}"
                    ] = modulation_plasticity
            parameter_groups = candidate.adaptation_parameter_groups()
            selected_base_layers: list[str] = []
            extra_parameter_names: set[str] = set()
            if strong_event and strong_base_layer_count > 0:
                selected_base_layers = list(
                    self._v17_recovery_base_layers
                )
                for layer_name in selected_base_layers:
                    layer_index = model.layer_names.index(layer_name)
                    group_name = f"base_{layer_name}"
                    base_parameters = list(
                        candidate.blocks[layer_index].parameters()
                    )
                    for parameter in base_parameters:
                        parameter.requires_grad_(True)
                    parameter_groups[group_name] = base_parameters
                    plasticities[group_name] = (
                        min(
                            float(cfg.dasp.v3_backbone_plasticity),
                            max(
                                0.1,
                                float(
                                    event_decision.requirements.get(
                                        layer_name, 0.0
                                    )
                                ),
                            ),
                        )
                        if method in V3_BACKBONE_METHODS
                        else max(
                            0.75,
                            float(
                                event_decision.requirements.get(
                                    layer_name, 0.0
                                )
                            ),
                        )
                    )
                    prefix = f"blocks.{layer_index}."
                    extra_parameter_names.update(
                        name
                        for name, _ in candidate.named_parameters()
                        if name.startswith(prefix)
                    )

            current_unlabeled = (
                item.current_unlabeled_x
                if item.current_unlabeled_x is not None
                else item.window.x
            )
            query_count = min(
                len(current_unlabeled),
                max(1, cfg.dasp.v17_iwms_query_size),
            )
            if query_count < len(current_unlabeled):
                query_indices = torch.linspace(
                    0,
                    len(current_unlabeled) - 1,
                    steps=query_count,
                ).round().long()
                current_query = current_unlabeled[query_indices]
            else:
                current_query = current_unlabeled
            historical = memory.tensors(self.device)
            similarity_replay = (
                select_uncertainty_weighted_similarity_replay(
                    model,
                    historical[0],
                    historical[1],
                    current_query,
                    list(model.layer_names),
                    replay_size=(
                        (
                            cfg.dasp.fast_strong_replay_size
                            if strong_event
                            else cfg.dasp.fast_fast_replay_size
                        )
                        if method in FAST_DASP_METHODS
                        else (
                            cfg.dasp.v17_strong_replay_size
                            if strong_event
                            else cfg.dasp.v17_fast_replay_size
                        )
                    ),
                    device=self.device,
                    seed=(
                        cfg.data.random_seed
                        + item.window.index * 104729
                        + 43
                    ),
                    temperature=cfg.dasp.v17_iwms_temperature,
                    confidence_power=cfg.dasp.v17_iwms_confidence_power,
                    uniform_mix=(
                        cfg.dasp.safe_replay_uniform_mix
                        if method in SAFE_REPLAY_METHODS
                        else cfg.dasp.v17_iwms_uniform_mix
                    ),
                    max_uniform_mix=(
                        (
                            cfg.dasp.safe_replay_uniform_mix
                            if method in SAFE_REPLAY_METHODS
                            else cfg.dasp.v18_replay_max_uniform_mix
                        )
                        if method in V18_METHODS
                        else None
                    ),
                    js_threshold=cfg.dasp.v18_replay_js_threshold,
                    batch_size=max(1, cfg.training.batch_size),
                )
                if historical is not None
                else None
            )
            update_x = train_x.to(self.device)
            update_y = train_y.to(self.device)
            base_new_weight = (
                cfg.dasp.v17_strong_new_sample_weight
                if strong_event
                else cfg.dasp.v17_new_sample_weight
            )
            if method in V18_METHODS:
                sample_weights = _surprise_class_balanced_weights(
                    model,
                    train_x,
                    train_y,
                    base_new_weight,
                    base_new_weight
                    * max(1.0, cfg.dasp.v18_class_weight_multiplier),
                    self.device,
                ).to(self.device)
                row["v18_current_weight_mean"] = float(
                    sample_weights.mean().item()
                )
                row["v18_current_weight_max"] = float(
                    sample_weights.max().item()
                )
            else:
                sample_weights = torch.full(
                    (len(update_y),),
                    base_new_weight,
                    dtype=torch.float32,
                    device=self.device,
                )
            projection_memory = memory_probe
            if similarity_replay is not None:
                update_x = torch.cat([update_x, similarity_replay.x])
                update_y = torch.cat([update_y, similarity_replay.y])
                sample_weights = torch.cat(
                    [
                        sample_weights,
                        torch.ones(
                            len(similarity_replay.y),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                    ]
                )
                projection_memory = (
                    similarity_replay.x,
                    similarity_replay.y,
                )
            v4_logit_adjustment = (
                _causal_logit_adjustment(
                    self._v4_label_history,
                    n_classes=int(model.classifier.out_features),
                    tau=cfg.dasp.v4_logit_adjust_tau,
                    smoothing=cfg.dasp.v4_logit_prior_smoothing,
                    prior_floor=cfg.dasp.v4_logit_prior_floor,
                    device=self.device,
                )
                if method in V4_PRIOR_METHODS
                else None
            )
            if method in V4_METHODS:
                history_counts = (
                    torch.bincount(
                        self._v4_label_history.long(),
                        minlength=int(model.classifier.out_features),
                    )
                    if self._v4_label_history is not None
                    else torch.zeros(
                        int(model.classifier.out_features),
                        dtype=torch.long,
                    )
                )
                row["v4_observed_label_history"] = "|".join(
                    str(int(value)) for value in history_counts.tolist()
                )
                row["v4_logit_adjustment"] = (
                    "|".join(
                        f"{float(value):.6f}"
                        for value in v4_logit_adjustment.detach().cpu()
                    )
                    if v4_logit_adjustment is not None
                    else ""
                )
            row["gradient_evaluations"] = _train_dasp(
                candidate,
                update_x,
                update_y,
                plasticities,
                cfg,
                self.device,
                memory,
                projection_memory=projection_memory,
                eta_ceiling=cfg.dasp.eta_ceiling,
                strict_freeze=True,
                sample_weights=sample_weights,
                epochs=(
                    (
                        cfg.dasp.fast_strong_epochs
                        if strong_event
                        else cfg.dasp.fast_fast_epochs
                    )
                    if method in FAST_DASP_METHODS
                    else (
                        cfg.dasp.v17_strong_epochs
                        if strong_event
                        else cfg.dasp.v17_fast_epochs
                    )
                ),
                # V4 already rehearses informative balanced samples in the
                # update batch.  Omitting the second projection gradient
                # removes a redundant backward pass and is part of the
                # preregistered efficiency ablation.
                use_projection_memory=(method not in V4_METHODS),
                parameter_groups=parameter_groups,
                logit_adjustment=v4_logit_adjustment,
            )
            if (
                method in V3_BACKBONE_METHODS
                and method not in V4_METHODS
                and strong_event
                and performance_event
                and v3_covariate_event
            ):
                row.update(
                    _v3_prior_calibration(
                        candidate,
                        train_x,
                        train_y,
                        self.device,
                    )
                )
            candidate.freeze_backbone()
            commit_function = (
                _safe_soft_commit
                if method in SAFE_GATE_METHODS
                else _v17_soft_commit
            )
            chosen, alpha, diagnostics = commit_function(
                fallback,
                candidate,
                (guard_x, guard_y),
                memory_guard,
                cfg,
                self.device,
                extra_parameter_names=extra_parameter_names,
                alphas=(
                    cfg.dasp.fast_commit_alphas
                    if method in FAST_DASP_METHODS
                    else None
                ),
            )
            row.update(diagnostics)
            checkpoint_restored = False
            if (
                alpha <= 0.0
                and method in V3_CHECKPOINT_METHODS
            ):
                checkpoint_restored, checkpoint_diagnostics = (
                    _restore_v3_checkpoint(
                        chosen,
                        self._v3_checkpoints,
                        recent=(guard_x, guard_y),
                        memory=memory_guard,
                        cfg=cfg,
                        device=self.device,
                        require_current_guard=True,
                    )
                )
                row.update(checkpoint_diagnostics)
            model.load_state_dict(chosen.state_dict())
            model.set_adapter_rank(
                chosen.adapter_rank,
                trainable_rank=chosen.adapter_trainable_rank,
            )
            if method in FAST_DASP_METHODS:
                self._refresh_fast_inference_model(model)
            if alpha > 0.0:
                accepted = "dasp" if strong_event else "light"
                if method in SAFE_V2_METHODS:
                    self._safe_label_buffer = None
                row["selected_candidate"] = (
                    (
                        "v18_anchor_modulated_adapter"
                        if method in V18_METHODS
                        else "v17_anchor_adapter"
                    )
                    if item.recurrence
                    else (
                        (
                            "v18_strong_modulated_adapter"
                            if method in V18_METHODS
                            else "v17_strong_adapter"
                        )
                        if strong_event
                        else (
                            "v18_fast_modulated_adapter"
                            if method in V18_METHODS
                            else "v17_fast_adapter"
                        )
                    )
                )
            else:
                accepted = "none"
                row["selected_candidate"] = (
                    "v3_checkpoint_rollback"
                    if checkpoint_restored
                    else (
                        "v18_frozen"
                        if method in V18_METHODS
                        else "v17_frozen"
                    )
                )

            if method in V18_METHODS:
                # Refresh from all acquired labels, including the current
                # holdout, only after the commit decision has been made.
                refresh_x, refresh_y = queried_x, queried_y
            else:
                refresh_x = torch.cat([train_x, guard_x])
                refresh_y = torch.cat([train_y, guard_y])
            ranked_refresh_layers = sorted(
                model.layer_names,
                key=lambda name: float(
                    event_decision.requirements.get(name, 0.0)
                ),
                reverse=True,
            )
            refresh_layers = ranked_refresh_layers[
                : min(2, len(ranked_refresh_layers))
            ]
            refresh_plasticities = {
                name: (
                    max(
                        0.1,
                        float(event_decision.requirements.get(name, 0.0)),
                    )
                    if name in refresh_layers
                    else 0.0
                )
                for name in model.layer_names
            }
            refresh_plasticities["classifier"] = 0.0
            historical = memory.tensors(self.device)
            if historical is not None and (
                method not in SAFE_GATE_METHODS or alpha > 0.0
            ):
                refresh = propose_layer_adaptive_memory(
                    model,
                    historical[0],
                    historical[1],
                    refresh_x,
                    refresh_y,
                    refresh_plasticities,
                    capacity=cfg.dasp.memory_size,
                    label_budget=min(
                        cfg.dasp.v17_refresh_size, len(refresh_y)
                    ),
                    novelty_weight=cfg.dasp.v11_novelty_weight,
                    interference_weight=0.0,
                    balance_weight=cfg.dasp.v11_balance_weight,
                    redundancy_weight=cfg.dasp.v11_redundancy_weight,
                    min_per_class=cfg.dasp.v11_min_per_class,
                    device=self.device,
                    batch_size=max(1, cfg.training.batch_size),
                )
                memory.replace(refresh.x, refresh.y)
                memory_managed = True
                row.update(
                    {
                        "buffer_inserted": refresh.inserted,
                        "buffer_forgotten": refresh.evicted,
                        "buffer_size": len(memory),
                        "v17_refresh_inserted": refresh.inserted,
                        "v17_refresh_layers": "|".join(
                            refresh.active_layers
                        ),
                        "v17_refresh_mean_novelty": refresh.mean_novelty,
                        "v17_refresh_mean_balance": refresh.mean_balance,
                    }
                )
            if method in V18_METHODS and (
                method not in SAFE_GATE_METHODS or alpha > 0.0
            ):
                self._v18_guard_history = _update_v18_guard_history(
                    self._v18_guard_history,
                    queried_x,
                    queried_y,
                    (
                        cfg.dasp.fast_guard_history_size
                        if method in FAST_DASP_METHODS
                        else cfg.dasp.v18_guard_history_size
                    ),
                    cfg.data.random_seed
                    + item.window.index * 104729
                    + 71,
                )
            if method in V3_MEMORY_METHODS:
                updated_memory = _method_true_label_memory_update(
                    method,
                    model,
                    memory.tensors(torch.device("cpu")),
                    item.window.x[query_indices],
                    item.window.y[query_indices],
                    cfg=cfg,
                    device=self.device,
                    seed=(
                        cfg.data.random_seed
                        + item.window.index * 104729
                        + 67
                    ),
                )
                memory.replace(*updated_memory)
                memory_managed = True
                memory_counts = torch.bincount(
                    updated_memory[1].long(),
                    minlength=int(model.classifier.out_features),
                )
                row["v3_memory_per_class"] = "|".join(
                    str(int(value)) for value in memory_counts.tolist()
                )
            row.update(
                {
                    "v17_path": "strong" if strong_event else "fast",
                    "v17_active_rank": model.adapter_rank,
                    "v17_trainable_rank": trainable_rank,
                    "v17_detector_triggered": event_decision.triggered,
                    "v17_performance_triggered": performance_event,
                    "v17_recurrence_triggered": item.recurrence,
                    "v17_selected_base_layers": "|".join(
                        selected_base_layers
                    ),
                    "v17_raw_strong_event": raw_strong_event,
                    "v17_strong_remaining": self._v17_strong_remaining,
                    "v17_delayed_macro_f1": performance["macro_f1"],
                    "v17_delayed_attack_recall": performance[
                        "attack_recall"
                    ],
                    "v17_reference_macro_f1": reference_macro,
                    "v17_reference_attack_recall": reference_recall,
                    "v17_macro_f1_drop": macro_drop,
                    "v17_attack_recall_drop": recall_drop,
                    "v17_iwms_mean_similarity": (
                        similarity_replay.mean_similarity
                        if similarity_replay is not None
                        else np.nan
                    ),
                    "v17_iwms_replay_size": (
                        len(similarity_replay.y)
                        if similarity_replay is not None
                        else 0
                    ),
                    "v17_iwms_selected_per_class": (
                        "|".join(
                            str(value)
                            for value in similarity_replay.selected_per_class
                        )
                        if similarity_replay is not None
                        else ""
                    ),
                    "v18_iwms_predicted_per_class": (
                        "|".join(
                            str(value)
                            for value in similarity_replay.predicted_per_class
                        )
                        if similarity_replay is not None
                        else ""
                    ),
                    "v18_iwms_effective_uniform_mix": (
                        similarity_replay.effective_uniform_mix
                        if similarity_replay is not None
                        else np.nan
                    ),
                    "v18_iwms_prior_js_divergence": (
                        similarity_replay.prior_js_divergence
                        if similarity_replay is not None
                        else np.nan
                    ),
                    "v18_iwms_mean_confidence": (
                        similarity_replay.mean_confidence
                        if similarity_replay is not None
                        else np.nan
                    ),
                    "v18_guard_history_size": (
                        len(self._v18_guard_history[1])
                        if self._v18_guard_history is not None
                        else 0
                    ),
                }
            )
        elif method in V3_METHODS:
            assert self._stateful_optimizer is not None
            if method == "stateful_online":
                row["gradient_evaluations"] = _train_stateful_online(
                    model,
                    self._stateful_optimizer,
                    train_x,
                    train_y,
                    cfg,
                    self.device,
                    item.window.index,
                )
                accepted = "online"
            elif method == "compute_online_v3":
                if item.decision.triggered:
                    probe_steps = _spend_probe_compute_budget(
                        model,
                        self._stateful_optimizer,
                        train_x,
                        train_y,
                        len(model.layer_names),
                        cfg,
                        self.device,
                    )
                    train_steps = _train_stateful_online(
                        model,
                        self._stateful_optimizer,
                        train_x,
                        train_y,
                        cfg,
                        self.device,
                        item.window.index,
                        epochs_multiplier=cfg.dasp.compute_match_epochs,
                    )
                    row["gradient_evaluations"] = probe_steps + train_steps
                else:
                    row["gradient_evaluations"] = _train_stateful_online(
                        model,
                        self._stateful_optimizer,
                        train_x,
                        train_y,
                        cfg,
                        self.device,
                        item.window.index,
                    )
                accepted = "online"
            elif not item.decision.triggered:
                row["gradient_evaluations"] = _train_stateful_online(
                    model,
                    self._stateful_optimizer,
                    train_x,
                    train_y,
                    cfg,
                    self.device,
                    item.window.index,
                )
                accepted = "online"
            else:
                gains, costs, plasticities = _counterfactual_utilities(
                    model,
                    train_x,
                    train_y,
                    guard_x,
                    guard_y,
                    memory_probe,
                    cfg,
                    self.device,
                )
                if method == "utility_global_v3":
                    scalar = float(np.mean(list(plasticities.values())))
                    plasticities = {name: scalar for name in plasticities}
                for name in model.layer_names:
                    row[f"adaptation_gain_{name}"] = gains[name]
                    row[f"forgetting_cost_{name}"] = costs[name]
                    row[f"utility_{name}"] = (
                        gains[name] - cfg.dasp.utility_forgetting_weight * costs[name]
                    )
                    row[f"plasticity_{name}"] = plasticities[name]

                online_candidate = copy.deepcopy(model)
                utility_candidate = copy.deepcopy(model)
                online_optimizer = _clone_stateful_optimizer(
                    self._stateful_optimizer, online_candidate, cfg
                )
                utility_optimizer = _clone_stateful_optimizer(
                    self._stateful_optimizer, utility_candidate, cfg
                )
                online_steps = _train_stateful_online(
                    online_candidate,
                    online_optimizer,
                    train_x,
                    train_y,
                    cfg,
                    self.device,
                    item.window.index,
                )
                utility_steps = _train_dasp_stateful(
                    utility_candidate,
                    utility_optimizer,
                    train_x,
                    train_y,
                    plasticities,
                    cfg,
                    self.device,
                    memory_probe,
                    item.window.index,
                )
                use_utility, diagnostics = _dual_guard(
                    online_candidate,
                    utility_candidate,
                    (guard_x, guard_y),
                    memory_guard,
                    cfg,
                    self.device,
                )
                row.update(diagnostics)
                row["guard_lcb"] = diagnostics["guard_combined_lcb"]
                row["guard_mean_delta"] = diagnostics["guard_recent_mean_delta"]
                chosen_model = utility_candidate if use_utility else online_candidate
                chosen_optimizer = utility_optimizer if use_utility else online_optimizer
                model.load_state_dict(chosen_model.state_dict())
                self._stateful_optimizer = _clone_stateful_optimizer(
                    chosen_optimizer, model, cfg
                )
                row["gradient_evaluations"] = (
                    len(model.layer_names) + online_steps + utility_steps
                )
                accepted = "dasp" if use_utility else "online"
        elif method in (
            V7_METHODS
            | V8_METHODS
            | V9_METHODS
            | V91_METHODS
            | V92_METHODS
            | V10_METHODS
        ):
            event_decision = (
                item.current_decision
                if (
                    method in V15_METHODS | V16_METHODS
                    and item.current_decision is not None
                )
                else item.decision
            )
            if method in V15_METHODS | V16_METHODS:
                row["v15_event_context_severity"] = event_decision.severity
                row["v15_event_context_triggered"] = event_decision.triggered
            replay_size = max(1, cfg.dasp.v7_replay_size)
            if cfg.dasp.v7_balanced_replay:
                replay_batch = memory.sample_balanced(replay_size, self.device)
            elif replay_size == 2 * cfg.dasp.memory_batch_size:
                if memory_probe is None:
                    replay_batch = memory_guard
                elif memory_guard is None:
                    replay_batch = memory_probe
                else:
                    replay_batch = (
                        torch.cat([memory_probe[0], memory_guard[0]]),
                        torch.cat([memory_probe[1], memory_guard[1]]),
                    )
            else:
                replay_batch = memory.sample(replay_size, self.device)
            v12_base_guard_metrics: dict[str, float] | None = None
            v12_warning_event = False
            if method in V12_METHODS:
                v12_base_guard_metrics = _guard_classification(
                    model, (guard_x, guard_y), self.device
                )
                v12_warning_event = _v12_warning_eligible(
                    event_decision,
                    severity_threshold=cfg.drift.severity_threshold,
                    guard_macro_f1=v12_base_guard_metrics["macro_f1"],
                    guard_attack_recall=v12_base_guard_metrics["attack_recall"],
                    minimum_macro_f1=cfg.dasp.v12_guard_min_macro_f1,
                    minimum_attack_recall=cfg.dasp.v12_guard_min_attack_recall,
                )
                row.update({
                    "v12_warning_eligible": v12_warning_event,
                    "v12_warning_attempted": v12_warning_event,
                    "v12_base_guard_macro_f1": (
                        v12_base_guard_metrics["macro_f1"]
                    ),
                    "v12_base_guard_attack_recall": (
                        v12_base_guard_metrics["attack_recall"]
                    ),
                })
            v16_performance_event = False
            v16_raw_performance_decline = False
            v16_oracle_event = False
            if method in V16_METHODS:
                performance = _guard_classification(
                    model, (guard_x, guard_y), self.device
                )
                history = self._v16_performance_history[
                    -max(1, cfg.dasp.v16_performance_reference_windows) :
                ]
                if history:
                    reference_macro = float(
                        np.median([value[0] for value in history])
                    )
                    finite_recalls = [
                        value[1] for value in history if np.isfinite(value[1])
                    ]
                    reference_recall = (
                        float(np.median(finite_recalls))
                        if finite_recalls
                        else np.nan
                    )
                else:
                    reference_macro = performance["macro_f1"]
                    reference_recall = performance["attack_recall"]
                macro_drop = reference_macro - performance["macro_f1"]
                recall_drop = (
                    reference_recall - performance["attack_recall"]
                    if (
                        np.isfinite(reference_recall)
                        and np.isfinite(performance["attack_recall"])
                    )
                    else np.nan
                )
                enough_history = (
                    len(history) >= cfg.dasp.v16_performance_min_history
                )
                v16_raw_performance_decline = bool(
                    enough_history
                    and (
                        macro_drop >= cfg.dasp.v16_macro_f1_drop
                        or (
                            np.isfinite(recall_drop)
                            and recall_drop
                            >= cfg.dasp.v16_attack_recall_drop
                        )
                    )
                )
                if self._v16_performance_cooldown > 0:
                    self._v16_performance_cooldown -= 1
                v16_performance_event = bool(
                    method == "dasp_v16"
                    and v16_raw_performance_decline
                    and self._v16_performance_cooldown == 0
                )
                if v16_performance_event:
                    self._v16_performance_cooldown = max(
                        0, cfg.dasp.v16_performance_cooldown
                    )
                if not v16_raw_performance_decline:
                    self._v16_performance_history.append(
                        (
                            performance["macro_f1"],
                            performance["attack_recall"],
                        )
                    )
                v16_oracle_event = bool(
                    method == "dasp_v16_oracle"
                    and _v16_oracle_event(
                        cfg.data.scenario, item.window.index
                    )
                )
                row.update(
                    {
                        "v16_delayed_macro_f1": performance["macro_f1"],
                        "v16_delayed_attack_recall": performance[
                            "attack_recall"
                        ],
                        "v16_reference_macro_f1": reference_macro,
                        "v16_reference_attack_recall": reference_recall,
                        "v16_macro_f1_drop": macro_drop,
                        "v16_attack_recall_drop": recall_drop,
                        "v16_performance_history": len(history),
                        "v16_raw_performance_decline": (
                            v16_raw_performance_decline
                        ),
                        "v16_performance_triggered": (
                            v16_performance_event
                        ),
                        "v16_oracle_triggered": v16_oracle_event,
                    }
                )
            fallback = copy.deepcopy(model)
            if method in NO_FULL_UPDATE_METHODS:
                # v13 has no whole-model fallback update. Between confirmed
                # drifts it only refreshes bounded memory; a PSLA candidate
                # may update selected layers when the alarm is confirmed.
                fallback_steps = 0
            else:
                fallback_steps = _train_online(
                    fallback,
                    train_x,
                    train_y,
                    cfg,
                    self.device,
                    replay=replay_batch,
                )
            balanced_candidate: OnlineMLP | None = None
            balanced_diagnostics: dict[str, float] | None = None
            balanced_steps = 0
            if method in V8_METHODS:
                balanced_batch = memory.sample_balanced(replay_size, self.device)
                balanced_candidate = copy.deepcopy(model)
                balanced_steps = _train_online(
                    balanced_candidate,
                    train_x,
                    train_y,
                    cfg,
                    self.device,
                    replay=balanced_batch,
                )
                _, balanced_diagnostics = _dual_guard(
                    fallback,
                    balanced_candidate,
                    (guard_x, guard_y),
                    memory_guard,
                    cfg,
                    self.device,
                    class_balanced=cfg.dasp.v8_balanced_guard,
                )
            if method == "dasp_v16_no_detector":
                adaptation_event = False
            elif method == "dasp_v16_oracle":
                adaptation_event = v16_oracle_event
            elif method == "dasp_v16":
                adaptation_event = bool(
                    event_decision.triggered or v16_performance_event
                )
            else:
                adaptation_event = bool(
                    event_decision.triggered
                    or v12_warning_event
                    or (
                        method in V9_METHODS | V91_METHODS | V92_METHODS
                        and item.recurrence
                    )
                )
            if not adaptation_event:
                if method in LIGHT_UPDATE_METHODS:
                    context_decision = item.current_decision or item.decision
                    ranked = sorted(
                        context_decision.requirements.items(),
                        key=lambda pair: float(pair[1]),
                        reverse=True,
                    )
                    active_layers = [
                        name
                        for name, _ in ranked[
                            : max(1, cfg.dasp.v14_light_layers)
                        ]
                        if name in model.layer_names
                    ]
                    if not active_layers:
                        active_layers = list(model.layer_names[-1:])
                    light_plasticities = {
                        name: float(name in active_layers)
                        for name in model.layer_names
                    }
                    light_plasticities["classifier"] = 1.0
                    current_batch = _balanced_tensor_sample(
                        train_x,
                        train_y,
                        cfg.dasp.v14_light_current_size,
                        self.device,
                        cfg.data.random_seed
                        + item.window.index * 104729
                        + 401,
                    )
                    historical = memory.tensors(self.device)
                    light_replay = None
                    if (
                        historical is not None
                        and item.current_unlabeled_x is not None
                    ):
                        selected_replay = select_feature_similarity_replay(
                            model,
                            historical[0],
                            historical[1],
                            item.current_unlabeled_x,
                            active_layers,
                            replay_size=(
                                cfg.dasp.v15_light_replay_size
                                if method in V15_METHODS | V16_METHODS
                                else cfg.dasp.v14_light_replay_size
                            ),
                            device=self.device,
                            batch_size=max(1, cfg.training.batch_size),
                        )
                        if selected_replay is not None:
                            light_replay = (
                                selected_replay.x,
                                selected_replay.y,
                            )
                            row["v14_light_similarity"] = (
                                selected_replay.mean_similarity
                            )
                    if light_replay is None:
                        light_replay = memory.sample_balanced(
                            (
                                cfg.dasp.v15_light_replay_size
                                if method in V15_METHODS | V16_METHODS
                                else cfg.dasp.v14_light_replay_size
                            ),
                            self.device,
                        )
                    assert current_batch is not None
                    light_x, light_y = current_batch
                    if light_replay is not None:
                        light_x = torch.cat([light_x, light_replay[0]], dim=0)
                        light_y = torch.cat([light_y, light_replay[1]], dim=0)
                    light_candidate = copy.deepcopy(model)
                    light_steps = _train_dasp(
                        light_candidate,
                        light_x,
                        light_y,
                        light_plasticities,
                        cfg,
                        self.device,
                        memory,
                        projection_memory=light_replay,
                        eta_ceiling=1.0,
                        strict_freeze=True,
                        epochs=cfg.dasp.v14_light_epochs,
                        use_projection_memory=method not in V15_METHODS,
                    )
                    _, light_diagnostics = _dual_guard(
                        fallback,
                        light_candidate,
                        (guard_x, guard_y),
                        memory_guard,
                        cfg,
                        self.device,
                        class_balanced=True,
                        attack_tolerance=cfg.dasp.v10_attack_guard_tolerance,
                    )
                    fallback_metrics = _guard_classification(
                        fallback, (guard_x, guard_y), self.device
                    )
                    candidate_metrics = _guard_classification(
                        light_candidate, (guard_x, guard_y), self.device
                    )
                    light_diagnostics.update({
                        "guard_candidate_macro_f1": candidate_metrics[
                            "macro_f1"
                        ],
                        "guard_fallback_macro_f1": fallback_metrics[
                            "macro_f1"
                        ],
                        "guard_candidate_attack_recall": candidate_metrics[
                            "attack_recall"
                        ],
                        "guard_fallback_attack_recall": fallback_metrics[
                            "attack_recall"
                        ],
                        "guard_candidate_min_per_class_recall": (
                            candidate_metrics["min_per_class_recall"]
                        ),
                        "guard_fallback_min_per_class_recall": (
                            fallback_metrics["min_per_class_recall"]
                        ),
                    })
                    for key, value in candidate_metrics.items():
                        if key.startswith("recall_class_"):
                            light_diagnostics[
                                f"guard_candidate_{key}"
                            ] = value
                            light_diagnostics[
                                f"guard_fallback_{key}"
                            ] = fallback_metrics[key]
                    light_score = _v13_class_balanced_score(
                        light_diagnostics,
                        macro_f1_tolerance=cfg.dasp.v13_macro_f1_tolerance,
                        per_class_recall_tolerance=(
                            cfg.dasp.v13_per_class_recall_tolerance
                        ),
                    )
                    attack_lcb = light_diagnostics["guard_attack_lcb"]
                    light_accepted = bool(
                        light_diagnostics["guard_recent_lcb"]
                        >= -cfg.dasp.recent_guard_tolerance
                        and light_diagnostics["guard_combined_lcb"] > 0.0
                        and (
                            not np.isfinite(attack_lcb)
                            or attack_lcb
                            >= -cfg.dasp.v10_attack_guard_tolerance
                        )
                        and light_score is not None
                    )
                    model.load_state_dict(
                        (
                            light_candidate
                            if light_accepted
                            else fallback
                        ).state_dict()
                    )
                    row.update(light_diagnostics)
                    row["guard_lcb"] = light_diagnostics[
                        "guard_combined_lcb"
                    ]
                    row["guard_mean_delta"] = light_diagnostics[
                        "guard_recent_mean_delta"
                    ]
                    row["gradient_evaluations"] = light_steps
                    row["selected_candidate"] = (
                        f"light_{active_layers[0]}"
                        if light_accepted
                        else "frozen_fallback"
                    )
                    row["v14_light_attempted"] = True
                    row["v14_light_committed"] = light_accepted
                    row["v14_light_context_severity"] = (
                        context_decision.severity
                    )
                    row["v14_light_layers"] = "|".join(active_layers)
                    accepted = "light" if light_accepted else "none"
                else:
                    use_balanced = bool(
                        balanced_candidate is not None
                        and balanced_diagnostics is not None
                        and balanced_diagnostics["guard_recent_lcb"]
                        >= -cfg.dasp.recent_guard_tolerance
                        and balanced_diagnostics["guard_combined_lcb"] > 0.0
                    )
                    chosen = balanced_candidate if use_balanced else fallback
                    model.load_state_dict(chosen.state_dict())
                    row["gradient_evaluations"] = (
                        fallback_steps + balanced_steps
                    )
                    row["selected_candidate"] = (
                        "balanced_replay"
                        if use_balanced
                        else (
                            "frozen_fallback"
                            if method in NO_FULL_UPDATE_METHODS
                            else "replay_fallback"
                        )
                    )
                    if balanced_diagnostics is not None:
                        row.update(balanced_diagnostics)
                        row["guard_lcb"] = balanced_diagnostics[
                            "guard_combined_lcb"
                        ]
                        row["guard_mean_delta"] = balanced_diagnostics[
                            "guard_recent_mean_delta"
                        ]
                    accepted = (
                        "none"
                        if (
                            method in NO_FULL_UPDATE_METHODS
                            and not use_balanced
                        )
                        else "online"
                    )
            else:
                gains, costs, layer_plasticities = _counterfactual_utilities(
                    model,
                    train_x,
                    train_y,
                    guard_x,
                    guard_y,
                    memory_probe,
                    cfg,
                    self.device,
                )
                scalar = float(np.mean(list(layer_plasticities.values())))
                global_plasticities = {
                    name: scalar for name in layer_plasticities
                }
                for name in model.layer_names:
                    row[f"adaptation_gain_{name}"] = gains[name]
                    row[f"forgetting_cost_{name}"] = costs[name]
                    row[f"utility_{name}"] = (
                        gains[name] - cfg.dasp.utility_forgetting_weight * costs[name]
                    )
                    row[f"plasticity_{name}"] = layer_plasticities[name]

                candidate_replay = replay_batch
                if method in V9_METHODS | V91_METHODS:
                    candidate_replay = memory.sample(
                        max(1, cfg.dasp.v9_candidate_replay_size),
                        self.device,
                    )
                if (
                    cfg.dasp.v7_rehearsal_candidates
                    and candidate_replay is not None
                ):
                    candidate_x = torch.cat(
                        [train_x.to(self.device), candidate_replay[0]], dim=0
                    )
                    candidate_y = torch.cat(
                        [train_y.to(self.device), candidate_replay[1]], dim=0
                    )
                else:
                    candidate_x, candidate_y = train_x, train_y
                candidate_sample_weights: torch.Tensor | None = None
                if method in V101_METHODS | V102_METHODS:
                    candidate_sample_weights = torch.ones(
                        len(candidate_x),
                        dtype=torch.float32,
                    )
                    if method in V103_METHODS:
                        new_weights = _surprise_class_balanced_weights(
                            model,
                            train_x,
                            train_y,
                            cfg.dasp.v10_1_new_sample_weight,
                            cfg.dasp.v10_3_max_new_sample_weight,
                            self.device,
                        )
                        candidate_sample_weights[: len(train_x)] = new_weights
                        row["v10_3_mean_new_sample_weight"] = float(
                            new_weights.mean().item()
                        )
                        row["v10_3_max_new_sample_weight"] = float(
                            new_weights.max().item()
                        )
                    else:
                        candidate_sample_weights[: len(train_x)] = max(
                            1.0,
                            cfg.dasp.v10_1_new_sample_weight,
                        )
                    row["v10_1_new_sample_weight"] = float(
                        cfg.dasp.v10_1_new_sample_weight
                    )
                v10_attack_tolerance = (
                    cfg.dasp.v10_attack_guard_tolerance
                    + (
                        (
                            cfg.dasp.v10_2_attack_guard_extra_budget
                            if method in V102_METHODS
                            else cfg.dasp.v10_1_attack_guard_extra_budget
                        )
                        * event_decision.severity
                        if method in V101_METHODS | V102_METHODS
                        else 0.0
                    )
                    if method in V10_METHODS
                    else None
                )
                if method in V101_METHODS | V102_METHODS:
                    row["v10_1_guard_lcb_floor"] = float(
                        -(
                            cfg.dasp.v10_2_guard_lcb_budget
                            if method in V102_METHODS
                            else cfg.dasp.v10_1_guard_lcb_budget
                        )
                        * event_decision.severity
                    )
                    row["v10_1_attack_tolerance"] = float(v10_attack_tolerance)

                candidates: dict[str, OnlineMLP] = {}
                diagnostics_by_name: dict[str, dict[str, float]] = {}
                v11_memory_proposals: dict[str, MemoryProposal] = {}
                gradient_evaluations = (
                    len(model.layer_names) + fallback_steps + balanced_steps
                )
                if balanced_candidate is not None and balanced_diagnostics is not None:
                    candidates["balanced_replay"] = balanced_candidate
                    diagnostics_by_name["balanced_replay"] = balanced_diagnostics
                candidate_specs: list[
                    tuple[str, dict[str, float], OnlineMLP | None]
                ]
                v10_progressive_specs: list[tuple[str, dict[str, float]]] = []
                if (
                    method in V9_METHODS | V91_METHODS | V92_METHODS
                    and item.recurrence
                ):
                    candidate_specs = [("anchor", layer_plasticities, None)]
                    row["v9_preselected_candidate"] = "anchor"
                elif method in V10_METHODS:
                    v10_progressive_specs, v10_diagnostics = (
                        _progressive_layer_candidates(
                            gains,
                            costs,
                            layer_plasticities,
                            list(model.adaptation_parameter_groups()),
                            cfg.dasp.utility_forgetting_weight,
                            cfg.dasp.v10_min_layer_utility,
                            (
                                cfg.dasp.v12_max_adaptive_layers
                                if method in V12_METHODS
                                else (
                                    cfg.dasp.v16_max_adaptive_layers
                                    if method in V16_METHODS
                                    else (
                                        cfg.dasp.v10_1_max_adaptive_layers
                                        if method in V101_METHODS | V102_METHODS
                                        else cfg.dasp.v10_max_adaptive_layers
                                    )
                                )
                            ),
                            (
                                cfg.dasp.v10_2_min_active_plasticity
                                if method in V102_METHODS
                                else cfg.dasp.v10_min_active_plasticity
                            ),
                            cfg.dasp.v10_classifier_plasticity,
                        )
                    )
                    row.update(v10_diagnostics)
                    row["v10_preselected_candidate"] = (
                        v10_progressive_specs[0][0]
                        if v10_progressive_specs
                        else "none"
                    )
                    candidate_specs = (
                        []
                        if not v10_progressive_specs
                        else [
                            (
                                v10_progressive_specs[0][0],
                                v10_progressive_specs[0][1],
                                copy.deepcopy(model),
                            )
                        ]
                    )
                elif method in V92_METHODS:
                    candidate_specs = [
                        ("global", global_plasticities, copy.deepcopy(model))
                    ]
                    row["v9_preselected_candidate"] = "global_lazy_first"
                elif method in V9_METHODS | V91_METHODS:
                    candidate_name, plasticities, v9_diagnostics = (
                        _select_v9_candidate(
                            gains,
                            costs,
                            layer_plasticities,
                            cfg.dasp.utility_forgetting_weight,
                            cfg.dasp.v9_min_candidate_utility,
                            cfg.dasp.v9_global_utility_ratio,
                        )
                    )
                    row.update(v9_diagnostics)
                    row["v9_preselected_candidate"] = (
                        "none" if candidate_name is None else candidate_name
                    )
                    if method in V91_METHODS and candidate_name is None:
                        candidate_name = "layer"
                        plasticities = dict(layer_plasticities)
                        row["v9_preselected_candidate"] = "layer_probe_fallback"
                    candidate_specs = (
                        []
                        if candidate_name is None
                        else [(candidate_name, plasticities, copy.deepcopy(model))]
                    )
                else:
                    candidate_specs = [
                        ("global", global_plasticities, copy.deepcopy(model)),
                        ("layer", layer_plasticities, copy.deepcopy(model)),
                    ]

                def v11_training_state(
                    candidate_name: str,
                    candidate_plasticities: dict[str, float],
                ) -> tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor | None,
                    tuple[torch.Tensor, torch.Tensor] | None,
                ]:
                    if method in NO_FULL_UPDATE_METHODS:
                        historical = memory.tensors(self.device)
                        replay = None
                        if historical is not None:
                            active_layers = [
                                name
                                for name, value in candidate_plasticities.items()
                                if name != "classifier" and value > 0.0
                            ]
                            if (
                                method in SIMILARITY_REPLAY_METHODS
                                and item.current_unlabeled_x is not None
                            ):
                                similarity_replay = select_feature_similarity_replay(
                                    model,
                                    historical[0],
                                    historical[1],
                                    item.current_unlabeled_x,
                                    active_layers,
                                    replay_size=cfg.dasp.v13_similarity_replay_size,
                                    device=self.device,
                                    batch_size=max(1, cfg.training.batch_size),
                                )
                                if similarity_replay is not None:
                                    replay = (
                                        similarity_replay.x,
                                        similarity_replay.y,
                                    )
                                    row[
                                        f"{candidate_name}_v13_similarity"
                                    ] = similarity_replay.mean_similarity
                                    row[
                                        f"{candidate_name}_v13_replay_layers"
                                    ] = "|".join(similarity_replay.active_layers)
                            if replay is None:
                                replay = memory.sample_balanced(
                                    cfg.dasp.v13_similarity_replay_size,
                                    self.device,
                                )
                        current_x = train_x.to(self.device)
                        current_y = train_y.to(self.device)
                        if replay is not None:
                            current_x = torch.cat([current_x, replay[0]], dim=0)
                            current_y = torch.cat([current_y, replay[1]], dim=0)
                        weights = torch.ones(len(current_x), dtype=torch.float32)
                        weights[: len(train_x)] = max(
                            1.0, cfg.dasp.v10_1_new_sample_weight
                        )
                        return current_x, current_y, weights, replay
                    if method not in V11_METHODS:
                        return (
                            candidate_x,
                            candidate_y,
                            candidate_sample_weights,
                            memory_probe,
                        )
                    historical = memory.tensors(self.device)
                    assert historical is not None
                    proposal = propose_layer_adaptive_memory(
                        model,
                        historical[0],
                        historical[1],
                        train_x,
                        train_y,
                        candidate_plasticities,
                        capacity=cfg.dasp.memory_size,
                        label_budget=cfg.dasp.v11_label_budget,
                        novelty_weight=cfg.dasp.v11_novelty_weight,
                        interference_weight=(
                            cfg.dasp.v11_interference_weight
                            if method in V11_FULL_METHODS
                            else 0.0
                        ),
                        balance_weight=cfg.dasp.v11_balance_weight,
                        redundancy_weight=cfg.dasp.v11_redundancy_weight,
                        min_per_class=cfg.dasp.v11_min_per_class,
                        device=self.device,
                        batch_size=max(1, cfg.training.batch_size),
                    )
                    v11_memory_proposals[candidate_name] = proposal
                    proposal_replay = _balanced_tensor_sample(
                        proposal.x,
                        proposal.y,
                        cfg.dasp.v11_candidate_replay_size,
                        self.device,
                        cfg.data.random_seed
                        + item.window.index * 104729
                        + len(v11_memory_proposals) * 17,
                    )
                    proposal_projection = _balanced_tensor_sample(
                        proposal.x,
                        proposal.y,
                        cfg.dasp.memory_batch_size,
                        self.device,
                        cfg.data.random_seed
                        + item.window.index * 104729
                        + len(v11_memory_proposals) * 31,
                    )
                    current_x, current_y = train_x.to(self.device), train_y.to(self.device)
                    if (
                        cfg.dasp.v7_rehearsal_candidates
                        and proposal_replay is not None
                    ):
                        current_x = torch.cat([current_x, proposal_replay[0]], dim=0)
                        current_y = torch.cat([current_y, proposal_replay[1]], dim=0)
                    weights = torch.ones(len(current_x), dtype=torch.float32)
                    weights[: len(train_x)] = max(
                        1.0, cfg.dasp.v10_1_new_sample_weight
                    )
                    if method in V11_RECOVERY_METHODS:
                        labels, inverse, counts = torch.unique(
                            current_y.detach().cpu(),
                            return_inverse=True,
                            return_counts=True,
                        )
                        del labels
                        beta = float(
                            np.clip(
                                cfg.dasp.v11_effective_number_beta,
                                0.0,
                                1.0 - 1e-9,
                            )
                        )
                        effective = (
                            1.0 - torch.pow(
                                torch.full_like(counts, beta, dtype=torch.float32),
                                counts.float(),
                            )
                        ) / max(1e-9, 1.0 - beta)
                        class_weights = effective.clamp_min(1e-12).reciprocal()
                        class_weights = class_weights / class_weights.mean()
                        weights *= class_weights[inverse]
                    return current_x, current_y, weights, proposal_projection

                recovery_fallback_metrics = (
                    _guard_classification(
                        fallback, (guard_x, guard_y), self.device
                    )
                    if (
                        method in V11_RECOVERY_METHODS
                        or v12_warning_event
                        or method in NO_FULL_UPDATE_METHODS
                    )
                    else None
                )
                for candidate_name, plasticities, candidate_base in candidate_specs:
                    if candidate_name == "anchor":
                        assert self._anchor_model is not None
                        candidate = _soft_reset_toward_reference(
                            model,
                            self._anchor_model,
                            cfg.dasp.recurrence_anchor_strength,
                        ).to(self.device)
                    else:
                        assert candidate_base is not None
                        candidate = candidate_base
                    (
                        current_candidate_x,
                        current_candidate_y,
                        current_sample_weights,
                        current_projection_memory,
                    ) = v11_training_state(candidate_name, plasticities)
                    gradient_evaluations += _train_dasp(
                        candidate,
                        current_candidate_x,
                        current_candidate_y,
                        plasticities,
                        cfg,
                        self.device,
                        memory,
                        projection_memory=current_projection_memory,
                        eta_ceiling=cfg.dasp.eta_ceiling,
                        strict_freeze=method in V10_METHODS,
                        sample_weights=current_sample_weights,
                        epochs=(
                            cfg.dasp.v11_recovery_epochs
                            if method in V11_RECOVERY_METHODS
                            else (
                                cfg.dasp.v10_2_adaptation_epochs
                                if method in V102_METHODS
                                else None
                            )
                        ),
                        focal_gamma=(
                            cfg.dasp.v11_focal_gamma
                            if method in V11_RECOVERY_METHODS
                            else 0.0
                        ),
                    )
                    _, diagnostics = _dual_guard(
                        fallback,
                        candidate,
                        (guard_x, guard_y),
                        memory_guard,
                        cfg,
                        self.device,
                        class_balanced=(
                            (
                                method in V8_METHODS
                                and cfg.dasp.v8_balanced_guard
                            )
                            or (
                                method in V10_METHODS
                                and cfg.dasp.v10_class_balanced_guard
                            )
                        ),
                        attack_tolerance=(
                            v10_attack_tolerance
                            if method in V10_METHODS
                            else None
                        ),
                    )
                    if recovery_fallback_metrics is not None:
                        candidate_metrics = _guard_classification(
                            candidate, (guard_x, guard_y), self.device
                        )
                        diagnostics.update({
                            "guard_candidate_macro_f1": candidate_metrics[
                                "macro_f1"
                            ],
                            "guard_candidate_attack_recall": candidate_metrics[
                                "attack_recall"
                            ],
                            "guard_fallback_macro_f1": recovery_fallback_metrics[
                                "macro_f1"
                            ],
                            "guard_fallback_attack_recall": (
                                recovery_fallback_metrics["attack_recall"]
                            ),
                            "guard_candidate_min_per_class_recall": (
                                candidate_metrics["min_per_class_recall"]
                            ),
                            "guard_fallback_min_per_class_recall": (
                                recovery_fallback_metrics[
                                    "min_per_class_recall"
                                ]
                            ),
                        })
                        for key, value in candidate_metrics.items():
                            if key.startswith("recall_class_"):
                                diagnostics[
                                    f"guard_candidate_{key}"
                                ] = value
                                diagnostics[
                                    f"guard_fallback_{key}"
                                ] = recovery_fallback_metrics[key]
                    candidates[candidate_name] = candidate
                    diagnostics_by_name[candidate_name] = diagnostics
                    for key, value in diagnostics.items():
                        row[f"{candidate_name}_{key}"] = value

                if method in V92_METHODS and not item.recurrence:
                    global_diagnostics = diagnostics_by_name["global"]
                    strong_global = bool(
                        global_diagnostics["guard_recent_lcb"]
                        >= -cfg.dasp.recent_guard_tolerance
                        and global_diagnostics["guard_combined_lcb"]
                        >= cfg.dasp.v9_lazy_accept_lcb
                    )
                    row["v9_lazy_global_accepted"] = strong_global
                    if not strong_global:
                        layer_candidate = copy.deepcopy(model)
                        gradient_evaluations += _train_dasp(
                            layer_candidate,
                            candidate_x,
                            candidate_y,
                            layer_plasticities,
                            cfg,
                            self.device,
                            memory,
                            projection_memory=memory_probe,
                            eta_ceiling=cfg.dasp.eta_ceiling,
                        )
                        _, layer_diagnostics = _dual_guard(
                            fallback,
                            layer_candidate,
                            (guard_x, guard_y),
                            memory_guard,
                            cfg,
                            self.device,
                        )
                        candidates["layer"] = layer_candidate
                        diagnostics_by_name["layer"] = layer_diagnostics
                        for key, value in layer_diagnostics.items():
                            row[f"layer_{key}"] = value

                if method in V10_METHODS:
                    first_name = (
                        v10_progressive_specs[0][0]
                        if v10_progressive_specs
                        else None
                    )
                    first_diagnostics = (
                        diagnostics_by_name.get(first_name)
                        if first_name is not None
                        else None
                    )

                    def strong_v10_guard(
                        diagnostics: dict[str, float] | None,
                    ) -> bool:
                        return bool(
                            diagnostics is not None
                            and diagnostics["guard_recent_lcb"]
                            >= -cfg.dasp.recent_guard_tolerance
                            and diagnostics["guard_combined_lcb"]
                            >= cfg.dasp.v10_lazy_accept_lcb
                            and (
                                not np.isfinite(diagnostics["guard_attack_lcb"])
                                or diagnostics["guard_attack_lcb"]
                                >= -float(v10_attack_tolerance)
                            )
                        )

                    strong_first = strong_v10_guard(first_diagnostics)
                    strong_progressive = strong_first
                    force_full_expansion = bool(
                        method in NO_FULL_UPDATE_METHODS
                        or (
                            method in V12_METHODS
                            and v12_warning_event
                        )
                        or (
                            method in V102_METHODS
                            and event_decision.severity
                            >= cfg.dasp.v10_2_full_expansion_severity
                        )
                    )
                    should_expand = bool(
                        len(v10_progressive_specs) > 1
                        and (not strong_progressive or force_full_expansion)
                    )
                    row["v10_lazy_first_accepted"] = strong_first
                    row["v10_expanded"] = should_expand
                    evaluated_width = 1 if first_diagnostics is not None else 0
                    for expanded_name, expanded_plasticities in (
                        v10_progressive_specs[1:] if should_expand else []
                    ):
                        width = int(expanded_name.removeprefix("layer_top"))
                        if (
                            method not in V101_METHODS | V102_METHODS
                            and width > 2
                        ):
                            break
                        if (
                            method in V101_METHODS | V102_METHODS
                            and width >= 3
                            and method not in NO_FULL_UPDATE_METHODS
                            and not (
                                method in V12_METHODS
                                and v12_warning_event
                            )
                            and event_decision.severity
                            < (
                                cfg.dasp.v10_2_full_expansion_severity
                                if method in V102_METHODS
                                else cfg.dasp.v10_1_full_expansion_severity
                            )
                        ):
                            break
                        expanded_candidate = copy.deepcopy(model)
                        (
                            expanded_x,
                            expanded_y,
                            expanded_weights,
                            expanded_projection,
                        ) = v11_training_state(
                            expanded_name, expanded_plasticities
                        )
                        gradient_evaluations += _train_dasp(
                            expanded_candidate,
                            expanded_x,
                            expanded_y,
                            expanded_plasticities,
                            cfg,
                            self.device,
                            memory,
                            projection_memory=expanded_projection,
                            eta_ceiling=cfg.dasp.eta_ceiling,
                            strict_freeze=True,
                            sample_weights=expanded_weights,
                            epochs=(
                                cfg.dasp.v11_recovery_epochs
                                if method in V11_RECOVERY_METHODS
                                else (
                                    cfg.dasp.v10_2_adaptation_epochs
                                    if method in V102_METHODS
                                    else None
                                )
                            ),
                            focal_gamma=(
                                cfg.dasp.v11_focal_gamma
                                if method in V11_RECOVERY_METHODS
                                else 0.0
                            ),
                        )
                        _, expanded_diagnostics = _dual_guard(
                            fallback,
                            expanded_candidate,
                            (guard_x, guard_y),
                            memory_guard,
                            cfg,
                            self.device,
                            class_balanced=cfg.dasp.v10_class_balanced_guard,
                            attack_tolerance=v10_attack_tolerance,
                        )
                        if recovery_fallback_metrics is not None:
                            expanded_metrics = _guard_classification(
                                expanded_candidate,
                                (guard_x, guard_y),
                                self.device,
                            )
                            expanded_diagnostics.update({
                                "guard_candidate_macro_f1": expanded_metrics[
                                    "macro_f1"
                                ],
                                "guard_candidate_attack_recall": expanded_metrics[
                                    "attack_recall"
                                ],
                                "guard_fallback_macro_f1": (
                                    recovery_fallback_metrics["macro_f1"]
                                ),
                                "guard_fallback_attack_recall": (
                                    recovery_fallback_metrics["attack_recall"]
                                ),
                                "guard_candidate_min_per_class_recall": (
                                    expanded_metrics["min_per_class_recall"]
                                ),
                                "guard_fallback_min_per_class_recall": (
                                    recovery_fallback_metrics[
                                        "min_per_class_recall"
                                    ]
                                ),
                            })
                            for key, value in expanded_metrics.items():
                                if key.startswith("recall_class_"):
                                    expanded_diagnostics[
                                        f"guard_candidate_{key}"
                                    ] = value
                                    expanded_diagnostics[
                                        f"guard_fallback_{key}"
                                    ] = recovery_fallback_metrics[key]
                        candidates[expanded_name] = expanded_candidate
                        diagnostics_by_name[expanded_name] = expanded_diagnostics
                        evaluated_width = width
                        for key, value in expanded_diagnostics.items():
                            row[f"{expanded_name}_{key}"] = value
                        strong_progressive = strong_v10_guard(
                            expanded_diagnostics
                        )
                        if strong_progressive and not force_full_expansion:
                            break
                    if method in V101_METHODS | V102_METHODS:
                        row["v10_1_max_evaluated_width"] = float(evaluated_width)

                eligible: list[tuple[object, str]] = []
                for candidate_name, diagnostics in diagnostics_by_name.items():
                    combined_floor = (
                        -(
                            cfg.dasp.v10_2_guard_lcb_budget
                            if method in V102_METHODS
                            else cfg.dasp.v10_1_guard_lcb_budget
                        )
                        * event_decision.severity
                        if method in V101_METHODS | V102_METHODS
                        else 0.0
                    )
                    recent_improves = (
                        diagnostics["guard_recent_mean_delta"] > 0.0
                        if (
                            method in V101_METHODS | V102_METHODS
                            and method not in NO_FULL_UPDATE_METHODS
                        )
                        else True
                    )
                    attack_safe = (
                        method not in V10_METHODS
                        or not np.isfinite(diagnostics["guard_attack_lcb"])
                        or diagnostics["guard_attack_lcb"]
                        >= -float(v10_attack_tolerance)
                    )
                    recovery_safe = True
                    if method in V11_RECOVERY_METHODS:
                        candidate_recall = diagnostics[
                            "guard_candidate_attack_recall"
                        ]
                        fallback_recall = diagnostics[
                            "guard_fallback_attack_recall"
                        ]
                        if np.isfinite(candidate_recall) and np.isfinite(
                            fallback_recall
                        ):
                            recovery_safe = bool(
                                candidate_recall + 1e-12 >= fallback_recall
                            )
                            if (
                                event_decision.severity
                                >= cfg.dasp.v11_recovery_severity
                            ):
                                recovery_safe = bool(
                                    recovery_safe
                                    and candidate_recall
                                    >= cfg.dasp.v11_min_guard_attack_recall
                                )
                    if method in V12_METHODS and v12_warning_event:
                        candidate_recall = diagnostics.get(
                            "guard_candidate_attack_recall", np.nan
                        )
                        fallback_recall = diagnostics.get(
                            "guard_fallback_attack_recall", np.nan
                        )
                        recovery_safe = bool(
                            np.isfinite(candidate_recall)
                            and np.isfinite(fallback_recall)
                            and candidate_recall + 1e-12 >= fallback_recall
                        )
                    class_balanced_safe = True
                    class_balanced_score: (
                        tuple[float, ...] | None
                    ) = None
                    if method in NO_FULL_UPDATE_METHODS:
                        if method in V16_METHODS:
                            class_balanced_score = (
                                _v16_class_balanced_score(
                                    diagnostics,
                                    macro_f1_tolerance=(
                                        cfg.dasp.v13_macro_f1_tolerance
                                    ),
                                    attack_recall_tolerance=(
                                        cfg.dasp.v16_attack_recall_tolerance
                                    ),
                                    min_per_class_recall_tolerance=(
                                        cfg.dasp
                                        .v16_min_per_class_recall_tolerance
                                    ),
                                )
                            )
                        else:
                            class_balanced_score = (
                                _v13_class_balanced_score(
                                    diagnostics,
                                    macro_f1_tolerance=(
                                        cfg.dasp.v13_macro_f1_tolerance
                                    ),
                                    per_class_recall_tolerance=(
                                        cfg.dasp
                                        .v13_per_class_recall_tolerance
                                    ),
                                )
                            )
                        class_balanced_safe = (
                            class_balanced_score is not None
                        )
                    if (
                        diagnostics["guard_recent_lcb"]
                        >= -cfg.dasp.recent_guard_tolerance
                        and diagnostics["guard_combined_lcb"] > combined_floor
                        and recent_improves
                        and attack_safe
                        and recovery_safe
                        and class_balanced_safe
                    ):
                        eligible.append(
                            (
                                class_balanced_score
                                if method in NO_FULL_UPDATE_METHODS
                                else (
                                    diagnostics["guard_candidate_macro_f1"]
                                    if method in V11_RECOVERY_METHODS
                                    else (
                                        diagnostics["guard_recent_mean_delta"]
                                        if method in V102_METHODS
                                        else diagnostics["guard_combined_lcb"]
                                    )
                                ),
                                candidate_name,
                            )
                        )
                if eligible:
                    _, selected = max(eligible)
                    model.load_state_dict(candidates[selected].state_dict())
                    accepted = (
                        "online" if selected == "balanced_replay" else "dasp"
                    )
                    selected_diagnostics = diagnostics_by_name[selected]
                else:
                    selected = (
                        "frozen_fallback"
                        if method in NO_FULL_UPDATE_METHODS
                        else "replay_fallback"
                    )
                    model.load_state_dict(fallback.state_dict())
                    accepted = (
                        "none"
                        if method in NO_FULL_UPDATE_METHODS
                        else "online"
                    )
                    selected_diagnostics = (
                        max(
                            diagnostics_by_name.values(),
                            key=lambda values: values["guard_combined_lcb"],
                        )
                        if diagnostics_by_name
                        else None
                    )
                row["selected_candidate"] = selected
                if method in V11_METHODS:
                    selected_proposal = v11_memory_proposals.get(selected)
                    committed = selected_proposal is not None
                    row["v11_memory_committed"] = committed
                    if method in V12_METHODS:
                        row["v12_warning_committed"] = bool(
                            v12_warning_event and committed
                        )
                        row["v12_alarm_committed"] = bool(
                            event_decision.triggered and committed
                        )
                        row["v12_selected_width"] = (
                            int(selected.removeprefix("layer_top"))
                            if selected.startswith("layer_top")
                            else 0
                        )
                    if selected_proposal is not None:
                        memory.replace(selected_proposal.x, selected_proposal.y)
                        memory_managed = True
                        row.update({
                            "buffer_inserted": selected_proposal.inserted,
                            "buffer_forgotten": selected_proposal.evicted,
                            "buffer_size": len(memory),
                            "v11_mean_novelty": selected_proposal.mean_novelty,
                            "v11_mean_interference": (
                                selected_proposal.mean_interference
                            ),
                            "v11_mean_balance": selected_proposal.mean_balance,
                            "v11_active_layers": "|".join(
                                selected_proposal.active_layers
                            ),
                        })
                if selected_diagnostics is not None:
                    row.update(selected_diagnostics)
                    row["guard_lcb"] = selected_diagnostics[
                        "guard_combined_lcb"
                    ]
                    row["guard_mean_delta"] = selected_diagnostics[
                        "guard_recent_mean_delta"
                    ]
                row["gradient_evaluations"] = gradient_evaluations
        elif method in V6_METHODS:
            adaptation_pressure = max(
                item.decision.severity,
                1.0 if item.recurrence else 0.0,
            )
            row["adaptation_pressure"] = adaptation_pressure
            if not (item.decision.triggered or item.recurrence):
                row["gradient_evaluations"] = _train_online(
                    model, train_x, train_y, cfg, self.device
                )
                row["selected_candidate"] = "online"
                accepted = "online"
            else:
                forgetting_weight = _severity_weighted_value(
                    cfg.dasp.utility_forgetting_weight,
                    cfg.dasp.severity_forgetting_floor,
                    adaptation_pressure,
                )
                memory_weight = _severity_weighted_value(
                    cfg.dasp.memory_guard_weight,
                    cfg.dasp.severity_memory_guard_floor,
                    adaptation_pressure,
                )
                row["effective_forgetting_weight"] = forgetting_weight
                gains, costs, layer_plasticities = _counterfactual_utilities(
                    model,
                    train_x,
                    train_y,
                    guard_x,
                    guard_y,
                    memory_probe,
                    cfg,
                    self.device,
                    forgetting_weight=forgetting_weight,
                )
                scalar = float(np.mean(list(layer_plasticities.values())))
                global_plasticities = {
                    name: scalar for name in layer_plasticities
                }
                for name in model.layer_names:
                    row[f"adaptation_gain_{name}"] = gains[name]
                    row[f"forgetting_cost_{name}"] = costs[name]
                    row[f"utility_{name}"] = gains[name] - forgetting_weight * costs[name]
                    row[f"plasticity_{name}"] = layer_plasticities[name]

                online_candidate = copy.deepcopy(model)
                online_steps = _train_online(
                    online_candidate, train_x, train_y, cfg, self.device
                )
                candidates: dict[str, OnlineMLP] = {}
                diagnostics_by_name: dict[str, dict[str, float]] = {}
                gradient_evaluations = len(model.layer_names) + online_steps
                for candidate_name, plasticities in (
                    ("global", global_plasticities),
                    ("layer", layer_plasticities),
                ):
                    candidate = copy.deepcopy(model)
                    gradient_evaluations += _train_dasp(
                        candidate,
                        train_x,
                        train_y,
                        plasticities,
                        cfg,
                        self.device,
                        memory,
                        projection_memory=memory_probe,
                        eta_ceiling=cfg.dasp.eta_ceiling,
                    )
                    _, diagnostics = _dual_guard(
                        online_candidate,
                        candidate,
                        (guard_x, guard_y),
                        memory_guard,
                        cfg,
                        self.device,
                        memory_weight=memory_weight,
                    )
                    candidates[candidate_name] = candidate
                    diagnostics_by_name[candidate_name] = diagnostics

                if item.recurrence:
                    assert self._anchor_model is not None
                    anchor_candidate = _soft_reset_toward_reference(
                        model,
                        self._anchor_model,
                        cfg.dasp.recurrence_anchor_strength,
                    ).to(self.device)
                    gradient_evaluations += _train_dasp(
                        anchor_candidate,
                        train_x,
                        train_y,
                        layer_plasticities,
                        cfg,
                        self.device,
                        memory,
                        projection_memory=memory_probe,
                        eta_ceiling=cfg.dasp.eta_ceiling,
                    )
                    _, diagnostics = _dual_guard(
                        online_candidate,
                        anchor_candidate,
                        (guard_x, guard_y),
                        memory_guard,
                        cfg,
                        self.device,
                        memory_weight=memory_weight,
                    )
                    candidates["anchor"] = anchor_candidate
                    diagnostics_by_name["anchor"] = diagnostics

                eligible: list[tuple[float, str]] = []
                for candidate_name, diagnostics in diagnostics_by_name.items():
                    for key, value in diagnostics.items():
                        row[f"{candidate_name}_{key}"] = value
                    if (
                        diagnostics["guard_recent_lcb"]
                        >= -cfg.dasp.recent_guard_tolerance
                        and diagnostics["guard_combined_lcb"] > 0.0
                    ):
                        eligible.append(
                            (diagnostics["guard_combined_lcb"], candidate_name)
                        )
                if eligible:
                    _, selected = max(eligible)
                    model.load_state_dict(candidates[selected].state_dict())
                    accepted = "dasp"
                else:
                    selected = "online"
                    model.load_state_dict(online_candidate.state_dict())
                    accepted = "online"
                row["selected_candidate"] = selected
                selected_diagnostics = (
                    diagnostics_by_name[selected]
                    if selected in diagnostics_by_name
                    else max(
                        diagnostics_by_name.values(),
                        key=lambda values: values["guard_combined_lcb"],
                    )
                )
                row.update(selected_diagnostics)
                row["guard_lcb"] = selected_diagnostics["guard_combined_lcb"]
                row["guard_mean_delta"] = selected_diagnostics[
                    "guard_recent_mean_delta"
                ]
                row["gradient_evaluations"] = gradient_evaluations
        elif method in V5_METHODS:
            if not item.decision.triggered:
                row["gradient_evaluations"] = _train_online(
                    model, train_x, train_y, cfg, self.device
                )
                row["selected_candidate"] = "online"
                accepted = "online"
            else:
                gains, costs, layer_plasticities = _counterfactual_utilities(
                    model,
                    train_x,
                    train_y,
                    guard_x,
                    guard_y,
                    memory_probe,
                    cfg,
                    self.device,
                )
                scalar = float(np.mean(list(layer_plasticities.values())))
                global_plasticities = {
                    name: scalar for name in layer_plasticities
                }
                for name in model.layer_names:
                    row[f"adaptation_gain_{name}"] = gains[name]
                    row[f"forgetting_cost_{name}"] = costs[name]
                    row[f"utility_{name}"] = (
                        gains[name] - cfg.dasp.utility_forgetting_weight * costs[name]
                    )
                    row[f"plasticity_{name}"] = layer_plasticities[name]

                online_candidate = copy.deepcopy(model)
                online_steps = _train_online(
                    online_candidate, train_x, train_y, cfg, self.device
                )
                requested = (
                    ["global"]
                    if method == "utility_global_v5"
                    else ["layer"]
                    if method == "utility_layer_v5"
                    else ["global", "layer"]
                )
                candidates: dict[str, OnlineMLP] = {}
                diagnostics_by_name: dict[str, dict[str, float]] = {}
                gradient_evaluations = len(model.layer_names) + online_steps
                for candidate_name in requested:
                    candidate = copy.deepcopy(model)
                    plasticities = (
                        global_plasticities
                        if candidate_name == "global"
                        else layer_plasticities
                    )
                    gradient_evaluations += _train_dasp(
                        candidate,
                        train_x,
                        train_y,
                        plasticities,
                        cfg,
                        self.device,
                        memory,
                        projection_memory=memory_probe,
                    )
                    _, diagnostics = _dual_guard(
                        online_candidate,
                        candidate,
                        (guard_x, guard_y),
                        memory_guard,
                        cfg,
                        self.device,
                    )
                    candidates[candidate_name] = candidate
                    diagnostics_by_name[candidate_name] = diagnostics
                    if method in {"utility_portfolio_v5", "dasp"}:
                        for key, value in diagnostics.items():
                            row[f"{candidate_name}_{key}"] = value

                eligible = []
                for candidate_name, diagnostics in diagnostics_by_name.items():
                    recent_lcb = diagnostics["guard_recent_lcb"]
                    combined_lcb = diagnostics["guard_combined_lcb"]
                    if (
                        recent_lcb >= -cfg.dasp.recent_guard_tolerance
                        and combined_lcb > 0.0
                    ):
                        eligible.append((combined_lcb, candidate_name))
                if eligible:
                    _, selected = max(eligible)
                    model.load_state_dict(candidates[selected].state_dict())
                    accepted = "dasp"
                else:
                    selected = "online"
                    model.load_state_dict(online_candidate.state_dict())
                    accepted = "online"
                row["selected_candidate"] = selected
                selected_diagnostics = (
                    diagnostics_by_name[selected]
                    if selected in diagnostics_by_name
                    else max(
                        diagnostics_by_name.values(),
                        key=lambda values: values["guard_combined_lcb"],
                    )
                )
                row.update(selected_diagnostics)
                row["guard_lcb"] = selected_diagnostics["guard_combined_lcb"]
                row["guard_mean_delta"] = selected_diagnostics[
                    "guard_recent_mean_delta"
                ]
                row["gradient_evaluations"] = gradient_evaluations
        elif method == "online_ewc":
            assert self._ewc_reference is not None
            assert self._ewc_fisher is not None
            row["gradient_evaluations"] = _train_online_ewc(
                model,
                train_x,
                train_y,
                self._ewc_reference,
                self._ewc_fisher,
                cfg,
                self.device,
                item.window.index,
            )
            current_batch: tuple[torch.Tensor, torch.Tensor] = (
                train_x[: cfg.dasp.memory_batch_size].to(self.device),
                train_y[: cfg.dasp.memory_batch_size].to(self.device),
            )
            if memory_probe is None:
                fisher_batch = current_batch
            else:
                fisher_batch = (
                    torch.cat([memory_probe[0], current_batch[0]]),
                    torch.cat([memory_probe[1], current_batch[1]]),
                )
            new_fisher = _estimate_fisher(model, fisher_batch, self.device)
            decay = float(np.clip(cfg.dasp.ewc_decay, 0.0, 1.0))
            self._ewc_fisher = {
                name: decay * self._ewc_fisher[name] + (1.0 - decay) * value
                for name, value in new_fisher.items()
            }
            self._ewc_reference = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
            row["selected_candidate"] = "online_ewc"
            accepted = "online"
        elif method == "agem":
            row["gradient_evaluations"] = train_agem(
                model,
                train_x,
                train_y,
                memory,
                cfg,
                self.device,
                item.window.index,
            )
            row["selected_candidate"] = "agem"
            accepted = "online"
        elif method == "periodic_ft":
            interval = max(1, int(cfg.baselines.periodic_ft_interval))
            if (item.window.index - cfg.data.warmup_windows) % interval == 0:
                row["gradient_evaluations"] = _train_online(
                    model, train_x, train_y, cfg, self.device
                )
                row["selected_candidate"] = "periodic_ft"
                accepted = "online"
            else:
                row["selected_candidate"] = "frozen_fallback"
                accepted = "none"
        elif method in {"online", "online_mlp"}:
            row["gradient_evaluations"] = _train_online(
                model, train_x, train_y, cfg, self.device
            )
            row["selected_candidate"] = "online"
            accepted = "online"
        elif method in {"replay", "experience_replay"}:
            if memory_probe is None:
                replay_batch = memory_guard
            elif memory_guard is None:
                replay_batch = memory_probe
            else:
                replay_batch = (
                    torch.cat([memory_probe[0], memory_guard[0]]),
                    torch.cat([memory_probe[1], memory_guard[1]]),
                )
            row["gradient_evaluations"] = _train_online(
                model, train_x, train_y, cfg, self.device, replay=replay_batch
            )
            row["selected_candidate"] = "replay"
            accepted = "online"
        elif method in {"utility_global", "utility_dasp", "utility_full"}:
            if not item.decision.triggered:
                _train_online(model, train_x, train_y, cfg, self.device)
                accepted = "online"
            else:
                gains, costs, plasticities = _counterfactual_utilities(
                    model,
                    train_x,
                    train_y,
                    guard_x,
                    guard_y,
                    memory_probe,
                    cfg,
                    self.device,
                )
                if method == "utility_global":
                    scalar = float(np.mean(list(plasticities.values())))
                    plasticities = {name: scalar for name in plasticities}
                for name in model.layer_names:
                    row[f"adaptation_gain_{name}"] = gains[name]
                    row[f"forgetting_cost_{name}"] = costs[name]
                    row[f"utility_{name}"] = (
                        gains[name] - cfg.dasp.utility_forgetting_weight * costs[name]
                    )
                    row[f"plasticity_{name}"] = plasticities[name]
                if method == "utility_full":
                    online_candidate = copy.deepcopy(model)
                    utility_candidate = copy.deepcopy(model)
                    _train_online(online_candidate, train_x, train_y, cfg, self.device)
                    _train_dasp(
                        utility_candidate,
                        train_x,
                        train_y,
                        plasticities,
                        cfg,
                        self.device,
                        memory,
                        projection_memory=memory_probe,
                    )
                    use_utility, diagnostics = _dual_guard(
                        online_candidate,
                        utility_candidate,
                        (guard_x, guard_y),
                        memory_guard,
                        cfg,
                        self.device,
                    )
                    row.update(diagnostics)
                    row["guard_lcb"] = diagnostics["guard_combined_lcb"]
                    row["guard_mean_delta"] = diagnostics["guard_recent_mean_delta"]
                    chosen = utility_candidate if use_utility else online_candidate
                    model.load_state_dict(chosen.state_dict())
                    accepted = "dasp" if use_utility else "online"
                else:
                    _train_dasp(
                        model,
                        train_x,
                        train_y,
                        plasticities,
                        cfg,
                        self.device,
                        memory,
                        projection_memory=memory_probe,
                    )
                    accepted = "dasp"
        else:
            should_adapt = item.decision.triggered or method in {"global_protection", "conflict_only"}
            if not should_adapt:
                if method == "modular_dasp":
                    calibration_performance = _guard_classification(
                        model, (guard_x, guard_y), self.device
                    )
                    emergency_repair = bool(
                        calibration_performance["macro_f1"]
                        < cfg.dasp.v12_guard_min_macro_f1
                    )
                    if emergency_repair:
                        ranked = sorted(
                            conflicts.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                        active = [
                            name
                            for name, _ in ranked[: max(1, cfg.drift.top_k)]
                        ]
                        if not active:
                            active = list(model.layer_names[: max(1, cfg.drift.top_k)])
                        plasticities = {
                            name: (
                                min(1.0, max(cfg.dasp.v3_backbone_plasticity, conflicts.get(name, 0.0)))
                                if name in active
                                else 0.0
                            )
                            for name in model.layer_names
                        }
                        plasticities["classifier"] = 1.0
                        row["selected_candidate"] = "semi_local_repair"
                        row["localized_repair_modules"] = "|".join(active)
                        row["localized_repair_module_count"] = len(active)
                        repair_epochs = cfg.dasp.modular_emergency_epochs
                    else:
                        ranked = sorted(
                            conflicts.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                        active = [
                            name
                            for name, _ in ranked[: max(1, cfg.drift.top_k)]
                        ]
                        if not active:
                            active = list(model.layer_names[: max(1, cfg.drift.top_k)])
                        plasticities = {
                            name: (
                                cfg.dasp.modular_background_plasticity
                                if name in active
                                else 0.0
                            )
                            for name in model.layer_names
                        }
                        plasticities["classifier"] = 1.0
                        row["selected_candidate"] = "background_local_consolidation"
                        row["localized_repair_modules"] = "|".join(active)
                        row["localized_repair_module_count"] = len(active)
                        repair_epochs = cfg.dasp.modular_repair_epochs
                    row["modular_calibration_macro_f1"] = calibration_performance[
                        "macro_f1"
                    ]
                    row["modular_calibration_attack_recall"] = calibration_performance[
                        "attack_recall"
                    ]
                    for name, value in plasticities.items():
                        row[f"plasticity_{name}"] = value
                    rehearsal_x, rehearsal_y, rehearsal_weights = _localized_rehearsal_batch(
                        train_x,
                        train_y,
                        memory,
                        cfg,
                        self.device,
                    )
                    row["localized_rehearsal_size"] = len(rehearsal_y)
                    row["localized_replay_size"] = len(rehearsal_y) - len(train_y)
                    row["gradient_evaluations"] = _train_dasp(
                        model,
                        rehearsal_x,
                        rehearsal_y,
                        plasticities,
                        cfg,
                        self.device,
                        memory,
                        strict_freeze=True,
                        sample_weights=rehearsal_weights,
                        epochs=repair_epochs,
                        use_projection_memory=False,
                    )
                    accepted = "dasp"
                else:
                    _train_online(model, train_x, train_y, cfg, self.device)
                    accepted = "online"
            else:
                plasticities = _plasticities(
                    method,
                    item.decision.requirements,
                    conflicts,
                    cfg.dasp.conflict_weight,
                )
                if method == "modular_dasp":
                    ranked_positive = sorted(
                        (
                            (name, value)
                            for name, value in plasticities.items()
                            if value > 0.0
                        ),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    positive = dict(ranked_positive[: max(1, cfg.drift.top_k)])
                    if not positive and plasticities:
                        name, value = max(plasticities.items(), key=lambda item: item[1])
                        positive = {name: value}
                    plasticities = {
                        name: float(positive.get(name, 0.0))
                        for name in model.layer_names
                    }
                    plasticities["classifier"] = 1.0
                    row["localized_repair_modules"] = "|".join(positive)
                    row["localized_repair_module_count"] = len(positive)
                for name, value in plasticities.items():
                    row[f"plasticity_{name}"] = value
                if method == "full_dasp":
                    online_candidate = copy.deepcopy(model)
                    dasp_candidate = copy.deepcopy(model)
                    _train_online(online_candidate, train_x, train_y, cfg, self.device)
                    _train_dasp(
                        dasp_candidate, train_x, train_y, plasticities, cfg, self.device, memory
                    )
                    lcb, mean_delta = _guard_lcb(
                        online_candidate,
                        dasp_candidate,
                        guard_x,
                        guard_y,
                        cfg.dasp.lcb_z,
                        self.device,
                    )
                    row["guard_lcb"] = lcb
                    row["guard_mean_delta"] = mean_delta
                    chosen = dasp_candidate if lcb > 0 else online_candidate
                    model.load_state_dict(chosen.state_dict())
                    accepted = "dasp" if lcb > 0 else "online"
                else:
                    if method == "modular_dasp":
                        rehearsal_x, rehearsal_y, rehearsal_weights = _localized_rehearsal_batch(
                            train_x,
                            train_y,
                            memory,
                            cfg,
                            self.device,
                        )
                        row["localized_rehearsal_size"] = len(rehearsal_y)
                        row["localized_replay_size"] = len(rehearsal_y) - len(train_y)
                        training_x, training_y = rehearsal_x, rehearsal_y
                        training_weights = rehearsal_weights
                        training_epochs = cfg.dasp.modular_repair_epochs
                        projection_enabled = False
                    else:
                        training_x, training_y = train_x, train_y
                        training_weights = None
                        training_epochs = None
                        projection_enabled = True
                    row["gradient_evaluations"] = _train_dasp(
                        model,
                        training_x,
                        training_y,
                        plasticities,
                        cfg,
                        self.device,
                        memory,
                        strict_freeze=(method == "modular_dasp"),
                        sample_weights=training_weights,
                        epochs=training_epochs,
                        use_projection_memory=projection_enabled,
                    )
                    accepted = "dasp"
        if method in LIGHT_REFRESH_METHODS:
            historical = memory.tensors(self.device)
            if historical is not None:
                refresh_decision = item.current_decision or item.decision
                ranked_layers = sorted(
                    (
                        (name, float(value))
                        for name, value in refresh_decision.requirements.items()
                        if name in model.layer_names
                    ),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                active = [
                    name
                    for name, _ in ranked_layers[
                        : max(1, cfg.dasp.v13_refresh_layers)
                    ]
                ]
                if not active:
                    active = list(model.layer_names[-1:])
                refresh_plasticities = {
                    name: float(name in active) for name in model.layer_names
                }
                refresh_plasticities["classifier"] = 0.0
                refresh = propose_layer_adaptive_memory(
                    model,
                    historical[0],
                    historical[1],
                    train_x,
                    train_y,
                    refresh_plasticities,
                    capacity=cfg.dasp.memory_size,
                    label_budget=cfg.dasp.v13_refresh_size,
                    novelty_weight=cfg.dasp.v11_novelty_weight,
                    interference_weight=0.0,
                    balance_weight=cfg.dasp.v11_balance_weight,
                    redundancy_weight=cfg.dasp.v11_redundancy_weight,
                    min_per_class=cfg.dasp.v11_min_per_class,
                    device=self.device,
                    batch_size=max(1, cfg.training.batch_size),
                )
                memory.replace(refresh.x, refresh.y)
                memory_managed = True
                row.update({
                    "buffer_inserted": refresh.inserted,
                    "buffer_forgotten": refresh.evicted,
                    "buffer_size": len(memory),
                    "v13_refresh_inserted": refresh.inserted,
                    "v13_refresh_layers": "|".join(refresh.active_layers),
                    "v13_refresh_mean_novelty": refresh.mean_novelty,
                    "v13_refresh_mean_balance": refresh.mean_balance,
                })
        row["accepted"] = accepted
        if not memory_managed:
            memory.add(train_x, train_y)
        row["global_update"] = row.get("selected_candidate") in {
            "global",
            "online",
            "replay",
            "periodic_ft",
            "replay_fallback",
            "balanced_replay",
            "ssf",
            "ssf_budget",
            "ssf_faithful",
            "repshield",
            "cdda_md",
        }
        if method in V16_METHODS | ADAPTER_METHODS and row["global_update"]:
            raise RuntimeError(
                f"{method} violated the selective-only invariant with "
                f"candidate={row.get('selected_candidate')!r}"
            )
        if method in ADAPTER_METHODS and any(
            parameter.requires_grad for parameter in model.blocks.parameters()
        ):
            raise RuntimeError(
                f"{method} backbone was unfrozen after warm-up"
            )
        row["rss_bytes"] = max(
            int(row.get("rss_bytes", 0)),
            int(self._process.memory_info().rss),
        )
        # CUDA kernels are asynchronous. Synchronize before closing the
        # wall-clock interval so update and test-then-train latency are not
        # systematically underestimated.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        row["update_seconds"] = time.perf_counter() - started
        return accepted

    def _write_outputs(
        self, rows: list[dict[str, object]], summary: dict[str, object]
    ) -> Path:
        cfg = self.config
        name = cfg.output.run_name or (
            f"{cfg.data.source}_{cfg.data.scenario}_{cfg.dasp.baseline}_seed{cfg.data.random_seed}"
        )
        output_dir = Path(cfg.output.directory) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        if any("full_cycle_seconds" in row for row in rows):
            summary["total_full_cycle_seconds"] = sum(float(row.get("full_cycle_seconds", 0.)) for row in rows)
            summary["terminal_update_seconds"] = sum(float(row.get("terminal_update_seconds", 0.)) for row in rows)
            summary["full_cycle_scope"] = "online prediction+metrics+drift+delayed update+reference refresh; warmup and terminal updates reported separately"
        original_review.annotate_outputs(self, rows, summary)
        if cfg.dasp.baseline == "modular_dasp" and cfg.original_review.enabled:
            (output_dir / "provenance.json").write_text(
                json.dumps(self._original_provenance, indent=2), encoding="utf-8")
            if cfg.output.save_checkpoints:
                temporary = output_dir / "checkpoint.tmp"
                torch.save(self._original_checkpoint, temporary)
                os.replace(temporary, output_dir / "checkpoint.pt")
        frame = pd.DataFrame(rows).sort_values("window")
        frame.to_csv(output_dir / "windows.csv", index=False)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        if cfg.output.save_checkpoints and not (cfg.dasp.baseline == "modular_dasp" and cfg.original_review.enabled):
            torch.save({"method": cfg.dasp.baseline}, output_dir / "metadata.pt")
        return output_dir
