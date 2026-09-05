from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class RevisionCondition:
    identifier: str
    method: str
    routing_policy: str
    routing_k: int
    comparator: str | None = None
    required_architecture: str = "modular"
    replay_enabled: bool = True
    new_sample_weight: float | None = None
    performance_gate_enabled: bool | None = None
    label_balanced_selection: bool = True
    final_representation_only: bool = False


REVISION_CONDITIONS = {
    "revision_a1_combined": RevisionCondition("A1", "revision_a1_combined", "combined", 3),
    "revision_a2_random": RevisionCondition("A2", "revision_a2_random", "random", 3),
    "revision_a3_fixed": RevisionCondition("A3", "revision_a3_fixed", "fixed", 3),
    "revision_a4_mmd": RevisionCondition("A4", "revision_a4_mmd", "mmd", 3),
    "revision_a5_conflict": RevisionCondition("A5", "revision_a5_conflict", "conflict", 3),
    "revision_a6_all": RevisionCondition("A6", "revision_a6_all", "all", 4),
    "revision_a7_k1": RevisionCondition("A7", "revision_a7_k1", "combined", 1),
    "revision_a8_k2": RevisionCondition("A8", "revision_a8_k2", "combined", 2),
    "revision_a9_classifier": RevisionCondition("A9", "revision_a9_classifier", "classifier", 0),
    "revision_a10_ssf": RevisionCondition("A10", "revision_a10_ssf", "ssf", 0, "ssf"),
    "revision_b1_no_replay": RevisionCondition(
        "B1", "revision_b1_no_replay", "combined", 3, replay_enabled=False
    ),
    "revision_b2_no_weighting": RevisionCondition(
        "B2", "revision_b2_no_weighting", "combined", 3, new_sample_weight=1.0
    ),
    "revision_b3_no_gate": RevisionCondition(
        "B3", "revision_b3_no_gate", "combined", 3, performance_gate_enabled=False
    ),
    "revision_b4_uniform_selection": RevisionCondition(
        "B4",
        "revision_b4_uniform_selection",
        "combined",
        3,
        label_balanced_selection=False,
    ),
    "revision_b5_flat_global": RevisionCondition(
        "B5",
        "revision_b5_flat_global",
        "all",
        2,
        required_architecture="sequential",
        final_representation_only=True,
    ),
}

REVISION_METHODS = frozenset(REVISION_CONDITIONS)
REVISION_COMPONENT_METHODS = frozenset(
    method for method, condition in REVISION_CONDITIONS.items() if condition.identifier.startswith("B")
)


@dataclass(frozen=True)
class LabelPoolIndices:
    query: torch.Tensor
    repair_candidates: torch.Tensor
    gate: torch.Tensor
    query_seed: int
    selection_seed: int


@dataclass(frozen=True)
class RoutingSelection:
    active_modules: tuple[str, ...]
    scores: dict[str, float]
    plasticities: dict[str, float]
    seed: int | None


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def index_digest(indices: torch.Tensor | Iterable[int]) -> str:
    if isinstance(indices, torch.Tensor):
        values = indices.detach().cpu().long().numpy()
    else:
        values = np.asarray(list(indices), dtype=np.int64)
    return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


def equal_label_pools(
    n_samples: int,
    label_budget: int,
    repair_candidate_size: int,
    gate_size: int,
    dataset_key: str,
    scenario_key: str,
    stream_seed: int,
    delayed_window_index: int,
    protocol_version: str,
) -> LabelPoolIndices:
    if min(label_budget, repair_candidate_size, gate_size) < 0:
        raise ValueError("label budgets must be non-negative")
    if repair_candidate_size + gate_size != label_budget:
        raise ValueError("repair_candidate_size + gate_size must equal label_budget")
    query_seed = stable_seed(
        "equal-label-query",
        dataset_key,
        scenario_key,
        stream_seed,
        delayed_window_index,
        protocol_version,
    )
    selection_seed = stable_seed(
        "equal-label-training-selection",
        dataset_key,
        scenario_key,
        stream_seed,
        delayed_window_index,
        protocol_version,
    )
    count = min(max(0, int(n_samples)), int(label_budget))
    generator = torch.Generator().manual_seed(query_seed)
    query = torch.randperm(max(0, int(n_samples)), generator=generator)[:count]
    repair_count = min(int(repair_candidate_size), len(query))
    gate_count = min(int(gate_size), max(0, len(query) - repair_count))
    repair = query[:repair_count]
    gate = query[repair_count : repair_count + gate_count]
    return LabelPoolIndices(query, repair, gate, query_seed, selection_seed)


def select_training_indices(
    labels: torch.Tensor,
    candidate_indices: torch.Tensor,
    budget: int,
    seed: int,
    label_balanced: bool = True,
) -> torch.Tensor:
    candidates = candidate_indices.detach().cpu().long()
    count = min(max(0, int(budget)), len(candidates))
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    if not label_balanced:
        return candidates[torch.randperm(len(candidates), generator=generator)[:count]]
    candidate_labels = labels.detach().cpu().long()[candidates]
    unique_labels = torch.unique(candidate_labels, sorted=True).tolist()
    quota, remainder = divmod(count, max(1, len(unique_labels)))
    selected_positions: list[int] = []
    for position, label in enumerate(unique_labels):
        available = torch.nonzero(candidate_labels == label, as_tuple=False).flatten()
        take = min(len(available), quota + int(position < remainder))
        if take:
            order = torch.randperm(len(available), generator=generator)[:take]
            selected_positions.extend(available[order].tolist())
    if len(selected_positions) < count:
        selected_set = set(selected_positions)
        remaining = torch.tensor(
            [index for index in range(len(candidates)) if index not in selected_set],
            dtype=torch.long,
        )
        take = min(count - len(selected_positions), len(remaining))
        if take:
            order = torch.randperm(len(remaining), generator=generator)[:take]
            selected_positions.extend(remaining[order].tolist())
    selected = torch.tensor(selected_positions, dtype=torch.long)
    if len(selected) > 1:
        selected = selected[torch.randperm(len(selected), generator=generator)]
    return candidates[selected]


def select_routing(
    policy: str,
    layer_names: list[str],
    requirements: dict[str, float],
    conflicts: dict[str, float],
    k: int,
    conflict_weight: float,
    representation_plasticity: float,
    classifier_plasticity: float,
    fixed_modules: list[str],
    random_seed: int | None = None,
) -> RoutingSelection:
    names = list(layer_names)
    if not 0 <= int(k) <= len(names):
        raise ValueError("routing k must be between zero and the module count")
    compatibility = {
        name: 1.0 / (1.0 + float(conflict_weight) * float(conflicts.get(name, 0.0)))
        for name in names
    }
    if policy == "combined":
        scores = {
            name: float(requirements.get(name, 0.0)) * compatibility[name]
            for name in names
        }
        active = tuple(sorted(names, key=lambda name: (-scores[name], names.index(name)))[:k])
    elif policy == "mmd":
        scores = {name: float(requirements.get(name, 0.0)) for name in names}
        active = tuple(sorted(names, key=lambda name: (-scores[name], names.index(name)))[:k])
    elif policy == "conflict":
        scores = compatibility
        active = tuple(sorted(names, key=lambda name: (-scores[name], names.index(name)))[:k])
    elif policy == "random":
        if random_seed is None:
            raise ValueError("random routing requires a deterministic seed")
        generator = torch.Generator().manual_seed(int(random_seed))
        order = torch.randperm(len(names), generator=generator).tolist()
        active = tuple(names[index] for index in order[:k])
        scores = {name: float(name in active) for name in names}
    elif policy == "fixed":
        unknown = [name for name in fixed_modules if name not in names]
        if unknown:
            raise ValueError(f"fixed routing contains unknown modules: {unknown}")
        active = tuple(fixed_modules[:k])
        scores = {name: float(name in active) for name in names}
    elif policy == "all":
        active = tuple(names)
        scores = {name: 1.0 for name in names}
    elif policy == "classifier":
        active = ()
        scores = {name: 0.0 for name in names}
    else:
        raise ValueError(f"unknown revision routing policy: {policy}")
    plasticities = {
        name: float(representation_plasticity if name in active else 0.0)
        for name in names
    }
    plasticities["classifier"] = float(classifier_plasticity)
    return RoutingSelection(active, scores, plasticities, random_seed)


def parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def locality_diagnostics(
    model: torch.nn.Module,
    before: dict[str, torch.Tensor],
    active_modules: tuple[str, ...],
    gradient_evaluations: int,
    tolerance: float,
) -> dict[str, float | int | str]:
    groups = model.adaptation_parameter_groups()
    representation_names = [name for name in model.layer_names if name in groups]
    total = sum(parameter.numel() for parameter in model.parameters())
    representation_total = sum(
        parameter.numel() for name in representation_names for parameter in groups[name]
    )
    active_representation = sum(
        parameter.numel() for name in active_modules for parameter in groups[name]
    )
    classifier = sum(parameter.numel() for parameter in groups.get("classifier", []))
    optimizer_included = active_representation + classifier
    changed = 0
    inactive_max = 0.0
    diagnostics: dict[str, float | int | str] = {}
    for name, parameter in model.named_parameters():
        difference = (parameter.detach().cpu() - before[name]).abs()
        changed += int((difference > float(tolerance)).sum().item())
    for module_name in representation_names:
        if module_name in active_modules:
            continue
        module_max = 0.0
        prefix = f"modules_list.{representation_names.index(module_name)}."
        for name, parameter in model.named_parameters():
            if name.startswith(prefix):
                module_max = max(
                    module_max,
                    float((parameter.detach().cpu() - before[name]).abs().max().item()),
                )
        diagnostics[f"inactive_{module_name}_max_abs_change"] = module_max
        inactive_max = max(inactive_max, module_max)
    diagnostics.update(
        {
            "active_representation_modules": "|".join(active_modules),
            "active_representation_module_count": len(active_modules),
            "active_representation_module_fraction": len(active_modules) / max(1, len(representation_names)),
            "total_model_parameter_count": total,
            "optimizer_included_parameter_count": optimizer_included,
            "optimizer_included_parameter_fraction": optimizer_included / max(1, total),
            "representation_parameter_count": representation_total,
            "active_representation_parameter_count": active_representation,
            "active_representation_parameter_fraction": active_representation / max(1, representation_total),
            "changed_parameter_count": changed,
            "changed_parameter_fraction": changed / max(1, total),
            "inactive_module_max_abs_change": inactive_max,
            "classifier_parameter_fraction": classifier / max(1, total),
            "parameter_gradient_element_evaluations": int(gradient_evaluations) * optimizer_included,
        }
    )
    return diagnostics


def routing_jaccard(previous: tuple[str, ...], current: tuple[str, ...]) -> float:
    left, right = set(previous), set(current)
    return len(left & right) / max(1, len(left | right))
