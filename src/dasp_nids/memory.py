from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MemoryProposal:
    """Pure candidate memory state produced before a guarded DASP commit."""

    x: torch.Tensor
    y: torch.Tensor
    inserted: int
    evicted: int
    mean_novelty: float
    mean_interference: float
    mean_balance: float
    active_layers: tuple[str, ...]


@dataclass(frozen=True)
class SimilarityReplay:
    """Labeled replay selected to resemble the current unlabeled window."""

    x: torch.Tensor
    y: torch.Tensor
    mean_similarity: float
    active_layers: tuple[str, ...]
    selected_per_class: tuple[int, ...]
    predicted_per_class: tuple[int, ...] = ()
    effective_uniform_mix: float = 0.0
    prior_js_divergence: float = 0.0
    mean_confidence: float = 1.0


def _rank01(values: torch.Tensor) -> torch.Tensor:
    """Stable percentile ranks in [0, 1], robust to score scale."""

    if not len(values):
        return values.float()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(len(values), dtype=torch.float32, device=values.device)
    return ranks / max(1, len(values) - 1)


def _model_outputs(
    model: Any,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits: list[torch.Tensor] = []
    representations: dict[str, list[torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), max(1, batch_size)):
            batch_logits, batch_representations = model(
                x[start : start + batch_size].to(device),
                return_representations=True,
            )
            logits.append(batch_logits.detach().cpu())
            for name, value in batch_representations.items():
                representations.setdefault(name, []).append(value.detach().cpu())
    return (
        torch.cat(logits),
        {name: torch.cat(values) for name, values in representations.items()},
    )


def _class_conditional_scores(
    old_representation: torch.Tensor,
    new_representation: torch.Tensor,
    old_y: torch.Tensor,
    new_y: torch.Tensor,
    old_error: torch.Tensor,
    new_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return novelty, gradient-conflict, and redundancy proxies for new data."""

    novelty = torch.zeros(len(new_y), dtype=torch.float32)
    interference = torch.zeros_like(novelty)
    redundancy = torch.zeros_like(novelty)
    for label in torch.unique(new_y).tolist():
        new_idx = torch.nonzero(new_y == label, as_tuple=False).flatten()
        old_idx = torch.nonzero(old_y == label, as_tuple=False).flatten()
        if not len(new_idx):
            continue
        new_rep = F.normalize(new_representation[new_idx].float(), dim=1)
        if len(old_idx):
            old_rep = F.normalize(old_representation[old_idx].float(), dim=1)
            centroid = F.normalize(old_rep.mean(dim=0, keepdim=True), dim=1)
            distance = 1.0 - (new_rep @ centroid.T).flatten()
            novelty[new_idx] = _rank01(distance)

            # A compact per-sample gradient sketch for a hypothetical linear
            # probe at this layer: dL/dW = (p-y) outer h. This avoids one
            # backward pass per sample while retaining directionality.
            old_sketch = torch.einsum(
                "bi,bj->bij", old_error[old_idx].float(), old_rep
            ).flatten(1)
            new_sketch = torch.einsum(
                "bi,bj->bij", new_error[new_idx].float(), new_rep
            ).flatten(1)
            protected = F.normalize(old_sketch.mean(dim=0, keepdim=True), dim=1)
            conflict = torch.clamp(
                -(F.normalize(new_sketch, dim=1) @ protected.T).flatten(),
                min=0.0,
                max=1.0,
            )
            interference[new_idx] = _rank01(conflict)
        else:
            novelty[new_idx] = 1.0
            interference[new_idx] = 1.0

        if len(new_idx) > 1:
            similarity = new_rep @ new_rep.T
            similarity.fill_diagonal_(-1.0)
            neighbours = min(5, len(new_idx) - 1)
            density = similarity.topk(neighbours, dim=1).values.mean(dim=1)
            redundancy[new_idx] = _rank01(density)
    return novelty, interference, redundancy


def _select_old_retention(
    old_representations: dict[str, torch.Tensor],
    new_representations: dict[str, torch.Tensor],
    old_logits: torch.Tensor,
    old_y: torch.Tensor,
    new_y: torch.Tensor,
    active_layers: list[str],
    layer_weights: torch.Tensor,
    retain_count: int,
    min_per_class: int,
) -> torch.Tensor:
    """Select a class-safe historical subset compatible with the new regime."""

    if retain_count >= len(old_y):
        return torch.arange(len(old_y))
    probabilities = torch.softmax(old_logits.float(), dim=1)
    old_loss = F.nll_loss(
        torch.log(probabilities.clamp_min(1e-12)),
        old_y,
        reduction="none",
    )
    keep_score = 0.25 * _rank01(old_loss)
    compatibility = torch.zeros(len(old_y), dtype=torch.float32)
    for weight, layer in zip(layer_weights.tolist(), active_layers):
        old_rep = F.normalize(old_representations[layer].float(), dim=1)
        new_rep = F.normalize(new_representations[layer].float(), dim=1)
        layer_score = torch.zeros(len(old_y), dtype=torch.float32)
        for label in torch.unique(old_y).tolist():
            old_idx = torch.nonzero(old_y == label, as_tuple=False).flatten()
            new_idx = torch.nonzero(new_y == label, as_tuple=False).flatten()
            if len(new_idx):
                centroid = F.normalize(
                    new_rep[new_idx].mean(dim=0, keepdim=True), dim=1
                )
            else:
                centroid = F.normalize(
                    old_rep[old_idx].mean(dim=0, keepdim=True), dim=1
                )
            layer_score[old_idx] = (old_rep[old_idx] @ centroid.T).flatten()
        compatibility += float(weight) * _rank01(layer_score)
    keep_score += compatibility

    protected: list[int] = []
    protected_set: set[int] = set()
    classes = torch.unique(old_y).tolist()
    feasible_floor = min(
        max(0, int(min_per_class)),
        retain_count // max(1, len(classes)),
    )
    for label in classes:
        candidates = torch.nonzero(old_y == label, as_tuple=False).flatten()
        take = min(feasible_floor, len(candidates))
        if take:
            chosen = candidates[
                torch.argsort(keep_score[candidates], descending=True, stable=True)[:take]
            ].tolist()
            protected.extend(chosen)
            protected_set.update(chosen)
    remaining = retain_count - len(protected)
    if remaining > 0:
        candidates = torch.tensor(
            [index for index in range(len(old_y)) if index not in protected_set],
            dtype=torch.long,
        )
        chosen = candidates[
            torch.argsort(keep_score[candidates], descending=True, stable=True)[:remaining]
        ].tolist()
        protected.extend(chosen)
    return torch.tensor(protected, dtype=torch.long)


def propose_layer_adaptive_memory(
    model: Any,
    old_x: torch.Tensor,
    old_y: torch.Tensor,
    new_x: torch.Tensor,
    new_y: torch.Tensor,
    plasticities: dict[str, float],
    *,
    capacity: int,
    label_budget: int,
    novelty_weight: float,
    interference_weight: float,
    balance_weight: float,
    redundancy_weight: float,
    min_per_class: int,
    device: torch.device,
    batch_size: int = 512,
) -> MemoryProposal:
    """Build a deterministic layer-conditioned memory without mutating input.

    The proposal is safe to discard. Callers commit it with
    ``ReservoirMemory.replace`` only after the paired model candidate passes
    DASP's guards.
    """

    old_x = old_x.detach().cpu()
    old_y = old_y.detach().cpu().long()
    new_x = new_x.detach().cpu()
    new_y = new_y.detach().cpu().long()
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if len(new_x) != len(new_y) or len(old_x) != len(old_y):
        raise ValueError("feature and label lengths must match")
    active_layers = [
        name for name, value in plasticities.items() if name != "classifier" and value > 0
    ]
    if not active_layers:
        active_layers = [
            name for name in getattr(model, "layer_names", []) if name in plasticities
        ]
    if not active_layers or not len(new_y):
        take = min(capacity, len(old_y))
        return MemoryProposal(
            old_x[:take].clone(),
            old_y[:take].clone(),
            0,
            max(0, len(old_y) - take),
            0.0,
            0.0,
            0.0,
            tuple(active_layers),
        )

    old_logits, old_representations = _model_outputs(
        model, old_x, device, batch_size
    )
    new_logits, new_representations = _model_outputs(
        model, new_x, device, batch_size
    )
    active_layers = [
        name
        for name in active_layers
        if name in old_representations and name in new_representations
    ]
    raw_weights = torch.tensor(
        [max(0.0, float(plasticities[name])) for name in active_layers],
        dtype=torch.float32,
    )
    layer_weights = raw_weights / raw_weights.sum().clamp_min(1e-12)
    n_classes = old_logits.shape[1]
    old_error = torch.softmax(old_logits.float(), dim=1) - F.one_hot(
        old_y, n_classes
    ).float()
    new_error = torch.softmax(new_logits.float(), dim=1) - F.one_hot(
        new_y, n_classes
    ).float()

    novelty = torch.zeros(len(new_y), dtype=torch.float32)
    interference = torch.zeros_like(novelty)
    redundancy = torch.zeros_like(novelty)
    for weight, layer in zip(layer_weights.tolist(), active_layers):
        layer_novelty, layer_interference, layer_redundancy = (
            _class_conditional_scores(
                old_representations[layer],
                new_representations[layer],
                old_y,
                new_y,
                old_error,
                new_error,
            )
        )
        novelty += float(weight) * layer_novelty
        interference += float(weight) * layer_interference
        redundancy += float(weight) * layer_redundancy

    classes, counts = torch.unique(old_y, return_counts=True)
    count_map = {int(label): int(count) for label, count in zip(classes, counts)}
    inverse = torch.tensor(
        [1.0 / max(1, count_map.get(int(label), 0)) for label in new_y],
        dtype=torch.float32,
    )
    balance = _rank01(inverse)
    score = (
        float(novelty_weight) * novelty
        + float(interference_weight) * interference
        + float(balance_weight) * balance
        - float(redundancy_weight) * redundancy
    )
    insert_count = min(
        max(0, int(label_budget)),
        len(new_y),
        capacity,
    )
    selected_new = torch.argsort(score, descending=True, stable=True)[:insert_count]
    retain_count = min(len(old_y), capacity - insert_count)
    selected_old = _select_old_retention(
        old_representations,
        new_representations,
        old_logits,
        old_y,
        new_y,
        active_layers,
        layer_weights,
        retain_count,
        min_per_class,
    )
    proposed_x = torch.cat([old_x[selected_old], new_x[selected_new]], dim=0)
    proposed_y = torch.cat([old_y[selected_old], new_y[selected_new]], dim=0)
    return MemoryProposal(
        x=proposed_x,
        y=proposed_y,
        inserted=int(len(selected_new)),
        evicted=int(max(0, len(old_y) - len(selected_old))),
        mean_novelty=float(novelty[selected_new].mean().item()) if len(selected_new) else 0.0,
        mean_interference=(
            float(interference[selected_new].mean().item()) if len(selected_new) else 0.0
        ),
        mean_balance=float(balance[selected_new].mean().item()) if len(selected_new) else 0.0,
        active_layers=tuple(active_layers),
    )


def select_feature_similarity_replay(
    model: Any,
    memory_x: torch.Tensor,
    memory_y: torch.Tensor,
    current_unlabeled_x: torch.Tensor,
    active_layers: list[str],
    *,
    replay_size: int,
    device: torch.device,
    batch_size: int = 512,
) -> SimilarityReplay | None:
    """Select deterministic class-aware replay by current feature similarity.

    This is a lightweight, deterministic variant of IWMS: current labels are
    never read. Predicted classes define class-conditional current centroids;
    labeled memory samples are ranked by cosine similarity in the selected
    representation layers. Half of the quota follows the predicted current
    class mix and half is distributed uniformly across available classes to
    avoid a majority-class replay collapse.
    """

    if replay_size <= 0 or not len(memory_y) or not len(current_unlabeled_x):
        return None
    memory_x = memory_x.detach().cpu()
    memory_y = memory_y.detach().cpu().long()
    current_unlabeled_x = current_unlabeled_x.detach().cpu()
    _, memory_representations = _model_outputs(
        model, memory_x, device, batch_size
    )
    current_logits, current_representations = _model_outputs(
        model, current_unlabeled_x, device, batch_size
    )
    layers = [
        name
        for name in active_layers
        if name in memory_representations and name in current_representations
    ]
    if not layers:
        shared = [
            name for name in memory_representations if name in current_representations
        ]
        layers = shared[-1:] if shared else []
    if not layers:
        return None

    predicted = current_logits.argmax(dim=1)
    available_classes = [
        int(label)
        for label in torch.unique(memory_y).tolist()
        if int((memory_y == int(label)).sum().item()) > 0
    ]
    if not available_classes:
        return None
    count = min(int(replay_size), len(memory_y))
    predicted_counts = torch.bincount(
        predicted, minlength=max(available_classes) + 1
    ).float()
    predicted_mass = predicted_counts[available_classes]
    if float(predicted_mass.sum().item()) <= 0.0:
        predicted_mass = torch.ones(len(available_classes))
    predicted_mass /= predicted_mass.sum()
    uniform_mass = torch.full_like(predicted_mass, 1.0 / len(available_classes))
    target_mass = 0.5 * predicted_mass + 0.5 * uniform_mass
    raw_quota = target_mass * count
    quota = torch.floor(raw_quota).long()
    remainder = count - int(quota.sum().item())
    if remainder:
        fractional = raw_quota - quota.float()
        order = torch.argsort(fractional, descending=True, stable=True)
        quota[order[:remainder]] += 1

    scores = torch.zeros(len(memory_y), dtype=torch.float32)
    for layer in layers:
        memory_rep = F.normalize(memory_representations[layer].float(), dim=1)
        current_rep = F.normalize(current_representations[layer].float(), dim=1)
        for label in available_classes:
            memory_idx = torch.nonzero(
                memory_y == label, as_tuple=False
            ).flatten()
            current_idx = torch.nonzero(
                predicted == label, as_tuple=False
            ).flatten()
            source = (
                current_rep[current_idx]
                if len(current_idx)
                else current_rep
            )
            centroid = F.normalize(source.mean(dim=0, keepdim=True), dim=1)
            scores[memory_idx] += (
                memory_rep[memory_idx] @ centroid.T
            ).flatten() / len(layers)

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_per_class: list[int] = []
    for position, label in enumerate(available_classes):
        candidates = torch.nonzero(memory_y == label, as_tuple=False).flatten()
        take = min(int(quota[position].item()), len(candidates))
        chosen = candidates[
            torch.argsort(
                scores[candidates], descending=True, stable=True
            )[:take]
        ].tolist()
        selected.extend(chosen)
        selected_set.update(chosen)
        selected_per_class.append(len(chosen))
    if len(selected) < count:
        remaining_indices = torch.tensor(
            [idx for idx in range(len(memory_y)) if idx not in selected_set],
            dtype=torch.long,
        )
        extra = remaining_indices[
            torch.argsort(
                scores[remaining_indices], descending=True, stable=True
            )[: count - len(selected)]
        ].tolist()
        selected.extend(extra)
        for idx in extra:
            selected_per_class[available_classes.index(int(memory_y[idx]))] += 1
    indices = torch.tensor(selected, dtype=torch.long)
    return SimilarityReplay(
        x=memory_x[indices].to(device),
        y=memory_y[indices].to(device),
        mean_similarity=float(scores[indices].mean().item()),
        active_layers=tuple(layers),
        selected_per_class=tuple(selected_per_class),
    )


def select_uncertainty_weighted_similarity_replay(
    model: Any,
    memory_x: torch.Tensor,
    memory_y: torch.Tensor,
    current_unlabeled_x: torch.Tensor,
    active_layers: list[str],
    *,
    replay_size: int,
    device: torch.device,
    seed: int,
    temperature: float = 0.10,
    confidence_power: float = 1.0,
    uniform_mix: float = 0.25,
    max_uniform_mix: float | None = None,
    js_threshold: float = 0.10,
    batch_size: int = 512,
) -> SimilarityReplay | None:
    """Direct per-sample IWMS with uncertainty and class-balance correction.

    Every current feature vector votes for labeled memory items in the selected
    representation layers.  Entropy downweights uncertain pseudo-classes, a
    uniform component limits confirmation bias, and seeded weighted sampling
    avoids repeatedly replaying only the same deterministic nearest neighbours.
    No current labels are accessed.
    """

    if replay_size <= 0 or not len(memory_y) or not len(current_unlabeled_x):
        return None
    temperature = max(float(temperature), 1e-4)
    confidence_power = max(float(confidence_power), 0.0)
    uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
    maximum_mix = (
        uniform_mix
        if max_uniform_mix is None
        else float(np.clip(max_uniform_mix, uniform_mix, 1.0))
    )
    memory_x = memory_x.detach().cpu()
    memory_y = memory_y.detach().cpu().long()
    current_unlabeled_x = current_unlabeled_x.detach().cpu()
    _, memory_representations = _model_outputs(
        model, memory_x, device, batch_size
    )
    current_logits, current_representations = _model_outputs(
        model, current_unlabeled_x, device, batch_size
    )
    layers = [
        name
        for name in active_layers
        if name in memory_representations and name in current_representations
    ]
    if not layers:
        shared = [
            name for name in memory_representations if name in current_representations
        ]
        layers = shared[-1:] if shared else []
    if not layers:
        return None

    probabilities = current_logits.float().softmax(dim=1)
    predicted = probabilities.argmax(dim=1)
    if probabilities.shape[1] > 1:
        entropy = -torch.sum(
            probabilities * probabilities.clamp_min(1e-12).log(), dim=1
        ) / float(np.log(probabilities.shape[1]))
        confidence = (1.0 - entropy).clamp(0.0, 1.0).pow(confidence_power)
    else:
        confidence = torch.ones(len(probabilities))
    scores = torch.zeros(len(memory_y), dtype=torch.float32)
    normalizer = 0.0
    for layer in layers:
        memory_rep = F.normalize(memory_representations[layer].float(), dim=1)
        current_rep = F.normalize(current_representations[layer].float(), dim=1)
        for start in range(0, len(current_rep), max(1, int(batch_size))):
            stop = min(len(current_rep), start + max(1, int(batch_size)))
            similarity = current_rep[start:stop] @ memory_rep.T
            kernel = torch.exp((similarity - 1.0) / temperature)
            class_match = (
                predicted[start:stop, None] == memory_y[None, :]
            ).float()
            class_weight = uniform_mix + (1.0 - uniform_mix) * class_match
            sample_weight = confidence[start:stop, None]
            scores += torch.sum(kernel * class_weight * sample_weight, dim=0)
            normalizer += float(confidence[start:stop].sum().item())
    scores /= max(normalizer * len(layers), 1e-12)

    available_classes = [int(value) for value in torch.unique(memory_y).tolist()]
    count = min(int(replay_size), len(memory_y))
    predicted_counts = torch.bincount(
        predicted, minlength=max(available_classes) + 1
    ).float()[available_classes]
    if float(predicted_counts.sum().item()) <= 0.0:
        predicted_counts = torch.ones_like(predicted_counts)
    predicted_mass = predicted_counts / predicted_counts.sum()
    memory_counts = torch.tensor(
        [int((memory_y == label).sum().item()) for label in available_classes],
        dtype=torch.float32,
    )
    memory_mass = memory_counts / memory_counts.sum().clamp_min(1.0)
    midpoint = 0.5 * (predicted_mass + memory_mass)
    js_divergence = 0.5 * torch.sum(
        predicted_mass
        * (
            predicted_mass.clamp_min(1e-12).log()
            - midpoint.clamp_min(1e-12).log()
        )
    ) + 0.5 * torch.sum(
        memory_mass
        * (
            memory_mass.clamp_min(1e-12).log()
            - midpoint.clamp_min(1e-12).log()
        )
    )
    mean_confidence = float(confidence.mean().item())
    mismatch = float(
        np.clip(
            float(js_divergence.item()) / max(float(js_threshold), 1e-8),
            0.0,
            1.0,
        )
    )
    reliability_deficit = float(np.clip(1.0 - mean_confidence, 0.0, 1.0))
    robust_pressure = max(mismatch, reliability_deficit)
    effective_uniform_mix = uniform_mix + (
        maximum_mix - uniform_mix
    ) * robust_pressure
    uniform_mass = torch.full_like(predicted_mass, 1.0 / len(available_classes))
    target_mass = (
        (1.0 - effective_uniform_mix) * predicted_mass
        + effective_uniform_mix * uniform_mass
    )
    raw_quota = target_mass * count
    quota = torch.floor(raw_quota).long()
    remainder = count - int(quota.sum().item())
    if remainder:
        order = torch.argsort(raw_quota - quota.float(), descending=True, stable=True)
        quota[order[:remainder]] += 1

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected: list[int] = []
    selected_set: set[int] = set()
    for position, label in enumerate(available_classes):
        candidates = torch.nonzero(memory_y == label, as_tuple=False).flatten()
        take = min(int(quota[position].item()), len(candidates))
        if take:
            weights = scores[candidates].clamp_min(0.0) + 1e-12
            local = torch.multinomial(
                weights, take, replacement=False, generator=generator
            )
            chosen = candidates[local].tolist()
            selected.extend(chosen)
            selected_set.update(chosen)
    if len(selected) < count:
        remaining = torch.tensor(
            [index for index in range(len(memory_y)) if index not in selected_set],
            dtype=torch.long,
        )
        take = min(count - len(selected), len(remaining))
        weights = scores[remaining].clamp_min(0.0) + 1e-12
        local = torch.multinomial(
            weights, take, replacement=False, generator=generator
        )
        selected.extend(remaining[local].tolist())

    indices = torch.tensor(selected, dtype=torch.long)
    selected_per_class = tuple(
        int((memory_y[indices] == label).sum().item())
        for label in available_classes
    )
    return SimilarityReplay(
        x=memory_x[indices].to(device),
        y=memory_y[indices].to(device),
        mean_similarity=float(scores[indices].mean().item()),
        active_layers=tuple(layers),
        selected_per_class=selected_per_class,
        predicted_per_class=tuple(int(value.item()) for value in predicted_counts),
        effective_uniform_mix=float(effective_uniform_mix),
        prior_js_divergence=float(js_divergence.item()),
        mean_confidence=mean_confidence,
    )


class ReservoirMemory:
    """Bounded, unbiased reservoir of labeled historical samples."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        # Keep reservoir replacement and replay draws independent so sampling
        # frequency cannot silently change future memory contents.
        seed_sequence = np.random.SeedSequence(seed)
        update_seed, replay_seed = seed_sequence.spawn(2)
        self.update_rng = np.random.default_rng(update_seed)
        self.replay_rng = np.random.default_rng(replay_seed)
        self.seen = 0
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.x is None else len(self.x)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().cpu()
        y = y.detach().cpu()
        if self.x is None:
            take = min(self.capacity, len(x))
            self.x = x[:take].clone()
            self.y = y[:take].clone()
            self.seen = take
            x, y = x[take:], y[take:]
        for sample_x, sample_y in zip(x, y):
            self.seen += 1
            if len(self) < self.capacity:
                self.x = torch.cat([self.x, sample_x.unsqueeze(0)])
                self.y = torch.cat([self.y, sample_y.unsqueeze(0)])
                continue
            index = int(self.update_rng.integers(0, self.seen))
            if index < self.capacity:
                self.x[index] = sample_x
                self.y[index] = sample_y

    def sample(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not len(self):
            return None
        count = min(n, len(self))
        indices = self.replay_rng.choice(len(self), size=count, replace=False).tolist()
        assert self.x is not None and self.y is not None
        return self.x[indices].to(device), self.y[indices].to(device)

    def sample_balanced(
        self, n: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Draw a deterministic class-balanced replay batch when possible."""

        if not len(self):
            return None
        assert self.x is not None and self.y is not None
        count = min(n, len(self))
        classes = torch.unique(self.y).tolist()
        quota, remainder = divmod(count, len(classes))
        selected: list[int] = []
        for position, label in enumerate(classes):
            candidates = torch.nonzero(self.y == label, as_tuple=False).flatten().tolist()
            take = min(len(candidates), quota + int(position < remainder))
            if take:
                selected.extend(
                    self.replay_rng.choice(candidates, size=take, replace=False).tolist()
                )
        if len(selected) < count:
            selected_set = set(selected)
            remaining = np.asarray([i for i in range(len(self)) if i not in selected_set])
            take = min(count - len(selected), len(remaining))
            if take:
                selected.extend(
                    self.replay_rng.choice(remaining, size=take, replace=False).tolist()
                )
        self.replay_rng.shuffle(selected)
        return self.x[selected].to(device), self.y[selected].to(device)

    def replace(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Replace the reservoir with a method-managed fixed-size buffer."""

        take = min(self.capacity, len(x))
        self.x = x[:take].detach().cpu().clone()
        self.y = y[:take].detach().cpu().clone()
        self.seen = max(self.seen, take)

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not len(self):
            return None
        assert self.x is not None and self.y is not None
        return self.x.to(device), self.y.to(device)
