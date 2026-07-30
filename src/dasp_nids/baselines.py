from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp
from torch import nn

from .config import ExperimentConfig
from .memory import ReservoirMemory


def normal_probabilities(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return F.softmax(model(x.to(device)), dim=1)[:, 0].detach()


def _soft_histogram(values: torch.Tensor, weights: torch.Tensor, bins: int) -> torch.Tensor:
    # A triangular kernel keeps the SSF mask objective differentiable.
    centers = torch.linspace(0.0, 1.0, bins, device=values.device)
    width = 1.0 / max(1, bins - 1)
    memberships = (1.0 - (values[:, None] - centers[None, :]).abs() / width).clamp_min(0.0)
    histogram = (memberships * weights[:, None]).sum(dim=0)
    return (histogram + 1e-8) / (histogram.sum() + 1e-8 * bins)


def ssf_masks(
    old_probability: torch.Tensor,
    new_probability: torch.Tensor,
    steps: int,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize SSF old/common and new/drift representative masks."""

    old_probability = old_probability.detach()
    new_probability = new_probability.detach()
    target_new = _soft_histogram(new_probability, torch.ones_like(new_probability), bins).detach()

    old_logits = nn.Parameter(torch.full_like(old_probability, 1.0))
    old_optimizer = torch.optim.Adam([old_logits], lr=0.1)
    for _ in range(max(1, steps)):
        old_optimizer.zero_grad(set_to_none=True)
        old_hist = _soft_histogram(old_probability, old_logits.sigmoid(), bins)
        F.kl_div(old_hist.log(), target_new, reduction="sum").backward()
        old_optimizer.step()
    old_mask = old_logits.sigmoid().detach()

    new_logits = nn.Parameter(torch.full_like(new_probability, -1.0))
    new_optimizer = torch.optim.Adam([new_logits], lr=0.1)
    for _ in range(max(1, steps)):
        new_optimizer.zero_grad(set_to_none=True)
        new_weights = new_logits.sigmoid()
        combined_values = torch.cat([old_probability, new_probability])
        combined_weights = torch.cat([old_mask, new_weights])
        combined_hist = _soft_histogram(combined_values, combined_weights, bins)
        F.kl_div(combined_hist.log(), target_new, reduction="sum").backward()
        new_optimizer.step()
    return old_mask, new_logits.sigmoid().detach()


def select_ssf_buffer(
    old_x: torch.Tensor,
    old_y: torch.Tensor,
    new_x: torch.Tensor,
    new_y: torch.Tensor,
    old_scores: torch.Tensor,
    new_scores: torch.Tensor,
    capacity: int,
    label_budget: int,
    drift: bool,
    *,
    allow_pseudo_labels: bool = False,
    pseudo_y: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    """Strategically refresh SSF without silently exceeding the label budget.

    SSF's published drift path may replace more than ``label_budget`` samples,
    but every replacement after the budget is pseudo-labeled.  The benchmark's
    primary fairness mode disables pseudo-labels, so both forgetting and
    insertion are capped at the number of newly acquired ground-truth labels.
    """

    budget = min(max(0, label_budget), len(new_x), capacity)
    nonrepresentative = torch.nonzero(old_scores < 0.5, as_tuple=False).flatten()
    if drift and allow_pseudo_labels:
        remove_count = max(budget, len(nonrepresentative))
    else:
        remove_count = budget
    remove_count = min(remove_count, len(old_x), len(new_x))
    if remove_count > budget and pseudo_y is None:
        raise ValueError("pseudo_y is required when SSF inserts beyond label_budget")

    old_order = torch.argsort(old_scores)
    keep = torch.ones(len(old_x), dtype=torch.bool, device=old_x.device)
    keep[old_order[:remove_count]] = False
    new_order = torch.argsort(new_scores, descending=True)[:remove_count]
    replacement_y = new_y[new_order].clone()
    pseudo_count = max(0, remove_count - budget)
    if pseudo_count:
        assert pseudo_y is not None
        replacement_y[budget:] = pseudo_y[new_order[budget:]].to(
            replacement_y.device
        )
    buffer_x = torch.cat([old_x[keep], new_x[new_order]], dim=0)
    buffer_y = torch.cat([old_y[keep], replacement_y], dim=0)
    new_mask = torch.cat([
        torch.zeros(int(keep.sum()), device=old_x.device),
        torch.ones(len(new_order), device=old_x.device),
    ])

    if len(buffer_x) > capacity:
        buffer_x = buffer_x[-capacity:]
        buffer_y = buffer_y[-capacity:]
        new_mask = new_mask[-capacity:]
    ground_truth_count = min(budget, len(new_order))
    return (
        buffer_x,
        buffer_y,
        new_mask,
        remove_count,
        len(new_order),
        ground_truth_count,
        pseudo_count,
    )


def train_ssf(
    model: nn.Module,
    teacher: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    new_mask: torch.Tensor,
    drift: bool,
    cfg: ExperimentConfig,
    device: torch.device,
    seed_offset: int,
) -> int:
    model.train()
    teacher.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    x, y, new_mask = x.to(device), y.to(device), new_mask.to(device)
    evaluations = 0
    for epoch in range(cfg.training.epochs_per_window):
        generator = torch.Generator().manual_seed(
            cfg.data.random_seed + seed_offset * 104729 + epoch
        )
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.training.batch_size):
            idx = order[start : start + cfg.training.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            losses = F.cross_entropy(logits, y[idx], reduction="none")
            weights = 1.0 + new_mask[idx] * (cfg.baselines.ssf_new_sample_weight - 1.0)
            loss = (losses * weights).mean()
            if not drift:
                with torch.no_grad():
                    teacher_probability = F.softmax(teacher(x[idx]), dim=1)
                distillation = F.kl_div(
                    F.log_softmax(logits, dim=1), teacher_probability, reduction="batchmean"
                )
                loss = loss + cfg.baselines.ssf_lwf_lambda * distillation
            loss.backward()
            optimizer.step()
            evaluations += 1
    return evaluations


def pca_variance_distance(old_x: torch.Tensor, new_x: torch.Tensor) -> float:
    """RepShield PCA drift score: L1 change in normalized explained variance."""

    count = min(len(old_x), len(new_x), 1024)
    if count < 2:
        return 0.0
    old = old_x[:count].float()
    new = new_x[:count].float()
    old_values = torch.linalg.svdvals(old - old.mean(dim=0)) ** 2
    new_values = torch.linalg.svdvals(new - new.mean(dim=0)) ** 2
    size = min(len(old_values), len(new_values))
    old_ratio = old_values[:size] / old_values.sum().clamp_min(1e-12)
    new_ratio = new_values[:size] / new_values.sum().clamp_min(1e-12)
    return float(0.5 * torch.abs(old_ratio - new_ratio).sum().item())


def strategic_coreset(
    model: nn.Module,
    old_x: torch.Tensor,
    old_y: torch.Tensor,
    new_x: torch.Tensor,
    new_y: torch.Tensor,
    capacity: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced uncertainty/centroid coreset used by RepShield."""

    x = torch.cat([old_x, new_x], dim=0)
    y = torch.cat([old_y, new_y], dim=0)
    if len(x) <= capacity:
        return x, y
    model.eval()
    with torch.no_grad():
        logits, representations = model(x.to(device), return_representations=True)
        features = list(representations.values())[-1]
        probability = F.softmax(logits, dim=1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1)
    selected: list[torch.Tensor] = []
    classes = torch.unique(y)
    base = capacity // max(1, len(classes))
    remainder = capacity - base * len(classes)
    for position, label in enumerate(classes):
        indices = torch.nonzero(y.to(device) == label.to(device), as_tuple=False).flatten()
        quota = min(len(indices), base + int(position < remainder))
        if not quota:
            continue
        class_features = features[indices]
        centroid = class_features.mean(dim=0)
        distance = torch.linalg.vector_norm(class_features - centroid, dim=1)
        representativeness = 1.0 / (1.0 + distance)
        score = representativeness + entropy[indices] / np.log(max(2, probability.shape[1]))
        selected.append(indices[torch.topk(score, quota).indices].cpu())
    keep = torch.cat(selected) if selected else torch.arange(min(capacity, len(x)))
    if len(keep) < capacity:
        missing = torch.tensor(
            [i for i in range(len(x)) if i not in set(keep.tolist())][: capacity - len(keep)]
        )
        keep = torch.cat([keep, missing])
    return x[keep[:capacity]], y[keep[:capacity]]


def build_gpm_subspaces(
    model: nn.Module,
    memory_x: torch.Tensor,
    energy: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Construct historical input subspaces for every linear layer."""

    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(
                module.register_forward_pre_hook(
                    lambda _module, inputs, key=name: captured.__setitem__(
                        key, inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                    )
                )
            )
    model.eval()
    with torch.no_grad():
        model(memory_x.to(device))
    for handle in handles:
        handle.remove()

    subspaces: dict[str, torch.Tensor] = {}
    target = float(np.clip(energy, 0.0, 1.0))
    for name, values in captured.items():
        centered = values - values.mean(dim=0, keepdim=True)
        if min(centered.shape) < 1:
            continue
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        explained = singular.square()
        # torch.cumsum has no deterministic CUDA implementation in the
        # supported PyTorch build. Rank selection is tiny, so perform this
        # reduction on CPU while leaving the expensive SVD and training on GPU.
        explained_for_rank = explained.detach().cpu()
        cumulative = torch.cumsum(explained_for_rank, dim=0)
        cumulative = cumulative / explained_for_rank.sum().clamp_min(1e-12)
        rank = int(torch.searchsorted(cumulative, target).item()) + 1
        subspaces[name] = vh[:rank].T.detach()
    return subspaces


def train_repshield(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    subspaces: dict[str, torch.Tensor],
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
    linear = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    evaluations = 0
    for epoch in range(cfg.training.epochs_per_window):
        generator = torch.Generator().manual_seed(
            cfg.data.random_seed + seed_offset * 104729 + epoch
        )
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.training.batch_size):
            idx = order[start : start + cfg.training.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            with torch.no_grad():
                for name, basis in subspaces.items():
                    module = linear.get(name)
                    if module is None or module.weight.grad is None:
                        continue
                    basis = basis.to(module.weight.grad.device)
                    module.weight.grad.sub_((module.weight.grad @ basis) @ basis.T)
            optimizer.step()
            evaluations += 1
    return evaluations


def train_agem(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    memory: ReservoirMemory,
    cfg: ExperimentConfig,
    device: torch.device,
    seed_offset: int,
) -> int:
    """A-GEM with one global episodic-memory gradient constraint."""

    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    x, y = x.to(device), y.to(device)
    evaluations = 0
    for epoch in range(cfg.training.epochs_per_window):
        generator = torch.Generator().manual_seed(
            cfg.data.random_seed + seed_offset * 104729 + epoch
        )
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.training.batch_size):
            idx = order[start : start + cfg.training.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            new_gradients = [
                torch.zeros_like(parameter)
                if parameter.grad is None
                else parameter.grad.detach().clone()
                for parameter in parameters
            ]
            evaluations += 1

            episodic = memory.sample(cfg.dasp.memory_batch_size, device)
            if episodic is not None:
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(episodic[0]), episodic[1]).backward()
                reference_gradients = [
                    torch.zeros_like(parameter)
                    if parameter.grad is None
                    else parameter.grad.detach().clone()
                    for parameter in parameters
                ]
                evaluations += 1
                dot = sum(
                    torch.sum(current * reference)
                    for current, reference in zip(new_gradients, reference_gradients)
                )
                reference_norm = sum(
                    torch.sum(reference.square()) for reference in reference_gradients
                )
                coefficient = dot / reference_norm.clamp_min(1e-12) if dot < 0 else 0.0
                constrained = [
                    current - coefficient * reference
                    for current, reference in zip(new_gradients, reference_gradients)
                ]
            else:
                constrained = new_gradients
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, constrained):
                parameter.grad = gradient
            optimizer.step()
    return evaluations


@dataclass
class ErrorRateDriftDetector:
    threshold: float
    patience: int
    min_increase: float
    consecutive_warnings: int = 0

    @classmethod
    def from_warmup(
        cls,
        errors: list[float],
        z: float,
        patience: int,
        min_increase: float,
    ) -> "ErrorRateDriftDetector":
        values = np.asarray(errors, dtype=float)
        spread = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        threshold = float(values.mean() + z * max(spread, min_increase))
        return cls(min(1.0, threshold), max(1, patience), min_increase)

    def update(self, error: float) -> tuple[bool, bool, float]:
        warning = bool(error >= self.threshold and error - self.threshold >= self.min_increase)
        self.consecutive_warnings = self.consecutive_warnings + 1 if warning else 0
        drift = self.consecutive_warnings >= self.patience
        if drift:
            self.consecutive_warnings = 0
            self.threshold = min(1.0, error + self.min_increase)
        severity = max(0.0, error - self.threshold)
        return warning, drift, severity


def train_cdda(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
    seed_offset: int,
) -> int:
    """Budget-matched CDDA-MD-style incremental update.

    The benchmark deliberately uses the shared MLP and AdamW optimizer.  It
    therefore evaluates the CDDA-MD detector/adaptation policy, not a faithful
    reproduction of the paper's task-specific LSTM-attention architecture.
    """

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    x, y = x.to(device), y.to(device)
    evaluations = 0
    for epoch in range(cfg.training.epochs_per_window):
        generator = torch.Generator().manual_seed(
            cfg.data.random_seed + seed_offset * 104729 + epoch
        )
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), cfg.training.batch_size):
            idx = order[start : start + cfg.training.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            optimizer.step()
            evaluations += 1
    return evaluations


def clone_teacher(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).eval()
