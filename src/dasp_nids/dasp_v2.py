"""DASP-v2 (``modular_dasp_v2``): pre-registration section 4, components C1-C3.

C4 (a smaller shared head) is an architecture setting,
``model.modular_head_dim``. Every decision uses only the labels queried under
the common label-blind budget and gate labels that arrived in earlier
windows; no evaluator metric computed on the whole window is read.
"""
from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import original_review

VERSION = "modular-dasp-v2-20261018-r0"
GATE_POLICIES = ("wilson", "point", "off")
ROUTINGS = ("utility", "native", "all", "classifier", "random", "quarantine", "attributed")
BACKGROUNDS = ("classifier", "native", "all", "none")
PRIOR_CORRECTIONS = ("none", "em", "acc")
REALIGNS = ("none", "attributed", "random", "global")
# Methods whose predictions and delayed updates can use routed recalibration.
REALIGN_HOSTS = ("modular_dasp_v2", "experience_replay")


def em_prior(probabilities: torch.Tensor, clip: float, iterations: int = 200,
             tolerance: float = 1e-7) -> torch.Tensor:
    """Saerens-Latinne-Decaestecker EM for the target class prior.

    ``probabilities`` are posteriors of a classifier whose implicit training
    prior is uniform (logit-adjusted training with the batch prior), so the
    reweighting factor is simply the candidate prior.
    """

    n_classes = probabilities.shape[1]
    prior = torch.full((n_classes,), 1.0 / n_classes, dtype=probabilities.dtype)
    for _ in range(iterations):
        weighted = probabilities * prior
        weighted = weighted / weighted.sum(dim=1, keepdim=True).clamp_min(1e-12)
        updated = weighted.mean(dim=0).clamp(clip, 1 - clip)
        updated = updated / updated.sum()
        converged = float((updated - prior).abs().max()) < tolerance
        prior = updated
        if converged:
            break
    return prior


def acc_prior(predicted_attack_rate: float, tpr: float, fpr: float, clip: float) -> torch.Tensor | None:
    """Adjusted classify-and-count (Forman, 2008): binary hard-label BBSE."""

    if not np.isfinite(tpr) or not np.isfinite(fpr) or tpr - fpr < 0.05:
        return None
    attack = float(np.clip((predicted_attack_rate - fpr) / (tpr - fpr), clip, 1 - clip))
    return torch.tensor([1 - attack, attack])


def prediction_logits(experiment, model, x: torch.Tensor,
                      labels: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    """Corrected logits; ``labels`` are read only for warm-up windows (attribution)."""

    v2 = experiment.config.dasp_v2
    is_v2 = experiment.config.dasp.baseline == "modular_dasp_v2"
    realigning = experiment.config.dasp.baseline in REALIGN_HOSTS and v2.realign != "none"
    realigned = {}
    if realigning:
        x, realigned = realign_inputs(experiment, model, x)
    logits, diagnostics = _prediction_logits(experiment, model, x)
    if (is_v2 and v2.attribution != "none") or realigning:
        # Also the warm-up bookkeeping that recalibration needs.
        diagnostics = diagnostics | observe_attribution(experiment, model, x, logits, labels)
    return logits, diagnostics | realigned


def _prediction_logits(experiment, model, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Raw logits plus quarantine and label-shift corrections, and row diagnostics."""

    v2 = experiment.config.dasp_v2
    is_v2 = experiment.config.dasp.baseline == "modular_dasp_v2"
    quarantine_diagnostics = {}
    raw = None
    if is_v2 and v2.quarantine:
        raw, quarantine_diagnostics = quarantine_logits(experiment, model, x)
    if raw is None:
        raw = model(x)
    if not is_v2 or v2.prior_correction == "none":
        return raw, quarantine_diagnostics
    raw_logits, prior_diagnostics = _prior_corrected(experiment, raw)
    return raw_logits, quarantine_diagnostics | prior_diagnostics


def _prior_corrected(experiment, raw: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    v2 = experiment.config.dasp_v2
    if v2.prior_correction not in PRIOR_CORRECTIONS or v2.prior_source not in ("previous", "current"):
        raise ValueError("unknown DASP-v2 prior correction settings")
    state = experiment.__dict__.setdefault("_dasp_v2_prior", {})
    probabilities = F.softmax(raw, dim=1).detach().cpu()
    if v2.prior_correction == "em":
        observed = em_prior(probabilities, v2.prior_clip)
    else:
        rates = state.get("rates")
        attack_rate = float((probabilities.argmax(dim=1) == 1).float().mean())
        observed = None if rates is None else acc_prior(attack_rate, rates[0], rates[1], v2.prior_clip)
    estimate = observed if v2.prior_source == "current" else state.get("next")
    state["next"] = observed
    if estimate is None:
        return raw, {"prior_attack_estimate": float("nan")}
    offset = v2.prior_strength * torch.log(estimate.to(raw.device, raw.dtype))
    return raw + offset, {"prior_attack_estimate": float(estimate[-1])}


def _subsample(values: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if len(values) <= count:
        return values
    return values[torch.randperm(len(values), generator=generator)[:count]]


def _group_columns(model) -> list[torch.Tensor]:
    """Input columns read by each module (semantic groups or contiguous slices)."""

    if getattr(model, "feature_groups", None):
        return [torch.tensor(group, dtype=torch.long) for group in model.feature_groups]
    return [torch.arange(start, stop) for start, stop in model.feature_slices]


def _group_distance(reference: torch.Tensor, current: torch.Tensor, columns: torch.Tensor, samples: int) -> float:
    from .drift import multiscale_mmd

    return float(multiscale_mmd(reference[:, columns], current[:, columns], samples, [0.5, 1.0, 2.0]))


def _establish_trust(experiment, model, state) -> None:
    """Reference = pooled warm-up inputs; null = distance to each warm-up window.

    Warm-up windows are unlabeled here, so the null reflects real
    window-to-window variability, including class-mix fluctuation.
    """

    v2 = experiment.config.dasp_v2
    generator = torch.Generator().manual_seed(experiment.config.data.random_seed + 31337)
    windows = state.pop("warmup_inputs")
    pooled = torch.cat(windows)
    reference = _subsample(pooled, v2.quarantine_samples, generator)
    columns = _group_columns(model)
    null = {name: [] for name in model.layer_names}
    draws = [_subsample(window, v2.quarantine_samples, generator) for window in windows for _ in range(2)]
    for name, cols in zip(model.layer_names, columns):
        null[name] = [_group_distance(reference, draw, cols, v2.quarantine_samples) for draw in draws]
    state["reference"] = {name: reference for name in model.layer_names}
    state["median"] = {name: float(np.median(values)) for name, values in null.items()}
    state["mad"] = {name: max(float(np.median(np.abs(np.asarray(values) - np.median(values)))),
                              0.25 * float(np.median(values)), 1e-4) for name, values in null.items()}
    state["streak"] = {name: 0 for name in model.layer_names}
    state["quarantined"] = set()


def refresh_trust(experiment, model, x: torch.Tensor, modules: list[str]) -> None:
    """After a labeled repair, a repaired module's group is trusted on recent inputs."""

    state = experiment.__dict__.get("_dasp_v2_trust", {})
    if "reference" not in state or not modules:
        return
    generator = torch.Generator().manual_seed(experiment.config.data.random_seed + 104729)
    recent = _subsample(x.to(experiment.device), experiment.config.dasp_v2.quarantine_samples, generator)
    for name in modules:
        state["reference"][name] = recent
        state["streak"][name] = 0
        state["quarantined"].discard(name)


def quarantine_logits(experiment, model, x: torch.Tensor) -> tuple[torch.Tensor | None, dict]:
    """Replace persistently shifted feature groups' modules by their reference mean."""

    v2 = experiment.config.dasp_v2
    state = experiment.__dict__.setdefault("_dasp_v2_trust", {})
    if "reference" not in state:
        inputs = state.setdefault("warmup_inputs", [])
        if len(inputs) < experiment.config.data.warmup_windows:
            generator = torch.Generator().manual_seed(experiment.config.data.random_seed + len(inputs))
            inputs.append(_subsample(x.detach(), v2.quarantine_samples, generator))
            return None, {}
        _establish_trust(experiment, model, state)
    generator = torch.Generator().manual_seed(experiment.config.data.random_seed)
    current = _subsample(x.detach(), v2.quarantine_samples, generator)
    scores = {}
    for name, cols in zip(model.layer_names, _group_columns(model)):
        distance = _group_distance(state["reference"][name], current, cols, v2.quarantine_samples)
        z = max(0.0, (distance - state["median"][name]) / (1.4826 * state["mad"][name]))
        scores[name] = z
        state["streak"][name] = state["streak"][name] + 1 if z >= v2.quarantine_z else 0
    quarantined = {name for name in model.layer_names if state["streak"][name] >= 2}
    if len(quarantined) == len(model.layer_names):
        quarantined -= {min(quarantined, key=lambda n: scores[n])}  # never blank every module
    state["quarantined"] = quarantined
    _, representations = model(x, return_representations=True)
    if quarantined:
        for name in quarantined:
            _, reference = model(state["reference"][name], return_representations=True)
            representations[name] = reference[name].mean(dim=0, keepdim=True).expand_as(representations[name])
    return model.logits_from_representations(representations), {
        "quarantined_modules": "|".join(sorted(quarantined)),
        **{f"quarantine_z_{name}": round(value, 4) for name, value in scores.items()}}


def _class_conditional_z(state, columns, x: torch.Tensor, y: torch.Tensor, samples: int,
                         minimum: int, generator) -> dict[str, float]:
    """Count-weighted mean over classes of the robust z of group-input MMD."""

    scores = {}
    for name, cols in columns.items():
        total, weight = 0.0, 0
        for label, reference in state["class_reference"].items():
            members = x[y == label]
            if len(members) < minimum or name not in state["class_null"][label]:
                continue
            current = _subsample(members, samples, generator)
            median, mad = state["class_null"][label][name]
            distance = _group_distance(reference, current, cols, samples)
            total += len(members) * max(0.0, (distance - median) / (1.4826 * mad))
            weight += len(members)
        scores[name] = total / weight if weight else float("nan")
    return scores


def _attribution_reference(experiment, model, state) -> None:
    """Per-class warm-up references (warm-up labels are training data) and nulls."""

    v2 = experiment.config.dasp_v2
    generator = torch.Generator().manual_seed(experiment.config.data.random_seed + 4242)
    windows = state.pop("warmup_labeled")
    columns = dict(zip(model.layer_names, _group_columns(model)))
    state["class_reference"], state["class_null"] = {}, {}
    pooled_x = torch.cat([x for x, _ in windows])
    pooled_y = torch.cat([y for _, y in windows])
    state["class_moments"], state["moment_null"] = {}, {}
    for label in pooled_y.unique().tolist():
        members = pooled_x[pooled_y == label]
        if len(members) < 2 * v2.attribution_min_class:
            continue
        state["class_moments"][label] = _robust_moments(members)
        effects = [_moment_effects(state["class_moments"][label], _robust_moments(x[y == label]))
                   for x, y in windows if int((y == label).sum()) >= v2.attribution_min_class]
        if effects:
            state["moment_null"][label] = tuple(torch.stack(part).amax(dim=0) for part in zip(*effects))
        reference = _subsample(members, v2.quarantine_samples, generator)
        state["class_reference"][label] = reference
        state["class_null"][label] = {}
        draws = [_subsample(x[y == label], v2.quarantine_samples, generator)
                 for x, y in windows for _ in range(2) if int((y == label).sum()) >= v2.attribution_min_class]
        if len(draws) < 3:
            del state["class_reference"][label], state["class_null"][label]
            continue
        for name, cols in columns.items():
            null = np.asarray([_group_distance(reference, draw, cols, v2.quarantine_samples) for draw in draws])
            median = float(np.median(null))
            state["class_null"][label][name] = (median, max(float(np.median(np.abs(null - median))),
                                                            0.25 * median, 1e-4))


def observe_attribution(experiment, model, x: torch.Tensor, logits: torch.Tensor,
                        labels: torch.Tensor | None) -> dict[str, float]:
    """Record class-conditional attribution for this window; labels only in warm-up."""

    v2 = experiment.config.dasp_v2
    state = experiment.__dict__.setdefault("_dasp_v2_attr", {"calls": 0, "scores": {}})
    index = state["calls"]
    state["calls"] += 1
    warmup = experiment.config.data.warmup_windows
    if index < warmup:
        if labels is None:
            raise ValueError("class-conditional attribution needs warm-up labels")
        state.setdefault("warmup_labeled", []).append((x.detach().cpu(), labels.detach().cpu()))
        return {}
    if "class_reference" not in state:
        _attribution_reference(experiment, model, state)
    if v2.attribution != "pseudo":
        return {}
    generator = torch.Generator().manual_seed(experiment.config.data.random_seed + index)
    predicted = logits.argmax(dim=1).detach().cpu()
    columns = dict(zip(model.layer_names, _group_columns(model)))
    scores = _class_conditional_z(state, columns, x.detach().cpu(), predicted,
                                  v2.quarantine_samples, v2.attribution_min_class, generator)
    state["scores"][index] = scores
    return {f"attr_cc_z_{name}": (round(value, 4) if np.isfinite(value) else value) for name, value in scores.items()}


def _robust_moments(x: torch.Tensor) -> dict[str, torch.Tensor]:
    q25, q50, q75 = torch.quantile(x.double(), torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64), dim=0)
    return {"median": q50, "iqr": q75 - q25, "std": x.double().std(dim=0)}


def _scale(moments: dict[str, torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    return torch.where(moments["iqr"] > eps, moments["iqr"], moments["std"])


def _moment_effects(reference: dict, current: dict, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature location change in reference-scale units, and |log scale ratio|."""

    scale = _scale(reference)
    location = (current["median"] - reference["median"]).abs() / scale.clamp_min(eps)
    ratio = (torch.log(_scale(current).clamp_min(eps)) - torch.log(scale.clamp_min(eps))).abs()
    return location, ratio


def apply_maps(x: torch.Tensor, maps: dict) -> torch.Tensor:
    """Apply per-feature affine recalibrations, one per feature group."""

    if not maps:
        return x
    x = x.clone()
    for columns, scale, shift in maps.values():
        x[:, columns] = (x[:, columns].double() * scale + shift).to(x.dtype)
    return x


def estimate_map(raw: torch.Tensor, pseudo: torch.Tensor, moments: dict, columns: torch.Tensor,
                 minimum: int, null: dict | None = None, eps: float = 1e-6):
    """Class-conditional affine that maps the current group back to warm-up.

    Per class with enough pseudo-labelled members: scale from IQR (or the
    standard deviation when an IQR is zero), location from the median. A
    feature that collapsed to a constant is mapped to the reference median.
    Classes are combined weighted by their count. With a warm-up ``null``,
    a feature is recalibrated only where its change exceeds twice the largest
    warm-up window-to-window change; other features keep the identity.
    """

    total_scale, total_shift, weight = 0.0, 0.0, 0
    for label, reference in moments.items():
        members = raw[pseudo == label][:, columns]
        if len(members) < minimum:
            continue
        current = _robust_moments(members)
        ref = {key: value[columns] for key, value in reference.items()}
        use_iqr = (ref["iqr"] > eps) & (current["iqr"] > eps)
        use_std = ~use_iqr & (ref["std"] > eps) & (current["std"] > eps)
        scale = torch.where(use_iqr, ref["iqr"] / current["iqr"].clamp_min(eps),
                            torch.where(use_std, ref["std"] / current["std"].clamp_min(eps),
                                        torch.ones_like(ref["std"])))
        collapsed = (current["std"] <= eps) & (ref["std"] > eps)
        scale = torch.where(collapsed, torch.zeros_like(scale), scale)
        scale = scale.clamp(0.0, 1e4)
        shift = ref["median"] - scale * current["median"]
        if null is not None and label in null:
            location, ratio = _moment_effects(ref, current)
            null_location, null_ratio = (value[columns] for value in null[label])
            changed = (location > torch.clamp(2 * null_location, min=0.25)) \
                | (ratio > torch.clamp(2 * null_ratio, min=math.log(1.25))) | collapsed
            scale = torch.where(changed, scale, torch.ones_like(scale))
            shift = torch.where(changed, shift, torch.zeros_like(shift))
        total_scale = total_scale + len(members) * scale
        total_shift = total_shift + len(members) * shift
        weight += len(members)
    if not weight:
        return None
    return columns, total_scale / weight, total_shift / weight


def class_consistent(mapped: torch.Tensor, pseudo: torch.Tensor, moments: dict, columns: torch.Tensor,
                     changed: torch.Tensor, minimum: int, tolerance: float) -> tuple[bool, int]:
    """Does the shared map return every well-populated class to its own
    warm-up medians on the recalibrated features? Returns (consistent,
    number of classes checked); with fewer than two classes it cannot tell."""

    checked, consistent = 0, True
    if not bool(changed.any()):
        return True, 0
    for label, reference in moments.items():
        members = mapped[pseudo == label][:, columns]
        if len(members) < minimum:
            continue
        checked += 1
        ref = {key: value[columns] for key, value in reference.items()}
        location, _ = _moment_effects(ref, _robust_moments(members))
        if float((location[changed] > tolerance).float().mean()) > 0.25:
            consistent = False
    return (consistent if checked >= 2 else True), checked


def _leave_out_predictions(model, x: torch.Tensor, name: str, reference: torch.Tensor, device) -> torch.Tensor:
    """Predictions with one module replaced by its warm-up mean output, so the
    pseudo-labels used to recalibrate a group never read that group."""

    model.eval()
    with torch.no_grad():
        _, representations = model(x.to(device), return_representations=True)
        _, anchor = model(reference.to(device), return_representations=True)
        representations[name] = anchor[name].mean(dim=0, keepdim=True).expand_as(representations[name])
        return model.logits_from_representations(representations).argmax(dim=1).cpu()


def realign_inputs(experiment, model, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Routed, label-free recalibration of the attributed feature group."""

    cfg, v2 = experiment.config, experiment.config.dasp_v2
    if v2.realign not in REALIGNS:
        raise ValueError(f"unknown DASP-v2 realign mode {v2.realign!r}")
    attr = experiment.__dict__.setdefault("_dasp_v2_attr", {"calls": 0, "scores": {}})
    state = experiment.__dict__.setdefault("_dasp_v2_align", {"maps": {}, "by_window": {}, "changes": {}})
    index = attr["calls"]
    if index < cfg.data.warmup_windows:
        state["by_window"][index] = {}
        return x, {}
    if "class_reference" not in attr:
        _attribution_reference(experiment, model, attr)
    names = list(model.layer_names)
    columns = dict(zip(names, _group_columns(model)))
    raw = x.detach().cpu()
    previous = dict(state["maps"])
    maps = dict(previous)
    generator = torch.Generator().manual_seed(cfg.data.random_seed + 7919 + index)
    model.eval()
    dropped = []
    if maps and v2.realign != "global":
        # A map lives only while its raw group stays shifted (reverting faults).
        with torch.no_grad():
            predicted = model(apply_maps(raw, maps).to(experiment.device)).argmax(dim=1).cpu()
        raw_scores = _class_conditional_z(attr, {name: columns[name] for name in maps}, raw, predicted,
                                          v2.quarantine_samples, v2.attribution_min_class, generator)
        for name, value in raw_scores.items():
            if np.isfinite(value) and value < v2.realign_z:
                maps.pop(name)
                dropped.append(name)
    aligned = apply_maps(raw, maps)
    with torch.no_grad():
        predicted = model(aligned.to(experiment.device)).argmax(dim=1).cpu()
    scores = _class_conditional_z(attr, columns, aligned, predicted, v2.quarantine_samples,
                                  v2.attribution_min_class, generator)
    finite = {name: value for name, value in scores.items() if np.isfinite(value)}
    peak = max(finite.values()) if finite else float("nan")
    triggered = bool(finite) and peak >= v2.realign_z
    top = max(finite, key=lambda name: (finite[name], -names.index(name))) if triggered else None
    streak = state.get("streak", (None, 0))
    streak = (top, streak[1] + 1 if top is not None and streak[0] == top else int(top is not None))
    state["streak"] = streak
    persistent = triggered and (streak[1] >= v2.realign_persistence
                                or peak >= v2.realign_fast_factor * v2.realign_z)
    if v2.realign == "global":
        candidates = list(names)
    elif not persistent:
        candidates = []
    elif v2.realign == "attributed":
        candidates = [top]
    else:
        pick = torch.Generator().manual_seed(cfg.data.random_seed * 1009 + index)
        candidates = [names[int(torch.randint(len(names), (1,), generator=pick))]]
    reference = torch.cat(list(attr["class_reference"].values()))
    null = attr.get("moment_null") if v2.realign_feature_mask else None
    refreshed = [name for name in maps if name not in candidates] if v2.realign_refresh else []
    accepted, residual, inconsistent = [], {}, []
    for name in [*candidates, *refreshed]:
        pseudo = _leave_out_predictions(model, aligned, name, reference, experiment.device)
        new_map = estimate_map(raw, pseudo, attr["class_moments"], columns[name], v2.attribution_min_class, null)
        if new_map is None or not bool(((new_map[1] != 1) | (new_map[2] != 0)).any()):
            continue
        trial = maps | {name: new_map}
        if v2.realign_class_check and v2.realign != "global":
            changed = (new_map[1] != 1) | (new_map[2] != 0)
            ok, _ = class_consistent(apply_maps(raw, trial), pseudo, attr["class_moments"], columns[name],
                                     changed, v2.attribution_min_class, v2.realign_class_tolerance)
            if not ok:
                inconsistent.append(name)
                continue
        if v2.realign_accept and v2.realign != "global":
            after = _class_conditional_z(attr, {name: columns[name]}, apply_maps(raw, trial), pseudo,
                                         v2.quarantine_samples, v2.attribution_min_class, generator)[name]
            residual[name] = after
            if not (np.isfinite(after) and after < v2.realign_z):
                continue
        maps = trial
        accepted.append(name)
    state["maps"] = maps
    state["by_window"][index] = dict(maps)
    state["changes"][index] = (previous, accepted)
    return apply_maps(raw, maps).to(x.device), {
        "realign_trigger_z": round(peak, 4) if np.isfinite(peak) else peak,
        "realign_triggered": triggered,
        "realign_persistent": persistent,
        "realign_dropped": "|".join(dropped),
        "realign_accepted": "|".join(n for n in accepted if n in candidates),
        "realign_refreshed": "|".join(n for n in accepted if n not in candidates),
        "realign_rejected": "|".join(n for n in candidates if n not in accepted),
        "realign_class_inconsistent": "|".join(inconsistent),
        "realign_residual_z": json.dumps({k: round(float(v), 4) for k, v in residual.items()}),
        "realign_active": "|".join(sorted(maps)),
        "realign_parameters": int(sum(2 * len(cols) for cols, _, _ in maps.values())),
    }


def realigned_window(experiment, index: int, x: torch.Tensor) -> torch.Tensor:
    """The window as the model saw it at prediction time."""

    maps = experiment.__dict__.get("_dasp_v2_align", {}).get("by_window", {}).get(index, {})
    return apply_maps(x, maps)


def validate_realignment(experiment, model, index: int, raw_x: torch.Tensor, y: torch.Tensor) -> dict:
    """Delayed labels revert maps accepted at ``index`` that raised the loss."""

    v2 = experiment.config.dasp_v2
    state = experiment.__dict__.get("_dasp_v2_align", {})
    previous, accepted = state.get("changes", {}).get(index, ({}, []))
    if not accepted or not v2.realign_validate or v2.realign == "global":
        return {}
    n_classes = int(model.classifier.out_features)
    with_map = balanced_loss(model, apply_maps(raw_x, state["by_window"][index]), y, experiment.device, n_classes)
    without = balanced_loss(model, apply_maps(raw_x, previous), y, experiment.device, n_classes)
    reverted = []
    if with_map > without + v2.realign_margin:
        for name in accepted:
            if state["maps"].get(name) is state["by_window"][index].get(name):
                if name in previous:
                    state["maps"][name] = previous[name]
                else:
                    state["maps"].pop(name, None)
                reverted.append(name)
    return {"realign_validated_window": index, "realign_loss_with": round(with_map, 6),
            "realign_loss_without": round(without, 6), "realign_reverted": "|".join(reverted)}


def labeled_attribution(experiment, model, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Same statistic on the queried delayed labels (classes with enough labels only)."""

    v2 = experiment.config.dasp_v2
    state = experiment.__dict__.get("_dasp_v2_attr", {})
    if "class_reference" not in state:
        return {}
    generator = torch.Generator().manual_seed(experiment.config.data.random_seed + 977)
    columns = dict(zip(model.layer_names, _group_columns(model)))
    return _class_conditional_z(state, columns, x.cpu(), y.cpu(), v2.quarantine_samples,
                                max(8, v2.attribution_min_class // 4), generator)


def combined_attribution(experiment, item, model, labeled: dict[str, float]) -> dict[str, float]:
    """Class-conditional input z plus weighted representation z, per module."""

    v2 = experiment.config.dasp_v2
    if v2.attribution == "labels":
        conditional = labeled
    else:
        conditional = experiment.__dict__.get("_dasp_v2_attr", {}).get("scores", {}).get(item.window.index, {})
    representation = item.decision.z_scores or {}
    combined = {}
    for name in model.layer_names:
        value = conditional.get(name, float("nan"))
        value = 0.0 if not np.isfinite(value) else value
        combined[name] = value + v2.attribution_repr_weight * float(representation.get(name, 0.0))
    return combined


def record_operating_rates(experiment, model, x: torch.Tensor, y: torch.Tensor) -> None:
    """TPR/FPR of the raw (uncorrected) classifier on held-out labeled gate data."""

    model.eval()
    with torch.no_grad():
        predicted = model(x.to(experiment.device)).argmax(dim=1).cpu()
    y = y.cpu()
    attacks, benign = y == 1, y == 0
    tpr = float((predicted[attacks] == 1).float().mean()) if bool(attacks.any()) else float("nan")
    fpr = float((predicted[benign] == 1).float().mean()) if bool(benign.any()) else float("nan")
    experiment.__dict__.setdefault("_dasp_v2_prior", {})["rates"] = (tpr, fpr)


def wilson_lower_bound(successes: int, total: int, z: float) -> float:
    """Lower end of the Wilson score interval for a binomial proportion."""

    if total <= 0:
        return float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - margin) / denominator


def gate_assessment(
    y: torch.Tensor,
    prediction: torch.Tensor,
    n_classes: int,
    policy: str,
    min_support: int,
    threshold: float,
    z: float,
) -> dict[str, object]:
    """C1: per-class evidence instead of a fixed-class Macro-F1.

    A class is assessed only with at least ``min_support`` labels. Under
    ``wilson`` the model is degraded unless every assessed class has a
    recall whose Wilson lower bound reaches ``threshold``; ``point`` compares
    the point estimate. Without any assessable class the state is
    ``insufficient_support``, which is never read as evidence of health or of
    degradation.
    """

    if policy not in GATE_POLICIES:
        raise ValueError(f"unknown DASP-v2 gate policy: {policy}")
    y, prediction = y.cpu(), prediction.cpu()
    support, recall, lower = [], [], []
    for label in range(n_classes):
        mask = y == label
        count = int(mask.sum())
        correct = int((prediction[mask] == label).sum())
        support.append(count)
        recall.append(correct / count if count else float("nan"))
        lower.append(wilson_lower_bound(correct, count, z) if count else float("nan"))
    assessed = [label for label in range(n_classes) if support[label] >= min_support]
    if policy == "off":
        state, degraded = "disabled", False
    elif not assessed:
        state, degraded = "insufficient_support", False
    else:
        evidence = lower if policy == "wilson" else recall
        degraded = any(evidence[label] < threshold for label in assessed)
        state = "observed_degradation" if degraded else "certified"
    return {"state": state, "degraded": bool(degraded), "support": support,
            "recall": recall, "wilson_lower": lower, "assessed_classes": assessed}


def class_prior_adjustment(memory, n_classes: int, tau: float, batch=None) -> torch.Tensor | None:
    """C3: tau * log prior (Laplace-smoothed).

    With ``batch`` = (labels, weights) the prior is the weighted class
    frequency of the training batch itself; otherwise labels in memory.
    """

    if tau <= 0:
        return None
    counts = torch.ones(n_classes)
    if batch is not None:
        labels, weights = batch
        counts += torch.bincount(labels.cpu().long(), weights=weights.cpu().float(), minlength=n_classes)
    else:
        stored = memory.tensors(torch.device("cpu"))
        if stored is not None:
            counts += torch.bincount(stored[1].long(), minlength=n_classes).float()
    return tau * torch.log(counts / counts.sum())


def balanced_loss(model, x, y, device, n_classes: int) -> float:
    """Class-balanced cross-entropy, so a rare class is not drowned out."""

    model.eval()
    with torch.no_grad():
        losses = F.cross_entropy(model(x.to(device)), y.to(device), reduction="none").cpu()
    per_class = [losses[y.cpu() == label].mean() for label in range(n_classes) if bool((y == label).any())]
    return float(torch.stack(per_class).mean())


def utility_routing(experiment, model, names, batch, gate, adjustment, epochs, top_k, min_gain):
    """C2: route to modules whose short probe update helps on gate labels.

    Each module is probed together with the head and compared against a
    head-only probe, so the gain is what the module adds beyond the shared
    classifier. Probe updates run on copies and are charged to the
    ``routing_probe`` gradient phase.
    """

    from . import runner as r

    cfg, device = experiment.config, experiment.device
    bx, by, weights = batch
    n_classes = int(model.classifier.out_features)

    def probe(active):
        candidate = copy.deepcopy(model)
        plasticities = {name: float(name in active) for name in names}
        plasticities["classifier"] = 1.0
        r._train_dasp(candidate, bx, by, plasticities, cfg, device, None,
                      eta_ceiling=1.0, strict_freeze=True, sample_weights=weights,
                      epochs=epochs, use_projection_memory=False, logit_adjustment=adjustment)
        return balanced_loss(candidate, gate[0], gate[1], device, n_classes)

    head_only = probe(set())
    gains = {name: head_only - probe({name}) for name in names}
    ranked = sorted((name for name in names if gains[name] > min_gain),
                    key=lambda name: (-gains[name], names.index(name)))
    return ranked[:max(0, int(top_k))], gains, head_only


def update(experiment, model, memory, item, row) -> str:
    from . import runner as r

    cfg, v2 = experiment.config, experiment.config.dasp_v2
    if v2.routing not in ROUTINGS or v2.background not in BACKGROUNDS:
        raise ValueError("unknown DASP-v2 routing or background policy")
    device = experiment.device
    original_review.synchronize(device)
    started = time.perf_counter()
    names = list(model.layer_names)
    n_classes = int(model.classifier.out_features)

    # Common label-blind query, the same indices every budget-matched method uses.
    query = experiment._query_indices(item.window)
    if len(query) < 2:
        raise ValueError("DASP-v2 needs at least two queried labels")
    gate_count = min(len(query) - 1, max(1, round(len(query) * v2.gate_fraction)))
    train_ids, gate_ids = query[:-gate_count], query[-gate_count:]
    realign_row = {}
    if v2.realign != "none":
        realign_row = validate_realignment(experiment, model, item.window.index,
                                           item.window.x[query], item.window.y[query])
    # Training and gating see the window exactly as it was predicted.
    window_x = realigned_window(experiment, item.window.index, item.window.x) if v2.realign != "none" \
        else item.window.x
    tx, ty = window_x[train_ids], item.window.y[train_ids]
    gx, gy = window_x[gate_ids], item.window.y[gate_ids]

    history = getattr(experiment, "_dasp_v2_gate_history", [])
    recent = history[-v2.gate_buffer_windows:] if v2.gate_buffer_windows > 0 else []
    pool_x = torch.cat([gx, *[x for x, _ in recent]])
    pool_y = torch.cat([gy, *[y for _, y in recent]])

    model.eval()
    with torch.no_grad():
        pool_prediction = model(pool_x.to(device)).argmax(1).cpu()
    gate = gate_assessment(pool_y, pool_prediction, n_classes, v2.gate_policy,
                           v2.gate_min_support, v2.gate_threshold, v2.gate_z)
    drift = bool(item.decision.triggered)
    quarantined = sorted(experiment.__dict__.get("_dasp_v2_trust", {}).get("quarantined", set()))
    repair = drift or gate["degraded"] or bool(quarantined)

    attribution_x, attribution_y = torch.cat([tx, gx]), torch.cat([ty, gy])
    if v2.train_on_gate:
        if v2.routing == "utility":
            # Utility probes are scored on the gate pool; training on it too
            # would score candidates on their own training data.
            raise ValueError("train_on_gate cannot be combined with utility routing")
        # The gate decision is made; its labels now also train the model.
        tx, ty = attribution_x, attribution_y
    training_cfg = copy.deepcopy(cfg)
    training_cfg.dasp.modular_new_sample_weight = v2.new_sample_weight
    if not v2.replay_enabled:
        training_cfg.dasp.modular_replay_size = 0
    if v2.balanced_replay or not v2.replay_enabled:
        batch = r._localized_rehearsal_batch(tx, ty, memory, training_cfg, device)
    else:
        replay = memory.sample(training_cfg.dasp.modular_replay_size, device)
        new_x, new_y = tx.to(device), ty.to(device)
        new_w = torch.full((len(ty),), float(v2.new_sample_weight), device=device)
        batch = ((new_x, new_y, new_w) if replay is None else
                 (torch.cat([new_x, replay[0]]), torch.cat([new_y, replay[1]]),
                  torch.cat([new_w, torch.ones(len(replay[1]), device=device)])))
    if v2.logit_adjustment_prior not in ("memory", "batch"):
        raise ValueError("logit_adjustment_prior must be memory or batch")
    adjustment = class_prior_adjustment(
        memory, n_classes, v2.logit_adjustment_tau,
        (batch[1], batch[2]) if v2.logit_adjustment_prior == "batch" else None)

    counters: dict[str, dict[str, int]] = {}
    gains: dict[str, float] = {}
    conflicts: dict[str, float] = {}
    epochs = v2.repair_epochs
    if repair:
        candidate = "drift_repair" if drift else "gate_repair"
        if v2.routing == "utility":
            with original_review.gradient_scope(counters, "routing_probe"):
                active, gains, _ = utility_routing(
                    experiment, model, names, batch, (pool_x, pool_y), adjustment,
                    v2.utility_probe_epochs, v2.routing_top_k, v2.utility_min_gain)
        elif v2.routing == "native":
            memory_probe, _ = r._split_memory_for_probe_and_guard(memory, cfg.dasp.memory_batch_size, device)
            with original_review.gradient_scope(counters, "conflict"):
                conflicts = r._gradient_conflicts(model, tx[:cfg.dasp.memory_batch_size],
                                                  ty[:cfg.dasp.memory_batch_size], memory_probe, device)
            raw = r._plasticities("modular_dasp", item.decision.requirements, conflicts, cfg.dasp.conflict_weight)
            active = [name for name, value in sorted(raw.items(), key=lambda pair: -pair[1]) if value > 0][:v2.routing_top_k]
        elif v2.routing == "all":
            active = list(names)
        elif v2.routing == "random":
            generator = torch.Generator().manual_seed(cfg.data.random_seed * 1009 + item.window.index)
            active = [names[i] for i in torch.randperm(len(names), generator=generator)[:v2.routing_top_k].tolist()]
        elif v2.routing == "attributed":
            attribution = combined_attribution(
                experiment, item, model,
                labeled_attribution(experiment, model, attribution_x, attribution_y)
                if v2.attribution == "labels" else {})
            row["attribution_scores"] = json.dumps({k: round(v, 4) for k, v in attribution.items()})
            active = sorted(names, key=lambda n: (-attribution[n], names.index(n)))[:max(1, v2.routing_top_k)]
        elif v2.routing == "quarantine":
            # Repair what inference had to quarantine; otherwise the MMD attribution.
            if quarantined:
                active = quarantined[:max(1, v2.routing_top_k)]
            else:
                ranked = sorted(item.decision.requirements.items(), key=lambda pair: -pair[1])
                active = [name for name, value in ranked if value > 0][:v2.routing_top_k]
        else:
            active = []
        plasticities = {name: float(name in active) for name in names}
        plasticities["classifier"] = 1.0
    else:
        candidate = f"background_{v2.background}"
        if v2.background == "native":
            memory_probe, _ = r._split_memory_for_probe_and_guard(memory, cfg.dasp.memory_batch_size, device)
            with original_review.gradient_scope(counters, "conflict"):
                conflicts = r._gradient_conflicts(model, tx[:cfg.dasp.memory_batch_size],
                                                  ty[:cfg.dasp.memory_batch_size], memory_probe, device)
            active = sorted(names, key=lambda name: -conflicts.get(name, 0.0))[:v2.routing_top_k]
            plasticities = {name: (v2.background_plasticity if name in active else 0.0) for name in names}
        elif v2.background == "all":
            plasticities = {name: 1.0 for name in names}
        else:
            plasticities = {name: 0.0 for name in names}
        plasticities["classifier"] = 0.0 if v2.background == "none" else 1.0

    if "head_core" in model.adaptation_parameter_groups():
        plasticities["head_core"] = 1.0 if v2.head_core_update else 0.0
    before = ({name: value.detach().cpu().clone() for name, value in model.named_parameters()}
              if v2.verify_locality else None)
    calls = 0
    if any(value > 0 for value in plasticities.values()):
        with original_review.gradient_scope(counters, "training"):
            calls = r._train_dasp(model, *batch[:2], plasticities, training_cfg, device, memory,
                                  eta_ceiling=1.0, strict_freeze=True, sample_weights=batch[2],
                                  epochs=epochs, use_projection_memory=False,
                                  logit_adjustment=adjustment)
    memory.add(tx, ty)
    if v2.quarantine:
        refresh_trust(experiment, model, torch.cat([tx, gx]), [n for n in names if plasticities[n] > 0])
    if v2.prior_correction == "acc":
        record_operating_rates(experiment, model, pool_x, pool_y)
    experiment._dasp_v2_gate_history = [*history, (gx, gy)][-max(1, v2.gate_buffer_windows):]

    row.update(original_review._locality(model, before, plasticities))
    row.update({
        "dasp_v2_version": VERSION,
        "labels_exposed_total": len(query), "ground_truth_update_labels": len(query),
        "new_training_labels_used": len(ty), "labels_used_for_gating": len(gate_ids),
        "gate_pool_size": len(pool_y), "gate_state": gate["state"],
        "gate_support": json.dumps(gate["support"]),
        "gate_wilson_lower": json.dumps([None if math.isnan(v) else round(v, 6) for v in gate["wilson_lower"]]),
        "gate_degraded": gate["degraded"], "drift_triggered": drift,
        "selected_candidate": candidate, "accepted": "dasp",
        "routing_policy": v2.routing if repair else v2.background,
        "routing_gains": json.dumps({k: round(v, 6) for k, v in gains.items()}),
        "localized_repair_modules": "|".join(n for n in names if plasticities[n] > 0),
        "localized_repair_module_count": sum(plasticities[n] > 0 for n in names),
        "classifier_updated": plasticities["classifier"] > 0,
        "logit_adjustment": json.dumps(None if adjustment is None else [round(float(v), 6) for v in adjustment]),
        "localized_rehearsal_size": len(batch[1]), "localized_replay_size": len(batch[1]) - len(ty),
        "gradient_evaluations": calls, "training_gradient_evaluations": calls,
        "routing_probe_gradient_evaluations": counters.get("routing_probe", {}).get("calls", 0),
        "conflict_gradient_evaluations": counters.get("conflict", {}).get("calls", 0),
        "overall_gradient_evaluations": sum(c["calls"] for c in counters.values()),
        "parameter_gradient_element_evaluations": sum(c["requested_elements"] for c in counters.values()),
        "gradient_phases": json.dumps(counters, sort_keys=True),
        "quarantined_at_update": "|".join(quarantined),
        "global_update": False,
        "rss_bytes": int(experiment._process.memory_info().rss),
        "query_indices": json.dumps(query.tolist()),
        "label_query_policy": f"{cfg.baselines.query_policy}_label_blind",
    })
    row.update({f"plasticity_{name}": value for name, value in plasticities.items()})
    row.update({f"conflict_{name}": value for name, value in conflicts.items()})
    row.update(realign_row)
    original_review.synchronize(device)
    row["update_seconds"] = time.perf_counter() - started
    return "dasp"


def annotate_outputs(experiment, rows, summary) -> None:
    if experiment.config.dasp.baseline != "modular_dasp_v2":
        return
    evaluated = [row for row in rows if not row["warmup"]]
    summary.update({
        "dasp_v2_version": VERSION,
        "algorithm_id": f"{VERSION}/{experiment.config.dasp_v2.routing}/{experiment.config.dasp_v2.gate_policy}",
        "primary_comparison_scope": "matched_total_query_budget",
        "gradient_accounting_scope": "post-warmup adaptation: training, routing probes and conflicts; excludes warmup",
        "gradient_unit": "requested parameter-gradient tensor elements; not FLOPs",
        "total_overall_gradient_evaluations": sum(int(r.get("overall_gradient_evaluations", 0)) for r in evaluated),
        "total_routing_probe_gradient_evaluations": sum(int(r.get("routing_probe_gradient_evaluations", 0)) for r in evaluated),
        "total_parameter_gradient_element_evaluations": sum(int(r.get("parameter_gradient_element_evaluations", 0)) for r in evaluated),
        "repairs": sum(1 for r in evaluated if str(r.get("selected_candidate", "")).endswith("repair")),
        "gate_state_counts": {state: sum(1 for r in evaluated if r.get("gate_state") == state)
                              for state in ("certified", "observed_degradation", "insufficient_support", "disabled")},
        "frozen_parameters_do_not_imply_functional_or_causal_locality": True,
    })
    tp, fn, fp, tn = [sum(int(r[k]) for r in evaluated) for k in ("attack_tp", "attack_fn", "attack_fp", "attack_tn")]
    summary["cumulative_benign_fpr"] = fp / (fp + tn) if fp + tn else float("nan")
    rates = [r["attack_fp"] / (r["attack_fp"] + r["attack_tn"]) for r in evaluated if r["attack_fp"] + r["attack_tn"]]
    summary["mean_window_benign_fpr"] = float(np.mean(rates)) if rates else float("nan")
    summary["operational_error_counts"] = dict(tp=tp, fn=fn, fp=fp, tn=tn)
