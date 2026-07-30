from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    y_score: np.ndarray | None = None,
) -> dict[str, float | int]:
    labels = list(range(n_classes))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    observed = np.unique(y_true)
    balanced = float(np.mean([
        np.mean(y_pred[y_true == label] == label) for label in observed
    ]))
    attack_true = y_true != 0
    attack_pred = y_pred != 0
    tp = int(np.sum(attack_true & attack_pred))
    fn = int(np.sum(attack_true & ~attack_pred))
    fp = int(np.sum(~attack_true & attack_pred))
    tn = int(np.sum(~attack_true & ~attack_pred))
    attack_recall = float(tp / (tp + fn)) if tp + fn else np.nan
    attack_true_numeric = attack_true.astype(np.int8)
    pr_auc = (
        float(average_precision_score(attack_true_numeric, y_score))
        if y_score is not None and len(np.unique(attack_true_numeric)) == 2
        else np.nan
    )
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced),
        "attack_recall": attack_recall,
        "fnr": float(1.0 - attack_recall) if np.isfinite(attack_recall) else np.nan,
        "pr_auc": pr_auc,
        "mcc": float(
            matthews_corrcoef(
                attack_true_numeric, attack_pred.astype(np.int8)
            )
        ),
        "attack_tp": tp,
        "attack_fn": fn,
        "attack_fp": fp,
        "attack_tn": tn,
    }


def summarize_windows(rows: list[dict[str, object]], warmup_windows: int) -> dict[str, float | int]:
    evaluated = [row for row in rows if int(row["window"]) >= warmup_windows]
    if not evaluated:
        raise ValueError("No post-warmup windows were evaluated")
    result: dict[str, float | int] = {"evaluated_windows": len(evaluated)}
    for metric in [
        "macro_f1",
        "balanced_accuracy",
        "attack_recall",
        "fnr",
        "pr_auc",
        "mcc",
    ]:
        values = np.array([float(row[metric]) for row in evaluated])
        finite = values[np.isfinite(values)]
        result[f"defined_windows_{metric}"] = int(len(finite))
        if not len(finite):
            result[f"mean_{metric}"] = np.nan
            result[f"std_{metric}"] = np.nan
            result[f"p10_{metric}"] = np.nan
            result[f"worst_{metric}"] = np.nan
            continue
        result[f"mean_{metric}"] = float(finite.mean())
        result[f"std_{metric}"] = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
        result[f"p10_{metric}"] = float(np.quantile(finite, 0.10))
        result[f"worst_{metric}"] = float(finite.min() if metric != "fnr" else finite.max())
    macro = np.asarray([float(row["macro_f1"]) for row in evaluated], dtype=float)
    if len(macro) > 1:
        deltas = np.diff(macro)
        result["mean_absolute_window_delta_macro_f1"] = float(np.mean(np.abs(deltas)))
        result["max_window_drop_macro_f1"] = float(max(0.0, -float(deltas.min())))
        result["window_stability_macro_f1"] = float(
            np.clip(1.0 - np.mean(np.abs(deltas)), 0.0, 1.0)
        )
    else:
        result["mean_absolute_window_delta_macro_f1"] = 0.0
        result["max_window_drop_macro_f1"] = 0.0
        result["window_stability_macro_f1"] = 1.0
    running_best = np.maximum.accumulate(macro)
    forgetting_curve = running_best - macro
    result["mean_online_forgetting"] = float(forgetting_curve.mean())
    result["final_online_forgetting"] = float(forgetting_curve[-1])
    tp = sum(int(row["attack_tp"]) for row in evaluated)
    fn = sum(int(row["attack_fn"]) for row in evaluated)
    fp = sum(int(row["attack_fp"]) for row in evaluated)
    tn = sum(int(row["attack_tn"]) for row in evaluated)
    result["cumulative_attack_recall"] = float(tp / (tp + fn)) if tp + fn else np.nan
    result["cumulative_fnr"] = float(fn / (tp + fn)) if tp + fn else np.nan
    attack_f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else np.nan
    benign_f1 = float(2 * tn / (2 * tn + fp + fn)) if 2 * tn + fp + fn else np.nan
    defined_f1 = [value for value in [benign_f1, attack_f1] if np.isfinite(value)]
    result["cumulative_benign_f1"] = benign_f1
    result["cumulative_attack_f1"] = attack_f1
    result["cumulative_macro_f1"] = float(np.mean(defined_f1)) if defined_f1 else np.nan
    benign_recall = float(tn / (tn + fp)) if tn + fp else np.nan
    defined_recalls = [
        value for value in [benign_recall, result["cumulative_attack_recall"]]
        if np.isfinite(value)
    ]
    result["cumulative_benign_recall"] = benign_recall
    result["cumulative_balanced_accuracy"] = (
        float(np.mean(defined_recalls)) if defined_recalls else np.nan
    )
    result["cumulative_accuracy"] = float((tp + tn) / (tp + tn + fp + fn))
    result["adaptation_triggers"] = int(sum(bool(row.get("triggered")) for row in evaluated))
    result["dasp_commits"] = int(sum(row.get("accepted") == "dasp" for row in evaluated))
    result["online_commits"] = int(sum(row.get("accepted") == "online" for row in evaluated))
    result["light_commits"] = int(
        sum(row.get("accepted") == "light" for row in evaluated)
    )
    result["v16_performance_triggers"] = int(
        sum(bool(row.get("v16_performance_triggered", False)) for row in evaluated)
    )
    result["v16_oracle_triggers"] = int(
        sum(bool(row.get("v16_oracle_triggered", False)) for row in evaluated)
    )
    for candidate in [
        "online",
        "global",
        "layer",
        "layer_top1",
        "layer_top2",
        "layer_top3",
        "anchor",
        "replay_fallback",
        "frozen_fallback",
        "balanced_replay",
        "light_layer_1",
        "light_layer_2",
        "light_layer_3",
        "v17_fast_adapter",
        "v17_strong_adapter",
        "v17_anchor_adapter",
        "v17_frozen",
        "v18_fast_modulated_adapter",
        "v18_strong_modulated_adapter",
        "v18_anchor_modulated_adapter",
        "v18_frozen",
    ]:
        result[f"selected_{candidate}"] = int(
            sum(row.get("selected_candidate") == candidate for row in evaluated)
        )
    result["v17_soft_commits"] = int(
        sum(float(row.get("v17_soft_commit_alpha", 0.0)) > 0.0 for row in evaluated)
    )
    result["v17_partial_commits"] = int(
        sum(
            0.0 < float(row.get("v17_soft_commit_alpha", 0.0)) < 1.0
            for row in evaluated
        )
    )
    result["v17_strong_events"] = int(
        sum(row.get("v17_path") == "strong" for row in evaluated)
    )
    attempted = int(sum(row.get("accepted") in {"dasp", "online"} for row in evaluated))
    result["commit_acceptance_rate"] = float(result["dasp_commits"] / attempted) if attempted else 0.0
    phases = sorted({str(row["phase"]) for row in evaluated})
    for phase in phases:
        values = np.array([
            float(row["macro_f1"]) for row in evaluated if str(row["phase"]) == phase
        ])
        result[f"phase_{phase}_mean_macro_f1"] = float(values.mean())
    if "phase_A1_mean_macro_f1" in result and "phase_A2_mean_macro_f1" in result:
        result["forgetting_A1_to_A2"] = float(
            result["phase_A1_mean_macro_f1"] - result["phase_A2_mean_macro_f1"]
        )
    return result
