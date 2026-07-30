from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import friedmanchisquare, wilcoxon

from .config import ExperimentConfig
from .runner import StreamingExperiment


def _paired_statistics(frame: pd.DataFrame, method: str, reference: str) -> dict[str, object]:
    left = frame[frame["method"] == method].set_index("seed")["mean_macro_f1"]
    right = frame[frame["method"] == reference].set_index("seed")["mean_macro_f1"]
    paired = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    differences = paired["left"] - paired["right"]
    if len(differences) < 2:
        p_value = np.nan
        effect = np.nan
    elif np.allclose(differences.to_numpy(), 0.0):
        p_value = 1.0
        effect = 0.0
    else:
        try:
            p_value = float(wilcoxon(differences, alternative="greater").pvalue)
        except ValueError:
            p_value = 1.0
        sd = float(differences.std(ddof=1))
        effect = float(differences.mean() / sd) if sd > 0 else (np.inf if differences.mean() > 0 else 0.0)
    rng = np.random.default_rng(20260720)
    if len(differences):
        samples = rng.choice(differences.to_numpy(), size=(5000, len(differences)), replace=True).mean(axis=1)
        ci_low, ci_high = np.quantile(samples, [0.025, 0.975])
    else:
        ci_low = ci_high = np.nan
    return {
        "method": method,
        "reference": reference,
        "n_pairs": int(len(differences)),
        "mean_difference": float(differences.mean()) if len(differences) else np.nan,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_one_sided_p": p_value,
        "cohen_dz": effect,
    }


def run_comparison(
    config: ExperimentConfig,
    methods: list[str],
    seeds: list[int],
    output_dir: str | Path | None = None,
    resume: bool = False,
) -> Path:
    summaries: list[dict[str, object]] = []
    for seed in seeds:
        for method in methods:
            print(f"starting method={method} seed={seed}", flush=True)
            current = copy.deepcopy(config)
            current.data.random_seed = seed
            current.dasp.baseline = method
            current.output.run_name = (
                f"{current.data.source}_{current.data.scenario}_{method}_seed{seed}"
            )
            expected = Path(current.output.directory) / current.output.run_name
            cached_summary = expected / "summary.json"
            cached_config = expected / "config.yaml"
            if resume and cached_summary.exists() and cached_config.exists():
                resolved = yaml.safe_load(cached_config.read_text(encoding="utf-8"))
                if resolved != current.to_dict():
                    raise ValueError(
                        f"Refusing to resume {expected}: cached config does not match"
                    )
                summary = json.loads(cached_summary.read_text(encoding="utf-8"))
                print(f"resumed method={method} seed={seed}", flush=True)
            else:
                result = StreamingExperiment(current).run()
                summary = result.summary
            summaries.append(summary)
            print(
                f"completed method={method} seed={seed} "
                f"macro_f1={summary['mean_macro_f1']:.6f}",
                flush=True,
            )
    frame = pd.DataFrame(summaries)
    destination = Path(output_dir or config.output.directory) / (
        f"comparison_{config.data.source}_{config.data.scenario}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "summary_by_run.csv", index=False)
    aggregate = frame.groupby("method", as_index=False).agg(
        mean_macro_f1=("mean_macro_f1", "mean"),
        sd_macro_f1=("mean_macro_f1", "std"),
        mean_attack_recall=("mean_attack_recall", "mean"),
        mean_fnr=("mean_fnr", "mean"),
        mean_balanced_accuracy=("mean_balanced_accuracy", "mean"),
    )
    for column in [
        "forgetting_A1_to_A2",
        "mean_online_forgetting",
        "final_online_forgetting",
        "window_stability_macro_f1",
        "mean_absolute_window_delta_macro_f1",
        "max_window_drop_macro_f1",
        "cumulative_macro_f1",
        "cumulative_attack_f1",
        "cumulative_benign_f1",
        "cumulative_balanced_accuracy",
        "cumulative_accuracy",
        "total_gradient_evaluations",
        "total_update_seconds",
        "reference_resets",
        "total_sample_updates",
        "internal_drifts",
        "internal_warnings",
        "adaptation_triggers",
        "dasp_commits",
        "selected_global",
        "selected_layer",
        "selected_anchor",
        "selected_replay_fallback",
    ]:
        if column in frame.columns:
            values = frame.groupby("method", as_index=False)[column].mean()
            aggregate = aggregate.merge(values, on="method", how="left")
    aggregate.to_csv(destination / "aggregate.csv", index=False)
    comparisons: list[dict[str, object]] = []
    for method, reference in [
        ("layer_dasp", "global_dasp"),
        ("global_dasp", "online"),
        ("full_dasp", "online"),
        ("utility_dasp", "global_dasp"),
        ("utility_dasp", "utility_global"),
        ("utility_full", "online"),
        ("utility_full", "utility_dasp"),
        ("utility_layer_v3", "utility_global_v3"),
        ("utility_layer_v3", "compute_online_v3"),
        ("utility_global_v3", "compute_online_v3"),
        ("utility_global_v5", "online"),
        ("utility_layer_v5", "online"),
        ("utility_layer_v5", "utility_global_v5"),
        ("utility_portfolio_v5", "online"),
        ("utility_portfolio_v5", "utility_global_v5"),
        ("utility_portfolio_v5", "utility_layer_v5"),
        ("reactive_dasp_v6", "online"),
        ("reactive_dasp_v6", "replay"),
        ("reactive_dasp_v6", "utility_portfolio_v5"),
        ("dasp", "online_mlp"),
        ("dasp", "experience_replay"),
        ("dasp", "online_ewc"),
        ("dasp", "agem"),
        ("dasp", "ssf"),
        ("dasp", "repshield"),
        ("dasp", "cdda_md"),
        ("dasp_v9", "dasp_v7"),
        ("dasp_v9", "experience_replay"),
        ("dasp_v9", "repshield"),
        ("dasp_v9_1", "dasp_v7"),
        ("dasp_v9_1", "experience_replay"),
        ("dasp_v9_1", "repshield"),
        ("dasp_v9_2", "dasp_v7"),
        ("dasp_v9_2", "experience_replay"),
        ("dasp_v9_2", "repshield"),
        ("dasp_v10", "dasp_v7"),
        ("dasp_v10", "experience_replay"),
        ("dasp_v10", "repshield"),
        ("dasp_v10_1", "experience_replay"),
        ("dasp_v10_1", "online_ewc"),
        ("dasp_v10_1", "online_mlp"),
        ("dasp_v10_1", "agem"),
        ("dasp_v10_1", "ssf"),
        ("dasp_v10_1", "repshield"),
        ("dasp_v10_1", "cdda_md"),
        ("dasp_v10_2", "experience_replay"),
        ("dasp_v10_2", "ssf"),
        ("dasp_v10_2", "repshield"),
        ("dasp_v10_3", "experience_replay"),
        ("dasp_v10_3", "ssf"),
        ("dasp_v10_3", "repshield"),
        ("dasp_v11_novelty", "dasp_v10_2"),
        ("dasp_v11_novelty", "ssf"),
        ("dasp_v11", "dasp_v10_2"),
        ("dasp_v11", "ssf"),
        ("dasp_v11_1", "dasp_v11_novelty"),
        ("dasp_v11_1", "ssf"),
        ("dasp_v12", "dasp_v10_2"),
        ("dasp_v12", "dasp_v11_novelty"),
        ("dasp_v12", "ssf"),
        ("dasp_v13_iwms", "dasp_v12"),
        ("dasp_v13_refresh", "dasp_v12"),
        ("dasp_v13", "dasp_v12"),
        ("dasp_v13", "ssf"),
        ("dasp_v14_light", "dasp_v12"),
        ("dasp_v14", "dasp_v12"),
        ("dasp_v14", "ssf"),
        ("dasp_v15_light", "dasp_v12"),
        ("dasp_v15", "dasp_v12"),
        ("dasp_v15", "ssf"),
    ]:
        if method in set(frame["method"]) and reference in set(frame["method"]):
            comparisons.append(_paired_statistics(frame, method, reference))
    (destination / "paired_statistics.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    benchmark_methods = [
        "online_mlp",
        "experience_replay",
        "online_ewc",
        "agem",
        "ssf",
        "repshield",
        "cdda_md",
        "dasp",
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
        "dasp_v13_iwms",
        "dasp_v13_refresh",
        "dasp_v13",
        "dasp_v14_light",
        "dasp_v14",
        "dasp_v15_light",
        "dasp_v15",
    ]
    present = [method for method in benchmark_methods if method in set(frame["method"])]
    proposed = (
        "dasp_v15"
        if "dasp_v15" in present
        else (
            "dasp_v14"
            if "dasp_v14" in present
            else (
                "dasp_v13"
                if "dasp_v13" in present
                else (
                    "dasp_v12"
                    if "dasp_v12" in present
                    else (
                        "dasp_v11_1"
                        if "dasp_v11_1" in present
                        else (
                            "dasp_v11"
                            if "dasp_v11" in present
                            else (
                                "dasp_v11_novelty"
                                if "dasp_v11_novelty" in present
                                else (
                                    "dasp_v10_3"
                                    if "dasp_v10_3" in present
                                    else (
                                        "dasp_v10_2"
                                        if "dasp_v10_2" in present
                                        else (
                                            "dasp_v10_1"
                                            if "dasp_v10_1" in present
                                            else (
                                                "dasp_v10"
                                                if "dasp_v10" in present
                                                else (
                                                    "dasp_v9_2"
                                                    if "dasp_v9_2" in present
                                                    else (
                                                        "dasp_v9_1"
                                                        if "dasp_v9_1" in present
                                                        else (
                                                            "dasp_v9"
                                                            if "dasp_v9" in present
                                                            else (
                                                                "dasp"
                                                                if "dasp" in present
                                                                else "reactive_dasp_v6"
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    if proposed in present and len(present) >= 3:
        pivot = frame.pivot(index="seed", columns="method", values="mean_macro_f1")
        complete = pivot[present].dropna()
        if len(complete) >= 2:
            friedman = friedmanchisquare(
                *[complete[method].to_numpy() for method in present]
            )
            average_ranks = (
                complete.rank(axis=1, ascending=False, method="average")
                .mean(axis=0)
                .to_dict()
            )
        else:
            friedman = None
            average_ranks = {}
        contrasts = [
            _paired_statistics(frame, proposed, baseline)
            for baseline in present
            if baseline != proposed
        ]
        order = sorted(
            range(len(contrasts)),
            key=lambda idx: float(contrasts[idx]["wilcoxon_one_sided_p"]),
        )
        running = 0.0
        family_size = len(order)
        for rank, idx in enumerate(order):
            raw = float(contrasts[idx]["wilcoxon_one_sided_p"])
            adjusted = min(1.0, (family_size - rank) * raw)
            running = max(running, adjusted)
            contrasts[idx]["holm_adjusted_p"] = running
        benchmark = {
            "methods": present,
            "n_complete_seeds": int(len(complete)),
            "friedman": None
            if friedman is None
            else {
                "statistic": float(friedman.statistic),
                "p_value": float(friedman.pvalue),
            },
            "average_ranks": {key: float(value) for key, value in average_ranks.items()},
            "proposed_method": proposed,
            "proposed_vs_baselines": contrasts,
        }
        (destination / "benchmark_statistics.json").write_text(
            json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return destination
