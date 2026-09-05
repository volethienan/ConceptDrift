from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


P0_ROOT = Path("outputs_revision_v1/full_p0_v1")
COMPONENT_ROOT = Path("outputs_revision_v1/p1_components_v1")
PERMUTATION_ROOT = Path("outputs_revision_v1/p1_feature_permutation_v1")
NATURAL_ROOT = Path("outputs_revision_v1/p1_natural_v1")
OUTPUT_ROOT = Path("outputs_revision_v1/cluster_aware_v1")
BOOTSTRAP_REPLICATES = 20_000
TIE_TOLERANCE = 1e-12


def _seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _holm_adjust(frame: pd.DataFrame, p_column: str, output_column: str) -> None:
    order = frame[p_column].sort_values(kind="mergesort").index.tolist()
    running = 0.0
    family_size = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (family_size - rank) * float(frame.at[index, p_column]))
        running = max(running, adjusted)
        frame.at[index, output_column] = running


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def _cluster_statistics(differences: pd.Series, label: str) -> dict[str, object]:
    values = pd.to_numeric(differences, errors="coerce").dropna().to_numpy(float)
    if not len(values):
        raise ValueError(f"no cluster differences for {label}")
    rng = np.random.default_rng(_seed(label))
    bootstrap = rng.choice(
        values,
        size=(BOOTSTRAP_REPLICATES, len(values)),
        replace=True,
    ).mean(axis=1)
    nonzero = values[np.abs(values) > TIE_TOLERANCE]
    wins = int((values > TIE_TOLERANCE).sum())
    losses = int((values < -TIE_TOLERANCE).sum())
    ties = int(len(values) - wins - losses)
    sign_p = (
        1.0
        if not len(nonzero)
        else float(binomtest(wins, len(nonzero), 0.5, alternative="two-sided").pvalue)
    )
    if not len(nonzero):
        wilcoxon_p = 1.0
    else:
        try:
            wilcoxon_p = float(
                wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    mean_difference = float(values.mean())
    if standard_deviation > 0:
        cohen_dz = mean_difference / standard_deviation
    elif math.isclose(mean_difference, 0.0, abs_tol=TIE_TOLERANCE):
        cohen_dz = 0.0
    else:
        cohen_dz = math.copysign(math.inf, mean_difference)
    return {
        "comparison": label,
        "clusters": int(len(values)),
        "mean_difference": mean_difference,
        "median_difference": float(np.median(values)),
        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p_two_sided": sign_p,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "cohen_dz": float(cohen_dz),
    }


def _p0_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(P0_ROOT / "full_summary.csv")
    cluster_means = (
        frame.groupby(["dataset", "scenario", "condition"], as_index=False)[
            "mean_macro_f1"
        ]
        .mean()
        .pivot(index=["dataset", "scenario"], columns="condition", values="mean_macro_f1")
        .reset_index()
    )
    rows = []
    delta_rows = []
    for baseline in [f"A{index}" for index in range(2, 11)]:
        label = f"A1_minus_{baseline}"
        differences = cluster_means["A1"] - cluster_means[baseline]
        rows.append(_cluster_statistics(differences, label))
        for index, value in differences.items():
            delta_rows.append(
                {
                    "dataset": cluster_means.at[index, "dataset"],
                    "scenario": cluster_means.at[index, "scenario"],
                    "comparison": label,
                    "macro_f1_difference": value,
                }
            )
    statistics = pd.DataFrame(rows)
    _holm_adjust(statistics, "sign_test_p_two_sided", "sign_test_holm_p")
    _holm_adjust(statistics, "wilcoxon_p_two_sided", "wilcoxon_holm_p")
    deltas = pd.DataFrame(delta_rows)
    leave_one_out_rows = []
    for comparison, group in deltas.groupby("comparison"):
        for omitted_dataset in sorted(group["dataset"].unique()):
            retained = group[group["dataset"] != omitted_dataset]
            values = retained["macro_f1_difference"]
            leave_one_out_rows.append(
                {
                    "comparison": comparison,
                    "omitted_dataset": omitted_dataset,
                    "clusters": len(retained),
                    "mean_difference": values.mean(),
                    "median_difference": values.median(),
                    "wins": int((values > TIE_TOLERANCE).sum()),
                    "ties": int((values.abs() <= TIE_TOLERANCE).sum()),
                    "losses": int((values < -TIE_TOLERANCE).sum()),
                }
            )
    return statistics, deltas, pd.DataFrame(leave_one_out_rows)


def _component_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    p0 = pd.read_csv(P0_ROOT / "full_summary.csv")
    p1 = pd.read_csv(COMPONENT_ROOT / "full_summary.csv")
    references = p0[p0["condition"].isin(["A1", "A6"])][
        ["dataset", "scenario", "stream_seed", "model_seed", "condition", "mean_macro_f1"]
    ]
    rows = []
    delta_rows = []
    for condition in ["B1", "B2", "B3", "B4", "B5"]:
        reference = "A6" if condition == "B5" else "A1"
        current = p1[p1["condition"] == condition]
        paired = current.merge(
            references[references["condition"] == reference],
            on=["dataset", "scenario", "stream_seed", "model_seed"],
            suffixes=("_condition", "_reference"),
            validate="one_to_one",
        )
        clusters = paired.assign(
            macro_f1_difference=(
                paired["mean_macro_f1_condition"] - paired["mean_macro_f1_reference"]
            )
        ).groupby(["dataset", "scenario"], as_index=False)["macro_f1_difference"].mean()
        label = f"{condition}_minus_{reference}"
        rows.append(_cluster_statistics(clusters["macro_f1_difference"], label))
        clusters["comparison"] = label
        delta_rows.append(clusters)
    statistics = pd.DataFrame(rows)
    _holm_adjust(statistics, "sign_test_p_two_sided", "sign_test_holm_p")
    _holm_adjust(statistics, "wilcoxon_p_two_sided", "wilcoxon_holm_p")
    return statistics, pd.concat(delta_rows, ignore_index=True)


def _permutation_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(PERMUTATION_ROOT / "feature_order_combined_with_p0.csv")
    rows = []
    delta_rows = []
    for method, group in frame.groupby("method"):
        original = (
            group[group["feature_order"] == "original"]
            .groupby(["dataset", "scenario"])["mean_macro_f1"]
            .mean()
        )
        permuted = (
            group[group["feature_order"] != "original"]
            .groupby(["dataset", "scenario"])["mean_macro_f1"]
            .mean()
        )
        paired = pd.concat(
            [original.rename("original"), permuted.rename("permuted")],
            axis=1,
        ).dropna()
        paired["macro_f1_difference"] = paired["permuted"] - paired["original"]
        label = f"{method}_permuted_minus_original"
        rows.append(_cluster_statistics(paired["macro_f1_difference"], label))
        current = paired.reset_index()
        current["comparison"] = label
        delta_rows.append(current)
    statistics = pd.DataFrame(rows)
    _holm_adjust(statistics, "sign_test_p_two_sided", "sign_test_holm_p")
    _holm_adjust(statistics, "wilcoxon_p_two_sided", "wilcoxon_holm_p")
    return statistics, pd.concat(delta_rows, ignore_index=True)


def _natural_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(NATURAL_ROOT / "full_summary.csv")
    descriptive = frame.groupby(["condition", "method"], as_index=False).agg(
        model_seeds=("model_seed", "nunique"),
        mean_macro_f1=("mean_macro_f1", "mean"),
        sd_macro_f1=("mean_macro_f1", "std"),
        minimum_macro_f1=("mean_macro_f1", "min"),
        maximum_macro_f1=("mean_macro_f1", "max"),
        mean_update_seconds=("total_update_seconds", "mean"),
        mean_throughput=("prediction_decision_throughput_flows_per_second", "mean"),
    )
    a1 = frame[frame["condition"] == "A1"].set_index("model_seed")["mean_macro_f1"]
    paired_rows = []
    for condition in ["A10", "N3", "N4"]:
        comparator = frame[frame["condition"] == condition].set_index("model_seed")[
            "mean_macro_f1"
        ]
        differences = (a1 - comparator).dropna()
        paired_rows.append(
            {
                "comparison": f"A1_minus_{condition}",
                "paired_model_seeds": len(differences),
                "mean_difference": differences.mean(),
                "minimum_difference": differences.min(),
                "maximum_difference": differences.max(),
                "inference_scope": "descriptive_only_single_operational_stream",
            }
        )
    return descriptive, pd.DataFrame(paired_rows)


def _artifact_manifest() -> None:
    rows = []
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if not path.is_file() or path.name == "artifact_sha256.csv":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "path": path.relative_to(OUTPUT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "artifact_sha256.csv", index=False)


def main() -> None:
    required = [
        P0_ROOT / "full_summary.csv",
        COMPONENT_ROOT / "full_summary.csv",
        PERMUTATION_ROOT / "feature_order_combined_with_p0.csv",
        NATURAL_ROOT / "full_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing analysis inputs: " + ", ".join(missing))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    p0_stats, p0_deltas, leave_one_out = _p0_analysis()
    component_stats, component_deltas = _component_analysis()
    permutation_stats, permutation_deltas = _permutation_analysis()
    natural_descriptive, natural_paired = _natural_analysis()
    outputs = {
        "p0_cluster_statistics.csv": p0_stats,
        "p0_cluster_deltas.csv": p0_deltas,
        "p0_leave_one_dataset_out.csv": leave_one_out,
        "p1_component_cluster_statistics.csv": component_stats,
        "p1_component_cluster_deltas.csv": component_deltas,
        "p1_feature_order_cluster_statistics.csv": permutation_stats,
        "p1_feature_order_cluster_deltas.csv": permutation_deltas,
        "p1_natural_descriptive.csv": natural_descriptive,
        "p1_natural_paired_differences.csv": natural_paired,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_ROOT / name, index=False)
    report = [
        "# Cluster-Aware Revision Analysis",
        "",
        "- P0 inference unit: 50 dataset-scenario clusters; seed/model cells are averaged within cluster.",
        "- P1 component and feature-order unit: 15 dataset-scenario clusters.",
        f"- Confidence intervals: {BOOTSTRAP_REPLICATES:,} cluster bootstrap replicates.",
        "- Multiplicity: Holm correction is applied separately to sign-test and Wilcoxon families.",
        "- Natural chronological evidence is descriptive because it contains one operational stream with three model seeds.",
        "",
        "## P0 A1 Contrasts",
        "",
        _markdown_table(p0_stats),
        "",
        "## P1 Component Contrasts",
        "",
        _markdown_table(component_stats),
        "",
        "## Feature-Order Robustness",
        "",
        _markdown_table(permutation_stats),
        "",
        "## Natural Chronological",
        "",
        _markdown_table(natural_descriptive),
    ]
    (OUTPUT_ROOT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    metadata = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "tie_tolerance": TIE_TOLERANCE,
        "p0_clusters": int(p0_deltas[["dataset", "scenario"]].drop_duplicates().shape[0]),
        "p1_clusters": int(
            component_deltas[["dataset", "scenario"]].drop_duplicates().shape[0]
        ),
        "natural_inference_scope": "descriptive_only_single_operational_stream",
    }
    (OUTPUT_ROOT / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _artifact_manifest()
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
