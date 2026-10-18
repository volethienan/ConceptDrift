from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from dasp_nids.drift import multiscale_mmd
DATASETS = [
    "nf-unsw-nb15-v3",
    "nf-cse-cic-ids2018-v3",
    "nf-ton-iot-v3",
    "nf-bot-iot-v3",
    "cic-ids2017",
]
LOCAL_LEVELS = {
    "S4-Low": 0.05,
    "S4-Medium": 0.10,
    "S4-High": 0.20,
    "S7-Local-Low": 0.05,
    "S7-Local-Medium": 0.10,
    "S7-Local-High": 0.20,
}
CANONICAL_SCENARIOS = [
    "S0",
    "S1-Sudden",
    "S2-Gradual",
    "S3-Incremental",
    "S4-Recurrent",
    "S5-Blip",
    "S6-Combined",
    "S7-Local-Low",
    "S7-Local-Medium",
    "S7-Local-High",
]
STRESS_SCENARIOS = {"S5", "S8-Stress"}
SCENARIOS = [
    *CANONICAL_SCENARIOS,
    "S8-Stress",
    "S1",
    "S2",
    "S3",
    "S4-Low",
    "S4-Medium",
    "S4-High",
    "S5",
]


def _construct_classification(
    scenario: str,
    attack_mmd: float,
    probe_drop: float,
    *,
    mmd_threshold: float,
    boundary_impact_threshold: float,
) -> str:
    if scenario == "S0":
        return "stationary_control"
    if (
        np.isfinite(probe_drop)
        and probe_drop >= boundary_impact_threshold
    ):
        return "boundary_impacting_proxy"
    if np.isfinite(attack_mmd) and attack_mmd >= mmd_threshold:
        return "distribution_shift_only_or_boundary_inconclusive"
    return "weak_or_undetected_shift"


def _mmd_attack_shift(
    payload: dict[str, np.ndarray], scenario: str
) -> float:
    x = payload["x"]
    y = payload["y"]
    rng = np.random.default_rng(20260726)
    a = x[7:12][y[7:12] == 1]
    if scenario in {"S2", "S3-Incremental"}:
        b_slice = slice(24, 29)
    elif scenario == "S2-Gradual":
        b_slice = slice(21, 24)
    elif scenario == "S5-Blip":
        b_slice = slice(12, 14)
    else:
        b_slice = slice(12, 17)
    b = x[b_slice][y[b_slice] == 1]
    count = min(512, len(a), len(b))
    a = a[rng.choice(len(a), size=count, replace=False)]
    b = b[rng.choice(len(b), size=count, replace=False)]
    import torch

    return float(
        multiscale_mmd(
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
            max_samples=count,
            scales=[0.5, 1.0, 2.0],
        )
    )


def _probe_drop(
    payload: dict[str, np.ndarray], scenario: str
) -> tuple[float, float, float]:
    x = payload["x"]
    y = payload["y"]
    train_x = x[4:10].reshape(-1, x.shape[-1])
    train_y = y[4:10].reshape(-1)
    a_x = x[10:12].reshape(-1, x.shape[-1])
    a_y = y[10:12].reshape(-1)
    if scenario in {"S2", "S3-Incremental"}:
        b_slice = slice(24, 29)
    elif scenario == "S2-Gradual":
        b_slice = slice(21, 24)
    elif scenario == "S5-Blip":
        b_slice = slice(12, 14)
    else:
        b_slice = slice(12, 17)
    b_x = x[b_slice].reshape(-1, x.shape[-1])
    b_y = y[b_slice].reshape(-1)
    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        random_state=20260726,
    )
    model.fit(train_x, train_y)
    a_f1 = float(
        f1_score(
            a_y,
            model.predict(a_x),
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )
    b_f1 = float(
        f1_score(
            b_y,
            model.predict(b_x),
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )
    return a_f1, b_f1, a_f1 - b_f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate controlled stream construction and drift construct."
    )
    parser.add_argument("--prepared-dir", default="prepared_streams")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--scenarios", default=",".join(CANONICAL_SCENARIOS)
    )
    parser.add_argument(
        "--stream-seeds",
        default="2001,2002,2003,2101,2102,2103,2104,2105",
    )
    parser.add_argument("--output", default="prepared_streams/validation_report.json")
    parser.add_argument("--mmd-threshold", type=float, default=0.005)
    parser.add_argument(
        "--boundary-impact-threshold", type=float, default=0.02
    )
    parser.add_argument(
        "--min-boundary-impact-fraction", type=float, default=0.80
    )
    parser.add_argument(
        "--technical-only",
        action="store_true",
        help=(
            "Check all construction, overlap, shape, label-rate, and phase "
            "invariants without fitting the redundant one-model construct "
            "probe. Use validate_final_constructs.py for that gate."
        ),
    )
    args = parser.parse_args()
    if args.mmd_threshold < 0:
        raise ValueError("mmd-threshold must be non-negative")
    if args.boundary_impact_threshold < 0:
        raise ValueError("boundary-impact-threshold must be non-negative")
    if not 0 <= args.min_boundary_impact_fraction <= 1:
        raise ValueError("min-boundary-impact-fraction must be in [0, 1]")
    root = Path(args.prepared_dir)
    selected_datasets = [
        value.strip()
        for value in args.datasets.split(",")
        if value.strip()
    ]
    unknown_datasets = set(selected_datasets).difference(DATASETS)
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {sorted(unknown_datasets)}")
    selected_scenarios = [
        value.strip()
        for value in args.scenarios.split(",")
        if value.strip()
    ]
    unknown = set(selected_scenarios).difference(SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    stream_seeds = [
        int(value) for value in args.stream_seeds.split(",") if value
    ]
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for dataset in selected_datasets:
        scenario_seed_ids: dict[str, dict[int, set[int]]] = {}
        for scenario in selected_scenarios:
            scenario_seed_ids[scenario] = {}
            for seed in stream_seeds:
                path = (
                    root
                    / dataset
                    / scenario
                    / f"stream_seed{seed}.npz"
                )
                if not path.exists():
                    errors.append(f"missing:{path}")
                    continue
                with np.load(path, allow_pickle=False) as loaded:
                    payload = {
                        name: np.asarray(loaded[name])
                        for name in loaded.files
                    }
                x = payload["x"]
                y = payload["y"]
                source_ids = payload["source_ids"]
                session_ids = payload.get("session_ids")
                phases = payload["phases"].astype(str)
                s5_audit: dict[str, object] = {}
                attack_rates = y.mean(axis=1)
                unique_rows = len(np.unique(source_ids))
                overlap = int(source_ids.size - unique_rows)
                if x.shape[:2] != (36, 4096):
                    errors.append(f"shape:{path}:{x.shape}")
                if overlap:
                    errors.append(f"within-overlap:{path}:{overlap}")
                session_overlap = 0
                if session_ids is None:
                    errors.append(f"session-ids-missing:{path}")
                else:
                    segment_sets = [
                        set(values.reshape(-1).tolist())
                        for values in [
                            session_ids[:4],
                            session_ids[4:12],
                            session_ids[12:24],
                            session_ids[24:36],
                        ]
                    ]
                    for left_index, left in enumerate(segment_sets):
                        for right in segment_sets[left_index + 1 :]:
                            session_overlap += len(
                                left.intersection(right)
                            )
                    if session_overlap:
                        errors.append(
                            f"session-overlap:{path}:{session_overlap}"
                        )
                if np.ptp(attack_rates) > 1 / 4096:
                    errors.append(f"attack-rate:{path}")
                if not np.all(phases[:4] == "warmup"):
                    errors.append(f"warmup-phase:{path}")
                if scenario in STRESS_SCENARIOS:
                    if not np.all(phases[12:] == "B_evasive"):
                        errors.append(f"s5-phase:{path}")
                    if "s5_evasiveness_score" not in payload:
                        errors.append(f"s5-score-missing:{path}")
                    else:
                        scores = payload["s5_evasiveness_score"]
                        expected_score_mask = np.zeros_like(
                            y, dtype=bool
                        )
                        expected_score_mask[12:] = y[12:] == 1
                        if scores.shape != y.shape:
                            errors.append(
                                f"s5-score-shape:{path}:{scores.shape}"
                            )
                        elif not np.array_equal(
                            np.isfinite(scores), expected_score_mask
                        ):
                            errors.append(f"s5-score-mask:{path}")
                    manifest_path = path.with_suffix(
                        ".manifest.json"
                    )
                    if not manifest_path.exists():
                        errors.append(f"s5-manifest-missing:{path}")
                    else:
                        manifest = json.loads(
                            manifest_path.read_text(encoding="utf-8")
                        )
                        audit = manifest.get("s5", {}).get(
                            "audit", {}
                        )
                        s5_audit = audit
                        if not audit.get("passed", False):
                            errors.append(f"s5-gate-failed:{path}")
                scenario_seed_ids[scenario][seed] = set(
                    source_ids.reshape(-1).tolist()
                )
                mmd = (
                    _mmd_attack_shift(payload, scenario)
                    if scenario != "S0" and not args.technical_only
                    else np.nan
                )
                a_f1, b_f1, probe_drop = (
                    _probe_drop(payload, scenario)
                    if scenario != "S0" and not args.technical_only
                    else (np.nan, np.nan, np.nan)
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "stream_seed": seed,
                        "attack_rate": float(attack_rates.mean()),
                        "attack_rate_range": float(np.ptp(attack_rates)),
                        "unique_source_rows": unique_rows,
                        "within_stream_overlap": overlap,
                        "within_stream_session_overlap": session_overlap,
                        "attack_mmd_A_B": mmd,
                        "probe_A_macro_f1": a_f1,
                        "probe_B_macro_f1": b_f1,
                        "probe_A_minus_B": probe_drop,
                        "construct_classification": (
                            "not_evaluated_technical_only"
                            if args.technical_only
                            else _construct_classification(
                                scenario,
                                mmd,
                                probe_drop,
                                mmd_threshold=args.mmd_threshold,
                                boundary_impact_threshold=(
                                    args.boundary_impact_threshold
                                ),
                            )
                        ),
                        "s5_evasion_uplift_vs_matched_random": (
                            s5_audit.get(
                                "evasion_uplift_vs_matched_random",
                                np.nan,
                            )
                        ),
                        "s5_recall_drop_uplift_vs_matched_random": (
                            s5_audit.get(
                                "median_attack_recall_drop_uplift_vs_matched_random",
                                np.nan,
                            )
                        ),
                        "s5_macro_f1_drop_uplift_vs_matched_random": (
                            s5_audit.get(
                                "median_macro_f1_drop_uplift_vs_matched_random",
                                np.nan,
                            )
                        ),
                    }
                )
            seeds = list(scenario_seed_ids[scenario])
            for left_index, left in enumerate(seeds):
                for right in seeds[left_index + 1 :]:
                    overlap = scenario_seed_ids[scenario][left].intersection(
                        scenario_seed_ids[scenario][right]
                    )
                    if overlap:
                        errors.append(
                            f"cross-seed-overlap:{dataset}:{scenario}:"
                            f"{left}:{right}:{len(overlap)}"
                        )
    import pandas as pd

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    construct_summary_frame = (
        frame.groupby(["dataset", "scenario"])
        .agg(
            mean_attack_mmd=("attack_mmd_A_B", "mean"),
            mean_probe_A_minus_B=("probe_A_minus_B", "mean"),
            boundary_impact_fraction=(
                "construct_classification",
                lambda values: float(
                    np.mean(
                        np.asarray(values)
                        == "boundary_impacting_proxy"
                    )
                ),
            ),
        )
        .reset_index()
        if len(frame) and not args.technical_only
        else pd.DataFrame()
    )
    if len(construct_summary_frame):
        construct_summary_frame["claim_eligible"] = (
            (
                construct_summary_frame["scenario"] == "S0"
            )
            | (
                (
                    construct_summary_frame[
                        "mean_probe_A_minus_B"
                    ]
                    >= args.boundary_impact_threshold
                )
                & (
                    construct_summary_frame[
                        "boundary_impact_fraction"
                    ]
                    >= args.min_boundary_impact_fraction
                )
            )
        )
    nonstationary = (
        construct_summary_frame[
            construct_summary_frame["scenario"] != "S0"
        ]
        if len(construct_summary_frame)
        else construct_summary_frame
    )
    construct_warnings = (
        [
            (
                f"{row.dataset}/{row.scenario}: distribution change is "
                "not sufficient evidence of a material binary decision-"
                "boundary impact"
            )
            for row in nonstationary.itertuples()
            if not bool(row.claim_eligible)
        ]
        if len(nonstationary)
        else []
    )
    report = {
        "artifacts_checked": len(frame),
        "expected_artifacts": len(selected_datasets)
        * len(selected_scenarios)
        * len(stream_seeds),
        "errors": errors,
        "validation_passed": bool(
            not errors
            and len(frame)
            == len(selected_datasets)
            * len(selected_scenarios)
            * len(stream_seeds)
        ),
        "construct_thresholds": {
            "attack_mmd": args.mmd_threshold,
            "probe_A_minus_B": args.boundary_impact_threshold,
            "minimum_artifact_fraction": (
                args.min_boundary_impact_fraction
            ),
        },
        "technical_only": bool(args.technical_only),
        "construct_summary": construct_summary_frame.to_dict(
            orient="records"
        ),
        "construct_warnings": construct_warnings,
        "real_concept_drift_claim_allowed_for_all_selected_scenarios": bool(
            len(nonstationary)
            and nonstationary["claim_eligible"].all()
        ),
        "interpretation_rule": (
            "A probe A-minus-B drop at or above the locked threshold is a "
            "boundary-impacting proxy, not direct proof that P(Y|X) changed. "
            "MMD without probe degradation is reported as distribution shift "
            "only or boundary-inconclusive and cannot support a real-concept-"
            "drift claim."
        ),
    }
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
