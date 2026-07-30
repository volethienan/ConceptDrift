from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PRIMARY_METHODS = {
    "static",
    "periodic_ft",
    "online_mlp",
    "experience_replay",
    "ssf_faithful",
    "ssf_budget",
    "repshield",
    "dasp_v16_no_detector",
    "dasp_v16_oracle",
    "dasp_v16",
}
PRIMARY_SCENARIOS = {"S0", "S1", "S2", "S3", "S4"}
NF_DATASETS = {
    "nf-unsw-nb15-v3",
    "nf-cse-cic-ids2018-v3",
    "nf-ton-iot-v3",
    "nf-bot-iot-v3",
}


def _matrix_gate(
    root: Path,
    *,
    expected_runs: int,
    expected_scenarios: set[str],
) -> dict[str, object]:
    summary_path = root / "summary_by_run.csv"
    if not summary_path.exists():
        return {
            "passed": False,
            "reason": f"missing {summary_path}",
            "completed_runs": 0,
            "expected_runs": expected_runs,
        }
    frame = pd.read_csv(summary_path)
    keys = ["dataset", "scenario", "stream_seed", "model_seed", "method"]
    missing_columns = sorted(set(keys).difference(frame.columns))
    if missing_columns:
        return {
            "passed": False,
            "reason": f"missing columns {missing_columns}",
            "completed_runs": len(frame),
            "expected_runs": expected_runs,
        }
    duplicates = int(frame.duplicated(keys).sum())
    datasets = set(frame["dataset"].astype(str))
    scenarios = set(frame["scenario"].astype(str))
    methods = set(frame["method"].astype(str))
    checks = {
        "exact_run_count": len(frame) == expected_runs,
        "zero_duplicate_cells": duplicates == 0,
        "four_nf_datasets": datasets == NF_DATASETS,
        "expected_scenarios": scenarios == expected_scenarios,
        "ten_locked_methods_with_causal_ablations": (
            methods == PRIMARY_METHODS
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "completed_runs": int(len(frame)),
        "expected_runs": expected_runs,
        "datasets": sorted(datasets),
        "scenarios": sorted(scenarios),
        "methods": sorted(methods),
        "duplicate_cells": duplicates,
        "checks": checks,
    }


def _construct_gate(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"passed": False, "reason": f"missing {path}"}
    report = json.loads(path.read_text(encoding="utf-8"))
    structural = bool(report.get("validation_passed", False))
    summary = report.get("construct_summary", [])
    if "real_concept_drift_claim_allowed_for_all_selected_scenarios" in report:
        claim_allowed = bool(
            report[
                "real_concept_drift_claim_allowed_for_all_selected_scenarios"
            ]
        )
        warnings = report.get("construct_warnings", [])
    else:
        ineligible = [
            row
            for row in summary
            if row.get("scenario") != "S0"
            and (
                row.get("mean_probe_A_minus_B") is None
                or float(row.get("mean_probe_A_minus_B")) < 0.02
            )
        ]
        claim_allowed = bool(summary and not ineligible)
        warnings = [
            (
                f"{row.get('dataset')}/{row.get('scenario')}: legacy "
                "construct report shows probe A-minus-B below 0.02"
            )
            for row in ineligible
        ]
    return {
        "passed": bool(structural and claim_allowed),
        "artifact_integrity_passed": structural,
        "real_concept_drift_claim_allowed": claim_allowed,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed audit for the evidence needed by the broad Q1 "
            "DASP manuscript claim."
        )
    )
    parser.add_argument(
        "--primary-output", default="outputs_dasp_v16_five_dataset"
    )
    parser.add_argument("--s5-output", default="outputs_dasp_v16_s5")
    parser.add_argument(
        "--construct-report",
        default="prepared_streams_v2/validation_report.json",
    )
    parser.add_argument(
        "--s5-construct-report",
        default="prepared_streams_s5_v1/validation_report.json",
    )
    parser.add_argument(
        "--manuscript",
        default=(
            "ACCESS_latex_template_20260513/"
            "ACCESS_ConceptDrift_Cyber.tex"
        ),
    )
    parser.add_argument("--repository-url", default="")
    parser.add_argument("--persistent-identifier", default="")
    parser.add_argument(
        "--output", default="q1_submission_readiness.json"
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manuscript = Path(args.manuscript)
    manuscript_text = (
        manuscript.read_text(encoding="utf-8")
        if manuscript.exists()
        else ""
    )
    gates = {
        "primary_four_dataset_matrix": _matrix_gate(
            Path(args.primary_output),
            expected_runs=2_000,
            expected_scenarios=PRIMARY_SCENARIOS,
        ),
        "secondary_s5_matrix": _matrix_gate(
            Path(args.s5_output),
            expected_runs=400,
            expected_scenarios={"S5"},
        ),
        "primary_construct_validity": _construct_gate(
            Path(args.construct_report)
        ),
        "s5_construct_and_matched_control": _construct_gate(
            Path(args.s5_construct_report)
        ),
        "metadata_complete": {
            "passed": bool(
                manuscript_text
                and "to be completed" not in manuscript_text.casefold()
            ),
            "manuscript_exists": manuscript.exists(),
            "contains_placeholders": (
                "to be completed" in manuscript_text.casefold()
            ),
        },
        "public_archive": {
            "passed": bool(
                args.repository_url.strip()
                and args.persistent_identifier.strip()
            ),
            "repository_url_supplied": bool(args.repository_url.strip()),
            "persistent_identifier_supplied": bool(
                args.persistent_identifier.strip()
            ),
        },
    }
    unresolved = [
        name for name, gate in gates.items() if not gate["passed"]
    ]
    report = {
        "submission_ready_for_broad_q1_claim": not unresolved,
        "policy": (
            "Broad cross-dataset superiority and real-concept-drift claims "
            "are blocked until every gate passes. Missing experiments are "
            "never replaced by planned/protocol language."
        ),
        "gates": gates,
        "unresolved_gates": unresolved,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and unresolved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
