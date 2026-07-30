from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = Path(__file__).resolve().parent
AUDIT = MANUSCRIPT / "audit"
CONFIRMATION = ROOT / "outputs_dasp_fast_nf_unsw_locked_7_gpu"
PERFORMANCE = ROOT / "outputs_dasp_fast_nf_unsw_locked7_performance_gpu_v2"
INTEGRATED = CONFIRMATION / "analysis_locked7_drift_performance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def row_by_method(path: Path) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in read_rows(path)}


def main() -> None:
    tex_path = MANUSCRIPT / "ACCESS_ConceptDrift_Cyber.tex"
    bib_path = MANUSCRIPT / "references.bib"
    pdf_path = MANUSCRIPT / "ACCESS_ConceptDrift_Cyber.pdf"
    log_path = MANUSCRIPT / "ACCESS_ConceptDrift_Cyber.log"
    tex = tex_path.read_text(encoding="utf-8")
    tex_numeric = tex.replace(",", "")
    bib = bib_path.read_text(encoding="utf-8")
    log = log_path.read_text(encoding="utf-8", errors="replace")

    overall = row_by_method(INTEGRATED / "drift_adaptation_overall.csv")
    performance = row_by_method(INTEGRATED / "gpu_performance_locked7.csv")
    paired = read_rows(CONFIRMATION / "analysis/paired_dasp_all_baselines_statistics.csv")
    confirmation_manifest = json.loads(
        (CONFIRMATION / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    performance_manifest = json.loads(
        (PERFORMANCE / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    performance_audit = json.loads(
        (PERFORMANCE / "clean_performance_audit.json").read_text(encoding="utf-8")
    )

    expected_adaptation = {
        method: {
            "early_macro_f1": f"{float(row['early_macro_f1']):.6f}",
            "early_attack_recall": f"{float(row['early_attack_recall']):.6f}",
            "minimum_post_drift_macro_f1": (
                f"{float(row['minimum_post_drift_macro_f1']):.6f}"
            ),
        }
        for method, row in overall.items()
    }
    adaptation_numeric_checks = [
        {
            "method": method,
            "metric": metric,
            "formatted_value": value,
            "pass": value in tex_numeric,
        }
        for method, metrics in expected_adaptation.items()
        for metric, value in metrics.items()
    ]

    expected_systems = {
        method: {
            "inference_p95_ms": f"{float(row['inference_p95_ms']):.3f}",
            "mean_end_to_end_ms": f"{float(row['mean_end_to_end_ms']):.3f}",
            "end_to_end_p95_ms": f"{float(row['end_to_end_p95_ms']):.3f}",
            "peak_cuda_allocated_mib": (
                f"{float(row['peak_cuda_allocated_mib']):.2f}"
            ),
        }
        for method, row in performance.items()
    }
    systems_numeric_checks = [
        {
            "method": method,
            "metric": metric,
            "formatted_value": value,
            "pass": value in tex_numeric,
        }
        for method, metrics in expected_systems.items()
        for metric, value in metrics.items()
    ]

    paired_checks = []
    for row in paired:
        expected = {
            "mean_difference": f"{abs(float(row['mean_difference'])):.6f}",
            "ci95_low": f"{abs(float(row['ci95_low'])):.6f}",
            "ci95_high": f"{abs(float(row['ci95_high'])):.6f}",
        }
        paired_checks.append(
            {
                "metric": row["metric"],
                "baseline": row["baseline"],
                "values": expected,
                "pass": all(value in tex for value in expected.values()),
            }
        )

    citation_keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        citation_keys.update(key.strip() for key in group.split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing_bib = sorted(citation_keys - bib_keys)

    required_figures = [
        MANUSCRIPT / "figures/nf_unsw_drift_episode_heatmap.pdf",
        MANUSCRIPT / "figures/nf_unsw_accuracy_performance_tradeoff.pdf",
    ]
    checks = {
        "confirmation_manifest_is_400_of_400": (
            confirmation_manifest["expected_runs"] == 400
            and confirmation_manifest["completed_runs"] == 400
        ),
        "confirmation_has_exact_locked_methods": (
            confirmation_manifest["new_algorithms_added"] is False
            and len(confirmation_manifest["comparators"]) == 7
            and confirmation_manifest["proposed_method"] == "dasp"
        ),
        "performance_manifest_is_80_of_80": (
            performance_manifest["expected_runs"] == 80
            and performance_manifest["completed_runs"] == 80
            and len(performance_manifest["methods"]) == 8
        ),
        "clean_performance_audit_passed": performance_audit["audit_passed"],
        "adaptation_values_match_artifacts": all(
            item["pass"] for item in adaptation_numeric_checks
        ),
        "systems_values_match_artifacts": all(
            item["pass"] for item in systems_numeric_checks
        ),
        "paired_intervals_match_artifacts": all(
            item["pass"] for item in paired_checks
        ),
        "all_citations_have_bib_entries": not missing_bib,
        "required_vector_figures_exist": all(
            path.exists() and path.stat().st_size > 0 for path in required_figures
        ),
        "compiled_pdf_exists": pdf_path.exists() and pdf_path.stat().st_size > 0,
        "no_undefined_citations_in_final_log": not re.search(
            r"Citation `[^']+' .* undefined", log
        ),
        "no_undefined_references_in_final_log": not re.search(
            r"Reference `[^']+' .* undefined", log
        ),
        "no_fatal_latex_error": all(
            marker not in log
            for marker in ["Fatal error occurred", "LaTeX Error", "Emergency stop"]
        ),
        "no_universal_or_sota_overclaim": all(
            phrase not in tex.lower()
            for phrase in [
                "outperforms all baselines",
                "achieves state-of-the-art",
                "achieved state-of-the-art",
                "universal superiority was established",
            ]
        ),
        "no_stale_previous_version_result_claim": all(
            phrase not in tex
            for phrase in [
                "2,000-run controlled confirmation",
                "outputs_dasp_v16r1",
                "outputs_dasp_main_screen",
                "80-run chronological study",
            ]
        ),
        "current_artifact_paths_named": all(
            value in tex
            for value in [
                "outputs_dasp_fast_nf_unsw_locked_7_gpu",
                "outputs_dasp_fast_nf_unsw_locked7_performance_gpu_v2",
            ]
        ),
    }

    tracked = [
        tex_path,
        bib_path,
        pdf_path,
        MANUSCRIPT / "audit_manuscript.py",
        *required_figures,
    ]
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_bibliography_keys": missing_bib,
        "adaptation_numeric_checks": adaptation_numeric_checks,
        "systems_numeric_checks": systems_numeric_checks,
        "paired_interval_checks": paired_checks,
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in tracked
        },
    }

    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Current DASP manuscript integrity report",
        "",
        f"Overall status: **{report['status']}**",
        "",
        "## Automated checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in checks.items()
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The manuscript reports the 400-run NF-UNSW confirmation and the "
            "separate CUDA-synchronized 80-run systems matrix. It does not pool "
            "results from earlier DASP implementations. The audit verifies "
            "artifact-to-manuscript consistency; it does not certify external "
            "validity or complete author metadata.",
            "",
            "## Metadata boundary",
            "",
            "Author identity, affiliations, funding, conflicts, CRediT roles, "
            "repository DOI, and publisher metadata remain placeholders.",
            "",
            "Full comparisons and SHA-256 hashes are in `audit_report.json`.",
            "",
        ]
    )
    (AUDIT / "INTEGRITY_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
