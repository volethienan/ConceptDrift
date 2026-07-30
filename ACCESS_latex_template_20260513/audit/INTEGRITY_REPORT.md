# Current DASP manuscript integrity report

Overall status: **PASS**

## Automated checks

- PASS: `confirmation_manifest_is_400_of_400`
- PASS: `confirmation_has_exact_locked_methods`
- PASS: `performance_manifest_is_80_of_80`
- PASS: `clean_performance_audit_passed`
- PASS: `adaptation_values_match_artifacts`
- PASS: `systems_values_match_artifacts`
- PASS: `paired_intervals_match_artifacts`
- PASS: `all_citations_have_bib_entries`
- PASS: `required_vector_figures_exist`
- PASS: `compiled_pdf_exists`
- PASS: `no_undefined_citations_in_final_log`
- PASS: `no_undefined_references_in_final_log`
- PASS: `no_fatal_latex_error`
- PASS: `no_universal_or_sota_overclaim`
- PASS: `no_stale_previous_version_result_claim`
- PASS: `current_artifact_paths_named`

## Interpretation boundary

The manuscript reports the 400-run NF-UNSW confirmation and the separate CUDA-synchronized 80-run systems matrix. It does not pool results from earlier DASP implementations. The audit verifies artifact-to-manuscript consistency; it does not certify external validity or complete author metadata.

## Metadata boundary

Author identity, affiliations, funding, conflicts, CRediT roles, repository DOI, and publisher metadata remain placeholders.

Full comparisons and SHA-256 hashes are in `audit_report.json`.
