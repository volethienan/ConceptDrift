# Q1 Reviewer Response and Revision Log

Date: 2026-07-29  
Manuscript: `ACCESS_latex_template_20260513/ACCESS_ConceptDrift_Cyber.tex`  
Current decision: **Major revision; not ready for a broad Q1 claim**

## Evidence boundary

The completed evidence is the locked 400-run NF-UNSW-NB15-v3 matrix and the
separate 80-run GPU systems matrix. The four-dataset primary matrix, the S5
matrix, detector-off/oracle confirmation, and natural-track confirmation are
not complete. This revision does not report planned experiments as results.

## Issue-by-issue response

| ID | Reviewer concern | Severity | Revision made | Status |
|---|---|---:|---|---|
| R1 | Attack-family composition may change \(P(X\mid Y)\) without changing \(P(Y\mid X)\) | Critical | Renamed completed episodes as controlled class-conditional distribution shifts; added MMD and A-trained-probe evidence to Methods, Abstract, Threats, and Conclusion; validator now classifies each artifact and emits a fail-closed real-drift claim gate | Manuscript/code resolved; boundary-impacting experiments still required |
| R2 | Five stream seeds are too few for the strength of the statistical language | Critical | Added exact two-sided stream-cluster sign-flip tests; retained hierarchical bootstrap as effect uncertainty; added Holm correction for exploratory all-baseline families | Resolved |
| R3 | Bootstrap CI excluded zero but independent-cluster evidence may not | Critical | Reanalysis gives primary DASP--SSF-budget \(p=0.0625\) and Attack Recall \(p=0.25\); no exploratory contrast survives Holm; superiority language removed | Resolved |
| R4 | No detector-off/oracle causal attribution | Critical | Locked runner already contains detector-off/proposed/oracle; analyzer now emits dedicated causal-ablation contrasts and oracle-gap closure | Code resolved; 2,000-run confirmation pending |
| R5 | Only one dataset supports the manuscript | Critical | Broad and cross-dataset claims removed; fail-closed audit requires the exact four-NF-dataset, ten-method, 2,000-run matrix | Claims resolved; experiment pending |
| R6 | Near-ceiling NF-UNSW outcomes make recovery and drift response non-discriminative | Major | Ceiling limitation is explicit; zero recovery is no longer presented as adaptation evidence; S5 remains a separate secondary stress track | Reporting resolved; stronger evidence pending |
| R7 | S5 could be a cherry-picked hard subset | Major | Added a source-disjoint random-B control matched on family weights, attack prevalence, and time quantiles; S5 must beat absolute severity gates and uplift gates against this control | Code resolved; S5 artifacts and 400 runs pending |
| R8 | One surrogate ensemble can overfit the S5 definition | Major | Selection and audit ensembles use disjoint design/audit splits and separately seeded Logistic Regression, Random Forest, and Extra Trees models; per-audit-model outcomes are recorded | Resolved in design; external-model sensitivity remains supplementary |
| R9 | Multiple baseline comparisons inflate false positives | Major | Added metric-wise Holm correction and manuscript columns for raw exact and adjusted p-values | Resolved |
| R10 | Windows/model seeds may be treated as independent | Major | Exact test aggregates at dataset--stream level; model seeds and episodes remain nested; manuscript states five independent clusters | Resolved |
| R11 | “Real-time” inference claim ignores update tail | Major | Manuscript separates forward inference from complete test-then-train latency and makes only a systems trade-off claim | Resolved |
| R12 | Metadata, author information, repository, and DOI are incomplete | Major | Added an automated readiness gate that blocks submission while placeholders or archive identifiers remain | Requires author/repository input |
| R13 | Prepared four-dataset streams are not equivalent to completed evidence | Major | Readiness audit requires `summary_by_run.csv`, exact run count, exact cells, all datasets/scenarios, and all ten methods; protocol text cannot pass the gate | Resolved |
| R14 | S4/local shift is weak or non-boundary-impacting in several datasets | Major | Validator reports scenario-level boundary-impact fraction and forbids a real-drift claim when the locked threshold is not met | Code resolved; generator/scenario redesign may be required |

## Statistical correction to the existing manuscript

The prior wording relied too heavily on percentile bootstrap intervals.
Reanalysis keeps the original effect estimates but adds the independent-cluster
check:

- DASP minus SSF-budget Early Macro-F1@5:
  \(\Delta=0.001726\), bootstrap 95% CI
  \([0.001522,0.002039]\), exact two-sided stream-cluster \(p=0.0625\).
- DASP minus SSF-budget Early Attack Recall@5:
  \(\Delta=0.003729\), bootstrap 95% CI
  \([0.000145,0.007186]\), exact two-sided \(p=0.25\).
- All seven exploratory DASP--baseline contrasts have Holm-adjusted
  \(p\ge 0.4375\).

Therefore, the current single-dataset evidence estimates favorable effects but
does not establish superiority at 0.05.

## Required experiment closure

1. Generate and validate S5. Any artifact failing the absolute or
   matched-random uplift gates must be rejected, not relabeled.
2. Run the locked primary matrix:
   four NF-v3 datasets × S0--S4 × five stream seeds × two model seeds × ten
   methods = 2,000 runs.
3. Run the separate S5 matrix:
   four NF-v3 datasets × S5 × five stream seeds × two model seeds × ten methods
   = 400 runs.
4. Inspect `causal_ablation_statistics.json`; do not claim trigger benefit
   unless proposed beats detector-off and closes a material, positive oracle
   gap.
5. Report only dataset/scenario cells that pass the construct-validity gate as
   boundary-impacting proxies. Report the remainder as distribution shift or
   inconclusive.
6. Complete author metadata and deposit code/configs/manifests/raw outputs in a
   public archive with a persistent identifier.
7. Run `tools/audit_q1_submission_readiness.py --strict`. Submission language
   may be broadened only when every gate passes.

## Files changed by this revision

- `tools/build_controlled_nf_streams.py`
- `tools/validate_controlled_streams.py`
- `tools/analyze_dasp_v16_five_dataset.py`
- `tools/audit_q1_submission_readiness.py`
- `tests/test_s5_stream_generator.py`
- `tests/test_analysis_statistics.py`
- `tests/test_construct_validation.py`
- `DASP_FIVE_DATASET_DRIFT_ADAPTATION_PROTOCOL.md`
- `README.md`
- `ACCESS_latex_template_20260513/ACCESS_ConceptDrift_Cyber.tex`

## Integrity statement

No S5, four-dataset, detector-off, oracle-trigger, or natural-track result was
invented or inferred from an incomplete matrix. Unresolved items remain marked
as pending experiments or pending author input.
