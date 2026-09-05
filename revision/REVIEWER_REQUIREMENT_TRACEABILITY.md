# Reviewer Requirement Traceability

## Status Legend

- `DESIGN COMPLETE`: protocol is sufficiently specified to implement without outcome-driven choices.
- `IMPLEMENTATION PENDING`: code and tests have not yet been completed.
- `MANUSCRIPT PENDING`: no experiment is required, but the revised paper or response letter must be updated.
- `AUTHOR INPUT`: factual declarations must be supplied and verified by the authors.

## Reviewer 1

| ID | Reviewer requirement | Planned evidence or action | Protocol location | Design status |
|---|---|---|---|---|
| R1-1 | Random, fixed, MMD-only, and conflict-only routing controls under equal repair/replay budgets | A1--A6 routing matrix plus event-matched routing diagnostic | Sections 5--6.1 | COMPLETE; 5,000 P0 runs validated |
| R1-2 | Equal total label exposure or protocol-specific claims | Common 512-label query, common reveal time, disjoint repair/gate pools, per-purpose label logs, DASP--SSF comparison | Section 4 | COMPLETE; label contract validated in P0 and natural P1 |
| R1-3a | Separate modular-backbone contribution | Exactly parameter-matched flat global-adaptation control B5 versus modular all-module and selective DASP | Section 13 | COMPLETE; B5 versus A6 analyzed |
| R1-3b | Separate attribution-routing contribution | A1 versus A2--A6 and event-matched diagnostic | Sections 5--6.1 | COMPLETE; cluster-aware P0 analysis available |
| R1-3c | Separate replay contribution | B1 no-replay comparison | Section 13 | COMPLETE; replay contribution supported |
| R1-3d | Separate new-sample weighting contribution | B2 weight-equals-one comparison | Section 13 | COMPLETE; no detectable P1 cluster effect |
| R1-3e | Separate performance-gating contribution | B3 trigger-only comparison | Section 13 | COMPLETE; gating contribution supported |
| R1-4 | Add natural chronological/operational streaming evaluation or moderate claims | Untouched timestamp-sorted NF-ToN-IoT-v3 evaluation and claim rules | Sections 12 and 15 | COMPLETE; manuscript claims revised |
| R1-5 | Complete repository and reproducibility package | Existing baseline lock plus pinned revision environment, exact commands, hardware, configs, checksums, and archive release | Section 11; baseline-lock directory | LOCAL PACKAGE AND MANUSCRIPT COMPLETE; TAGGED PUBLIC REVISION RELEASE PENDING |
| R1-6 | Complete author contributions, funding, and conflict declarations | Replace placeholders with author-verified statements | Outside experiment protocol | AUTHOR INPUT REQUIRED BEFORE RESUBMISSION |
| R1-7 | Improve conciseness | Move detailed invariants, implementation audits, extended configs, and reproduction commands to appendices/supplement | Outside experiment protocol | COMPLETE; repeated post-conclusion section removed from compiled article |

## Reviewer 2

| ID | Reviewer requirement | Planned evidence or action | Protocol location | Design status |
|---|---|---|---|---|
| R2-1 | Isolate routing, modularization, replay, class balance, weighting, gate, and selective optimization | A-series routing/locality controls and B-series component controls including B4 uniform selection | Sections 5--7 and 13 | COMPLETE; P0 and P1 controls analyzed |
| R2-2 | Common label-access contract for DASP and primary comparator | Equal-label DASP--SSF experiment with identical query indices and exposure budget | Section 4 | COMPLETE; validation passed |
| R2-3 | Trainable-parameter fraction, `k=1,2,3`, all-module, and classifier-only | A6--A9 and quantitative locality logs | Sections 5 and 7 | COMPLETE; P0 logs and contrasts available |
| R2-4 | Feature-order robustness | Original order plus three fixed random permutations for A1, A6, and A9 | Section 14 | COMPLETE; 405/405 runs validated |
| R2-5 | Explain pilot influence and add unseen confirmation | Hyperparameter provenance table; seed-held-out wording; untouched chronological drift configuration | Sections 15--16 | COMPLETE; provenance and scope clarified |
| R2-6 | Cluster-aware uncertainty | 50 dataset--scenario cluster analysis, bootstrap CI, effect sizes, Holm correction, leave-one-dataset-out sensitivity | Section 9 | COMPLETE |
| R2-7 | Unmodified chronological experiment | Timestamp-sorted natural NF-ToN-IoT-v3 stream without injected drift or per-window balancing | Section 15 | COMPLETE; descriptive single-stream scope |
| R2-8 | Clarify accuracy/adaptation/inference trade-off | Common systems audit and explicit claim restriction | Sections 12 and 17 | COMPLETE; manuscript reframed as a three-way trade-off |

## Editorial and Cross-Cutting Deliverables

| Deliverable | Status |
|---|---|
| Preserve original 4,000-run evidence | COMPLETE; SHA-256 verification passes |
| Primary P0 protocol | COMPLETE; executed without post-lock tuning |
| Hyperparameter provenance table | COMPLETE |
| Exact revision environment lock | COMPLETE; clean `.venv-revision` passes `pip check` |
| Revision implementation and focused tests | COMPLETE; final focused suite 25/25 passed |
| Smoke matrix | COMPLETE; 10/10 runs, validation passed |
| Reduced pilot | COMPLETE; 150/150 runs, 15/15 cells passed validation |
| Full 5,000-run confirmation | COMPLETE; 5,000/5,000 runs and 500/500 cells validated |
| P1 component ablations | COMPLETE; 225/225 runs and 45/45 groups validated |
| P1 feature-permutation robustness | COMPLETE; 405/405 runs and 135/135 groups validated |
| P1 untouched chronological evaluation | COMPLETE; 12/12 runs and 3/3 groups validated |
| Cluster-aware analysis | COMPLETE; 50 P0 clusters and 15 P1 clusters |
| Revised manuscript and response letter | COMPLETE; compilation/visual QA pending |

## Final Design Verdict

As of September 4, 2026, all pre-specified P0 and P1 experiments and the cluster-aware analysis have been executed and integrated into the revised manuscript and response letter. The evidence requires narrower claims: attribution-guided routing is not predictively superior to all-module routing under the locked protocol, while replay and performance gating show clearer component contributions. The remaining author-controlled actions are publication metadata verification and upload of the tagged revision archive; compilation and visual QA are the remaining technical checks.
