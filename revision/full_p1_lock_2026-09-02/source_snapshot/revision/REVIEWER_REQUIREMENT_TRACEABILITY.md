# Reviewer Requirement Traceability

## Status Legend

- `DESIGN COMPLETE`: protocol is sufficiently specified to implement without outcome-driven choices.
- `IMPLEMENTATION PENDING`: code and tests have not yet been completed.
- `MANUSCRIPT PENDING`: no experiment is required, but the revised paper or response letter must be updated.
- `AUTHOR INPUT`: factual declarations must be supplied and verified by the authors.

## Reviewer 1

| ID | Reviewer requirement | Planned evidence or action | Protocol location | Design status |
|---|---|---|---|---|
| R1-1 | Random, fixed, MMD-only, and conflict-only routing controls under equal repair/replay budgets | A1--A6 routing matrix plus event-matched routing diagnostic | Sections 5--6.1 | IMPLEMENTATION COMPLETE; RUNS PENDING |
| R1-2 | Equal total label exposure or protocol-specific claims | Common 512-label query, common reveal time, disjoint repair/gate pools, per-purpose label logs, DASP--SSF comparison | Section 4 | IMPLEMENTATION COMPLETE; RUNS PENDING |
| R1-3a | Separate modular-backbone contribution | Exactly parameter-matched flat global-adaptation control B5 versus modular all-module and selective DASP | Section 13 | DESIGN COMPLETE; IMPLEMENTATION PENDING |
| R1-3b | Separate attribution-routing contribution | A1 versus A2--A6 and event-matched diagnostic | Sections 5--6.1 | IMPLEMENTATION COMPLETE; RUNS PENDING |
| R1-3c | Separate replay contribution | B1 no-replay comparison | Section 13 | DESIGN COMPLETE; IMPLEMENTATION PENDING |
| R1-3d | Separate new-sample weighting contribution | B2 weight-equals-one comparison | Section 13 | DESIGN COMPLETE; IMPLEMENTATION PENDING |
| R1-3e | Separate performance-gating contribution | B3 trigger-only comparison | Section 13 | DESIGN COMPLETE; IMPLEMENTATION PENDING |
| R1-4 | Add natural chronological/operational streaming evaluation or moderate claims | Untouched timestamp-sorted NF-ToN-IoT-v3 evaluation and claim rules | Sections 12 and 15 | DESIGN COMPLETE; IMPLEMENTATION PENDING |
| R1-5 | Complete repository and reproducibility package | Existing baseline lock plus pinned revision environment, exact commands, hardware, configs, checksums, and archive release | Section 11; baseline-lock directory | ENVIRONMENT COMPLETE; RUN COMMANDS/CHECKSUMS/ARCHIVE PENDING |
| R1-6 | Complete author contributions, funding, and conflict declarations | Replace placeholders with author-verified statements | Outside experiment protocol | AUTHOR INPUT; MANUSCRIPT PENDING |
| R1-7 | Improve conciseness | Move detailed invariants, implementation audits, extended configs, and reproduction commands to appendices/supplement | Outside experiment protocol | MANUSCRIPT PENDING |

## Reviewer 2

| ID | Reviewer requirement | Planned evidence or action | Protocol location | Design status |
|---|---|---|---|---|
| R2-1 | Isolate routing, modularization, replay, class balance, weighting, gate, and selective optimization | A-series routing/locality controls and B-series component controls including B4 uniform selection | Sections 5--7 and 13 | A-SERIES IMPLEMENTATION COMPLETE; B-SERIES PENDING |
| R2-2 | Common label-access contract for DASP and primary comparator | Equal-label DASP--SSF experiment with identical query indices and exposure budget | Section 4 | IMPLEMENTATION COMPLETE; RUNS PENDING |
| R2-3 | Trainable-parameter fraction, `k=1,2,3`, all-module, and classifier-only | A6--A9 and quantitative locality logs | Sections 5 and 7 | IMPLEMENTATION COMPLETE; RUNS PENDING |
| R2-4 | Feature-order robustness | Original order plus three fixed random permutations for A1, A6, and A9 | Section 14 | DESIGN COMPLETE; IMPLEMENTATION PENDING |
| R2-5 | Explain pilot influence and add unseen confirmation | Hyperparameter provenance table; seed-held-out wording; untouched chronological drift configuration | Sections 15--16 | DESIGN COMPLETE; IMPLEMENTATION/MANUSCRIPT PENDING |
| R2-6 | Cluster-aware uncertainty | 50 dataset--scenario cluster analysis, bootstrap CI, effect sizes, Holm correction, leave-one-dataset-out sensitivity | Section 9 | DESIGN COMPLETE; ANALYSIS PENDING |
| R2-7 | Unmodified chronological experiment | Timestamp-sorted natural NF-ToN-IoT-v3 stream without injected drift or per-window balancing | Section 15 | DESIGN COMPLETE; IMPLEMENTATION PENDING |
| R2-8 | Clarify accuracy/adaptation/inference trade-off | Common systems audit and explicit claim restriction | Sections 12 and 17 | DESIGN COMPLETE; ANALYSIS/MANUSCRIPT PENDING |

## Editorial and Cross-Cutting Deliverables

| Deliverable | Status |
|---|---|
| Preserve original 4,000-run evidence | COMPLETE; SHA-256 verification passes |
| Primary P0 protocol | DESIGN COMPLETE |
| P1 component ablations | DESIGN COMPLETE in protocol version 1.1 |
| P1 feature-permutation robustness | DESIGN COMPLETE in protocol version 1.1 |
| P1 untouched chronological evaluation | DESIGN COMPLETE in protocol version 1.1 |
| Hyperparameter provenance table | CONTENT PENDING |
| Exact revision environment lock | COMPLETE; clean `.venv-revision` passes `pip check` |
| Revision implementation and tests | COMPLETE FOR PHASE B; P0/P1 RUNS PENDING |
| Smoke matrix | COMPLETE; 10/10 runs, validation passed |
| Reduced pilot | COMPLETE; 150/150 runs, 15/15 cells passed validation |
| Full 5,000-run confirmation | COMPLETE; 5,000/5,000 runs and 500/500 cells validated |
| P1 runs | PENDING |
| Revised manuscript and response letter | PENDING |

## Final Design Verdict

As of protocol version `1.1`, every experimental or analytical request made by Reviewers 1 and 2 has a pre-specified response. Publication metadata and concision are manuscript tasks rather than experiments and remain dependent on author input or later editing. No revision experiment has been executed yet.
