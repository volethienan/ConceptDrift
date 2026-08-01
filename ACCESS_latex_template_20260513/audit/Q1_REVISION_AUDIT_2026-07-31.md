# Q1-Style Revision Audit

Date: 2026-07-31
Manuscript: DASP: Drift-Attributed Subspace Plasticity for Localized Continual Adaptation in Streaming Network Intrusion Detection

## Revision constraint

The method, source-code behavior, experimental matrices, and reported numerical results were treated as locked. No experiment was rerun and no value was changed during this revision.

## Critical issues addressed

1. **Causal interpretation of attribution** — The manuscript now defines attribution as an operational routing responsibility derived from discrepancy and delayed-label conflict scores, not causal feature attribution.
2. **Router characterization** — The router is described as a fixed data-driven scoring-and-selection rule, not an end-to-end learned router.
3. **Construct validity of concept drift** — Controlled scenario transformations are distinguished from proof that every episode changes the real-world binary decision function.
4. **Statistical replication scope** — The paired cell is identified as the analysis unit, while dependence induced by shared datasets and scenario generators is disclosed. Nominal p-values are treated as sensitivity evidence for the locked matrix.
5. **Baseline fidelity scope** — Claims are restricted to the locked protocol-level implementations unless an official implementation is explicitly identified.

## Major issues addressed

1. Added explicit RQ1--RQ3 alignment between accuracy, structural locality/adaptation cost, and failure boundaries.
2. Added a Related Work synthesis that connects concept-drift detection, continual-learning protection, and continual-NIDS evaluation to the exact research gap.
3. Clarified that the classifier is a shared adaptation group and that the inactive-module invariant applies to representation modules.
4. Reframed fairness as equality of the external streaming contract rather than architectural or parameter-count equality.
5. Replaced leaderboard-style and causal wording with evidence-bounded interpretation.
6. Expanded Discussion to distinguish sequential development evidence from factorial causal ablation.
7. Expanded limitations for controlled-shift validity, implementation-specific throughput, dependence across analysis cells, baseline reproduction, and external replication.
8. Removed manuscript-facing phrases such as “Q1-level evidence” and “reviewer-oriented checks”; the corresponding material is now framed as experimental integrity and reproducibility.

## Residual pre-submission blockers

These items require verified author input and were not fabricated:

- Final author names, affiliations, email addresses, and corresponding-author metadata.
- Received/revised dates and DOI placeholder.
- Verified CRediT contributions, funding statement, and conflict-of-interest declaration.
- Public repository or archival DOI for code, streams, configurations, and row-level outputs.
- Dataset-license verification and final bibliographic metadata check.
- Pinned Python/PyTorch environment, CPU identification, and an author-independent archival reproduction.

## Editorial risk after revision

The scientific narrative is substantially stronger and more conservative. Remaining rejection risk is concentrated in external validity, official-baseline fidelity, controlled-versus-natural drift semantics, and the absence of independent reproduction. These risks are now disclosed rather than hidden, but they cannot be removed without new evidence, which is outside the locked revision scope.
