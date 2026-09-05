# P0 Hyperparameter Provenance

Status date: September 1, 2026.

This table was finalized before the first Gate 4 run. Gate 2 and Gate 3 outcomes were inspected only for correctness, runtime, invariant violations, degenerate label pools, and numerical failures. No P0 value below was changed in response to predictive results from either gate.

| Parameter | Locked value | Provenance category | Rationale and history | Pilot influenced? |
|---|---:|---|---|---|
| Representation modules | 4 | Inherited | Existing modular DASP backbone and original confirmation architecture | No |
| Module width | 64 | Inherited | Existing modular DASP backbone | No |
| Hidden dimensions | 128, 64 | Inherited | Existing primary MLP configuration | No |
| Routing `k` for combined DASP | 3 | Inherited | Existing modular DASP primary routing setting | No |
| MMD sample size | 128 | Inherited | Existing primary modular DASP configuration | No |
| MMD kernel scales | 0.5, 1.0, 2.0 | Inherited | Existing multi-scale MMD configuration | No |
| Calibration windows | 4 | Inherited | Existing primary modular DASP configuration | No |
| Severity threshold | 0.15 | Inherited | Existing primary modular DASP configuration | No |
| Emergency severity threshold | 0.60 | Inherited | Existing primary modular DASP configuration | No |
| Trigger persistence | 1 | Inherited | Existing primary modular DASP configuration | No |
| Cooldown windows | 0 | Inherited | Existing primary modular DASP configuration | No |
| Conflict weight | 1.0 | Inherited | Existing combined attribution score definition | No |
| Repair epochs | 2 | Inherited | Existing primary modular DASP repair schedule | No |
| Emergency repair epochs | 3 | Inherited | Existing primary modular DASP emergency repair schedule | No |
| Replay capacity | 5,000 | Revision design | Common revision capacity fixed before Gate 2 to avoid capacity truncation becoming a comparison confound | No |
| Repair replay size | 512 | Inherited | Existing bounded modular replay budget | No |
| Conflict memory batch | 128 | Inherited | Existing primary modular DASP configuration | No |
| New-sample weight | 3.0 | Inherited | Existing modular repair weighting | No |
| Background plasticity | 0.25 | Inherited | Existing modular repair configuration; strict P0 routing controls still exclude inactive modules from the optimizer | No |
| Performance gate | Enabled | Revision design | Required common repair decision mechanism fixed in Protocol v1.1 | No |
| Gate Macro-F1 threshold | 0.80 | Inherited | Existing guarded-repair threshold `v12_guard_min_macro_f1` | No |
| Learning rate | 0.001 | Inherited | Existing primary training configuration | No |
| Weight decay | 0.0001 | Inherited | Existing primary training configuration | No |
| Batch size | 128 | Inherited | Existing primary training configuration | No |
| Label query budget | 512 | Revision design | Equal-total-label-exposure contract | No |
| Repair-candidate pool | 384 | Revision design | Common label-blind candidate pool | No |
| Disjoint gate pool | 128 | Revision design | Common performance evidence unavailable to same-window optimization | No |
| Maximum new optimization labels | 128 | Inherited plus fairness lock | Preserves the original update-label budget under the common exposure contract | No |
| Fixed routing modules | modules 1, 2, 3 | Revision design | Pre-registered deterministic fixed-module control matching `k=3` | No |
| Random routing namespace | `revision-routing-primary-v1` | Revision design | Deterministic pre-registered random top-k realization | No |
| Protocol version | 1.1 | Revision design | Reviewer-response P0 protocol | No |

The executable source of truth is `configs/revision_p0.yaml` together with the fully serialized per-run `config.yaml`. Defaults not repeated in the YAML, including the 0.80 gate threshold, are locked through the recorded source SHA-256 values.
