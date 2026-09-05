# DASP Major-Revision Experiment Protocol

## Protocol Status

- Protocol version: `1.1`
- Protocol lock date: `2026-09-01`
- Original confirmation: `outputs_confirmation_v6_10scenario/full_summary.csv`
- Original confirmation size: 4,000 method runs over 500 paired cells
- Revision output root: `outputs_revision_v1`
- Status: pre-registered design; no revision result may be used to alter the primary definitions below

Version `1.1` completes the secondary protocols requested by the reviewers. This amendment was made before implementation or inspection of any revision-run outcome.

This document fixes the primary revision experiment before implementation and execution. Any later deviation must be recorded in a dated amendment that states the reason, affected runs, and whether the change was made before or after inspecting outcome metrics.

## 1. Objectives

The revision experiments address four primary questions.

1. Does combined MMD/conflict attribution outperform matched routing controls?
2. Does the DASP--SSF comparison remain favorable under equal total label exposure?
3. How does performance change as the representation-module budget varies from classifier-only to all-module adaptation?
4. Does representation-module locality correspond to a quantitatively meaningful restriction on trainable and changed parameters?

Component ablations, feature-partition robustness, and the untouched natural chronological evaluation are secondary experiments with designs fixed in Sections 13--15. They are not part of the 5,000-run primary matrix but remain required for the planned reviewer response.

## 2. Frozen Original Baseline

The original 4,000-run matrix remains a separate protocol-specific result. It must not be overwritten, merged with revision runs, or silently regenerated under the revised supervision contract.

- Preserve `outputs_full_v6_core`.
- Preserve `outputs_full_v6_10scenario_pilot`.
- Preserve `outputs_confirmation_v6_10scenario`.
- Store all revision runs under `outputs_revision_v1` or a later explicitly versioned root.
- Use `revision/baseline_lock_2026-09-01/evidence_sha256.csv` to detect changes to the original computational evidence.
- Describe the original matrix as the integrated-system confirmation under method-specific supervision semantics.

## 3. Common Experimental Cells

The primary revision confirmation uses the same paired cells as the original confirmation.

- Datasets: CIC-IDS2017, NF-BoT-IoT-v3, NF-CSE-CIC-IDS2018-v3, NF-ToN-IoT-v3, and NF-UNSW-NB15-v3.
- Scenarios: S0, S1-Sudden, S2-Gradual, S3-Incremental, S4-Recurrent, S5-Blip, S6-Combined, S7-Local-Low, S7-Local-Medium, and S7-Local-High.
- Stream seeds: 2001, 2002, 2003, 2101, and 2102.
- Model seeds: 91 and 92.
- Paired cells: `5 x 10 x 5 x 2 = 500`.
- Window order: prediction first, one-window delayed labels, then repair for future windows.
- Warm-up: four windows.
- Window size: 4,096 flows.

All methods within a paired cell receive the same prepared stream and the same label-query indices.

## 4. Equal-Total-Label-Exposure Contract

### 4.1 Budget

For every delayed non-warm-up window, each revision method receives at most 512 newly revealed ground-truth labels.

- Repair-candidate pool: 384 labels.
- Performance-gate pool: 128 labels.
- Maximum new examples used for optimization: 128.
- DASP replay examples per repair: 512, drawn only from previously revealed labels.
- The gate pool is disjoint from the repair-candidate pool and may not enter optimization or replay during the same window.

The 512-label budget is one eighth of a 4,096-flow window. It preserves the locked 128-example optimization budget while retaining a sufficiently sized disjoint gate and a bounded candidate pool for class-balanced selection.

### 4.2 Label-Blind Query

The 512 query indices must be generated before reading the corresponding labels. The query seed must be a deterministic function of:

- dataset key;
- scenario key;
- stream seed;
- delayed window index;
- protocol version.

The label-query seed must not depend on method, model predictions, labels, or outcome metrics. Therefore, every method in a paired cell receives identical exposed indices.

### 4.3 Pool Assignment

The queried indices are deterministically partitioned before labels are used:

1. the first 384 shuffled query indices form the repair-candidate pool;
2. the remaining 128 indices form the performance-gate pool.

After the labels have been revealed:

- DASP and all routing controls select at most 128 class-balanced optimization examples from the common 384-example repair-candidate pool;
- all routing controls use exactly the same selected optimization indices as combined DASP;
- SSF may apply its documented selection rule only within the common repair-candidate pool and may insert at most 128 ground-truth examples;
- no method may inspect labels outside the common 512-example exposed pool;
- pseudo-labels remain disabled in the primary equal-exposure comparison.

### 4.4 Gate Use

Combined DASP and its routing controls use the common 128-example gate pool for the same pre-update performance-gating rule. Routing-control comparisons must not alter the gate threshold or repair intensity logic.

SSF receives the same gate labels to satisfy the exposure contract. If the locked SSF procedure does not use them, they must be logged as exposed but unused. The manuscript must distinguish equal label availability from algorithm-specific label use.

### 4.5 Required Label Logs

Every window record must include:

- `label_contract_version`;
- `labels_exposed_total`;
- `repair_candidate_labels_exposed`;
- `gate_labels_exposed`;
- `new_training_labels_used`;
- `labels_used_for_selection`;
- `labels_used_for_gating`;
- `pseudo_labels_used`;
- `query_index_digest`;
- `repair_candidate_index_digest`;
- `gate_index_digest`;
- `training_index_digest`.

The primary validation must fail if exposed-label counts or query digests differ across methods within a paired cell.

## 5. Primary Routing and Locality Conditions

The primary revision matrix contains ten conditions and 5,000 runs.

| ID | Condition | Representation routing rule | Purpose |
|---|---|---|---|
| A1 | Combined DASP, `k=3` | rank `MMD requirement / (1 + conflict)` | Full method |
| A2 | Random top-3 | deterministic random choice of three modules per repair | Attribution-free routing control |
| A3 | Fixed top-3 | always open modules 1, 2, and 3 | Fixed routing control |
| A4 | MMD-only top-3 | rank MMD requirement without conflict attenuation | Isolate discrepancy contribution |
| A5 | Conflict-only top-3 | rank conflict-derived compatibility without MMD requirement | Isolate conflict contribution |
| A6 | All-module, `k=4` | open all four representation modules | Locality ablation |
| A7 | Combined DASP, `k=1` | combined score, top one | Strong locality sensitivity |
| A8 | Combined DASP, `k=2` | combined score, top two | Intermediate locality sensitivity |
| A9 | Classifier-only, `k=0` | freeze all representation modules | Test necessity of representation repair |
| A10 | SSF | locked SSF logic under the common label-access contract | Primary comparator |

### 5.1 Conflict-Only Definition

Conflict is measured as the non-negative part of negative cosine similarity between new-example and replay gradients. Higher conflict indicates greater interference. The conflict-only routing score must use the same compatibility transformation as the combined router, `1 / (1 + conflict_weight * conflict)`, and rank modules by this score. The manuscript must describe this as conflict-aware compatibility routing rather than routing toward the most conflicting module.

### 5.2 Random Routing Definition

Random top-3 routing uses a deterministic seed derived from the paired-cell identifiers, delayed window index, and a routing-policy namespace. The random seed must not be tuned or rerolled after inspecting results. The primary result uses one deterministic routing realization per paired cell. A secondary robustness analysis may add multiple routing seeds but may not replace the pre-registered primary realization.

### 5.3 Fixed Routing Definition

The primary fixed routing mask is modules 1, 2, and 3, selected before outcome inspection. A secondary leave-one-module-out analysis may report all four fixed three-module masks. The primary fixed mask must not be selected from the best-performing secondary mask.

## 6. Controls Held Constant Across Routing Variants

Conditions A1--A9 use the same:

- modular backbone initialization;
- prepared input windows and prediction-first evaluation order;
- exposed label indices;
- 128-example selected repair subset;
- 128-example gate subset;
- 512-example replay subset;
- replay memory state at the start of the repair;
- MMD computation and trigger rule;
- performance-gate threshold and repair-mode rule;
- optimizer type, learning rate, weight decay, batch size, and epoch count;
- new-sample weight;
- reference-bank refresh rule;
- classifier plasticity, except that classifier-only freezes all representation modules;
- strict exclusion of inactive representation modules from the optimizer.

Only the representation routing rule or the pre-registered value of `k` may differ. Any implementation detail that violates this rule converts the experiment from a routing ablation into a system ablation and must be reported separately.

The end-to-end variants use the same causal trigger and performance-gate rules, but their repair-event counts may diverge because their learned model states and predictions diverge. The number of repair events, cumulative replay use, and cumulative parameter-gradient work must therefore be reported rather than assumed equal.

### 6.1 Event-Matched Routing Diagnostic

To isolate the immediate effect of the routing mask from differences in later model trajectories, the revision also includes an event-matched diagnostic based on combined-DASP repair events.

For each repair event in A1:

1. save the exact pre-repair model, optimizer-independent state, replay memory, reference bank, repair indices, gate indices, and replay indices;
2. clone that state once for each routing control A2--A6 and A9;
3. apply one repair with the same samples, weights, epochs, optimizer settings, and repair intensity, changing only the routing mask;
4. evaluate every branch on the same next prediction window;
5. discard the diagnostic branches so they do not alter the end-to-end streams.

The event-matched analysis reports the paired next-window change in Macro-F1, attack recall, loss, update time, and changed-parameter fraction. It is a local mechanism diagnostic and must not replace the end-to-end comparison.

## 7. Locality Measurements

Module count alone is insufficient because the shared fusion/classification head contains most model parameters. Every repair must report:

- active representation-module count;
- active representation-module fraction;
- total model parameter count;
- optimizer-included parameter count;
- optimizer-included parameter fraction;
- representation parameter count;
- active representation parameter count;
- active representation parameter fraction;
- parameter elements changed above tolerance after repair;
- changed-parameter fraction;
- maximum absolute change in each inactive module;
- classifier parameter fraction;
- gradient evaluations and parameter-gradient element evaluations.

The default numerical change tolerance is `1e-12` in the stored parameter dtype. Exact equality remains the strict-freezing invariant for inactive modules; the tolerance-based statistic is an additional summary for active parameters.

The manuscript must define locality as representation-subspace intervention scope. It must not imply that `k=3` updates only a small fraction of all parameters unless the reported parameter fractions support that statement.

## 8. Primary Outcomes

### 8.1 Predictive Outcomes

- Mean prequential Macro-F1.
- Cumulative attack recall.
- Cumulative false-negative rate.
- Online forgetting.
- Window-level stability.
- Worst paired cell and worst dataset--scenario cluster.

### 8.2 Systems Outcomes

- Total and median update time.
- Prediction-and-decision throughput.
- Gradient evaluations.
- Parameter-gradient element evaluations.
- Peak RSS increment.
- Exposed labels and labels used by purpose.

### 8.3 Routing Outcomes

- Active-module count and fraction.
- Per-module selection frequency.
- Consecutive-window routing-set Jaccard overlap.
- Fraction of repairs matching each fixed mask.
- Performance by drift family and local-drift severity.

## 9. Statistical Analysis

The primary uncertainty unit is the dataset--scenario cluster, not an individual window or seed-level cell.

1. Average paired method differences over the ten stream-seed/model-seed replications within each dataset--scenario combination.
2. Report the 50 cluster-level differences, their mean, median, quartiles, wins, and worst cluster.
3. Report a cluster-level sign test and Wilcoxon signed-rank sensitivity analysis.
4. Report a bootstrap confidence interval by resampling dataset--scenario clusters with replacement while preserving method pairing.
5. Report dataset-level aggregates and leave-one-dataset-out sensitivity.
6. Retain 500-cell paired tests only as secondary protocol-specific analyses.
7. Apply Holm correction across the pre-registered routing comparisons against combined DASP.
8. Report effect sizes and confidence intervals rather than relying on nominal p-values alone.

The primary routing contrasts are A1 versus A2, A3, A4, A5, A6, and A9. The `k` sensitivity comparison across A7, A8, A1, and A6 is descriptive unless a separate ordered-trend test is pre-specified before execution.

## 10. Execution Gates

### Gate 1: Unit and Invariant Tests

The implementation may not enter a matrix run until tests verify:

- label queries are label-blind and identical across methods;
- repair and gate pools are disjoint;
- routing variants receive identical repair and replay indices;
- classifier-only leaves every representation parameter unchanged;
- `k=1`, `k=2`, `k=3`, and `k=4` activate the expected maximum number of representation modules;
- inactive modules are excluded from the optimizer and remain exactly unchanged;
- random routing is deterministic under the declared seed;
- all required label and locality logs are populated.

### Gate 2: Smoke Matrix

- One dataset.
- One scenario.
- One stream seed.
- One model seed.
- All ten conditions.

The smoke matrix is for correctness only and may not be used for parameter selection.

### Gate 3: Reduced Pilot

- Five datasets.
- S1-Sudden, S2-Gradual, and S7-Local-High.
- Stream seed 2001.
- Model seed 91.
- All ten conditions.

The reduced pilot checks runtime, missing outputs, invariant violations, degenerate labels, and numerical failures. Outcome differences may be inspected for debugging but may not be used to change the primary routing definitions, label budget, thresholds, replay size, or weighting.

### Gate 4: Full Revision Confirmation

- 500 paired cells.
- Ten conditions.
- 5,000 method runs.

The configuration and code commit must be recorded before the first full run. After execution starts, outcome-driven changes require a new protocol version and a complete rerun of affected conditions.

## 11. Reproducibility Requirements for Revision Runs

Before Gate 4, archive:

- exact repository commit;
- clean working-tree status or a patch file covering all changes;
- pinned Python and dependency versions;
- operating system, CPU, RAM, GPU, and execution device;
- all YAML configuration files;
- label-query and routing seed derivation rules;
- smoke, pilot, full-run, analysis, table, and figure commands;
- per-run config, summary, window log, and source-stream checksums;
- resume behavior and duplicate-run detection;
- final aggregate SHA-256 manifest.

The current global inspection environment does not satisfy `numpy<2` from `pyproject.toml`. The existing `.venv` and `.venv-cpu` environments also use `include-system-site-packages = true`, and `.venv-cpu` fails `pip check` because unrelated system packages and the current River/NumPy requirements conflict. None of these environments may be declared as the revision environment.

Before Gate 1, create a clean isolated `.venv-revision` with `include-system-site-packages = false`. The P0/P1 method list does not use the optional River baselines, so River must be removed from the core revision dependency set or installed in a separate optional environment with a compatible NumPy contract. The revision environment must pass `python -m pip check` before tests or experiments are accepted.

## 12. Claim Rules Before Results

Until the revision results are available:

- do not claim that attribution-guided routing independently causes the reported gain;
- describe the original DASP result as an integrated-system result;
- describe the original confirmation as seed-held-out robustness over the same datasets and scenario families;
- describe locality as representation-module selectivity, not parameter sparsity;
- describe efficiency as an adaptation-cost versus inference-throughput trade-off;
- do not characterize controlled scenario transformations as live operational drift.

## 13. P1 Component-Isolation Protocol

The component-isolation experiment uses a locked subset rather than repeating every secondary variant over all 500 cells.

- Datasets: all five primary datasets.
- Scenarios: S1-Sudden, S2-Gradual, and S7-Local-High.
- Stream seeds: 2001, 2002, and 2003.
- Model seed: 91.
- Cells per condition: `5 x 3 x 3 x 1 = 45`.
- Reference: A1 combined DASP restricted to the same 45 cells.

The five component conditions are:

| ID | Condition | Change from A1 | Component isolated |
|---|---|---|---|
| B1 | No replay | replay size is zero | Replay contribution |
| B2 | No new-sample weighting | new-example weight is 1.0 | Weighting contribution |
| B3 | No performance gate | disable no-trigger gate repair; repair only after the stored MMD trigger | Performance-gate contribution |
| B4 | Uniform repair selection | choose 128 repair examples uniformly without using labels, from the common 384-example candidate pool | Class-balanced selection contribution |
| B5 | Flat matched global adaptation | use a parameter-matched flat MLP with global representation adaptation and matched replay, label, gate, epoch, and learning-rate budgets | Modular-backbone contribution |

Conditions B1--B4 retain the modular backbone, combined routing, equal-label query, trigger threshold, optimizer, and all other A1 settings. B5 is an architecture control. It uses hidden widths `[136, 96]` for the 49-feature NF-v3 datasets and `[184, 40]` for the 78-feature CIC-IDS2017 interface. These choices exactly match the corresponding modular model totals of 20,610 and 22,466 parameters and were selected analytically before outcome inspection.

B5 computes its MMD trigger from the final flat hidden representation using the same calibration, robust normalization, trigger threshold, persistence, and performance-gate rules as A6. All flat representation layers and the classifier are trainable during an applied repair. The comparison B5 versus A6 estimates the effect of the modular representation layout under matched total parameter count and common supervision/repair budgets; the manuscript must acknowledge that changing the representation layout also changes the representation on which MMD is computed.

The primary component analysis reports paired differences from A1 over the 45 locked cells. A component condition is expanded to all 500 cells only if it is needed for a main manuscript claim or if the locked-subset confidence interval is too wide to support a conclusion. Expansion rules must not depend on whether the point estimate is favorable to DASP.

## 14. P1 Feature-Partition Robustness Protocol

The feature-order experiment tests whether contiguous partition performance depends on the original column ordering.

- Orders: original order plus three fixed random permutations.
- Permutation seeds: 3101, 3102, and 3103.
- Permutations are generated once per dataset before model initialization.
- The same permutation is used for every method, window, stream seed, and model seed in the corresponding paired comparison.
- Labels, timestamps, and excluded identifier columns are never permuted into the predictor feature set.
- The feature order remains fixed for the complete stream; no window-specific permutation is allowed.

The locked evaluation subset is:

- all five datasets;
- S1-Sudden, S2-Gradual, and S7-Local-High;
- stream seeds 2001, 2002, and 2003;
- model seed 91;
- methods A1 combined DASP, A6 all-module, and A9 classifier-only.

The original-order results are reused from the corresponding primary cells. The three new permutations require `3 x 45 x 3 = 405` runs. Report the mean, range, standard deviation, worst permutation, and method-by-order interaction descriptively. A semantic grouping analysis is optional and may not replace the random-permutation result.

## 15. P1 Untouched Natural Chronological Protocol

NF-ToN-IoT-v3 is the primary untouched chronological evaluation. It was selected before revision outcomes were inspected because the raw file contains `FLOW_START_MILLISECONDS` and represents one of the more difficult datasets in the original confirmation.

Stream construction must:

1. sort the original flows by `FLOW_START_MILLISECONDS` using a stable sort;
2. exclude timestamps, IP addresses, labels, and identifiers from predictor features;
3. retain the natural class prevalence and attack-family sequence;
4. use consecutive non-overlapping 4,096-flow windows;
5. apply no artificial drift transformation, class-prevalence schedule, per-window balancing, or attack-family resampling;
6. fit preprocessing only on the first four warm-up windows;
7. use prediction-first evaluation and one-window delayed labels;
8. use the equal-label contract from Section 4 after warm-up.

The primary natural stream consists of the earliest 147,456 valid flows after the stable timestamp sort, producing exactly 36 consecutive windows of 4,096 flows. Row validity and non-finite-value handling follow one predeclared cleaning implementation shared with all four methods. If the selected prefix contains only one binary class after cleaning, the experiment is still reported as a degenerate natural prefix and may be supplemented, but not replaced, by the next non-overlapping 36-window chronological block.

The methods are A1 combined DASP, A10 SSF, experience replay, and online MLP. Model seeds are 91, 92, and 93. The natural stream has one deterministic chronological construction rather than multiple random stream seeds, giving 12 primary runs.

This evaluation is untouched with respect to method selection: it may not be used to revise `k`, thresholds, memory size, weighting, gate rules, or routing definitions. It serves simultaneously as the unmodified chronological experiment and an unseen drift-configuration confirmation. NF-CSE-CIC-IDS2018-v3 may be added as a secondary chronological replication if compute permits, but it may not replace NF-ToN-IoT-v3 based on observed results.

## 16. Hyperparameter Provenance and Confirmation Interpretation

Before the full revision confirmation, create a provenance table covering at least:

- module count and module width;
- `k`;
- MMD sample size, kernel scales, calibration windows, and severity threshold;
- persistence and cooldown;
- repair and emergency epochs;
- replay capacity and repair replay size;
- new-sample weight;
- performance-gate threshold;
- background plasticity;
- learning rate, weight decay, and batch size;
- label-query, repair-candidate, gate, and optimization budgets.

For each value, record whether it was theory/design derived, inherited from prior work, selected during development, or influenced by the pilot. The original confirmation must be described as seed-held-out robustness over the same datasets and scenario families, not validation on unseen environments. The untouched chronological protocol in Section 15 provides the pre-registered unseen drift configuration.

## 17. Systems-Measurement Audit

All primary and component comparisons must use the same declared execution environment within an analysis block.

- Record OS, CPU, RAM, GPU availability, execution device, package versions, PyTorch thread count, and process-level parallelism.
- Use the same batch size and timing boundaries for paired methods.
- Do not run competing training jobs during a timing block.
- Record total update time, median repair time, prediction-and-decision throughput, gradient evaluations, and parameter-gradient element evaluations.
- Report method medians and cluster-level uncertainty in addition to means.
- State separately whether a conclusion concerns adaptation cost, inference throughput, memory, or predictive performance.

The manuscript must not describe DASP as simply more efficient. The intended conclusion is an accuracy/adaptation-cost/inference-throughput trade-off whose suitability depends on whether deployment is adaptation-bound or prediction-throughput-bound.

## 18. P2 Optional Extensions

The following are optional after all P0 and P1 requirements are complete:

- multiple random-routing seeds beyond the pre-registered primary realization;
- all four fixed leave-one-module-out masks;
- semantic feature groups;
- a second untouched dataset or a completely external dataset;
- deployment-oriented fused inference variants.

Optional extensions may strengthen interpretation but may not redefine or replace an unfavorable pre-registered P0/P1 result.
