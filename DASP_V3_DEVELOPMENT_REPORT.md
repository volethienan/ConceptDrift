# DASP-v3 development assessment

## Material passport

- Origin skill: `academic-research-suite`
- Review modes: methodology review, devil's-advocate review, experiment
  validation
- Date: 2026-07-30
- Status: **development gate failed**
- Confirmation status: **locked; no 1,200-run or 4,000-run matrix executed**

## Executive decision

The DASP-v3 proposal was scientifically well motivated, but the implemented
algorithm is not ready for the full pilot. The causal protocol and resource
accounting are defensible; cross-dataset robustness is not.

The final repeated small pilot completed all 120 planned cells. DASP-v3 passed
on NF-UNSW-NB15-v3 and NF-CSE-CIC-IDS2018-v3, but failed on NF-ToN-IoT-v3,
NF-BoT-IoT-v3, and CIC-IDS2017. The failures were not hidden:

- ToN-IoT missed the non-inferiority margin and had one three-window class
  collapse;
- BoT-IoT met the performance margin and removed the earlier collapse, but
  exceeded the fixed compute budget;
- CIC-IDS2017 missed the non-inferiority margin.

Therefore neither the 1,200-run development matrix nor the 4,000-run
confirmation matrix is authorized.

## What was implemented

The implementation adds five cumulative variants:

1. `dasp_v3_memory`: revealed-true-label memory with deterministic
   class-balanced allocation and no replacement;
2. `dasp_v3_query`: adaptive label-blind query allocation;
3. `dasp_v3_backbone`: bounded backbone/calibration recovery;
4. `dasp_v3_checkpoint`: causally eligible scored checkpoints;
5. `dasp_v3`: insufficient-class fallback and early delayed-performance
   recovery.

The final code also:

- never exposes attack subtype to the classifier;
- captures checkpoint state at prediction time but makes it eligible only
  after delayed labels arrive;
- prevents a one-class gradient update;
- uses a disjoint historical guard when the current holdout lacks class
  support;
- restricts global backbone updates;
- records rollback, calibration, query, memory, label-budget, latency, RSS,
  and CUDA diagnostics.

## Verification before real-data pilot

The final synthetic mock used binary `Benign=0`, `Attack=1`; attack families
remained hidden metadata.

| Check | Result |
|---|---:|
| Unit/integration tests | 104 passed |
| Mock matrix | 96/96 complete |
| Integrity violations | 0 |
| DASP-v3 collapse runs | 0 |
| Total update time, v3/v2 | 1.249 |
| Peak RSS, v3/v2 | 0.964 |
| Peak CUDA, v3/v2 | 1.000 |
| Mock gate | Passed |

The earlier ratio-of-medians cost failure was retained separately. The
stream-level total update time is the registered compute-budget statistic;
median and tail latency remain diagnostics.

## Final small-pilot result

Matrix: five datasets × three scenarios (`S0`, `S1-Sudden`,
`S2-Gradual`) × stream seed 2001 × model seed 11 × eight methods = 120
complete runs.

| Dataset | DASP-v3 Macro-F1 | Stronger SSF/ER | Delta | Collapse runs | Update-time ratio | Decision |
|---|---:|---:|---:|---:|---:|---|
| NF-UNSW-NB15-v3 | 0.9729 | 0.9784 | -0.0055 | 0 | 1.636 | Pass |
| NF-CSE-CIC-IDS2018-v3 | 0.9003 | 0.9502 | -0.0499 | 0 | 0.780 | Pass, borderline |
| NF-ToN-IoT-v3 | 0.7613 | 0.8863 | -0.1250 | 1 | 1.113 | Fail |
| NF-BoT-IoT-v3 | 0.8839 | 0.8899 | -0.0061 | 0 | 2.754 | Fail cost |
| CIC-IDS2017 | 0.8440 | 0.9260 | -0.0820 | 0 | 0.832 | Fail |

All RAM ratios were at most 1.037 and CUDA ratios at most 1.018, so memory was
not the limiting resource.

## Ablation interpretation

| Dataset | v2 | Memory | Query | Backbone | Checkpoint | Final v3 |
|---|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15-v3 | 0.9730 | 0.9744 | 0.9724 | 0.9751 | 0.9751 | 0.9729 |
| NF-CSE-CIC-IDS2018-v3 | 0.9010 | 0.9267 | 0.9279 | 0.9255 | 0.9255 | 0.9003 |
| NF-ToN-IoT-v3 | 0.7527 | 0.7514 | 0.7684 | 0.7613 | 0.7613 | 0.7613 |
| NF-BoT-IoT-v3 | 0.8657 | 0.8644 | 0.8863 | 0.8827 | 0.8827 | 0.8839 |
| CIC-IDS2017 | 0.8228 | 0.8326 | 0.8618 | 0.8585 | 0.8585 | 0.8440 |

The cumulative sequence is not monotonic. In four datasets an intermediate
variant outperformed final DASP-v3. This is a major reviewer objection:
the final combination introduces mechanism interference, especially through
early recovery and insufficient-class checkpoint fallback.

The strict memory correction did solve a real implementation error and the
rare-class stable query removed the BoT-IoT gradual collapse. It also increased
BoT-IoT compute beyond budget. Thus the mechanism is effective but not yet
efficient.

## Remaining failure mechanisms

### 1. Warm-up-to-A collapse is still possible

The ToN-IoT sudden run collapsed on windows 4--6, before the intended A→B
transition. Attack recall was 1.0 while benign true negatives were 0, 2, and
3 out of 2,499 benign flows. Recovery began at window 7.

This is not evidence that DASP failed to adapt to the declared drift; it is
evidence of either warm-up→A discontinuity or an inadequate warm-up decision
boundary. A reviewer can correctly argue that the stream contains an
unreported transition at the evaluation boundary. Warm-up→A must be audited
with the same regime classifier, MMD, and cross-concept generalization checks
used for A→B.

### 2. Final fallback is too active

On stationary streams, final DASP-v3 still issued many strong recovery events:
three on UNSW, one on CSE, six on ToN, twelve on BoT, and ten on CIC. It also
restored checkpoints repeatedly when a queried batch lacked class support,
even when the current model itself had not collapsed.

Rollback should be an emergency action, not a general response to a sparse
holdout. The current fallback can replace a useful model with a historically
good but concept-mismatched checkpoint.

### 3. Detector evidence and predictive utility remain misaligned

The detector can trigger under ToN S0, while many large predictive declines
on CSE/CIC are detected only by delayed performance. Covariate distance alone
does not establish that backbone adaptation is useful. A detector-confirmed
shift must still compete against an adapter-only candidate on an independent
delayed guard.

### 4. Memory balance does not solve posterior calibration

Equal memory allocation changes the training prior. It helps minority
retention but can miscalibrate the decision threshold relative to the actual
stream prior. ToN remained biased toward Attack even after balancing. A
revealed-label-only temperature/threshold calibrator should be evaluated as a
separate ablation, with calibration error and class recalls reported.

### 5. Extreme-imbalance querying is computationally expensive

The rare-class query fixed BoT collapse but made total update/query time
2.754× v2. The stable rare-class path should use a cheap probability-tail
reserve and amortized prototype refresh instead of running the full diversity
pipeline every stable window. Its label budget must remain unchanged.

## Reviewer-style objections

A strict reviewer would likely raise the following:

1. **Final method not supported by its ablations.** The final combination is
   worse than the query-only or backbone/checkpoint variant on most datasets.
2. **Development-set reuse.** Mechanisms were revised after observing seed
   2001. Those runs are engineering evidence, not confirmatory evidence.
3. **Non-inferiority margin.** The 0.05 screening margin is permissive and
   needs a domain-based justification; the full-pilot 0.03 margin must remain
   fixed before new data are inspected.
4. **Boundary transition.** ToN collapses before the declared drift, so
   warm-up→A exchangeability is not established.
5. **False alarms omitted from the advancement gate.** S0 false alarms and
   unnecessary recovery events must have an explicit cap, not merely be
   reported.
6. **One seed is not evidence of stability.** Windows are correlated and may
   not be treated as independent replicates.
7. **Baseline advantage remains material.** SSF/ER outperform DASP-v3 on ToN
   and CIC, not merely by statistical noise visible in this one seed.
8. **Implementation status.** SSF, RepShield, and CDDA-MD reproduction scope
   still needs exact correspondence statements to the original papers before
   manuscript claims.

## Scientifically defensible next revision

Do not call the current implementation final DASP-v3. Freeze it as
`DASP-v3-dev`.

For a future DASP-v3.1:

1. audit and repair warm-up→A continuity before algorithm comparison;
2. make adapter-only and adapter+backbone candidates compete on a disjoint
   delayed guard under the same gradient budget;
3. allow checkpoint restore only when the current model has a
   class-supported lower-confidence-bound collapse, not merely an
   insufficient current query;
4. remove the one-window early-performance shortcut except for a
   class-collapse event;
5. add delayed temperature/threshold calibration as its own ablation;
6. replace full stable diversity querying with a cheap rare-class
   probability-tail reserve;
7. add fixed S0 caps for detector false alarms, strong updates, and rollback
   frequency;
8. validate the frozen revision on a new development seed before reopening
   the 1,200-run matrix.

No new rule should inspect a dataset name, current label, future label, or
confirmation seed.

## Statistical interpretation

This is an exploratory engineering pilot with one stream seed and one model
seed. The three scenarios are not guaranteed independent because they are
constructed from the same source pools. Therefore:

- no p-value, confidence interval, Friedman test, Wilcoxon-Holm test, or
  effect-size claim is valid at this stage;
- windows are not independent observations;
- numerical deltas are screening estimates only;
- the appropriate confidence label is **RED_FLAG for advancement**, not
  because the run is corrupted, but because the preregistered performance and
  collapse gates failed.

## Statistical fallacy scan — 11/11 checked

1. **Simpson's paradox:** no pooled superiority claim is made; dataset-level
   directions are reported because ablation effects are heterogeneous.
2. **Ecological fallacy:** no flow-level causal claim is inferred from
   dataset-level averages.
3. **Berkson's paradox:** selected controlled streams may not represent
   deployment traffic; this limits external validity.
4. **Collider bias:** no adjusted regression or conditioning claim is made.
5. **Base-rate neglect:** high BoT attack recall is interpreted with its
   extreme prevalence, Macro-F1, MCC, and benign collapse.
6. **Regression to the mean:** recovery after an extreme collapsed window is
   not treated alone as proof of mechanism efficacy.
7. **Survivorship bias:** all 120 planned cells and failed gates are retained;
   no failed run was excluded.
8. **Look-elsewhere effect:** all datasets, methods, and negative ablations are
   reported; no significance search was performed.
9. **Garden of forking paths:** present as a development risk because seed
   2001 informed revisions; confirmation remains locked.
10. **Correlation is not causation:** controlled paired streams support
    within-cell algorithm comparisons, but not broad deployment claims from
    one seed.
11. **Reverse causality:** not applicable to randomized algorithm assignment;
    temporal precedence is enforced by the delayed prequential protocol.

## Reproducibility artifacts

- preregistration and amendments: `DASP_V3_ADVANCEMENT_PREREG.md`
- final mock gate: `outputs_dasp_v3_mock/mock_gate.json`
- first failed pilot: `outputs_dasp_v3_small_pilot/small_pilot_gate.json`
- final repeated pilot: `outputs_dasp_v3_small_pilot_v3/small_pilot_gate.json`
- final run summaries:
  `outputs_dasp_v3_small_pilot_v3/summary_by_run.csv`
- dataset/method metrics:
  `outputs_dasp_v3_small_pilot_v3/metrics_by_dataset_method.csv`
- integrity and collapse audit:
  `outputs_dasp_v3_small_pilot_v3/run_integrity_and_collapse.csv`

