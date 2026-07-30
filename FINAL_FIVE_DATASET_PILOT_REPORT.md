# DASP five-dataset pilot: final engineering report

Date: 2026-07-30  
Status: **pilot completed; confirmation blocked**

## Decision

The cross-dataset pilot is technically complete, but it does not pass the
locked advancement gate. It must not be reported as a successful
cross-dataset confirmation of DASP.

- 400/400 prepared stream artifacts passed the technical audit.
- 1,200/1,200 pilot runs completed:
  5 datasets x 10 scenarios x 3 data seeds x 8 methods.
- There were no missing window files, duplicate run cells, non-finite primary
  outcomes, or DASP global-backbone update violations.
- The final software suite passed 109 tests; one test was skipped.
- SSF-budget used at most 256 ground-truth insertions per window and no
  pseudo-label insertions.
- The construct gate failed for all NF-UNSW-NB15-v3 scenarios and for several
  dataset-scenario cells elsewhere.
- DASP failed the predictive pilot gate and therefore the planned 4,000-run
  confirmation was not executed.

This is a protocol stop, not an omitted positive result.

## Data audit

| Dataset | Flows | Numeric features | Benign | Attack | Notes |
|---|---:|---:|---:|---:|---|
| NF-UNSW-NB15-v3 | 2,365,424 | 49 | 2,237,731 | 127,693 | Missing and Inf values present |
| NF-CSE-CIC-IDS2018-v3 | 20,115,529 | 49 | 17,514,626 | 2,600,903 | One constant feature |
| NF-ToN-IoT-v3 | 27,520,260 | 49 | 16,792,214 | 10,728,046 | Repeated directional sessions |
| NF-BoT-IoT-v3 | 16,933,808 | 49 | 51,989 | 16,881,819 | Extreme 99.69% Attack prevalence |
| CIC-IDS2017 | 2,830,743 | 78 | 2,273,097 | 557,646 | Separate schema; no usable session or timestamp field |

The full audit, including subtype distributions, missing/Inf values,
duplicates, constants, leakage checks, and partition limitations, is in
`artifacts/final_five_dataset/AUDIT_REPORT.md`.

## Stream design actually executed

Each stream has 36 windows of 4,096 flows:

- W0-W3: warm-up;
- W4-W11: Concept A;
- W12-W23: transition or Concept B;
- W24-W35: final concept or A-prime.

The ten scenarios are:

1. S0 Stationary;
2. S1 Sudden;
3. S2 Gradual;
4. S3 Incremental;
5. S4 Recurrent;
6. S5 Blip;
7. S6 Combined;
8. S7 Local-Low, 5% of Attack changed;
9. S7 Local-Medium, 10% of Attack changed;
10. S7 Local-High, 20% of Attack changed.

Local magnitude is reported both as a fraction of Attack and a fraction of
the complete stream. S5 is not silently merged into another scenario.

NF-ToN-IoT Concept A was locked at 71% DDoS and 29% DoS because 30% DoS was
not feasible for eight source/session-disjoint streams after removing
multi-family directional 5-tuples. NF-BoT-IoT uses a directional
5-tuple-by-30-minute proxy. CIC-IDS2017 uses source-row disjointness and a
local duplicate proxy (exact-feature hash by 10,000-row source-order block);
true session disjointness cannot be verified for CIC-IDS2017.

## Construct validity

The gate used two independent probes, Logistic Regression and Extra Trees.
It checked A-to-B macro-F1 and recall drops, attack and benign regime AUC,
prediction disagreement, local-subgroup recall, and whether A-prime is closer
to A than B.

The global pilot construct decision was `false`.

- NF-UNSW-NB15-v3: all non-stationary scenarios failed.
- NF-CSE-CIC-IDS2018-v3: S1-S5 and all three local levels passed; Combined
  failed.
- NF-ToN-IoT-v3: S1-S5, Local-Medium, and Local-High passed; Combined and
  Local-Low failed.
- NF-BoT-IoT-v3: S1-S6 passed; all three local levels failed.
- CIC-IDS2017: S1-S5 and all three local levels passed; Combined failed.
- The A-prime recurrence relation passed for Recurrent and Combined on all
  datasets.

A passed construct is still a boundary-impacting proxy, not direct proof
that the data-generating mechanism changed P(Y|X). Failed cells may only be
described as subtype/composition or covariate-shift stress tests.

## Pilot accuracy and performance

Mean values below average 30 pilot runs per method (5 datasets x 10
scenarios x 3 streams gives 150 runs per method in the pooled row).

| Method | Macro-F1 | Attack recall | Interpretation |
|---|---:|---:|---|
| SSF-budget | 0.8946 | 0.9335 | Best pooled Macro-F1 |
| Experience Replay | 0.8943 | 0.9300 | Nearly tied with SSF; much cheaper updates |
| RepShield | 0.8797 | 0.9239 | Strong but highest update cost |
| A-GEM | 0.8672 | 0.9162 | Middle group |
| Online MLP | 0.8617 | 0.9113 | Strong simple baseline |
| Online EWC | 0.8616 | 0.9113 | No pooled gain over Online MLP |
| CDDA-MD | 0.7636 | 0.8877 | Several severe failures |
| DASP | 0.7613 | 0.9020 | Failed pilot stability gate |

For the prespecified early post-drift Macro-F1 comparison, DASP minus
SSF-budget was -0.1404 with a 95% cluster-bootstrap interval
[-0.3724, -0.0040], win rate 0.2667, and exact cluster sign-flip
`p = 0.00696`. Early Attack Recall differed by -0.0237 with interval
[-0.1157, 0.0733] and `p = 0.4187`.

These are pilot/development statistics over 15 independent
dataset-by-stream clusters. They are not confirmation claims.

### Dataset-level DASP results and cost

| Dataset | Macro-F1 | Recall | PR-AUC | MCC | FNR | p95 decision ms/window | p95 update ms/window | Mean labels/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NF-UNSW-NB15-v3 | 0.9890 | 0.9867 | 0.9947 | 0.9796 | 0.0133 | 45.45 | 375.94 | 2,937.6 |
| NF-CSE-CIC-IDS2018-v3 | 0.9334 | 0.8340 | 0.8930 | 0.8841 | 0.1660 | 27.72 | 454.06 | 4,390.4 |
| NF-ToN-IoT-v3 | 0.3196 | 0.9959 | 0.8027 | 0.0710 | 0.0041 | 48.43 | 458.68 | 4,384.0 |
| NF-BoT-IoT-v3 | 0.6955 | 0.9811 | 0.9999 | 0.4217 | 0.0189 | 31.09 | 296.01 | 3,315.2 |
| CIC-IDS2017 | 0.8688 | 0.7124 | 0.8642 | 0.7675 | 0.2876 | 20.85 | 561.52 | 5,324.8 |

The very high Attack Recall but near-zero MCC on NF-ToN-IoT shows why recall
alone is insufficient: DASP often predicted nearly every flow as Attack.
NF-BoT-IoT likewise produced many one-class Macro-F1 values near 0.499 under
its true extreme prevalence.

## Failure diagnosis

The most severe example was
`NF-ToN-IoT/S3-Incremental/stream 2003/model 11`:

- DASP Macro-F1: 0.2903;
- balanced accuracy: 0.5030;
- MCC: 0.0127;
- cumulative Benign recall: 0.0096;
- Attack recall: 0.9963.

W1-W3 warm-up predictions reached F1=1.0, but W4 caused a large unintended
warm-up-to-A benign covariate shift. DASP predicted all 4,096 W4 flows as
Attack. The label-blind query then selected 256 samples whose revealed labels
were all Benign. The current-label guard consequently contained one class and
could not identify a safe balanced update. Experience Replay and SSF recovered
because they update every window; DASP mostly retained the failed frozen
model.

This exposes two separate defects:

1. the stream gate checked A-to-B but not warm-up-to-A continuity;
2. DASP lacks a safe two-class delayed guard/query fallback when a requested
   label batch contains only one class.

## Required corrections before confirmation

1. Add a locked warm-up-to-A continuity gate for both Benign and Attack:
   regime AUC, cross-segment generalization, prediction disagreement, and
   per-class performance must remain within preregistered tolerances.
2. Match whole source/session groups across warm-up, A, B, and final segments
   using label-blind feature/time strata. If this cannot be achieved without
   session leakage, declare that dataset/scenario unsuitable.
3. Require the delayed guard to contain both classes. If the current queried
   labels are one-class, augment only from already labeled, class-balanced
   memory and log the fallback; never evaluate macro-F1 on a one-class guard.
4. Add a safe bounded adaptation fallback after persistent one-class
   predictions, while preserving the common 256-label/update budget.
5. Add an executable W3-to-W4 parity/continuity test and a regression test for
   one-class delayed query batches.
6. Re-run only a development pilot after these changes. Freeze the revised
   algorithm and stream constructor before touching confirmation seeds
   2101-2105.
7. Run confirmation only for construct-eligible cells and report failed
   constructs as negative results or supplementary stress tests.

## Reproducibility artifacts

- Final protocol: `DASP_FINAL_FIVE_DATASET_PROTOCOL.md`
- Audit: `artifacts/final_five_dataset/AUDIT_REPORT.md`
- Technical validation:
  `artifacts/final_five_dataset/technical_validation.json`
- Construct validation:
  `artifacts/final_five_dataset/construct_validation.json`
- Combined run summary:
  `outputs_dasp_final_pilot_combined/summary_by_run.csv`
- Integrity audit:
  `outputs_dasp_final_pilot_combined/analysis/integrity_audit.json`
- Paired statistics:
  `outputs_dasp_final_pilot_combined/analysis/primary_statistics.json`
- Per-method performance:
  `outputs_dasp_final_pilot_combined/analysis/performance_by_method.csv`
- Episode metrics:
  `outputs_dasp_final_pilot_combined/analysis/episode_metrics.csv`
- Window-level metrics:
  `outputs_dasp_final_pilot_combined/analysis/all_windows.csv`
- Figures:
  `outputs_dasp_final_pilot_combined/analysis/`

## Manuscript status

The existing single-dataset manuscript was not overwritten with these pilot
numbers. Doing so would falsely present a failed development pilot as a
cross-dataset confirmation. Before submission, the manuscript must either:

- remain explicitly a single-dataset NF-UNSW distribution-shift study with no
  cross-dataset claim; or
- be revised only after the two defects above are fixed and an untouched
  confirmation passes.
