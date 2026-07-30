## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: dasp_v4_mock_result_v1

## Experiment Result

- **ID**: DASP-V4-MOCK-2026-07-30
- **Type**: synthetic continual-learning benchmark
- **Status**: completed
- **Primary command**: `.venv\Scripts\python.exe tools\run_dasp_v4_mock.py`
- **Working directory**: `C:\Users\AnThiwn\Desktop\Q!`
- **Matrix**: 4 scenarios × 2 new stream seeds × 6 methods = 48 runs
- **Scenarios**: S0 stationary, S1 sudden, S3 recurrent, S5 high-severity
- **Label protocol**: delayed prequential, one-window label delay
- **Real datasets used**: no
- **Confirmation seeds used**: no

### Implemented ablations

1. `dasp_v4_replay`: true-label class-quota memory with boundary,
   representation-diversity and recency strata; replay is trained in one pass
   without the redundant projection gradient.
2. `dasp_v4_prior`: `dasp_v4_replay` plus bounded logit adjustment estimated
   only from labels already revealed.
3. `dasp_v4`: `dasp_v4_prior` plus selectively unfrozen recent backbone
   layers, accepted through the existing delayed guard.

### Integrity

- Matrix complete: 48/48.
- Integrity violations: 0.
- DASP-v4 collapse: 0.
- Label budget violations: 0.
- Future-label use detected: 0.
- Global-update invariant violations: 0.
- Safety/core tests: 107 passed.

## Aggregate Results

| Method | Macro-F1 | PR-AUC | MCC | Mean update time (s) | Mean gradient evaluations | Max collapse streak |
|---|---:|---:|---:|---:|---:|---:|
| SSF-budget | 0.8111 | 0.8255 | 0.6667 | 2.189 | 96.00 | 1 |
| Experience Replay | 0.8029 | 0.8058 | 0.6461 | 0.202 | 36.00 | 0 |
| **DASP-v4 replay** | **0.7861** | **0.7828** | **0.5992** | **1.850** | **31.25** | **0** |
| DASP-v4 prior | 0.7841 | 0.7806 | 0.5880 | 2.029 | 33.00 | 0 |
| DASP-v3 query | 0.7823 | 0.7781 | 0.5920 | 1.969 | 64.50 | 0 |
| DASP-v4 full | 0.7786 | 0.7729 | 0.5773 | 2.021 | 32.75 | 0 |

### Paired DASP-v4 replay versus DASP-v3 query

Across the eight matched `scenario × seed` cells:

| Metric | Mean delta | Median delta | Worst delta | Best delta |
|---|---:|---:|---:|---:|
| Macro-F1 | +0.00382 | +0.00359 | −0.00849 | +0.01527 |
| PR-AUC | +0.00473 | +0.00305 | −0.00449 | +0.01601 |
| MCC | +0.00719 | +0.00593 | −0.01590 | +0.03177 |

Mean Macro-F1 improved in every scenario after averaging the two held-out
mock seeds:

| Scenario | DASP-v3 query | DASP-v4 replay | Delta |
|---|---:|---:|---:|
| S0 | 0.879819 | 0.881974 | +0.002155 |
| S1 | 0.792172 | 0.799554 | +0.007382 |
| S3 | 0.832155 | 0.834495 | +0.002340 |
| S5 | 0.624988 | 0.628374 | +0.003387 |

`dasp_v4_replay` used 0.940× the update time of `dasp_v3_query` and about
0.484× its gradient evaluations. It remained about 9.16× slower than
Experience Replay, so the computational-performance problem is not solved.

## Mechanistic Interpretation

The minimal replay revision produced small, directionally consistent gains in
Macro-F1, PR-AUC and MCC while reducing update work. This supports retaining
informative class-balanced replay and removing the second projection gradient
as the next development core.

The full combination did not improve:

- `dasp_v4_prior` improved F1 slightly over v3-query but reduced MCC.
- selective backbone adaptation added on top reduced mean F1, PR-AUC and MCC.
- the complete `dasp_v4` was worse than `dasp_v3_query` by 0.00372 Macro-F1,
  0.00521 PR-AUC and 0.01466 MCC.

Therefore, this mock supports the module-conflict diagnosis. It does not
support promoting the full DASP-v4 configuration.

## Validation Report

- **Overall confidence**: CAUTION
- **Reproducibility**: REPRODUCIBLE for the repeated
  `S5 × seed 9501 × dasp_v4_replay` cell. Macro-F1 and PR-AUC matched exactly;
  MCC differed only by `5.55e-17` from CSV floating-point serialization;
  gradient evaluations matched exactly.
- **Inference limit**: two seeds and synthetic data are insufficient for
  significance testing or real-dataset generalization.
- **Multiple comparisons**: no p-values or confirmatory hypothesis tests are
  claimed. All ablations, including negative results, are retained.

### Fallacy scan

- Coverage: 11/11 checked.
- Simpson's paradox: scenario directions were checked; no aggregate reversal
  for the selected replay core.
- Ecological fallacy: no individual-flow causal inference is made from
  scenario means.
- Berkson/collider bias: no post-outcome sample filtering or controlled
  collider was introduced.
- Base-rate neglect: Macro-F1 is accompanied by PR-AUC and MCC; the synthetic
  class prior is not treated as evidence for extreme real NIDS imbalance.
- Regression to the mean: paired baselines use identical scenario/seed cells.
- Survivorship bias: all 48 runs, failures and negative ablations are included.
- Look-elsewhere effect: the selected replay core is labelled exploratory; no
  significance claim is made.
- Garden of forking paths: configuration was held fixed during the 48-run
  matrix; results were not used to retune midway.
- Correlation/causation and reverse causality: ablation differences support a
  mechanism hypothesis only; they do not prove real-dataset causality.

## Gate Decision

- Mock pipeline integrity: **PASS**.
- DASP-v4 replay-core exploratory improvement: **PASS**.
- Full DASP-v4 improvement: **FAIL**.
- Real-data pilot: **LOCKED**.
- 1,200-run pilot and 4,000-run confirmation: **LOCKED**.

The next candidate is `dasp_v4_replay`, not `dasp_v4`. Before any full pilot,
it needs a small real-data development check on ToN-IoT and CIC-IDS2017 and an
additional efficiency pass aimed at reducing the remaining gap to Experience
Replay.
