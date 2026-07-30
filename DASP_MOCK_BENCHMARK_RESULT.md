# DASP revised-mechanism mock benchmark

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-26
- Verification Status: ANALYZED
- Version Label: dasp_mock_result_v1
- Preregistration: `DASP_MOCK_ADVANCEMENT_PREREG.md`

## Experiment result

- ID: `dasp_mock_diagnostic`
- Type: controlled synthetic streaming benchmark
- Status: completed
- Command: `python tools/run_dasp_mock_benchmark.py`
- Working directory: `C:\Users\AnThiwn\Desktop\Q!`
- Duration: 116 seconds
- Exit code: 0
- Completed cells: 120/120
- Real IDS data used: no
- NF confirmation seeds used: no

The locked matrix contains four scenarios, three paired stream seeds, one model
seed, and ten methods. Every adaptive budget-matched method receives at most 64
queried labels per window with a one-window delay.

## Mechanism changes tested

1. Prototype-aware, label-blind acquisition spreads queries using labeled-memory
   feature centroids when classifier pseudo-labels collapse.
2. A current-concept holdout is created only after the 64 labels are acquired.
   Candidate selection therefore sees delayed evidence from the new concept
   while keeping acquisition label-blind.
3. Current samples receive bounded surprise and inverse-frequency weights.
4. Budget-matched baselines now query labels uniformly before reading labels;
   SSF-budget no longer receives oracle class-balanced access to the full
   window.
5. The backbone remains structurally frozen after warm-up.

This correction also invalidates the fairness interpretation of the earlier
24-run NF DASP--SSF screen. Those historical numbers remain an audit record but
must not be pooled with or substituted for a corrected real-data rerun.

## Primary first-drift results

| Method | Early Macro-F1 | Early Attack Recall | Mean rank |
|---|---:|---:|---:|
| DASP | 0.57433 | 0.42343 | 1.67 |
| SSF-budget | 0.51740 | 0.36279 | 1.89 |
| Experience Replay | 0.50389 | 0.32084 | 3.89 |
| A-GEM | 0.49466 | 0.30543 | 5.56 |
| Online MLP | 0.49460 | 0.30543 | 5.56 |
| Online EWC | 0.49460 | 0.30543 | 5.56 |
| Periodic FT | 0.48186 | 0.27675 | 6.89 |
| Frozen | 0.47909 | 0.27891 | 7.44 |
| RepShield | 0.47737 | 0.29031 | 8.11 |
| CDDA-MD | 0.27382 | 0.00000 | 8.44 |

DASP minus SSF-budget:

- Early Macro-F1: +0.05693; diagnostic paired-bootstrap 95% interval
  [+0.02561, +0.09766].
- Early Attack Recall: +0.06064; diagnostic paired-bootstrap 95% interval
  [+0.01333, +0.12711].
- Macro-F1 wins: 8/9 paired first-drift cells.

DASP beat every comparator in mean first-drift Macro-F1. Its full-stream mean
Macro-F1 was 0.88483, compared with 0.85939 for SSF-budget and 0.78991 for
Experience Replay.

## Recurrence diagnostic

S3 must not be summarized only by the first transition:

| Episode | DASP Macro-F1 | SSF Macro-F1 | DASP Attack Recall | SSF Attack Recall |
|---|---:|---:|---:|---:|
| First drift A-to-B | 0.69880 | 0.63181 | 0.55725 | 0.48658 |
| Recurrence B-to-A | 0.88495 | 0.92512 | 0.80524 | 0.88283 |

The revision solves the mock first-drift bottleneck but still underperforms SSF
on mean recurrence. This prevents treating the mock result as evidence that
all drift-adaptation mechanisms are resolved.

## Efficiency and invariants

- DASP median prediction-plus-decision latency: 11.77 ms per 512-flow window.
- SSF-budget median prediction-plus-decision latency: 0.60 ms.
- DASP mean prediction throughput: 42,479 flows/s.
- DASP mean peak process memory: 394.6 MiB; SSF-budget: 364.0 MiB.
- DASP whole-model updates: 0.
- Maximum DASP acquired labels per window: 64.
- S0 detector alarms: 0 across all three streams.
- Automated tests: 88/88 passed.

## Advancement decision

The locked advancement rule is satisfied:

- positive Macro-F1 delta versus SSF: pass;
- Attack Recall non-inferiority: pass;
- at least 6/9 first-drift wins: 8/9, pass;
- mean rank at most 3.0: 1.67, pass;
- zero global updates: pass;
- label ceiling: pass.

`advances = true` means the mechanism may proceed to a **real-data development
rerun**. It does not authorize opening confirmation seeds or making a
superiority claim. Recurrence should be repaired or separately gated before a
full confirmatory matrix.

## Statistical validation and fallacy scan

- Confidence: CAUTION, because only three synthetic seeds and one generator
  family were used.
- No null-hypothesis significance claim is made. Bootstrap intervals describe
  these nine paired synthetic cells only.
- Reproducibility verdict: CANNOT VERIFY by an independent rerun in this stage;
  seeds, configs, logs, and all per-window outputs are retained.
- Fallacy coverage: 11/11 checked.

| Fallacy | Status | Finding |
|---|---|---|
| Simpson's paradox | CAUTION | First drift favors DASP, but recurrence favors SSF on average. |
| Ecological fallacy | Clear | No inference from synthetic aggregate to real IDS flows is made. |
| Berkson's paradox | CAUTION | Generator-defined campaigns limit external validity. |
| Collider bias | Clear | No post-treatment covariate adjustment is used. |
| Base-rate neglect | Clear | Synthetic class prevalence is fixed at 70/15/15 and Attack Recall is reported with Macro-F1. |
| Regression to mean | Clear | Paired methods use identical streams; cells were not selected by extreme outcomes. |
| Survivorship bias | Clear | All 120/120 cells completed and were retained. |
| Look-elsewhere effect | Clear | Endpoints and advancement thresholds were locked before execution. |
| Garden of forking paths | NOTE | The latency-column analysis bug was corrected and disclosed; experiment outputs were not rerun. |
| Correlation implies causation | Clear | Claims are restricted to randomized paired behavior inside this generator. |
| Reverse causality | Not applicable | The benchmark has explicit stream order and intervention timing. |

## Artifacts

- `outputs_dasp_mock_benchmark/summary_by_run.csv`
- `outputs_dasp_mock_benchmark/early_episodes.csv`
- `outputs_dasp_mock_benchmark/method_summary.csv`
- `outputs_dasp_mock_benchmark/paired_comparisons.csv`
- `outputs_dasp_mock_benchmark/advancement_criteria.json`
- `outputs_dasp_mock_benchmark/mock_early_macro_f1.png`
- `dasp_mock_benchmark.log`
