# DASP final five-dataset controlled-stream protocol

**Status:** frozen before the final pilot  
**Task:** binary Benign (0) versus Attack (1)  
**Datasets:** NF-UNSW-NB15-v3, NF-CSE-CIC-IDS2018-v3,
NF-ToN-IoT-v3, NF-BoT-IoT-v3, and CIC-IDS2017

## 1. Scope of the claim

The temporal form of a stream and the statistical kind of distribution change
are reported separately. Sudden, gradual, incremental, recurrent, blip,
combined, and local describe schedules. They do not by themselves prove a
change in \(P(Y\mid X)\).

Each transition is classified after a locked construct audit as:

1. boundary-impacting controlled drift;
2. attack-family/covariate-composition shift; or
3. invalid/no detectable transition.

The manuscript must not rename category 2 as naturally occurring real concept
drift. A negative construct result is retained and reported.

## 2. Fixed methods

Exactly eight methods are used in the primary comparison:

1. Online MLP (`online_mlp`);
2. Experience Replay (`experience_replay`);
3. Online EWC (`online_ewc`);
4. A-GEM (`agem`);
5. SSF budget-matched (`ssf_budget`);
6. RepShield (`repshield`);
7. CDDA-MD (`cdda_md`);
8. DASP (`dasp`).

All methods share the same stream artifact, preprocessing, MLP initialization,
label delay, optimizer family, label budget, replay/update budget, and epoch
budget where the method permits. SSF is primary only in budget-matched mode.
RepShield and CDDA-MD are adjusted shared-backbone implementations and must not
be described as exact paper-faithful reproductions.

## 3. Controlled stream

Each stream contains 36 windows of 4,096 flows:

- W0-W3: warm-up;
- W4-W11: stable Concept A;
- W12: first drift onset;
- W12-W23: transition or Concept B;
- W24-W35: final concept or new-flow recurrence A-prime.

The primary prevalence is the audited prevalence of each dataset. No clipping
to 5% or 95% is applied. This is important for the highly imbalanced
NF-BoT-IoT-v3 dataset. Every window has the same rounded prevalence within a
dataset. Sampling is without replacement.

### Scenario identifiers

| ID | Name | Locked schedule |
|---|---|---|
| S0 | Stationary | A for W0-W35 |
| S1 | Sudden | A to B at W12 |
| S2 | Gradual | increasingly frequent pure-B windows during W12-W23 |
| S3 | Incremental | monotone A/B intermediate mixtures during W12-W23 |
| S4 | Recurrent | A to B at W12; B to new-flow A-prime at W24 |
| S5 | Blip | B only at W12-W13, followed by new-flow A-prime |
| S6 | Combined | sudden A-to-B, incremental B-to-C, then recurrence toward A-prime |
| S7-Low | Local 5% | 5% of Attack changes local subtype |
| S7-Medium | Local 10% | 10% of Attack changes local subtype |
| S7-High | Local 20% | 20% of Attack changes local subtype |

S5 is scored as a short-lived robustness event, not automatically as persistent
concept drift. S6 is a study-defined composite stress scenario, not a claimed
universal drift category. There is no separate S8 in the primary average.

For local drift the manifest and report contain both:

- percentage of Attack flows affected; and
- percentage of the complete stream window affected.

Subgroup Attack Recall/FNR is primary for S7; whole-window Macro-F1 is
secondary because the global affected fraction can be small.

## 4. Data isolation and preprocessing

Attack subtype is used only to construct and audit streams. It is never supplied
to a model. Imputation medians and `StandardScaler` parameters are fit only on
W0-W3.

Within each stream:

- source rows do not repeat;
- warm-up, A, B/transition, C, and A-prime do not share the available
  session/duplicate-group proxy;
- A-prime uses new flows;
- benign allocation follows the same procedure in every regime.

Pilot streams (2001-2003) and confirmation streams (2101-2105) are generated
in one joint allocation, so source rows and grouping proxies are disjoint
across all eight data seeds.

The four NF-v3 datasets use a directional five-tuple as a declared session
proxy. CIC-IDS2017 has no IP, flow ID, timestamp, or session annotation in the
supplied schema. It therefore uses a full-feature hash only to keep duplicate
groups together. This is not represented as ground-truth session disjointness.
CIC file/day and attack-family confounding is reported as a limitation.

## 5. Construct audit

Before method comparison, each endpoint must pass:

- constant \(P(Y)\) within rounding tolerance;
- attack-only regime classifier AUC;
- A-trained cross-concept generalization;
- prediction disagreement;
- Macro-F1 and Attack Recall performance drop;
- benign-only regime classifier near chance where applicable;
- A-prime closer to A than to B for recurrence.

Performance-drop quantities are:

\[
\Delta_{\mathrm{early}} =
F1_{\mathrm{reference}}-\operatorname{mean}(F1_{t_d:t_d+4}),
\]

\[
\Delta_{\mathrm{max}} =
F1_{\mathrm{reference}}-\min_{t\in episode} F1_t,
\]

and onset drop at \(t_d\). The design threshold is not changed after results
are observed.

## 6. Delayed-prequential order

Labels are delayed by one window:

1. predict Wt with the state available at its arrival;
2. make unsupervised detector decisions using only available information;
3. receive labels for Wt-1;
4. update using only labeled data and the locked budget;
5. activate the updated model from Wt+1.

No current-window label may affect its own prediction, alarm, sample query, or
candidate selection.

## 7. Metrics

Predictive:

- Macro-F1, Attack Recall, PR-AUC, MCC, FNR;
- Early/Post-drift Macro-F1@5 and Attack Recall@5;
- onset, early-mean, and maximum performance drop;
- minimum post-drift Macro-F1;
- recovery time to 90% of the W7-W11 reference for two consecutive windows;
- recovery deficit and forgetting/recurrence gain;
- subtype recall and minimum subtype recall.

Detection:

- event-matched detection rate, precision, delay, missed event;
- false alarms on S0;
- one alarm can match at most one event.

Operational performance:

- preprocessing, forward inference, drift scoring, decision, update, and
  end-to-end latency;
- median/p95/p99 latency, flows/s throughput;
- ground-truth labels, pseudo-labels, sample updates, gradient evaluations;
- peak RSS, CUDA allocated/reserved memory, buffer size, and model size.

Raw performance is reported without a “real-time” claim unless an independent
arrival-rate requirement is supplied.

## 8. Run stages and statistics

Pilot:

- three independent data seeds (2001-2003);
- one model seed (11);
- all ten conditions and all eight methods.

Confirmation, after a valid pilot:

- five independent data seeds (2101-2105);
- two model seeds nested per data stream (101-102);
- all ten conditions and all eight methods.

Windows are not treated as independent replicates. Analysis is paired by
dataset, scenario, data seed, and model seed. Report 95% confidence intervals,
paired effect sizes, hierarchical bootstrap, cluster sign-flip sensitivity,
Friedman sensitivity, and Wilcoxon-Holm comparisons. Failed runs, construct
failures, and negative effects remain in the report.

