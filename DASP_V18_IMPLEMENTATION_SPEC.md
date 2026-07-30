# DASP main-branch implementation specification

## Status

- Code status: upgraded; the public method key `dasp` resolves to
  `event-gated-fused-v1`
- Scientific status: 93 automated tests pass; isolated real-data
  accuracy--latency revalidation remains pending
- Benchmark status: 120/120 mock runs completed across ten methods; the older
  24-run NF screen is superseded by the label-acquisition fairness correction
- Parent: v17.5
- Whole-backbone update: forbidden after warm-up

## Event-gated/fused upgrade

The main method no longer trains a fast candidate in every delayed-label
window. A cheap input-space gate runs first. Stable windows acquire at most 64
label-blind labels for bounded memory/guard refresh and perform zero gradient
steps. Layer-representation MMD, active querying, replay, and candidate
training run only on warning, confirmed drift, recurrence, or sampled
performance decline.

After every accepted update, low-rank residual weights and feature modulation
are folded into a prediction-only cache. The trainable adapter model remains
unchanged and its backbone stays frozen. This separates deployment inference
cost from the auditable selective-learning state.

The upgraded path uses:

- stable label budget: 64;
- fast/strong replay: 128/256;
- fast/strong epochs: 1/2;
- historical guard capacity: 128;
- soft-commit candidates: `1.0` and `0.5`;
- full label budget on adaptation events: at most 256.

## Changes from v17.5

### 1. Feature-wise residual modulation

v18 retains the low-rank adapters and raw-input residual head, and adds a
zero-initialized bounded affine modulator after each hidden layer:

`h' = h * (1 + s * tanh(gain)) + s * tanh(bias)`.

Only `gain` and `bias` are trainable. Base linear layers, LayerNorm parameters,
and the original backbone remain frozen. This gives the selective path direct
control over post-normalization representations without a global update.

### 2. Drift-robust IWMS

The replay selector measures Jensen--Shannon divergence between:

- pseudo-class mass in the current unlabeled query; and
- labeled class mass in memory.

Low prediction confidence or a large prior mismatch increases the effective
uniform mixture from 0.25 up to 0.75. This prevents the first-drift replay quota
from collapsing to the pseudo-class distribution of an already failing model.

Logged fields include effective mix, JS divergence, mean confidence, predicted
class counts, and selected replay counts.

### 3. Label-blind active query selection

v18 fixes query indices before reading current labels. It ranks the full
delayed window using:

- normalized predictive entropy;
- feature novelty against labeled-memory class centroids;
- diversity-aware selection within a high-value candidate pool; and
- a pseudo-class-spread remainder quota.

Only the selected 256 indices have their labels dereferenced for the update.
The selection function receives no current `y` tensor.

### 4. Dual prequential guard

The label-blind query is split only after its labels arrive. By default, 25%
becomes a deterministic class-spread current-concept holdout and the remainder
trains the candidate. A bounded class-balanced history from earlier windows is
the secondary forgetting check. Current queried labels enter that history only
after the commit decision, so the candidate is never evaluated on samples used
to train that update.

### 5. Memory refresh

The refresh pool contains only the 256 model-selected newly labeled samples.
Historical guard samples are not reinserted as new observations. The existing
layer-conditioned old-sample retention remains unchanged.

### 6. Current-concept delayed guard

After label-blind acquisition, the queried labels are split into a deterministic
class-spread training set and current-concept holdout. Candidate selection now
uses delayed evidence from the affected concept instead of only a cumulative
historical guard. The historical guard remains a secondary forgetting check.

### 7. Prototype-aware acquisition and class weighting

Labeled-memory feature centroids provide a second pseudo-class view when the
classifier collapses after drift. Disagreement with the classifier enters the
query score, and the non-global query quota is spread by prototype class.
Training uses bounded surprise and inverse-frequency weights.

### 8. Fair budget-matched baselines

Budget-matched baselines query update labels uniformly before labels are read.
This removes the prior oracle class-balanced selection from the entire current
window. DASP retains its declared label-blind active query as part of the
proposed method.

## Invariants

1. Test-then-train order and configured label delay are unchanged.
2. `ground_truth_update_labels <= update_label_budget`.
3. Current labels are not read during query selection.
4. No base-block parameter has `requires_grad=True` after warm-up.
5. `global_update_count` must remain zero.
6. `dasp_v18` preserves the pre-upgrade technical path; the main `dasp` key
   resolves to the event-gated/fused implementation.
7. Fused weights are an inference cache and are never optimized directly.

## Configuration

Development configuration: `configs/dasp_v18_controlled.yaml`. Its method key
is `dasp`; “v18” remains only an internal implementation/configuration label.

Important new keys:

- `v18_modulation_scale`
- `v18_fast_modulation_plasticity`
- `v18_strong_modulation_plasticity`
- `v18_replay_max_uniform_mix`
- `v18_replay_js_threshold`
- `v18_guard_history_size`
- `v18_query_entropy_weight`
- `v18_query_novelty_weight`
- `v18_query_disagreement_weight`
- `v18_query_global_fraction`
- `v18_query_candidate_multiplier`
- `v18_current_guard_fraction`
- `v18_current_guard_min_per_class`
- `v18_class_weight_multiplier`
- `fast_revision`
- `fast_warning_severity`
- `fast_input_gate_samples`
- `fast_stable_label_budget`
- `fast_guard_history_size`
- `fast_fast_replay_size`
- `fast_strong_replay_size`
- `fast_fast_epochs`
- `fast_strong_epochs`
- `fast_commit_alphas`

## Verification performed

- All 93 automated tests passed, including fused-logit equivalence,
  stationary zero-gradient gating, warning-path adaptation, modulation, robust IWMS,
  label-blind querying, cumulative guard balance, the label budget, and the
  no-global-update invariant.
- A locked development screen completed 24/24 runs before the fairness fix:
  four NF datasets, S1/S3/S4, one stream seed, one model seed, and paired
  `dasp`/`ssf_budget` methods.
- DASP preserved `global_update_count = 0` and the 256-label ceiling.
- It did not advance numerically, but the DASP--SSF effect is now superseded
  because SSF's queried labels were selected with unavailable class labels.
- Confirmation seeds remain sealed. These screening estimates are diagnostic
  and must not be presented as confirmatory evidence.

The revised mechanism subsequently passed a preregistered synthetic gate:
120/120 runs over four scenarios, three stream seeds, and ten methods. DASP
beat SSF-budget by +0.05693 early Macro-F1 and +0.06064 early Attack Recall,
with 8/9 paired first-drift wins and mean rank 1.67. It still trailed SSF on
mean S3 recurrence, so the next authorized stage is a real-data development
rerun rather than confirmation.

Full result record: `DASP_MAIN_SCREEN_RESULT.md`.
Revised mock result: `DASP_MOCK_BENCHMARK_RESULT.md`.
