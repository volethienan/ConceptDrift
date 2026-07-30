## Material Passport

- Origin Skill: academic-research-suite
- Origin Modes: methodology-focus + experiment-agent
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: dasp_v3_development_v3

## Methodology assessment

The DASP-v3 proposal is directionally strong because every requested mechanism
maps to an observed DASP-v2 failure. It is not scientifically complete unless
the following constraints are explicit:

1. “True-label-aware” means labels already revealed under the one-window delay;
   current or future labels remain inaccessible to prediction, detection, and
   query selection.
2. Adaptive query allocation is based on past labeled prevalence, current
   unlabeled features, probabilities, and prototypes only. It may not inspect
   current true labels.
3. A “good checkpoint” is selected only from checkpoints whose performance has
   already been observed after label arrival. The retained horizon and scoring
   rule are fixed globally.
4. Memory quota, query budget, number of trainable backbone layers, checkpoint
   horizon, and fallback policy are identical on all datasets.
5. The full 1,200-run count fixes exactly eight methods:

   - `dasp_safe_v2`;
   - `dasp_v3_memory`;
   - `dasp_v3_query`;
   - `dasp_v3_backbone`;
   - `dasp_v3_checkpoint`;
   - `dasp_v3`;
   - `ssf_budget`;
   - `experience_replay`.

The five DASP-v3 variants are cumulative. The complete `dasp_v3` adds the
insufficient-class fallback after the checkpoint ablation.

## Fixed mechanisms

### True-label memory

- Update only after labels arrive.
- Class-balanced retention without replacement.
- Fixed minimum quota per represented class.
- A model rollback does not erase newly revealed labeled evidence, but the
  evidence cannot enter the guard used for the same candidate decision.

### Adaptive query

- Total new-label budget remains 256 per window.
- Allocation combines inverse historical class prevalence, soft output-class
  tails, raw-feature k-center diversity, uncertainty, novelty, and prototype
  coverage.
- No dataset names or dataset-specific thresholds are available to the policy.

### Backbone recovery

- At most two selected backbone layers.
- Reduced learning rate/plasticity relative to adapters.
- A bounded log-prior correction may adjust only the classifier bias from the
  revealed training split; the separate delayed guard still decides whether
  the candidate is committed.
- Activated only by a strong causal event and evaluated by delayed guards.
- Global backbone update remains forbidden.

### Causal checkpoint rollback

- Keep the five most recent class-supported prediction checkpoints.
- Score: `0.5 × Macro-F1 + 0.5 × min(Benign recall, Attack recall)`.
- A checkpoint becomes eligible only after labels for its prediction window
  arrive.
- Candidate rejection may restore the highest-scoring eligible checkpoint if
  it is non-inferior on already revealed guards.

### Insufficient-class fallback

- No gradient update from a one-class batch.
- Accumulate revealed labels within the same 256-sample adaptation buffer.
- Preserve class-aware memory.
- If the current holdout lacks the fixed class support but the delayed batch
  itself is trainable, train on that revealed batch and gate only against an
  older disjoint class-supported guard; record this fallback explicitly.
- After one causally scored post-warm-up window, permit an immediate recovery
  event when delayed-label performance falls past the same global v2
  threshold. This prevents collapsed windows from poisoning the reference
  history; it does not alter the threshold by dataset.
- If support remains insufficient, deploy the best eligible recent checkpoint;
  otherwise retain the current model. No pseudo-label training is allowed.

## Sequential gates

### Mock gate

- The mock model receives only `Benign=0` and `Attack=1`; synthetic attack
  families remain hidden stream-construction metadata.
- Unit and adversarial tests pass.
- No current-label influence on current prediction/query.
- Exact checkpoint restore and fused/unfused parity.
- No one-class gradient update.
- No budget or finite-metric violation.

### Small pilot gate

Matrix: five datasets × `S0`, `S1-Sudden`, `S2-Gradual` × stream seed 2001 ×
eight methods = 120 runs.

Required:

- zero DASP-v3 three-window collapse;
- no integrity violation;
- DASP-v3 mean Macro-F1 no more than 0.05 below the stronger SSF/ER comparator
  on every dataset;
- total update time per complete stream no more than 2× DASP-v2;
- peak RSS and CUDA memory no more than 1.25× DASP-v2.

Failure stops the full pilot.

### Development amendment after mock-v1

The first synthetic mock used the median of per-window update times and failed
the cost gate (`3.19×`) despite an aggregate total-update-time ratio near
`1.38×`. Update latency is a zero-inflated, bimodal quantity because most
stationary windows do not adapt; a ratio of medians can therefore divide an
adaptation time by a near-zero bookkeeping time and does not measure the
stream-level compute budget. Before any real-data pilot, the fixed cost gate
was amended to total update time over the complete, identical stream matrix.
Median and tail latency remain reported diagnostics. The original mock-v1
failure is retained in the report and is not reclassified silently.

### Development amendment after small-pilot-v1

The first 120-run real-data pilot failed on ToN-IoT and CIC-IDS2017 and
contained one BoT-IoT gradual collapse. Investigation identified three
implementation-level mismatches with the stated v3 design:

1. the memory enforced only a class floor and filled remaining capacity at the
   empirical prior instead of using an equal class allocation when possible;
2. stable windows bypassed adaptive query and therefore failed to acquire the
   rare class under extreme imbalance;
3. backbone recovery and checkpoint restoration were allowed without,
   respectively, strong covariate evidence and evidence that the current model
   had collapsed.

Before any additional real-data run, these rules were corrected globally:
equal allocation with deterministic deficit redistribution, adaptive
label-blind querying on stable windows, backbone/calibration only under a
detector-confirmed covariate event, and checkpoint rollback only when the
revealed guard shows minimum class recall at or below `0.10`. No dataset name
or dataset-specific threshold is available to these mechanisms. Mock-v3 must
pass again before the real-data pilot is repeated; small-pilot-v1 remains a
failed result.

### Full pilot gate

Matrix: five datasets × ten scenarios × three stream seeds × one model seed ×
eight methods = 1,200 runs.

Required:

- at most one collapse run out of 150 DASP-v3 runs;
- all five datasets satisfy the preregistered -0.03 Macro-F1 non-inferiority
  margin against the stronger SSF/ER baseline;
- paired improvement over DASP-v2 is directionally consistent in at least four
  of five datasets and not materially negative in the fifth;
- cost limits from the small pilot continue to hold;
- inference uses no future labels and every run cell is complete and unique.

## Confirmation lock

The 4,000-run confirmation is not part of this task. Even a passing full pilot
does not execute confirmation automatically; it only produces an eligible
recommendation for separate user authorization.
