## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-30
- Verification Status: VERIFIED
- Version Label: dasp_safety_pilot_v2

## Decision

The mock gate passed, but the five-dataset development pilot failed. The
4,000-run confirmation matrix remains locked and was not executed.

- Synthetic mock v2: 96/96 runs complete, 0 integrity violations.
- Real-data development pilot: 1,200/1,200 unique cells complete.
- Full test suite: 115 passed, 1 CUDA-only test skipped on the CPU test
  environment.
- Pilot pass: **no**.
- Dataset non-inferiority pass: 1/5 (`NF-UNSW-NB15-v3` only).
- DASP-v2 collapse runs: 17/150.

## Implemented revisions

The unchanged `dasp` method remains the current/reference implementation.
Cumulative development variants were added:

1. `dasp_safe_query`: label-blind soft-class query. It reserves samples with
   high relative probability for every output class even when argmax
   predictions collapse.
2. `dasp_safe_replay`: class-aware replay using only previously revealed true
   labels.
3. `dasp_safe_oneclass`: no model, guard, detector-reference, replay-memory, or
   fused-cache update from a newly revealed one-class batch.
4. `dasp_safe`: class-supported safety gate and bit-exact rollback.
5. `dasp_safe_v2`: balanced warm-up memory, bounded multi-window delayed-label
   accumulator, and one selectively unfrozen backbone layer during a strong
   event. It still updates only after all classes have sufficient revealed
   support.

All variants use the same streams, preprocessing, one-window delay, model
initialization, label budget, and optimizer budget. Current-window labels are
dereferenced only after prediction, detector evaluation, and query-index
selection.

## Root-cause analysis

### Gate and rollback

Rollback is correct but the gate can be too conservative. DASP-v1 made 826
exact rollbacks on BoT-IoT and retained a deployed model that was already
collapsed after the warm-up-to-A boundary. A safety gate prevents a harmful
candidate; it cannot repair a harmful fallback.

DASP-v2 improved this by preserving minority evidence and permitting one
guarded base layer, but still produced 879 exact rollbacks on BoT-IoT. No
inexact rollback was observed.

### Detector

Feature warnings and supervised performance evidence are not aligned under
extreme imbalance. One-class delayed samples make performance evidence
uninformative, while the input detector can repeatedly warn about a benign
covariate boundary that was unintentionally introduced between warm-up and
Concept A.

The existing stream validation checks concept transitions but not a strict
warm-up-to-A continuity gate. Therefore some failures attributed to online
drift adaptation are partly caused by a hidden evaluation boundary before the
declared drift onset.

### Query

The query is label-blind as required, but class-aware pseudo-strata cannot
guarantee true-class coverage after representation collapse. The effect is
most severe when a class is extremely rare:

- NF-BoT-IoT-v3 mean Attack rate: 99.68%.
- With a 256-label budget, an unstratified query contains fewer than one
  Benign sample in expectation.

DASP-v2 accumulated revealed labels across windows and waited for class
support. This obeys delayed prequential order and avoids training on a
one-class batch, but cannot fully solve an information deficit.

### Replay

Setting the replay target to 50/50 is insufficient when the reservoir has
already discarded the minority. On BoT-IoT, mean minority replay share changed
from approximately 0.095 in DASP-v1 to 0.402 in DASP-v2 after balanced warm-up
retention. This confirms that reservoir composition, not only replay sampling,
was a causal bottleneck.

### Adapter update

A fully frozen backbone was too weak for the severe warm-up-to-A shift on
ToN-IoT. Allowing one guarded base layer in DASP-v2 increased mean Macro-F1
from 0.502 to 0.677 and reduced ToN collapse runs from 10 to 2. The remaining
gap to SSF/ER shows that the adapter still does not recover fast enough.

### Delayed-label one-class feedback loop

The original loop was:

`collapsed prediction -> biased query -> one-class reveal -> invalid guard ->
frozen fallback -> repeated collapsed prediction`.

The accumulator and balanced memory weakened this loop but did not eliminate
it on BoT-IoT. In that dataset, the scarcity of Benign examples is a protocol
constraint as well as an algorithmic problem.

## Pilot results

Mean values are over 30 paired
`scenario × stream_seed × model_seed` cells per dataset, not over windows.

| Dataset | Current DASP | Safe v1 | Safe v2 | SSF-budget | ER | v2 − stronger baseline | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| NF-UNSW-NB15-v3 | 0.9890 | 0.9900 | 0.9886 | 0.9892 | 0.9899 | -0.0012 | Pass |
| NF-CSE-CIC-IDS2018-v3 | 0.9334 | 0.9172 | 0.9199 | 0.9642 | 0.9577 | -0.0443 | Fail |
| NF-ToN-IoT-v3 | 0.3196 | 0.5019 | 0.6775 | 0.8661 | 0.8554 | -0.1887 | Fail |
| NF-BoT-IoT-v3 | 0.6955 | 0.6635 | 0.6946 | 0.7246 | 0.7440 | -0.0494 | Fail |
| CIC-IDS2017 | 0.8688 | 0.8679 | 0.8675 | 0.9288 | 0.9245 | -0.0613 | Fail |

The preregistered non-inferiority margin was -0.03 on every dataset.

Paired 95% bootstrap confidence intervals for DASP-v2 minus the stronger
SSF/ER baseline:

| Dataset | Mean delta | 95% CI |
|---|---:|---:|
| NF-UNSW-NB15-v3 | -0.0012 | [-0.0026, -0.0000] |
| NF-CSE-CIC-IDS2018-v3 | -0.0443 | [-0.0609, -0.0282] |
| NF-ToN-IoT-v3 | -0.1887 | [-0.2476, -0.1377] |
| NF-BoT-IoT-v3 | -0.0494 | [-0.0901, -0.0112] |
| CIC-IDS2017 | -0.0613 | [-0.0793, -0.0436] |

## Collapse outcome

Collapse means at least three consecutive evaluated windows with Benign recall
or Attack recall at most 0.01.

| Method | Collapse runs / 150 |
|---|---:|
| Current DASP | 45 |
| DASP-safe v1 | 32 |
| DASP-safe v2 | 17 |

DASP-v2 collapse locations:

- NF-ToN-IoT-v3: 2/30.
- NF-BoT-IoT-v3: 15/30.
- Other three datasets: 0/30.

This is a material improvement, but it does not satisfy the preregistered
zero-collapse requirement.

## Scientifically valid next revision

Do not tune thresholds separately for ToN or BoT. The next development cycle
should:

1. Add an explicit warm-up-to-A continuity test for Benign and Attack,
   including a regime classifier and cross-segment generalization. Rebuild or
   reject streams that fail.
2. Seed the delayed performance detector with held-out warm-up predictions and
   attach confidence/support flags to one-class observations.
3. Replace pseudo-class-only query coverage with a fixed,
   dataset-independent mixture of raw-feature k-center diversity, model
   uncertainty, and prototype coverage.
4. Preserve a preregistered per-class minimum in memory for every method that
   claims class-aware replay.
5. Evaluate a fixed two-layer maximum selective recovery path under the same
   safety gate. Do not permit a global backbone update.
6. If a 256-label budget cannot provide minimum class support under the
   observed prevalence, report the scenario as budget-infeasible or increase
   the budget for all methods before looking at outcomes.
7. Use new development seeds for the next pilot. Confirmation seeds remain
   untouched.

## Artifacts

- Preregistered gate: `DASP_SAFETY_ADVANCEMENT_PREREG.md`
- Mock gate: `outputs_dasp_safety_mock/mock_gate.json`
- Final pilot gate:
  `outputs_dasp_safety_v2_pilot_combined/pilot_gate.json`
- Dataset/method metrics:
  `outputs_dasp_safety_v2_pilot_combined/metrics_by_dataset_method.csv`
- Paired confidence intervals:
  `outputs_dasp_safety_v2_pilot_combined/paired_macro_f1.csv`
- Collapse and integrity audit:
  `outputs_dasp_safety_v2_pilot_combined/run_integrity_and_collapse.csv`
- Mechanism diagnostics:
  `outputs_dasp_safety_v2_pilot_combined/mechanism_diagnostics.csv`
- Per-episode performance, drop, recovery, detector delay, and S0 false alarms:
  `outputs_dasp_safety_v2_pilot_combined/episode_performance.csv`

## Confirmation status

`confirmation_locked = true`

`confirmation_run_executed = false`

No confirmation result may be reported or inserted into the manuscript.
