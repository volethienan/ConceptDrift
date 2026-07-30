## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-30
- Verification Status: PREREGISTERED
- Version Label: dasp_safety_pilot_v1

## Scope

This development pilot compares the unchanged current DASP, cumulative safety
ablations, the complete safety revision, SSF-budget, and Experience Replay:

1. `dasp`
2. `dasp_safe_query`
3. `dasp_safe_replay`
4. `dasp_safe_oneclass`
5. `dasp_safe`
6. `ssf_budget`
7. `experience_replay`

The matrix is fixed at five datasets, ten controlled scenarios, three stream
seeds, one model seed, and seven methods: 1,050 runs. Confirmation seeds are
forbidden. No parameter may be selected from pilot results per dataset,
scenario, or stream.

## Fixed safety definitions

- Label delay is exactly one window.
- A class collapse is at least three consecutive evaluated windows in which
  either Benign recall or Attack recall is at most 0.01.
- A rollback violation is any rejected safety commit whose restored model state
  is not bit-exact to the deployed fallback.
- Every method is limited to 256 newly revealed labels per update window.
- Current-window labels may not affect that window's prediction, detector
  decision, or query indices.
- Attack subtype is stream-construction metadata only and is never a model
  feature.

## Pilot pass gate

All conditions are required:

1. Exactly 1,050 unique run cells complete with finite primary metrics.
2. No label-budget, global-backbone-update, rollback, or ordering violation.
3. `dasp_safe` has no three-window class collapse in any run.
4. On each of the five datasets, paired mean Macro-F1 of `dasp_safe` is no more
   than 0.03 below the stronger of SSF-budget and Experience Replay.
5. On each dataset, both cumulative Benign recall and cumulative Attack recall
   for `dasp_safe` exceed 0.05 in every run.
6. Statistical uncertainty is computed over paired
   `dataset × scenario × stream_seed × model_seed` cells, never over windows.

Failure of any condition keeps confirmation locked. A negative result is
reported without retuning or deleting failed runs.

## Confirmation lock

The 4,000-run confirmation matrix is outside this task and must not run even if
the pilot passes. A separate explicit user authorization would still be
required after reviewing the pilot report.
