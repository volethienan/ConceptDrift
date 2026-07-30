# DASP mock-benchmark advancement preregistration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-26
- Verification Status: LOCKED-BEFORE-RUN
- Version Label: dasp_mock_prereg_v1

## Purpose

This is a synthetic diagnostic gate for the revised DASP mechanism. It is not
evidence about NF datasets and cannot replace the controlled or chronological
benchmark. No unused NF development or confirmation seed may be opened during
this gate.

## Locked matrix

- Source: the repository's synthetic controlled generator
- Scenarios: S0, S1, S3, S4
- Stream seeds: 3101, 3102, 3103
- Model seed: 41
- Windows: 24, with four warm-up windows
- Window size: 512 flows
- Label delay: one window
- Update-label ceiling: 64 labels per evaluated window
- Methods:
  - Frozen
  - Periodic fine-tuning
  - Online MLP
  - Experience Replay
  - Online EWC
  - A-GEM
  - SSF-budget
  - RepShield
  - CDDA-MD
  - DASP

All budget-matched baselines query labels uniformly before labels are read.
DASP uses its declared label-blind active query. No method may use the
unqueried labels for updating.

## Locked outcomes

The primary endpoint is mean Macro-F1 over the first four windows after each
synthetic phase transition. Attack Recall over the same windows is the safety
endpoint. S3 first drift and recurrence are reported separately. S0 is used for
false-alarm diagnostics. Full-stream Macro-F1, latency, memory, labels, and
global-update counts are secondary.

## Advancement rule

DASP advances to a real-data development rerun only if all conditions hold:

1. Mean paired early Macro-F1 difference DASP minus SSF-budget is positive.
2. Mean paired early Attack Recall difference versus SSF-budget is at least
   -0.01.
3. DASP wins early Macro-F1 in at least six of the nine first-drift paired
   cells against SSF-budget.
4. DASP's mean early-Macro-F1 rank among ten methods is at most 3.0.
5. DASP has zero whole-model updates in every run.
6. No DASP window exceeds 64 acquired labels.

Bootstrap intervals are diagnostic because three synthetic seeds are not a
confirmatory sample. Failure of this gate blocks a full experiment.
