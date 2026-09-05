# Phase C and D Execution Status

Status date: September 1, 2026.

## Phase C: Gate 2 Smoke Matrix

- Locked cell: NF-ToN-IoT-v3, S1-Sudden, stream seed 2001, model seed 91.
- Conditions: A1--A10.
- Completed runs: 10/10.
- Validation: passed with zero errors and zero label-pool warnings.
- Output: `outputs_revision_v1/smoke_v1`.
- Validation SHA-256: `2F1B817D468EE5A59AFB422C177C1046DC07B75EB2D3FCE0BBCF32896D485503`.
- Artifact-manifest SHA-256: `E8662EDD45F65DC86DEC9E829D2E4BE7639DBDDB76AFB9540996F3C63AD1BC1A`.

## Phase D: Gate 3 Reduced Pilot

- Datasets: all five primary datasets.
- Scenarios: S1-Sudden, S2-Gradual, and S7-Local-High.
- Stream seed: 2001.
- Model seed: 91.
- Conditions: A1--A10.
- Completed runs: 150/150 across 15 cells.
- Validation: 15/15 cells passed with zero missing artifacts, digest mismatches, strict-freeze violations, locality-field omissions, or non-finite primary summaries.
- Output: `outputs_revision_v1/reduced_pilot_v1`.
- Validation SHA-256: `F6028D9DD2483C02ADDF10683F3A69FCCF7A19EE29412B07999622827FD63F7A`.
- Artifact-manifest SHA-256: `0518517F21F4C11A84702C2EDB1F25963856FF85C6BB90F4C64EEA14D922F61D`.

## Degenerate-Label Audit

The 15 prepared streams contain 480 post-warm-up windows, and every full window contains both classes. Label-blind subsampling nevertheless produces one-class pools in the three NF-BoT-IoT-v3 cells:

- query pools: 15 windows;
- repair-candidate pools: 26 windows;
- gate pools: 61 windows;
- balanced-training pools: 26 windows.

This is recorded as a diagnostic warning rather than an invariant failure because the indices are label-blind and shared across methods. The locked query, label, routing, replay, gating, and weighting definitions were not changed after inspecting the pilot.

## Next Gate

Gate 4 preparation is complete. Its live execution status is tracked in `revision/GATE4_EXECUTION_STATUS.md`.
