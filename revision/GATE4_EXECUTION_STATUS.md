# Gate 4 Full P0 Execution Status

Status date: September 2, 2026.

## Locked Matrix

- Datasets: 5.
- Scenarios: 10.
- Stream seeds: 2001, 2002, 2003, 2101, and 2102.
- Model seeds: 91 and 92.
- Paired cells: 500.
- Conditions: A1--A10.
- Expected method runs: 5,000.
- Output directory: `outputs_revision_v1/full_p0_v1`.
- Resume command: `.venv-revision/Scripts/python.exe tools/run_revision_experiments.py --mode full_p0 --resume`.

## Pre-run Gates

- Gate 2 validation: passed.
- Gate 3 validation: passed.
- Prepared streams: 250/250 present.
- Revision environment: isolated; `pip check` passed.
- Revision tests: 13/13 passed.
- Outcome-driven tuning after Gate 3: none.

## Execution State

Gate 4 completed successfully on September 2, 2026. Source, configuration, routing, label, replay, gating, and weighting definitions remained frozen under Protocol v1.1 for the complete run.

- Completed runs: 5,000/5,000.
- Validated paired cells: 500/500.
- Failed cells: 0.
- Standard-error log: empty.
- Runtime: 53,136.3 seconds, approximately 14 hours 45 minutes 36 seconds.
- Per-run artifacts: 5,000 configs, 5,000 summaries, 5,000 window logs, and 5,000 provenance files.
- Label-pool diagnostic warnings: 100 cells, all from NF-BoT-IoT-v3 label-blind subsampling; no full window was one-class.
- Validation SHA-256: `4047D0C901B6506B81EC1E9C056B7F903700B0F66C2CEBAEF6882371A75BC4CB`.
- Artifact-manifest SHA-256: `661E6F3FB64954CB9AEFA79CE583ED7DFC5804070325B8C8F26D0843DAC0A5B3`.
- Final results: `outputs_revision_v1/full_p0_v1/full_summary.csv`.
- Final validation: `outputs_revision_v1/full_p0_v1/validation.json`.
- Gate report: `outputs_revision_v1/full_p0_v1/GATE_REPORT.md`.
