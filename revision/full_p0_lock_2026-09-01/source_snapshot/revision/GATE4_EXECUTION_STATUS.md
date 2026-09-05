# Gate 4 Full P0 Execution Status

Status date: September 1, 2026.

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

Gate 4 is ready to start. Once the first run begins, source, configuration, routing, label, replay, gating, and weighting definitions are frozen under Protocol v1.1.
