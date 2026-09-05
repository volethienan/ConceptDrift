# Pre-Revision Baseline Lock

This directory records the computational evidence used by the original 4,000-run confirmation before the major-revision experiments.

## Locked Scope

- 4,000 method runs.
- 500 paired dataset--scenario--stream-seed--model-seed cells.
- 250 prepared streams.
- 12,518 code, configuration, aggregate, prepared-stream, and run-record files.
- 16,481,836,196 locked bytes.

The matrix reuses 720 runs from `outputs_full_v6_core`, 840 runs from `outputs_full_v6_10scenario_pilot`, and 2,440 runs stored directly in `outputs_confirmation_v6_10scenario`.

## Files

- `baseline_lock.json`: repository, matrix, environment-inspection, and aggregate-digest metadata.
- `matrix_runs.csv`: maps every one of the 4,000 matrix rows to its actual config, summary, window log, and prepared stream.
- `evidence_sha256.csv`: per-file SHA-256 manifest.
- `git_status_at_lock.txt`: working-tree status recorded when the lock was created.

## Aggregate Digest

The aggregate SHA-256 over the sorted path, size, and per-file digest records is:

`85536b32aa1d6ee5c34ff3882c2fa36e834da0ca82375948f312222d6303aead`

The lock records a dirty working tree because manuscript build artifacts and manuscript-source edits existed before the revision experiment implementation. The lock does not claim that the inspection environment was the environment originally used to execute all 4,000 runs.

## Verification

Run:

```powershell
python tools/create_baseline_lock.py --repo-root . --lock-directory revision/baseline_lock_2026-09-01 --verify
```

The command recomputes all 12,518 file digests and fails if any locked file is missing or changed.
