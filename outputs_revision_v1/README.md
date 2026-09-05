# Revision Experiment Outputs

This directory is reserved for the major-revision experiments defined in `revision/REVISION_EXPERIMENT_PROTOCOL.md`.

- Do not copy or move original confirmation records into this directory.
- Do not overwrite `outputs_full_v6_core`, `outputs_full_v6_10scenario_pilot`, or `outputs_confirmation_v6_10scenario`.
- Store smoke, reduced-pilot, and full-confirmation outputs in separately named subdirectories.
- Record the protocol version, repository commit, configuration digest, and execution environment in every run root.
- Generated run artifacts remain ignored by Git; this README is the only tracked file in the output root.

Suggested subdirectories:

- `smoke_v1`
- `reduced_pilot_v1`
- `confirmation_v1`
