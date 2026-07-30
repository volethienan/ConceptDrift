# DASP-NIDS

Reproducible prequential experiments for binary network intrusion detection
under concept drift.

## Current research task

- Classes: `Benign` and `Attack`.
- Every attack subtype is mapped to the single `Attack` class.
- Main paper method: `DASP`.
- CLI method key: `dasp`.
- `dasp` resolves to the event-gated, fused-inference main implementation
  (`event-gated-fused-v1`).
- Whole-backbone updates after warm-up are forbidden.
- Evaluation order is test-then-train with a one-window label delay.

## Main comparison methods

The controlled full matrix compares:

1. Static model
2. Periodic fine-tuning
3. Online MLP
4. Experience Replay
5. Online EWC
6. A-GEM
7. Budget-matched SSF
8. RepShield
9. CDDA-MD
10. DASP

All update-capable methods use the same stream, model initialization, label
delay, ground-truth label budget and update ceiling.

## Project structure

| Path | Purpose |
|---|---|
| `src/dasp_nids/` | Current implementation and baselines |
| `tests/` | Unit and integration tests |
| `configs/dasp_v18_controlled.yaml` | Current controlled/natural experiment configuration |
| `configs/dasp_mock_benchmark.yaml` | Small synthetic verification configuration |
| `configs/smoke.yaml` | Test configuration |
| `data/` | Raw IDS datasets |
| `prepared_streams_v2/` | Binary controlled streams |
| `prepared_natural_streams_v2/` | Binary chronological streams |
| `outputs_dasp_full_binary_controlled/` | Legacy pre-upgrade CPU control matrix |
| `outputs_dasp_full_binary_natural/` | Legacy pre-upgrade natural results |
| `outputs_dasp_full_binary_controlled_gpu/` | Legacy pre-upgrade GPU control matrix; its concurrent timing is not reportable |
| `outputs_dasp_fast_full_binary_controlled_gpu/` | Clean upgraded-DASP GPU confirmation matrix |
| `outputs_dasp_mock_benchmark/` | Current mock benchmark record |
| `ACCESS_latex_template_20260513/` | Paper source |
| `DASP_FIVE_DATASET_DRIFT_ADAPTATION_PROTOCOL.md` | Locked experiment protocol |
| `DASP_V18_IMPLEMENTATION_SPEC.md` | Current DASP mechanism |

The filenames `run_dasp_v16_five_dataset.py`,
`run_dasp_v16_natural.py`, and their analysis counterparts are retained only
as established matrix runners. Passing the current configuration and method
key `dasp` runs the event-gated/fused implementation. Resolved configs include
`fast_revision`, so legacy cells cannot be resumed into an upgraded matrix.

## Full controlled experiment

The confirmation matrix contains 2,000 runs:

`4 datasets × 5 drift scenarios × 5 stream seeds × 2 model seeds × 10 methods`.

Resume it with:

```powershell
.venv\Scripts\python tools\run_dasp_v16_five_dataset.py `
  --config configs\dasp_v18_controlled.yaml `
  --prepared-dir prepared_streams_v2 `
  --stage confirmation `
  --output-dir outputs_dasp_fast_full_binary_controlled_gpu `
  --methods static,periodic_ft,online_mlp,experience_replay,online_ewc,agem,ssf_budget,repshield,cdda_md,dasp `
  --resume
```

## S5 high-severity stress track

S5 is an explicit secondary track and is not generated or run by the canonical
defaults. It keeps the real features, labels, class prevalence, and concept-B
family weights, but selects the most benign-like attack flows with a fixed
surrogate ensemble. A disjoint audit ensemble enforces preregistered severity
gates before an artifact is written.

Generate and validate S5 separately:

```powershell
.venv\Scripts\python tools\build_controlled_nf_streams.py `
  --scenarios S5 `
  --output-dir prepared_streams_s5_v1 `
  --stream-seeds 2101,2102,2103,2104,2105

.venv\Scripts\python tools\validate_controlled_streams.py `
  --prepared-dir prepared_streams_s5_v1 `
  --scenarios S5 `
  --stream-seeds 2101,2102,2103,2104,2105 `
  --output prepared_streams_s5_v1\validation_report.json
```

The S5 confirmation matrix adds 400 secondary runs:

`4 datasets × 1 scenario × 5 stream seeds × 2 model seeds × 10 methods`.

Run it with `--scenarios S5` and point `--prepared-dir` to the validated S5
artifact directory. Do not pool S5 into the canonical S1–S4 primary contrast.
The generator now rejects S5 unless it is also materially harder than a
source-disjoint, family/prevalence/time-matched random-B control. This prevents
the stress track from being justified only by selecting an unusually hard
subset.

Before any broad Q1 claim, run the fail-closed evidence audit:

```powershell
$env:PYTHONPATH='src'
python tools\audit_q1_submission_readiness.py
```

The generated `q1_submission_readiness.json` keeps planned matrices, completed
evidence, construct-validity eligibility, manuscript metadata, and archival
requirements separate.

## Full natural chronological experiment

The natural matrix contains 400 runs:

`5 datasets × 10 model seeds × 8 main methods`.

Resume it with:

```powershell
.venv\Scripts\python tools\run_dasp_v16_natural.py `
  --config configs\dasp_v18_controlled.yaml `
  --prepared-dir prepared_natural_streams_v2 `
  --output-dir outputs_dasp_fast_full_binary_natural_gpu `
  --methods online_mlp,experience_replay,online_ewc,agem,ssf_budget,repshield,cdda_md,dasp `
  --model-seeds 101,102,103,104,105,106,107,108,109,110 `
  --resume
```

The current NF-CSE-CIC-IDS2018 chronological prefix contains only benign
flows. It must either be reported as a benign-only false-alarm segment or be
excluded from natural classification aggregation. Attack Recall for that
prefix is `N/A`; it must not be encoded as zero.

## Verification

```powershell
.venv\Scripts\python -m pytest -q
```

Each completed experiment cell contains its resolved `config.yaml`,
`windows.csv`, and `summary.json`. The resume runners reuse a cell only when
its stored resolved configuration exactly matches the requested configuration.
Latency runs must execute with no concurrent CPU/GPU benchmark; report forward
inference, adaptation-event latency, and end-to-end window latency separately.
