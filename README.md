# DASP-NIDS

Reproducible prequential experiments for binary network intrusion detection under concept drift.

## Current Research Task

- Main method: `modular_dasp` — Drift-Attributed Subspace Plasticity.
- Core idea: localize drift at module/subspace level, then repair only affected modules.
- Architecture: `ModularOnlineMLP` with disjoint feature-group modules.
- Evaluation: test-then-train with one-window label delay.
- Main comparator: `ssf` under the same stream, label budget, and model family.

## Kept Experiment Scenarios

| Path | Purpose |
|---|---|
| `configs/modular_dasp_main.yaml` | Main controlled partial-drift pilot against SSF |
| `configs/modular_dasp_smoke.yaml` | Small auxiliary smoke run/test config |
| `configs/smoke.yaml` | Legacy lightweight unit-test fixture |
| `src/dasp_nids/` | Algorithm, baselines, stream runner, metrics |
| `tests/` | Unit and regression tests |
| `data/` | Optional raw/prepared IDS data inputs |

## Run Smoke Test

```powershell
$env:PYTHONPATH='src'
python -m dasp_nids.cli run --config configs\modular_dasp_smoke.yaml
```

## Compare Against SSF

```powershell
$env:PYTHONPATH='src'
python -m dasp_nids.cli compare `
  --config configs\modular_dasp_main.yaml `
  --methods modular_dasp ssf `
  --seeds 9501
```

## Key Evidence Columns

Each run writes `windows.csv` with:

- `requirement_module_*`: localized drift attribution by module.
- `plasticity_module_*`: routed update intensity by module.
- `localized_repair_modules`: modules selected for repair.
- `localized_repair_module_count`: number of repaired subspaces.
- `global_update`: should remain false for `modular_dasp`.


## Small Pilot

```powershell
$env:PYTHONPATH='src'
python toolsun_modular_dasp_small_pilot.py
```

This runs 5 datasets × 3 scenarios × 1 stream seed × 1 model seed × 8 methods and writes `outputs_pilot_small/pilot_summary.csv` plus `outputs_pilot_small/PILOT_SMALL_REPORT.md`.

## Verification

```powershell
$env:PYTHONPATH='src'
pytest tests/test_core.py::test_modular_dasp_strict_freeze_updates_only_routed_module -q
```

For the current global Python install, PyTorch emits a NumPy 2.x compatibility warning, but the smoke run and focused unit test complete successfully.
