# Reproducibility Release Checklist

## Verified Public Base

- Repository: https://github.com/volethienan/ConceptDrift
- Remote main commit: 42d9cf3cbce4133ba4b1f13e3a787f46b9182cd9
- Original evidence lock: revision/baseline_lock_2026-09-01
- Original aggregate manifest: 85536b32aa1d6ee5c34ff3882c2fa36e834da0ca82375948f312222d6303aead

The public base commit predates the reviewer-requested P0/P1 files. Do not state that the public repository contains the revision bundle until those files are uploaded.

## Revision Bundle to Publish

- configs/revision_p0.yaml
- requirements-revision-lock.txt
- src/dasp_nids/revision.py
- revised src/dasp_nids/config.py and src/dasp_nids/runner.py
- tools/run_revision_experiments.py
- tools/run_revision_p1.py
- tools/prepare_revision_natural_stream.py
- tools/analyze_revision_cluster_aware.py
- revision/full_p0_lock_2026-09-01
- revision/full_p1_lock_2026-09-02
- outputs_revision_v1/full_p0_v1 summary, validation, environment, and manifest files
- outputs_revision_v1/p1_components_v1 summary, validation, environment, and manifest files
- outputs_revision_v1/p1_feature_permutation_v1 summary, validation, environment, and manifest files
- outputs_revision_v1/p1_natural_v1 summary, validation, environment, and manifest files
- outputs_revision_v1/cluster_aware_v1
- focused revision tests
- revised manuscript source, generated revision tables, and response letter

If the full run-level archive is too large for Git, publish it through an archival service and place a manifest plus stable DOI in the Git release. Do not omit the per-run configurations or SHA-256 manifest.

## Exact Environment

- CPython 3.12.7
- PyTorch 2.3.1+cpu
- NumPy 1.26.4
- pandas 2.3.3
- scikit-learn 1.7.2
- SciPy 1.16.2
- PyYAML 6.0.2
- psutil 7.1.0
- pytest 8.4.2
- Windows 11 build 26200
- Lenovo 82XV
- Intel Core i5-13420H
- 8 physical cores, 12 logical cores
- approximately 16 GB RAM
- CPU-only, one PyTorch thread per run

The normative dependency record is requirements-revision-lock.txt. The normative machine records are revision/environment_revision.json and the execution_environment.json files stored with each experiment family.

## Commands

Create and validate the environment:

    py -3.12 -m venv .venv-revision
    .\.venv-revision\Scripts\python.exe -m pip install -r requirements-revision-lock.txt
    .\.venv-revision\Scripts\python.exe -m pip check

Verify the original evidence lock:

    .\.venv-revision\Scripts\python.exe tools\create_baseline_lock.py --repo-root . --lock-directory revision\baseline_lock_2026-09-01 --verify

Run or resume P0:

    .\.venv-revision\Scripts\python.exe tools\run_revision_experiments.py --mode full_p0 --resume

Run or resume P1:

    .\.venv-revision\Scripts\python.exe tools\run_revision_p1.py --mode components --resume
    .\.venv-revision\Scripts\python.exe tools\run_revision_p1.py --mode permutations --resume
    .\.venv-revision\Scripts\python.exe tools\run_revision_p1.py --mode natural --resume

Regenerate cluster-aware statistics:

    .\.venv-revision\Scripts\python.exe tools\analyze_revision_cluster_aware.py

Run focused validation:

    .\.venv-revision\Scripts\python.exe -m pytest tests\test_revision_protocol.py tests\test_revision_runner_validation.py tests\test_revision_p1_orchestration.py tests\test_revision_cluster_aware.py -q

## Author Actions Before Resubmission

1. Upload the revision bundle as a tagged Git release or stable external archive.
2. Record the release URL, tag, and DOI if available.
3. Replace the archive marker in the Data Availability section.
4. Update Reviewer 1, Comment 5 in the response letter with the final identifier.
5. Verify that an anonymous reviewer can download the files and run the smoke/reproduction commands.
6. Confirm that the uploaded checksums match the locked local manifests.
