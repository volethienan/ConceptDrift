# Purity-certified same-window verification for NIDS under label delay

Code and result snapshots for experiments on network intrusion detection when
analyst labels arrive several windows late. A small part of each window's
labelling budget is verified at once (the *fast lane*); the fast labels are
spent on clusters of the flows the model calls benign, and a cluster is raised
as one alert group only when `r` uniformly drawn members are all attacks, which
bounds the chance of raising a cluster whose benign share exceeds a tolerance.

## Contents

| Path | What it holds |
|---|---|
| `src/dasp_nids/` | stream runner, learners, label oracle, fast lanes (`verified_propagation.py`) |
| `tools/build_natural_streams.py`, `tools/build_natural_cic_stream.py`, `tools/build_natural_cidds_stream.py` | chronological stream construction (`--drop-conflicts` for the conflict-free NF-ToN-IoT rebuild) |
| `tools/run_dasp_v2.py` | experiment stages: `verify`, `verify2`, `tradeoff`, `audit`, `revision`, `dedup`, `memory`, `acceptance`, `pilot`, `pilot2` |
| `tools/analyze_dasp_v2.py` | per-run metrics and stage analyses |
| `paper/jtit/*.py` | scripts that turn the result snapshots into the tables and figures |
| `results_snapshot/` | per-run metrics and summaries used for every reported number (`verify_20261109`, `verify2_20261110`, `audit_20261112`, `revision_20261113`); `dev/dev_selection.json` holds the selected learner configurations |
| `tests/` | unit tests, including label-leakage checks that flip every unqueried label |

## Data

All datasets are public and are not redistributed here: the NetFlow v3 releases
of UNSW-NB15, CSE-CIC-IDS2018, ToN-IoT and BoT-IoT (Luay et al., 2025),
CIC-IDS2017 (MachineLearningCVE feature files) and CIDDS-001 (OpenStack weeks
1-2). Place them under `data/` as expected by `tools/build_controlled_nf_streams.py`
and the natural-stream builders.

## Reproducing

```bash
python -m venv .venv-revision
.venv-revision/Scripts/pip install -r requirements-revision-lock.txt   # Windows; use bin/ on Linux
.venv-revision/Scripts/pip install -e .

# build chronological streams (example: NF-ToN-IoT)
python tools/build_natural_streams.py --datasets nf-ton-iot-v3 --stream-seeds 3001,3002,3003 \
    --output-dir prepared_streams_v2/natural

# plan, then run one stage (shards are independent processes)
python tools/run_dasp_v2.py --stage revision --selection results_snapshot/dev/dev_selection.json \
    --prepared-root prepared_streams_v2 --output-root results/revision
python tools/run_dasp_v2.py --stage revision --selection results_snapshot/dev/dev_selection.json \
    --prepared-root prepared_streams_v2 --output-root results/revision --execute --resume --shard 1/4

# tables from the snapshots (or from fresh results with --results results)
python paper/jtit/make_revision_tables.py
python -m pytest tests
```

Held-out streams (CIC-IDS2017, CIDDS-001) are planned with `--heldout`.
Every stream is ordered in time, preprocessing is fitted on warm-up windows
only, and each window is scored only on the flows the fast lane did not query.

## License

MIT, see `LICENSE`. Citation metadata is in `CITATION.cff`.
