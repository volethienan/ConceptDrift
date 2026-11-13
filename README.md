# Certified Same-Window Verification for Network Intrusion Detection under Label Delay

Code, tests and result snapshots for the study *Certified Same-Window
Verification for Network Intrusion Detection under Label Delay*.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24203046.svg)](https://doi.org/10.5281/zenodo.24203046)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

## Overview

Learned network intrusion detectors are updated from analyst labels, and those
labels arrive hours or days after the traffic they describe. Attacks in the
benchmark captures arrive as bursts of one to three observation windows, so a
detector that waits for delayed labels learns each new attack after it has ended.

This repository implements a **fast lane** that moves a small part of each
window's labelling budget forward in time:

1. The flows that the current model calls benign are clustered.
2. Each cluster is screened with one uniformly drawn label.
3. A cluster whose screening label is an attack is confirmed with further uniform
   draws and raised as one alert group only when all `r` draws are attacks.
   A cluster whose benign share exceeds a tolerance `ε` therefore passes with
   probability at most `(1 − ε)^r`.
4. Clusters whose draws disagree are split and tested again.

Every method spends the same 128 labels per window; the fast lane only changes
when 16 of them arrive. Each window is scored only on the flows the fast lane did
not query, and a label oracle refuses any query beyond the budget.

**Headline result.** With labels delayed by five windows, the certified fast lane
raises the cumulative Macro-F1 of experience replay from 0.49 to 0.92 on the
held-out CIC-IDS2017 stream (benign FPR 1.1%), within 0.02 of the same lane with
immediate labels. All per-run metrics are in `results_snapshot/`.

## Repository layout

```
src/dasp_nids/                 stream runner, learners, metrics
  verified_propagation.py      fast lanes: purity-certified, representative,
                               uniform + kNN, hierarchical sampling, ILAB-style,
                               label oracle and analyst-error models
  runner.py                    prequential loop with delayed and fast labels
tools/
  build_natural_streams.py     chronological NetFlow v3 streams (--drop-conflicts
                               rebuilds NF-ToN-IoT without contradictory duplicates)
  build_natural_cic_stream.py  chronological CIC-IDS2017 stream
  build_natural_cidds_stream.py chronological CIDDS-001 stream
  run_dasp_v2.py               experiment stages (plan, shard, resume)
  analyze_dasp_v2.py           per-run metrics and stage analyses
paper/jtit/*.py                scripts that build the tables and figures
results_snapshot/              per-run metrics behind every reported number
tests/                         unit tests, incl. label-leakage checks
configs/                       learner configuration
```

## Installation

Python 3.12 was used for all experiments (CPU only).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-revision-lock.txt
pip install -e .
python -m pytest tests             # about 350 tests, a few minutes on CPU
```

Pinned versions include PyTorch 2.3.1, scikit-learn 1.7.2, NumPy 1.26.4 and
pandas 2.3.3 (`requirements-revision-lock.txt`).

## Data

The datasets are public and are not redistributed here. Place them under `data/`:

| Dataset | Expected location | Role |
|---|---|---|
| NF-UNSW-NB15-v3 | `data/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv` | development |
| NF-CSE-CIC-IDS2018-v3 | `data/NF-CSE-CIC-IDS2018-v3/NF-CICIDS2018-v3.csv` | development |
| NF-ToN-IoT-v3 | `data/NF-ToN-IoT-v3/NF-ToN-IoT-v3.csv` | development |
| CIC-IDS2017 (MachineLearningCVE files) | `data/CIC-IDS2017/` | held-out |
| CIDDS-001 (OpenStack, weeks 1–2) | `data/CIDDS-001/CIDDS-001/traffic/OpenStack/` | held-out |

## Reproducing the experiments

**1. Build the streams.** Each capture is ordered in time, cut into 36 segments
of equal flow count, and each window is a uniform sample of 4,096 flows of its
segment. Three disjoint stream seeds are drawn per dataset.

```bash
python tools/build_natural_streams.py --datasets nf-unsw-nb15-v3 --stream-seeds 2001,2002,2003 --output-dir prepared_streams_v2/natural
python tools/build_natural_streams.py --datasets nf-cse-cic-ids2018-v3,nf-ton-iot-v3 --stream-seeds 3001,3002,3003 --output-dir prepared_streams_v2/natural
python tools/build_natural_cic_stream.py --stream-seeds 4001,4002,4003
python tools/build_natural_cidds_stream.py --stream-seeds 5001,5002,5003
```

**2. Run a stage.** A stage is first planned, then executed in independent
shards that can be resumed. Held-out streams are planned with `--heldout`.

```bash
ARGS="--selection results_snapshot/dev/dev_selection.json --prepared-root prepared_streams_v2"
python tools/run_dasp_v2.py --stage verify2 $ARGS --output-root results/verify2
python tools/run_dasp_v2.py --stage verify2 $ARGS --output-root results/verify2 --execute --resume --shard 1/4
# ... shards 2/4 to 4/4, then the same with --heldout for CIC-IDS2017 and CIDDS-001
```

| Stage | Contents |
|---|---|
| `verify` | learners without a fast lane (ER, IWMS, uncertainty sampling, ARF) at delays 1 and 5; immediate-label reference |
| `tradeoff` | number of draws `r` of the certified lane and clusters of the representative lane |
| `verify2` | all fast lanes scored on unqueried flows; delays 2 and 10; prospective variant; anomaly lane; fast lane on ARF; budget and design ablations; independent analyst errors |
| `audit` | certificate audit against ground truth; clustering ablations |
| `revision` | correlated analyst errors; self-training and ILAB-style baselines; fast lanes with immediate labels; volume-matched controls |
| `dedup` | main comparison on the conflict-free NF-ToN-IoT rebuild |
| `memory` | reuse of raised clusters in later windows |
| `acceptance` | acceptance rule on identical clusters (one label, majority, certificate) |
| `pilot`, `pilot2` | development-only pilots of variants that were not adopted |

**3. Build the tables.** The scripts in `paper/jtit/` read the run directories
under `results/` and write LaTeX tables, figures and CSV summaries:

```bash
python paper/jtit/make_figures.py          # trade-off and burst-onset figures
python paper/jtit/make_audit_tables.py     # budget and design ablations, certificate audit, prospective variant
python paper/jtit/make_revision_tables.py  # main comparison, analyst-error models, controls, stream-level contrasts
python paper/jtit/operational_load.py      # false alarms per shift, per-family recall
python paper/jtit/dedup_summary.py
python paper/jtit/memory_table.py
python paper/jtit/acceptance_table.py
```

## Result snapshots

`results_snapshot/` holds the per-run metrics (one row per run: dataset, stream
seed, model seed, condition) and stage summaries used for every reported number:

| Snapshot | Stages |
|---|---|
| `verify_20261109/` | `verify`, `tradeoff` |
| `verify2_20261110/` | `verify2` (development, CIC-IDS2017, CIDDS-001) |
| `audit_20261112/` | `audit`, incl. the per-cluster certificate records |
| `revision_20261113/` | `revision`, `dedup`, `memory`, `acceptance`, operational load |
| `dev/dev_selection.json` | learner configurations selected on separate development streams |

## Evaluation protocol

- Streams are chronological; imputation and scaling are fitted on the four warm-up windows only.
- 3 stream seeds × 5 model seeds per condition; the stream is the unit of evidence.
- Design choices were fixed on the development streams before the held-out streams were run.
- Each window is scored only on the flows that the fast lane did not query.
- A unit test flips every label outside the queried flows and checks that no prediction changes.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff); GitHub shows a
*Cite this repository* button for it. The code and result snapshots are archived
on Zenodo: [10.5281/zenodo.24203046](https://doi.org/10.5281/zenodo.24203046).

## License

Released under the [MIT License](LICENSE).
