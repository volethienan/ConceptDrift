from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from dasp_nids.config import load_config
from dasp_nids.runner import StreamingExperiment


SCENARIOS = ["S0", "S1", "S3", "S4"]
STREAM_SEEDS = [3101, 3102, 3103]
MODEL_SEED = 41
METHODS = [
    "static",
    "periodic_ft",
    "online_mlp",
    "experience_replay",
    "online_ewc",
    "agem",
    "ssf_budget",
    "repshield",
    "cdda_md",
    "dasp",
]


def main() -> None:
    base = load_config("configs/dasp_mock_benchmark.yaml")
    root = Path("outputs_dasp_mock_benchmark")
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    expected = len(SCENARIOS) * len(STREAM_SEEDS) * len(METHODS)
    completed = 0
    for scenario in SCENARIOS:
        for stream_seed in STREAM_SEEDS:
            for method in METHODS:
                config = copy.deepcopy(base)
                config.data.scenario = scenario
                config.data.random_seed = stream_seed
                config.data.stream_seed = stream_seed
                config.training.model_seed = MODEL_SEED
                config.dasp.baseline = method
                config.output.directory = str(root / scenario)
                config.output.run_name = (
                    f"{method}_stream{stream_seed}_model{MODEL_SEED}"
                )
                result = StreamingExperiment(config).run()
                record = dict(result.summary)
                record.update(
                    {
                        "scenario": scenario,
                        "stream_seed": stream_seed,
                        "model_seed": MODEL_SEED,
                        "method": method,
                        "run_dir": str(result.output_dir),
                    }
                )
                records.append(record)
                completed += 1
                pd.DataFrame(records).to_csv(
                    root / "summary_by_run.partial.csv", index=False
                )
                print(
                    f"completed {completed}/{expected} "
                    f"scenario={scenario} stream={stream_seed} "
                    f"method={method} "
                    f"macro={result.summary['mean_macro_f1']:.6f}",
                    flush=True,
                )
    frame = pd.DataFrame(records)
    key = ["scenario", "stream_seed", "model_seed", "method"]
    if len(frame) != expected or frame.duplicated(key).any():
        raise RuntimeError("Incomplete or duplicated mock matrix")
    frame.to_csv(root / "summary_by_run.csv", index=False)
    (root / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "stage": "dasp_mock_diagnostic",
                "scenarios": SCENARIOS,
                "stream_seeds": STREAM_SEEDS,
                "model_seeds": [MODEL_SEED],
                "methods": METHODS,
                "expected_runs": expected,
                "completed_runs": len(frame),
                "real_dataset_used": False,
                "confirmation_seed_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
