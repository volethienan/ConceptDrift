from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import yaml

from dasp_nids.config import load_config
from dasp_nids.runner import StreamingExperiment


DATASETS = [
    "nf-unsw-nb15-v3",
    "nf-cse-cic-ids2018-v3",
    "nf-ton-iot-v3",
    "nf-bot-iot-v3",
    "luflow",
]
METHODS = [
    "static",
    "periodic_ft",
    "online_mlp",
    "experience_replay",
    "ssf_budget",
    "repshield",
    "dasp_v16_no_detector",
    "dasp_v16",
]
MODEL_SEEDS = [101, 102]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded exact-prefix natural chronological track."
    )
    parser.add_argument(
        "--config", default="configs/dasp_v16_controlled.yaml"
    )
    parser.add_argument(
        "--prepared-dir", default="prepared_natural_streams"
    )
    parser.add_argument("--output-dir", default="outputs_dasp_v16_natural")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--model-seeds", default="101,102")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    datasets = [value for value in args.datasets.split(",") if value]
    methods = [value for value in args.methods.split(",") if value]
    model_seeds = [
        int(value) for value in args.model_seeds.split(",") if value
    ]
    base = load_config(args.config)
    root = Path(args.output_dir)
    summaries: list[dict[str, object]] = []
    expected = len(datasets) * len(methods) * len(model_seeds)
    completed = 0
    for dataset in datasets:
        stream_path = (
            Path(args.prepared_dir) / dataset / "natural_prefix.npz"
        )
        if not stream_path.exists():
            raise FileNotFoundError(stream_path)
        for model_seed in model_seeds:
            for method in methods:
                config = copy.deepcopy(base)
                config.data.path = str(stream_path)
                config.data.scenario = "natural"
                config.data.stream_seed = 0
                config.data.random_seed = 0
                config.training.model_seed = model_seed
                config.dasp.baseline = method
                config.output.directory = str(root / dataset)
                config.output.run_name = f"{method}_model{model_seed}"
                run_dir = Path(config.output.directory) / config.output.run_name
                summary_path = run_dir / "summary.json"
                config_path = run_dir / "config.yaml"
                if args.resume and summary_path.exists() and config_path.exists():
                    cached = yaml.safe_load(
                        config_path.read_text(encoding="utf-8")
                    )
                    if cached != config.to_dict():
                        raise ValueError(
                            f"Resolved config mismatch at {run_dir}"
                        )
                    summary = json.loads(
                        summary_path.read_text(encoding="utf-8")
                    )
                    state = "resumed"
                else:
                    summary = StreamingExperiment(config).run().summary
                    state = "completed"
                summary = dict(summary)
                summary.update(
                    {
                        "dataset": dataset,
                        "track": "natural_prefix",
                        "model_seed": model_seed,
                        "run_dir": str(run_dir),
                    }
                )
                summaries.append(summary)
                completed += 1
                print(
                    f"{state} {completed}/{expected} dataset={dataset} "
                    f"model={model_seed} method={method} "
                    f"macro_f1={summary['mean_macro_f1']:.6f}",
                    flush=True,
                )
                root.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(summaries).to_csv(
                    root / "summary_by_run.partial.csv", index=False
                )
    frame = pd.DataFrame(summaries)
    keys = ["dataset", "model_seed", "method"]
    if len(frame) != expected or frame.duplicated(keys).any():
        raise ValueError("Natural-track matrix is incomplete or duplicated")
    frame.to_csv(root / "summary_by_run.csv", index=False)
    print(f"natural matrix complete: {root}", flush=True)


if __name__ == "__main__":
    main()
