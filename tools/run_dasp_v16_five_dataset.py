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
    "cic-ids2017",
]
CANONICAL_SCENARIOS = [
    "S0",
    "S1-Sudden",
    "S2-Gradual",
    "S3-Incremental",
    "S4-Recurrent",
    "S5-Blip",
    "S6-Combined",
    "S7-Local-Low",
    "S7-Local-Medium",
    "S7-Local-High",
]
SCENARIOS = [
    *CANONICAL_SCENARIOS,
    "S8-Stress",
    "S1",
    "S2",
    "S3",
    "S4-Low",
    "S4-Medium",
    "S4-High",
    "S5",
]
PRIMARY_METHODS = [
    "online_mlp",
    "experience_replay",
    "online_ewc",
    "agem",
    "ssf_budget",
    "repshield",
    "cdda_md",
    "dasp",
]
SUPPLEMENTARY_METHODS = ["dasp_v12", "river_hat", "river_arf"]
STAGES = {
    "v3-small": ([2001], [11]),
    "pilot": ([2001, 2002, 2003], [11]),
    "confirmation": ([2101, 2102, 2103, 2104, 2105], [101, 102]),
}


def _read_cached_summary(
    output_dir: Path, expected_config: dict[str, object]
) -> dict[str, object]:
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "config.yaml"
    if not summary_path.exists() or not config_path.exists():
        raise FileNotFoundError(output_dir)
    cached_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cached_config != expected_config:
        raise ValueError(
            f"Refusing to resume {output_dir}: resolved config mismatch"
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the locked paired Selective-DASP-v16 controlled matrix."
    )
    parser.add_argument(
        "--config", default="configs/dasp_v18_controlled.yaml"
    )
    parser.add_argument(
        "--prepared-dir", default="prepared_streams"
    )
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--scenarios", default=",".join(CANONICAL_SCENARIOS)
    )
    parser.add_argument("--methods", default=",".join(PRIMARY_METHODS))
    parser.add_argument("--include-supplementary", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    datasets = [value for value in args.datasets.split(",") if value]
    scenarios = [value for value in args.scenarios.split(",") if value]
    methods = [value for value in args.methods.split(",") if value]
    if args.include_supplementary:
        methods.extend(
            method for method in SUPPLEMENTARY_METHODS if method not in methods
        )
    unknown_datasets = set(datasets).difference(DATASETS)
    unknown_scenarios = set(scenarios).difference(SCENARIOS)
    if unknown_datasets or unknown_scenarios:
        raise ValueError(
            f"Unknown datasets={sorted(unknown_datasets)}, "
            f"scenarios={sorted(unknown_scenarios)}"
        )

    stream_seeds, model_seeds = STAGES[args.stage]
    base = load_config(args.config)
    root = Path(
        args.output_dir or f"outputs_dasp_v16_{args.stage}"
    )
    summaries: list[dict[str, object]] = []
    expected_runs = (
        len(datasets)
        * len(scenarios)
        * len(stream_seeds)
        * len(model_seeds)
        * len(methods)
    )
    completed = 0
    for dataset in datasets:
        for scenario in scenarios:
            artifact = (
                Path(args.prepared_dir)
                / dataset
                / scenario
            )
            for stream_seed in stream_seeds:
                stream_path = artifact / f"stream_seed{stream_seed}.npz"
                if not stream_path.exists():
                    raise FileNotFoundError(
                        f"Missing prepared stream: {stream_path}"
                    )
                for model_seed in model_seeds:
                    for method in methods:
                        config = copy.deepcopy(base)
                        config.data.path = str(stream_path)
                        config.data.scenario = scenario
                        config.data.stream_seed = stream_seed
                        config.data.random_seed = stream_seed
                        config.training.model_seed = model_seed
                        config.dasp.baseline = method
                        config.output.directory = str(root / dataset / scenario)
                        config.output.run_name = (
                            f"{method}_stream{stream_seed}_model{model_seed}"
                        )
                        run_dir = (
                            Path(config.output.directory)
                            / config.output.run_name
                        )
                        if args.resume and run_dir.exists():
                            summary = _read_cached_summary(
                                run_dir, config.to_dict()
                            )
                            state = "resumed"
                        else:
                            result = StreamingExperiment(config).run()
                            summary = result.summary
                            state = "completed"
                        summary = dict(summary)
                        summary.update(
                            {
                                "dataset": dataset,
                                "stage": args.stage,
                                "stream_seed": stream_seed,
                                "model_seed": model_seed,
                                "run_dir": str(run_dir),
                            }
                        )
                        summaries.append(summary)
                        completed += 1
                        print(
                            f"{state} {completed}/{expected_runs} "
                            f"dataset={dataset} scenario={scenario} "
                            f"stream={stream_seed} model={model_seed} "
                            f"method={method} "
                            f"macro_f1={summary['mean_macro_f1']:.6f}",
                            flush=True,
                        )
                        pd.DataFrame(summaries).to_csv(
                            root / "summary_by_run.partial.csv",
                            index=False,
                        )
    frame = pd.DataFrame(summaries)
    if len(frame) != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} runs, completed {len(frame)}"
        )
    key = ["dataset", "scenario", "stream_seed", "model_seed", "method"]
    if frame.duplicated(key).any():
        raise ValueError("Duplicate run cells detected")
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "summary_by_run.csv", index=False)
    (root / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "stage": args.stage,
                "datasets": datasets,
                "scenarios": scenarios,
                "stream_seeds": stream_seeds,
                "model_seeds": model_seeds,
                "methods": methods,
                "expected_runs": expected_runs,
                "completed_runs": len(frame),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"matrix complete: {root}", flush=True)


if __name__ == "__main__":
    main()
