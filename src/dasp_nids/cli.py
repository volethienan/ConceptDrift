from __future__ import annotations

import argparse
import json

from .analysis import run_comparison
from .config import load_config
from .runner import METHODS, PRIMARY_METHODS, StreamingExperiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DASP-NIDS streaming experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one method and seed")
    run.add_argument("--config", required=True)
    run.add_argument("--output-dir")
    run.add_argument("--seed", type=int)
    compare = subparsers.add_parser("compare", help="run paired methods and seeds")
    compare.add_argument("--config", required=True)
    compare.add_argument("--output-dir")
    compare.add_argument(
        "--methods",
        nargs="+",
        default=PRIMARY_METHODS,
        choices=sorted(METHODS),
    )
    compare.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    compare.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed runs only when their resolved configs match exactly",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.output.directory = args.output_dir
    if args.command == "run" and args.seed is not None:
        config.data.random_seed = args.seed
    if args.command == "run":
        result = StreamingExperiment(config).run()
        print(json.dumps(result.summary, indent=2, ensure_ascii=False))
        print(result.output_dir)
    else:
        output = run_comparison(config, args.methods, args.seeds, resume=args.resume)
        print(output)


if __name__ == "__main__":
    main()
