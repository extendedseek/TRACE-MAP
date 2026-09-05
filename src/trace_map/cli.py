"""Command-line interface for TRACE-MAP."""

from __future__ import annotations

import argparse
import json
from typing import Any

from trace_map.config import load_config
from trace_map.evaluate import aggregate_directory, run_smoke


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Base YAML configuration")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="YAML overlay; may be supplied more than once",
    )
    parser.add_argument(
        "--set",
        dest="assignments",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted configuration override; may be supplied more than once",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace-map",
        description="TRACE-MAP language-guided multi-agent economic policy learning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run the dependency-light end-to-end check")
    _common_config(smoke)
    smoke.add_argument("--output", required=True)

    train = subparsers.add_parser("train", help="Train numerical and language-aware components")
    _common_config(train)
    train.add_argument("--output", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint")
    _common_config(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)

    inspect = subparsers.add_parser("inspect-config", help="Validate and print a resolved config")
    _common_config(inspect)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate generated metrics.json files")
    aggregate.add_argument("--input", required=True)
    aggregate.add_argument("--output", required=True)
    return parser


def _load(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, args.override, args.assignments)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "smoke":
        result = run_smoke(_load(args), args.output)
    elif args.command == "train":
        try:
            from trace_map.trainer import TraceMapTrainer
        except ImportError as error:
            parser.error(f"Training dependencies are unavailable: {error}")
        result = TraceMapTrainer(_load(args), args.output).train()
    elif args.command == "evaluate":
        try:
            from trace_map.trainer import TraceMapTrainer
        except ImportError as error:
            parser.error(f"Evaluation dependencies are unavailable: {error}")
        trainer = TraceMapTrainer(_load(args), args.output)
        trainer.load_checkpoint(args.checkpoint)
        result = trainer.evaluate()
    elif args.command == "inspect-config":
        result = _load(args)
    elif args.command == "aggregate":
        result = aggregate_directory(args.input, args.output)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
