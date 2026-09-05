#!/usr/bin/env python3
"""Aggregate scenario metrics into CSV and Markdown without copying paper targets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect(root: Path):
    grouped = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.rglob("metrics.json")):
        relative = path.relative_to(root)
        scenario = relative.parts[0] if len(relative.parts) > 1 else "default"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                grouped[scenario][key].append(float(value))
    return grouped


def rows(grouped):
    for scenario in sorted(grouped):
        for metric in sorted(grouped[scenario]):
            values = grouped[scenario][metric]
            yield {
                "scenario": scenario,
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    table = list(rows(collect(args.input)))
    if not table:
        raise SystemExit(f"No metrics.json files found below {args.input}")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(table)
    with args.markdown.open("w", encoding="utf-8") as handle:
        handle.write("| Scenario | Metric | Mean | Std. | n |\n")
        handle.write("| --- | --- | ---: | ---: | ---: |\n")
        for row in table:
            handle.write(
                f"| {row['scenario']} | {row['metric']} | {row['mean']:.6g} | "
                f"{row['std']:.6g} | {row['n']} |\n"
            )


if __name__ == "__main__":
    main()
