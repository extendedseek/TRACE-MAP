#!/usr/bin/env python3
"""Print a compact audit view from a TRACE-MAP decision_trace.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    displayed = 0
    with args.trace.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if args.agent and record.get("agent_id") != args.agent:
                continue
            audits = record.get("received_audits", [])
            print(
                json.dumps(
                    {
                        "episode": record.get("episode"),
                        "step": record["step"],
                        "agent": record["agent_id"],
                        "regime": record["regime"],
                        "mode": record["reasoning_mode"],
                        "memories": record["selected_memory_ids"],
                        "messages": [
                            {
                                "sender": audit["sender_id"],
                                "factual": audit["factual_support"],
                                "risk": audit["strategic_risk"],
                                "trust": audit["trust_weight"],
                                "harmful": audit["harmful"],
                            }
                            for audit in audits
                        ],
                        "action": record["action"],
                        "reward": record.get("step_reward"),
                    },
                    indent=2,
                )
            )
            displayed += 1
            if displayed >= args.limit:
                break


if __name__ == "__main__":
    main()
