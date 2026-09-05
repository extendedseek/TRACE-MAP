"""Dependency-light smoke evaluation and seed-log aggregation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from trace_map.config import config_digest, save_config
from trace_map.envs import make_environment
from trace_map.language import make_language_components
from trace_map.metrics import (
    aggregate_seed_metrics,
    brier_score,
    recovery_time,
    trusted_harmful_message_rate,
    useful_at_k,
)
from trace_map.pipeline import TraceMapPipeline
from trace_map.types import CounterfactualAudit
from trace_map.utils import append_jsonl, atomic_write_json, run_metadata, seed_everything


def run_smoke(config: dict[str, Any], output_directory: str | Path) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["run"]["seed"])
    seed_everything(seed, bool(config["run"].get("deterministic", True)))
    save_config(config, output / "resolved_config.yaml")
    metadata = {**run_metadata(), "config_sha256": config_digest(config), "mode": "smoke"}
    atomic_write_json(output / "run_metadata.json", metadata)

    environment = make_environment(config)
    language_policy, encoder = make_language_components(config)
    pipeline = TraceMapPipeline(config, environment, language_policy, encoder)
    episode_count = int(config.get("evaluation", {}).get("episodes", 1))
    all_audits: list[CounterfactualAudit] = []
    rewards_by_agent = {agent_id: [] for agent_id in environment.agent_ids}
    social_welfare_series: list[float] = []
    attribution_values: list[float] = []
    decision_latencies_ms: list[float] = []
    total_steps = 0

    try:
        for episode in range(episode_count):
            pipeline.reset_episode()
            observation = environment.reset(seed + episode)
            done = False
            while not done:
                start = time.perf_counter()
                prepared = pipeline.prepare(observation, training=False)
                proposed_actions = pipeline.heuristic_actions(prepared)
                actual_actions = pipeline.apply_strategic_deviations(proposed_actions, prepared)
                result = environment.step(actual_actions)
                records = pipeline.finalize_step(prepared, result)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                decision_latencies_ms.append(elapsed_ms)
                all_audits.extend(prepared.audits)
                for agent_id, reward in result.rewards.items():
                    rewards_by_agent[agent_id].append(float(reward))
                social_welfare_series.append(float(result.info.get("social_welfare", 0.0)))
                for record in records:
                    attribution_values.extend(float(value) for value in record["attribution"].values())
                    append_jsonl(
                        output / "decision_trace.jsonl",
                        {"episode": episode, **record},
                    )
                append_jsonl(
                    output / "steps.jsonl",
                    {
                        "episode": episode,
                        "step": observation.step,
                        "regime": observation.regime.name,
                        "rewards": result.rewards,
                        "info": result.info,
                        "audit_count": len(prepared.audits),
                        "latency_ms": elapsed_ms,
                    },
                )
                observation = result.observation
                done = bool(result.terminated or result.truncated)
                total_steps += 1
    finally:
        environment.close()

    household_ids = [agent for agent in rewards_by_agent if agent.startswith("household_")]
    household_returns = [sum(rewards_by_agent[agent]) for agent in household_ids]
    predicted_safe = [1.0 - audit.strategic_risk for audit in all_audits]
    safe_labels = [not audit.harmful for audit in all_audits]
    shift = config["environment"].get("shift", {})
    recovery = None
    if shift.get("enabled", False):
        recovery = recovery_time(
            social_welfare_series,
            int(shift["step"]),
            int(config["evaluation"]["recovery_window"]),
            float(config["evaluation"]["recovery_fraction"]),
        )
    metrics: dict[str, Any] = {
        "mode": "smoke",
        "seed": seed,
        "episodes": episode_count,
        "steps": total_steps,
        "average_household_return": float(np.mean(household_returns)) if household_returns else 0.0,
        "government_return": float(sum(rewards_by_agent.get("government", []))),
        "mean_social_welfare": float(np.mean(social_welfare_series)) if social_welfare_series else 0.0,
        "brier_score": brier_score(predicted_safe, safe_labels) if all_audits else None,
        "thmr": trusted_harmful_message_rate(
            all_audits, float(config["communication"]["trust_threshold"])
        ),
        "useful_at_k_proxy": useful_at_k(attribution_values),
        "recovery_time": recovery,
        "mean_decision_latency_ms": float(np.mean(decision_latencies_ms)),
        "p95_decision_latency_ms": float(np.quantile(decision_latencies_ms, 0.95)),
        "memory_sizes": {agent: len(bank) for agent, bank in pipeline.memory_banks.items()},
        "note": "Engineering smoke metrics are not comparable to manuscript results.",
    }
    atomic_write_json(output / "metrics.json", metrics)
    atomic_write_json(output / "pipeline_state.json", pipeline.state_dict())
    return metrics


def aggregate_directory(input_directory: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(input_directory)
    output = Path(output_path).resolve()
    records: list[dict[str, float]] = []
    sources: list[str] = []
    for candidate in sorted(root.rglob("metrics.json")):
        if candidate.resolve() == output:
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        numeric = {
            key: float(value)
            for key, value in payload.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        records.append(numeric)
        sources.append(str(candidate))
    if not records:
        raise FileNotFoundError(f"No metrics.json files found below {root}")
    summary = {"runs": len(records), "sources": sources, "metrics": aggregate_seed_metrics(records)}
    atomic_write_json(output_path, summary)
    return summary
