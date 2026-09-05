"""Trajectory-level language attribution and slow textual-policy revision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trace_map.language.base import LanguagePolicy


@dataclass
class LanguageCredit:
    agent_id: str
    credits: dict[str, float]
    individual_return: float
    social_return: float


@dataclass
class TextualPolicyStore:
    policies: dict[str, str]
    revision_history: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for agent_id in self.policies:
            self.revision_history.setdefault(agent_id, [])


class CentralizedLanguageCritic:
    """Assign credit from matched removals, or an auditable proxy when unavailable.

    Exact numerical reproduction should provide the original centralized-critic
    targets. The fallback never uses future information during action selection;
    it runs only after a complete trajectory has ended.
    """

    def __init__(self, individual_weight: float = 0.6, social_weight: float = 0.4):
        if not np.isclose(individual_weight + social_weight, 1.0):
            raise ValueError("Language-credit weights must sum to one")
        self.individual_weight = individual_weight
        self.social_weight = social_weight

    def assign(
        self,
        agent_id: str,
        trajectory: list[dict[str, Any]],
        individual_return: float,
        social_return: float,
        matched_removals: dict[str, tuple[float, float]] | None = None,
    ) -> LanguageCredit:
        if matched_removals:
            baseline = self.individual_weight * individual_return + self.social_weight * social_return
            credits = {
                decision_id: baseline
                - (self.individual_weight * removed_individual + self.social_weight * removed_social)
                for decision_id, (removed_individual, removed_social) in matched_removals.items()
            }
            return LanguageCredit(agent_id, credits, individual_return, social_return)

        credits: dict[str, list[float]] = {
            "reasoning": [],
            "communication": [],
            "commitment": [],
            "memory": [],
        }
        for record in trajectory:
            if record.get("agent_id") != agent_id:
                continue
            attribution = record.get("attribution", {})
            credits["memory"].extend(float(value) for value in attribution.values())
            for audit in record.get("received_audits", []):
                risk = float(audit.get("strategic_risk", 0.0))
                trust = float(audit.get("trust_weight", 0.0))
                harmful = bool(audit.get("harmful", False))
                credits["communication"].append(trust * (-1.0 if harmful else 1.0))
                credits["commitment"].append(1.0 - risk)
            mode = record.get("reasoning_mode", "inactive")
            if mode != "inactive":
                credits["reasoning"].append(float(record.get("step_reward", 0.0)))
        reduced = {
            category: float(np.mean(values)) if values else 0.0
            for category, values in credits.items()
        }
        return LanguageCredit(agent_id, reduced, individual_return, social_return)


class TextualPolicyReviser:
    def __init__(self, language_policy: LanguagePolicy, store: TextualPolicyStore):
        self.language_policy = language_policy
        self.store = store

    def revise_batch(
        self,
        trajectories: list[list[dict[str, Any]]],
        credits: list[LanguageCredit],
    ) -> dict[str, str]:
        by_agent: dict[str, list[tuple[list[dict[str, Any]], LanguageCredit]]] = {}
        for trajectory, credit in zip(trajectories, credits):
            by_agent.setdefault(credit.agent_id, []).append((trajectory, credit))
        updated: dict[str, str] = {}
        for agent_id, examples in by_agent.items():
            current = self.store.policies[agent_id]
            proposals = [
                self.language_policy.propose_revision(current, trajectory, credit.credits)
                for trajectory, credit in examples
            ]
            revised = self.language_policy.consolidate_revisions(current, proposals)
            self.store.policies[agent_id] = revised
            self.store.revision_history[agent_id].append(revised)
            updated[agent_id] = revised
        return updated
