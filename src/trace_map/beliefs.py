"""Credibility-calibrated probabilistic opponent inference (Eq. 5)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def entropy(probabilities: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(values * np.log(values)))


def certainty(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size <= 1:
        return 1.0
    return float(np.clip(1.0 - entropy(values) / np.log(values.size), 0.0, 1.0))


def action_profile_likelihood(
    action: np.ndarray,
    profile_count: int,
    sigma: float = 0.45,
) -> np.ndarray:
    """Likelihood under ordered latent prototypes spanning conservative to aggressive."""

    observed = np.asarray(action, dtype=np.float64).reshape(-1)
    centers = np.linspace(-0.8, 0.8, profile_count)
    mean_action = float(np.mean(observed)) if observed.size else 0.0
    likelihood = np.exp(-0.5 * ((mean_action - centers) / sigma) ** 2)
    return likelihood / np.maximum(np.sum(likelihood), 1e-12)


class OpponentBeliefModel:
    def __init__(self, agent_ids: list[str], config: dict[str, Any]):
        cfg = config["opponent"]
        self.agent_ids = list(agent_ids)
        self.profile_count = int(cfg["profile_count"])
        self.weak_likelihood_mix = float(cfg.get("weak_likelihood_mix", 1.0))
        self.behavioral_update_strength = float(cfg.get("behavioral_update_strength", 1.0))
        self.ablation = config.get("ablation", {})
        uniform = np.ones(self.profile_count, dtype=np.float64) / self.profile_count
        self.beliefs = {
            (observer, target): uniform.copy()
            for observer in self.agent_ids
            for target in self.agent_ids
            if observer != target
        }

    def posterior(self, observer: str, target: str) -> np.ndarray:
        return self.beliefs[(observer, target)].copy()

    def update_from_message(
        self,
        observer: str,
        target: str,
        type_likelihood: np.ndarray,
        credibility: float,
    ) -> np.ndarray:
        prior = self.beliefs[(observer, target)]
        likelihood = np.asarray(type_likelihood, dtype=np.float64).reshape(-1)
        if likelihood.size != self.profile_count:
            raise ValueError("Type likelihood dimension does not match profile count")
        likelihood = np.maximum(likelihood, 1e-12)
        if not self.ablation.get("opponent_belief", True):
            return prior.copy()
        if self.ablation.get("credibility_gated_influence", True):
            effective = float(np.clip(credibility, 0.0, 1.0)) ** self.weak_likelihood_mix
        else:
            effective = 1.0
        adjusted = effective * likelihood + (1.0 - effective) * np.ones_like(likelihood)
        posterior = adjusted * prior
        posterior /= np.maximum(np.sum(posterior), 1e-12)
        self.beliefs[(observer, target)] = posterior
        return posterior.copy()

    def refine_from_behavior(
        self, observer: str, target: str, observed_action: np.ndarray
    ) -> np.ndarray:
        prior = self.beliefs[(observer, target)]
        if not self.ablation.get("behavioral_refinement", True):
            return prior.copy()
        likelihood = action_profile_likelihood(observed_action, self.profile_count)
        likelihood = np.power(np.maximum(likelihood, 1e-12), self.behavioral_update_strength)
        posterior = likelihood * prior
        posterior /= np.maximum(np.sum(posterior), 1e-12)
        self.beliefs[(observer, target)] = posterior
        return posterior.copy()

    def uncertainty(self, observer: str, target: str) -> float:
        return entropy(self.beliefs[(observer, target)])

    def certainty(self, observer: str, target: str) -> float:
        return certainty(self.beliefs[(observer, target)])

    def aggregate(self, observer: str, targets: Iterable[str] | None = None) -> np.ndarray:
        selected = [target for target in (targets or self.agent_ids) if target != observer]
        if not selected:
            posterior = np.ones(self.profile_count) / self.profile_count
        else:
            posterior = np.mean([self.beliefs[(observer, target)] for target in selected], axis=0)
        return np.concatenate([posterior, [certainty(posterior)]]).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        return {f"{observer}->{target}": value.tolist() for (observer, target), value in self.beliefs.items()}

    def load_state_dict(self, state: dict[str, list[float]]) -> None:
        for key, value in state.items():
            observer, target = key.split("->", 1)
            array = np.asarray(value, dtype=np.float64)
            if array.size != self.profile_count:
                raise ValueError("Stored opponent belief has the wrong dimension")
            self.beliefs[(observer, target)] = array / np.maximum(np.sum(array), 1e-12)
