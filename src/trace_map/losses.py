"""Fast-timescale objectives (manuscript Eqs. 6, 8, and 11)."""

from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


def value_loss(predicted: torch.Tensor, reward: torch.Tensor, discount: float, target: torch.Tensor) -> torch.Tensor:
    td_target = reward + discount * target
    return F.mse_loss(predicted, td_target.detach())


def actor_loss(action_values: torch.Tensor) -> torch.Tensor:
    return -action_values.mean()


def selector_loss(logits: torch.Tensor, candidate_qualities: torch.Tensor) -> torch.Tensor:
    targets = torch.softmax(candidate_qualities.detach(), dim=-1)
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def credibility_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(predicted.clamp(1e-6, 1 - 1e-6), target)


def influence_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted, target)


def belief_loss(profile_logits: torch.Tensor, profile_targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(profile_logits, profile_targets.long())


def surrogate_loss(predicted: torch.Tensor, centralized_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted, centralized_target.detach())


def deviation_regularizer(
    selected_values: torch.Tensor,
    deviation_values: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Mean sampled unilateral advantage above ``epsilon_dev``.

    ``selected_values`` has shape ``[batch, agents]`` and
    ``deviation_values`` has shape ``[batch, agents, deviations]``.
    """

    if deviation_values.ndim != 3 or selected_values.shape != deviation_values.shape[:2]:
        raise ValueError("Deviation value tensors have incompatible shapes")
    best = deviation_values.max(dim=-1).values
    return torch.relu(best - selected_values - tolerance).mean()


def reinforce_memory_loss(log_probabilities: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    return -(log_probabilities * advantages.detach()).mean()


def weighted_fast_objective(
    components: Mapping[str, torch.Tensor], weights: Mapping[str, float]
) -> torch.Tensor:
    missing = set(components) - set(weights)
    if missing:
        raise KeyError(f"Missing fast-objective weights for: {sorted(missing)}")
    return sum(float(weights[name]) * value for name, value in components.items())
