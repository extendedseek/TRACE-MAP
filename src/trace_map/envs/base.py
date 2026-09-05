"""Common environment contract for TaxAI and the offline test economy."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from trace_map.types import ActionSpec, ObservationBundle, StepResult


class MultiAgentEconomy(ABC):
    @property
    @abstractmethod
    def agent_ids(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_specs(self) -> dict[str, ActionSpec]:
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed: int | None = None) -> ObservationBundle:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: dict[str, np.ndarray]) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def sample_actions(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        return {
            agent_id: rng.uniform(spec.low, spec.high).astype(np.float32)
            for agent_id, spec in self.action_specs.items()
        }
