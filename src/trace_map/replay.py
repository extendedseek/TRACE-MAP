"""Fixed-capacity replay buffer for multi-agent transitions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trace_map.types import Transition


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self._storage: list[Transition] = []
        self._cursor = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: Transition) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, batch_size: int) -> Sequence[Transition]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._storage):
            raise ValueError("Cannot sample more transitions than the replay buffer contains")
        indices = self.rng.choice(len(self._storage), size=batch_size, replace=False)
        return [self._storage[int(index)] for index in indices]
