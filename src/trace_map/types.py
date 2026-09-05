"""Shared typed records used across TRACE-MAP.

The actor-facing :class:`ObservationBundle` contains only per-agent local
observations and public economic context. Training-only privileged variables
belong in environment ``info`` dictionaries and are never passed to the
pipeline's actor feature builder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

import numpy as np


class ReasoningMode(str, Enum):
    INACTIVE = "inactive"
    SHORT = "short"
    LONG = "long"


@dataclass(frozen=True)
class Regime:
    name: str
    depreciation_rate: float
    consumption_tax_rate: float
    interest_rate: float

    def vector(self) -> np.ndarray:
        return np.asarray(
            [self.depreciation_rate, self.consumption_tax_rate, self.interest_rate],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class ActionSpec:
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.high, dtype=np.float32).reshape(-1)
        if low.shape != high.shape or np.any(low >= high):
            raise ValueError("ActionSpec requires matching low/high arrays with low < high")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def dim(self) -> int:
        return int(self.low.size)

    def clip(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32).reshape(-1)
        if value.size != self.dim:
            raise ValueError(f"Expected action dimension {self.dim}, got {value.size}")
        return np.clip(value, self.low, self.high)


@dataclass
class ObservationBundle:
    local: dict[str, np.ndarray]
    global_state: np.ndarray
    public: dict[str, float]
    regime: Regime
    step: int

    def validate(self, agent_ids: list[str]) -> None:
        if set(self.local) != set(agent_ids):
            raise ValueError("Local observation keys do not match environment agent IDs")
        for agent_id, observation in self.local.items():
            array = np.asarray(observation)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"Invalid local observation for {agent_id}")
        if np.asarray(self.global_state).ndim != 1:
            raise ValueError("Global state must be a one-dimensional vector")


@dataclass
class StepResult:
    observation: ObservationBundle
    rewards: dict[str, float]
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredClaim:
    variable: str
    direction: str
    horizon: int
    sender_commitment: np.ndarray | None
    receiver_recommendation: np.ndarray | None
    extraction_confidence: float

    def validate(self) -> None:
        if self.direction not in {"increase", "decrease", "stable", "unknown"}:
            raise ValueError(f"Unsupported claim direction: {self.direction}")
        if self.horizon < 1:
            raise ValueError("Claim horizon must be positive")
        if not 0.0 <= self.extraction_confidence <= 1.0:
            raise ValueError("Extraction confidence must lie in [0, 1]")


@dataclass
class CandidateMessage:
    candidate_id: str
    sender_id: str
    text: str
    claim: StructuredClaim
    embedding: np.ndarray
    quality: float = 0.0


@dataclass
class CounterfactualAudit:
    sender_id: str
    receiver_id: str
    candidate_id: str
    factual_support: float
    bait: float
    switch: float
    edge: float
    commitment_consistency: float
    strategic_risk: float
    influence: float
    relevance: float
    posterior_certainty: float
    trust_weight: float
    harmful: bool


@dataclass
class DecisionTrace:
    step: int
    agent_id: str
    reasoning_mode: str
    reasoning_text: str
    selected_memory_ids: list[str]
    memory_scores: list[float]
    sent_candidate_id: str | None
    received_audits: list[CounterfactualAudit]
    belief_entropy: float
    action: np.ndarray
    regime: str
    attribution: dict[str, float] = field(default_factory=dict)


@dataclass
class Transition:
    global_state: np.ndarray
    local_features: dict[str, np.ndarray]
    language_features: dict[str, np.ndarray]
    actions: dict[str, np.ndarray]
    rewards: dict[str, float]
    next_global_state: np.ndarray
    next_local_features: dict[str, np.ndarray]
    next_language_features: dict[str, np.ndarray]
    done: bool
    auxiliary: dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses and NumPy values to JSON-compatible data."""

    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
