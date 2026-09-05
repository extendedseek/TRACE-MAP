"""Protocols and requests for pluggable language backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from trace_map.types import ReasoningMode, Regime, StructuredClaim


@dataclass
class ReasoningRequest:
    agent_id: str
    mode: ReasoningMode
    local_observation: np.ndarray
    public: dict[str, float]
    previous_public: dict[str, float]
    regime: Regime
    memory_summaries: list[str] = field(default_factory=list)
    textual_policy: str = ""
    max_tokens: int = 256


@dataclass
class MessageRequest:
    sender_id: str
    reasoning: str
    public: dict[str, float]
    previous_public: dict[str, float]
    regime: Regime
    sender_action_dim: int
    receiver_action_dim: int
    horizon: int
    max_tokens: int = 96
    temperature: float = 0.6
    top_p: float = 0.9


@dataclass
class GeneratedStatement:
    text: str
    claim: StructuredClaim


class TextEncoder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LanguagePolicy(Protocol):
    def reason(self, request: ReasoningRequest) -> str: ...

    def generate_statements(self, request: MessageRequest, count: int) -> list[GeneratedStatement]: ...

    def propose_revision(
        self, textual_policy: str, trajectory: list[dict[str, Any]], credits: dict[str, float]
    ) -> str: ...

    def consolidate_revisions(self, textual_policy: str, proposals: list[str]) -> str: ...
