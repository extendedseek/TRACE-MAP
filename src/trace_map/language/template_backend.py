"""Deterministic language and embedding components used by tests."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from trace_map.language.base import (
    GeneratedStatement,
    MessageRequest,
    ReasoningRequest,
)
from trace_map.types import StructuredClaim
from trace_map.utils import stable_hash


class HashingTextEncoder:
    """Signed feature hashing with deterministic L2 normalization."""

    def __init__(self, dimension: int):
        if dimension < 2:
            raise ValueError("HashingTextEncoder dimension must be at least two")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9_]+", text.lower())
            for token in tokens:
                token_hash = stable_hash(token)
                column = token_hash % self.dimension
                sign = 1.0 if (token_hash // self.dimension) % 2 == 0 else -1.0
                output[row, column] += sign
            norm = float(np.linalg.norm(output[row]))
            if norm > 0:
                output[row] /= norm
        return output


def _dominant_change(
    public: dict[str, float], previous: dict[str, float]
) -> tuple[str, str, float]:
    candidates = [
        "output",
        "social_welfare",
        "wage",
        "public_debt",
        "aggregate_expenditure",
        "interest_rate",
        "consumption_tax_rate",
    ]
    best_name = "output"
    best_change = 0.0
    for name in candidates:
        current = float(public.get(name, 0.0))
        old = float(previous.get(name, current))
        change = (current - old) / (abs(old) + 1e-6)
        if abs(change) >= abs(best_change):
            best_name, best_change = name, change
    if abs(best_change) < 0.01:
        direction = "stable"
    else:
        direction = "increase" if best_change > 0 else "decrease"
    return best_name, direction, best_change


class TemplateLanguagePolicy:
    """Auditable fallback that generates economic text without an LLM."""

    def reason(self, request: ReasoningRequest) -> str:
        variable, direction, change = _dominant_change(request.public, request.previous_public)
        memories = (
            " Relevant retained experience: " + request.memory_summaries[0]
            if request.memory_summaries
            else " No sufficiently relevant prior experience is available."
        )
        time_scope = "persistent conditions" if request.mode.value == "long" else "the immediate shock"
        policy = f" Follow this policy: {request.textual_policy}" if request.textual_policy else ""
        return (
            f"Assess {time_scope}: {variable} is {direction} "
            f"(relative change {change:+.3f}) in regime {request.regime.name}."
            f"{memories} Preserve liquidity, avoid an abrupt labor response, and condition any "
            f"coordination on observed commitment behavior.{policy}"
        )

    def generate_statements(self, request: MessageRequest, count: int) -> list[GeneratedStatement]:
        variable, direction, _ = _dominant_change(request.public, request.previous_public)
        pressure = (
            request.regime.interest_rate
            + request.regime.consumption_tax_rate
            + request.regime.depreciation_rate
        )
        base_receiver = np.zeros(request.receiver_action_dim, dtype=np.float32)
        if request.receiver_action_dim >= 1:
            base_receiver[0] = 0.25 if pressure > 0.18 else 0.05
        if request.receiver_action_dim >= 2:
            base_receiver[1] = 0.05
        base_sender = np.zeros(request.sender_action_dim, dtype=np.float32)

        styles = [
            (
                "cautious",
                "Preserve a liquidity buffer and make only gradual adjustments until commitments are observed.",
                base_sender,
                base_receiver,
                0.96,
            ),
            (
                "coordinating",
                "Coordinate a moderate saving response while avoiding a disproportionate labor expansion.",
                np.clip(base_sender + 0.10, -1, 1),
                np.clip(base_receiver + 0.10, -1, 1),
                0.92,
            ),
            (
                "influential",
                "Reduce saving and expand labor immediately to support aggregate demand.",
                np.clip(base_sender + 0.35, -1, 1),
                np.clip(base_receiver - 0.35, -1, 1),
                0.88,
            ),
            (
                "neutral",
                "Monitor the next observations and retain the current policy for now.",
                base_sender,
                np.zeros_like(base_receiver),
                0.90,
            ),
        ]
        statements: list[GeneratedStatement] = []
        for index in range(count):
            style, advice, sender_action, receiver_action, confidence = styles[index % len(styles)]
            text = (
                f"{variable.replace('_', ' ').title()} is expected to remain {direction} over the "
                f"next {request.horizon} periods. {advice} Sender stance: {style}."
            )
            claim = StructuredClaim(
                variable=variable,
                direction=direction,
                horizon=request.horizon,
                sender_commitment=sender_action.copy(),
                receiver_recommendation=receiver_action.copy(),
                extraction_confidence=confidence,
            )
            claim.validate()
            statements.append(GeneratedStatement(text, claim))
        return statements

    def propose_revision(
        self, textual_policy: str, trajectory: list[dict[str, Any]], credits: dict[str, float]
    ) -> str:
        negative = [key for key, value in credits.items() if value < 0]
        positive = [key for key, value in credits.items() if value > 0]
        return (
            "Retain positively attributed decisions "
            f"({', '.join(positive) or 'none'}); reduce reliance on negatively attributed decisions "
            f"({', '.join(negative) or 'none'}). Prioritize regime compatibility, observed commitments, "
            "and receiver loss under sender deviation over factual plausibility alone."
        )

    def consolidate_revisions(self, textual_policy: str, proposals: list[str]) -> str:
        unique: list[str] = []
        for proposal in proposals:
            normalized = " ".join(proposal.split())
            if normalized and normalized not in unique:
                unique.append(normalized)
        if not unique:
            return textual_policy
        prefix = textual_policy.strip()
        additions = " ".join(unique)
        return f"{prefix} {additions}".strip()
