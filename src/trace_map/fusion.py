"""Trust-gated pooling and local policy feature construction (Eq. 7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from trace_map.types import CandidateMessage, CounterfactualAudit


@dataclass
class FusedFeatures:
    actor_input: np.ndarray
    language_context: np.ndarray
    communication: np.ndarray
    belief: np.ndarray


class BeliefProjector:
    def __init__(self, input_dimension: int, output_dimension: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        matrix = rng.normal(
            0.0, 1.0 / np.sqrt(input_dimension), (input_dimension, output_dimension)
        )
        self.matrix = matrix.astype(np.float32)

    def __call__(self, belief: np.ndarray) -> np.ndarray:
        vector = np.asarray(belief, dtype=np.float32).reshape(-1)
        if vector.size != self.matrix.shape[0]:
            raise ValueError("Opponent belief feature has an unexpected dimension")
        projected = np.tanh(vector @ self.matrix)
        return projected.astype(np.float32)


def trusted_message_pool(
    messages: Iterable[tuple[CandidateMessage, CounterfactualAudit]],
    dimension: int,
    trust_gated: bool = True,
    epsilon: float = 1e-8,
) -> np.ndarray:
    entries = list(messages)
    if not entries:
        return np.zeros(dimension, dtype=np.float32)
    embeddings = np.stack([candidate.embedding for candidate, _ in entries]).astype(np.float32)
    if embeddings.shape[1] != dimension:
        raise ValueError("Message embedding dimension does not match fusion dimension")
    if trust_gated:
        weights = np.asarray([audit.trust_weight for _, audit in entries], dtype=np.float32)
    else:
        weights = np.ones(len(entries), dtype=np.float32)
    denominator = float(np.sum(weights)) + epsilon
    return np.sum(embeddings * weights[:, None], axis=0) / denominator


class PolicyFusion:
    def __init__(self, config: dict[str, Any], seed: int = 0):
        self.dimension = int(config["language"]["embedding_dim"])
        profile_count = int(config["opponent"]["profile_count"])
        self.belief_projector = BeliefProjector(profile_count + 1, self.dimension, seed)
        self.trust_gated = bool(config.get("ablation", {}).get("trust_gated_fusion", True))

    def build(
        self,
        local_observation: np.ndarray,
        reasoning_embedding: np.ndarray,
        memory_embedding: np.ndarray,
        received_messages: Iterable[tuple[CandidateMessage, CounterfactualAudit]],
        belief_feature: np.ndarray,
    ) -> FusedFeatures:
        observation = np.asarray(local_observation, dtype=np.float32).reshape(-1)
        reasoning = np.asarray(reasoning_embedding, dtype=np.float32).reshape(-1)
        memory = np.asarray(memory_embedding, dtype=np.float32).reshape(-1)
        if reasoning.size != self.dimension or memory.size != self.dimension:
            raise ValueError("Reasoning and memory representations must match embedding_dim")
        communication = trusted_message_pool(
            received_messages, self.dimension, trust_gated=self.trust_gated
        )
        belief = self.belief_projector(belief_feature)
        language = np.concatenate([reasoning, memory, communication, belief]).astype(np.float32)
        actor_input = np.concatenate([observation, language]).astype(np.float32)
        return FusedFeatures(actor_input, language, communication, belief)
