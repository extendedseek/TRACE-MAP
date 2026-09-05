"""Attribution-guided reflective memory (manuscript Eq. 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from trace_map.types import Regime
from trace_map.utils import cosine_similarity, sigmoid, softmax


@dataclass
class MemoryItem:
    memory_id: str
    event_summary: str
    regime: Regime
    reasoning: str
    communication: str
    action: np.ndarray
    realized_return: float
    reliability: float
    stored_step: int
    embedding: np.ndarray
    use_count: int = 0
    attribution_history: list[float] = field(default_factory=list)


@dataclass
class MemorySelection:
    items: list[MemoryItem]
    scores: np.ndarray
    probabilities: np.ndarray

    @property
    def ids(self) -> list[str]:
        return [item.memory_id for item in self.items]


def regime_compatibility(current: Regime, stored: Regime) -> float:
    """Smooth similarity normalized by the E1--E3 intervention span."""

    scale = np.asarray([0.055, 0.045, 0.050], dtype=np.float64)
    distance = np.abs(current.vector().astype(np.float64) - stored.vector().astype(np.float64))
    return float(np.exp(-np.mean(distance / scale)))


class RegimeAwareMemoryBank:
    def __init__(self, config: dict[str, Any], seed: int = 0):
        self.config = config
        memory_cfg = config["memory"]
        self.capacity = int(memory_cfg["capacity_per_agent"])
        self.candidate_count = int(memory_cfg["candidate_count"])
        self.selected_count = int(memory_cfg["selected_count"])
        self.reliability_ema = float(memory_cfg["reliability_ema"])
        self.semantic_weight = float(memory_cfg.get("semantic_weight", 1.0))
        self.regime_weight = float(memory_cfg["regime_weight"])
        self.reliability_weight = float(memory_cfg["reliability_weight"])
        self.age_weight = float(memory_cfg["age_weight"])
        self.gumbel_temperature = float(memory_cfg["gumbel_temperature"])
        self.unused_penalty = float(memory_cfg.get("unused_penalty", 0.02))
        self.ablation = config.get("ablation", {})
        self.rng = np.random.default_rng(seed)
        self.items: list[MemoryItem] = []

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: MemoryItem, merge_threshold: float = 0.94) -> None:
        item.embedding = np.asarray(item.embedding, dtype=np.float32).reshape(-1)
        item.action = np.asarray(item.action, dtype=np.float32).reshape(-1)
        item.reliability = float(np.clip(item.reliability, 0.0, 1.0))
        for existing in self.items:
            if (
                regime_compatibility(existing.regime, item.regime) >= 0.90
                and cosine_similarity(existing.embedding, item.embedding) >= merge_threshold
            ):
                total_uses = max(existing.use_count + item.use_count, 1)
                combined = existing.embedding + item.embedding
                existing.embedding = combined / max(float(np.linalg.norm(combined)), 1e-12)
                existing.realized_return = 0.5 * (existing.realized_return + item.realized_return)
                existing.reliability = 0.5 * (existing.reliability + item.reliability)
                existing.stored_step = max(existing.stored_step, item.stored_step)
                existing.use_count = total_uses
                if item.event_summary not in existing.event_summary:
                    existing.event_summary = f"{existing.event_summary} | {item.event_summary}"
                return
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda memory: (memory.reliability, memory.stored_step))
            del self.items[: len(self.items) - self.capacity]

    def retrieve(
        self,
        query_embedding: np.ndarray,
        regime: Regime,
        step: int,
        training: bool = False,
    ) -> MemorySelection:
        if not self.ablation.get("memory", True) or not self.items:
            return MemorySelection([], np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32))
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        semantic = np.asarray(
            [cosine_similarity(query, item.embedding) for item in self.items], dtype=np.float64
        )
        dense_count = min(self.candidate_count, len(self.items))
        dense_indices = np.argsort(-semantic, kind="stable")[:dense_count]
        combined: list[float] = []
        for index in dense_indices:
            item = self.items[int(index)]
            score = self.semantic_weight * semantic[index]
            if self.ablation.get("regime_compatibility", True):
                score += self.regime_weight * regime_compatibility(regime, item.regime)
            if self.ablation.get("memory_reliability", True):
                score += self.reliability_weight * item.reliability
            if self.ablation.get("staleness_penalty", True):
                age = max(0, int(step) - int(item.stored_step))
                score -= self.age_weight * np.log1p(age)
            combined.append(float(score))
        scores = np.asarray(combined, dtype=np.float64)
        probabilities = softmax(scores, self.gumbel_temperature)
        keep = min(self.selected_count, dense_count)
        if training and keep < dense_count:
            uniforms = np.clip(self.rng.uniform(size=dense_count), 1e-9, 1 - 1e-9)
            gumbel = -np.log(-np.log(uniforms))
            chosen_local = np.argsort(-(scores + gumbel) / self.gumbel_temperature)[:keep]
        else:
            chosen_local = np.argsort(-scores, kind="stable")[:keep]
        selected_items = [self.items[int(dense_indices[index])] for index in chosen_local]
        selected_scores = scores[chosen_local].astype(np.float32)
        selected_probabilities = probabilities[chosen_local].astype(np.float32)
        for item in selected_items:
            item.use_count += 1
        return MemorySelection(selected_items, selected_scores, selected_probabilities)

    @staticmethod
    def pool(selection: MemorySelection, dimension: int) -> np.ndarray:
        if not selection.items:
            return np.zeros(dimension, dtype=np.float32)
        embeddings = np.stack([item.embedding for item in selection.items]).astype(np.float32)
        if embeddings.shape[1] != dimension:
            raise ValueError("Memory embedding dimension does not match the policy encoder")
        weights = softmax(selection.scores, 1.0).astype(np.float32)
        return np.sum(embeddings * weights[:, None], axis=0)

    def apply_removal_attribution(
        self,
        selection: MemorySelection,
        full_value: float,
        removal_values: dict[str, float],
        used_memory_ids: Iterable[str] | None = None,
    ) -> dict[str, float]:
        if not self.ablation.get("memory_attribution", True):
            return {}
        used = set(used_memory_ids or selection.ids)
        attributions: dict[str, float] = {}
        for item in selection.items:
            if item.memory_id in used:
                contribution = float(full_value - removal_values.get(item.memory_id, full_value))
                target = sigmoid(5.0 * contribution)
            else:
                contribution = -self.unused_penalty
                target = 0.0
            item.reliability = float(
                np.clip(
                    self.reliability_ema * item.reliability
                    + (1.0 - self.reliability_ema) * target,
                    0.0,
                    1.0,
                )
            )
            item.attribution_history.append(contribution)
            attributions[item.memory_id] = contribution
        return attributions

    def state_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "memory_id": item.memory_id,
                "event_summary": item.event_summary,
                "regime": {
                    "name": item.regime.name,
                    "depreciation_rate": item.regime.depreciation_rate,
                    "consumption_tax_rate": item.regime.consumption_tax_rate,
                    "interest_rate": item.regime.interest_rate,
                },
                "reasoning": item.reasoning,
                "communication": item.communication,
                "action": item.action.tolist(),
                "realized_return": item.realized_return,
                "reliability": item.reliability,
                "stored_step": item.stored_step,
                "embedding": item.embedding.tolist(),
                "use_count": item.use_count,
                "attribution_history": item.attribution_history,
            }
            for item in self.items
        ]

    def load_state_dict(self, state: list[dict[str, Any]]) -> None:
        self.items.clear()
        for payload in state:
            regime = Regime(**payload["regime"])
            self.items.append(
                MemoryItem(
                    memory_id=payload["memory_id"],
                    event_summary=payload["event_summary"],
                    regime=regime,
                    reasoning=payload["reasoning"],
                    communication=payload["communication"],
                    action=np.asarray(payload["action"], dtype=np.float32),
                    realized_return=float(payload["realized_return"]),
                    reliability=float(payload["reliability"]),
                    stored_step=int(payload["stored_step"]),
                    embedding=np.asarray(payload["embedding"], dtype=np.float32),
                    use_count=int(payload.get("use_count", 0)),
                    attribution_history=list(payload.get("attribution_history", [])),
                )
            )
