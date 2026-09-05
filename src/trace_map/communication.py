"""Structured strategic communication and counterfactual credibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from trace_map.types import CandidateMessage, CounterfactualAudit, StructuredClaim
from trace_map.utils import cosine_similarity, sigmoid, softmax


@dataclass
class CounterfactualValues:
    receiver_baseline: float
    receiver_cooperate: float
    receiver_sender_deviates: float
    sender_cooperate: float
    sender_deviates: float


CounterfactualEvaluator = Callable[
    [StructuredClaim, np.ndarray, str], CounterfactualValues
]


class FactualVerifier:
    @staticmethod
    def support(
        claim: StructuredClaim,
        current_public: dict[str, float],
        previous_public: dict[str, float],
    ) -> float:
        if claim.variable not in current_public:
            return 0.5 * claim.extraction_confidence
        current = float(current_public[claim.variable])
        previous = float(previous_public.get(claim.variable, current))
        relative = (current - previous) / (abs(previous) + 1e-6)
        observed = "stable" if abs(relative) <= 0.01 else ("increase" if relative > 0 else "decrease")
        if claim.direction == "unknown":
            agreement = 0.5
        elif claim.direction == observed:
            agreement = 1.0
        elif observed == "stable" or claim.direction == "stable":
            agreement = 0.35
        else:
            agreement = 0.05
        return float(np.clip(agreement * claim.extraction_confidence, 0.0, 1.0))


def commitment_consistency(
    declared: np.ndarray | None, predicted: np.ndarray | None
) -> float:
    if declared is None or predicted is None:
        return 0.5
    declared_array = np.asarray(declared, dtype=np.float64).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if declared_array.size != predicted_array.size:
        return 0.0
    normalized_distance = float(np.mean(np.abs(declared_array - predicted_array)) / 2.0)
    return float(np.exp(-3.0 * normalized_distance))


def trust_weight(
    factual_support: float,
    strategic_risk: float,
    posterior_certainty: float,
    relevance: float,
) -> float:
    return float(
        np.clip(
            factual_support
            * (1.0 - strategic_risk)
            * posterior_certainty
            * relevance,
            0.0,
            1.0,
        )
    )


class CounterfactualCredibility:
    def __init__(self, config: dict[str, Any]):
        communication = config["communication"]
        self.risk_weights = communication["risk_weights"]
        self.quality_weights = communication["quality_weights"]
        self.trust_threshold = float(communication["trust_threshold"])
        self.max_message_tokens = int(config["language"]["message_max_tokens"])
        self.ablation = config.get("ablation", {})

    @staticmethod
    def _proxy_values(
        claim: StructuredClaim,
        predicted_sender_action: np.ndarray,
        factual_support: float,
    ) -> CounterfactualValues:
        declared = (
            np.asarray(claim.sender_commitment, dtype=np.float64).reshape(-1)
            if claim.sender_commitment is not None
            else np.zeros_like(predicted_sender_action, dtype=np.float64)
        )
        predicted = np.asarray(predicted_sender_action, dtype=np.float64).reshape(-1)
        if declared.size != predicted.size:
            mismatch = 1.0
        else:
            mismatch = float(np.clip(np.mean(np.abs(declared - predicted)) / 2.0, 0.0, 1.0))
        recommendation = (
            np.asarray(claim.receiver_recommendation, dtype=np.float64).reshape(-1)
            if claim.receiver_recommendation is not None
            else np.zeros(1, dtype=np.float64)
        )
        receiver_exposure = float(np.clip(np.mean(np.abs(recommendation)), 0.0, 1.0))
        baseline = 0.0
        cooperate = 0.20 * factual_support - 0.05 * receiver_exposure
        switched = cooperate - mismatch * (0.30 + 0.70 * receiver_exposure)
        sender_cooperate = 0.10 * factual_support
        sender_deviates = sender_cooperate + mismatch * (0.25 + 0.50 * receiver_exposure)
        return CounterfactualValues(
            baseline,
            cooperate,
            switched,
            sender_cooperate,
            sender_deviates,
        )

    def audit(
        self,
        candidate: CandidateMessage,
        receiver_id: str,
        current_public: dict[str, float],
        previous_public: dict[str, float],
        predicted_sender_action: np.ndarray,
        posterior_certainty: float,
        relevance: float,
        influence: float,
        evaluator: CounterfactualEvaluator | None = None,
    ) -> CounterfactualAudit:
        factual = FactualVerifier.support(candidate.claim, current_public, previous_public)
        if not self.ablation.get("factuality", True):
            factual_for_policy = 1.0
        else:
            factual_for_policy = factual
        consistency = commitment_consistency(
            candidate.claim.sender_commitment, predicted_sender_action
        )
        if not self.ablation.get("commitment_consistency", True):
            consistency_for_policy = 1.0
        else:
            consistency_for_policy = consistency
        values = (
            evaluator(candidate.claim, predicted_sender_action, receiver_id)
            if evaluator is not None
            else self._proxy_values(candidate.claim, predicted_sender_action, factual)
        )
        bait = float(values.receiver_cooperate - values.receiver_baseline)
        switch = float(max(0.0, values.receiver_cooperate - values.receiver_sender_deviates))
        edge = float(max(0.0, values.sender_deviates - values.sender_cooperate))
        harmful = bool(switch > max(bait, 0.0) + 1e-9)

        if self.ablation.get("counterfactual_credibility", True):
            weights = self.risk_weights
            risk_logit = (
                float(weights["switch"]) * switch
                + float(weights["edge"]) * edge
                + float(weights["inconsistency"]) * (1.0 - consistency_for_policy)
                - float(weights["bait"]) * max(bait, 0.0)
                - 1.0
            )
            risk = sigmoid(candidate.claim.extraction_confidence * risk_logit)
        else:
            risk = 0.0
        effective_influence = (
            influence * factual_for_policy * (1.0 - risk)
            if self.ablation.get("credibility_gated_influence", True)
            else influence
        )
        weight = trust_weight(factual_for_policy, risk, posterior_certainty, relevance)
        return CounterfactualAudit(
            sender_id=candidate.sender_id,
            receiver_id=receiver_id,
            candidate_id=candidate.candidate_id,
            factual_support=factual_for_policy,
            bait=bait,
            switch=switch,
            edge=edge,
            commitment_consistency=consistency_for_policy,
            strategic_risk=risk,
            influence=float(np.clip(effective_influence, 0.0, 1.0)),
            relevance=float(np.clip(relevance, 0.0, 1.0)),
            posterior_certainty=float(np.clip(posterior_certainty, 0.0, 1.0)),
            trust_weight=weight,
            harmful=harmful,
        )

    def quality(
        self,
        candidate: CandidateMessage,
        audits: Iterable[CounterfactualAudit],
        history_embeddings: Iterable[np.ndarray] = (),
    ) -> float:
        audit_list = list(audits)
        if not audit_list:
            return -np.inf
        factual = float(np.mean([audit.factual_support for audit in audit_list]))
        influence = float(np.mean([audit.influence for audit in audit_list]))
        risk = float(np.mean([audit.strategic_risk for audit in audit_list]))
        similarities = [
            max(0.0, cosine_similarity(candidate.embedding, embedding))
            for embedding in history_embeddings
        ]
        redundancy = max(similarities, default=0.0)
        length = min(len(candidate.text.split()) / max(self.max_message_tokens, 1), 2.0)
        weights = self.quality_weights
        value = (
            float(weights["factuality"]) * factual
            + float(weights["influence"]) * influence
            - float(weights["strategic_risk"]) * risk
            - float(weights["redundancy"]) * redundancy
            - float(weights["length"]) * length
        )
        candidate.quality = float(value)
        return float(value)

    def select(
        self,
        candidates: list[CandidateMessage],
        audits_by_candidate: dict[str, list[CounterfactualAudit]],
        history_embeddings: Iterable[np.ndarray] = (),
        training: bool = False,
        temperature: float = 0.7,
        rng: np.random.Generator | None = None,
    ) -> CandidateMessage:
        if not candidates:
            raise ValueError("At least one candidate message is required")
        history = list(history_embeddings)
        qualities = np.asarray(
            [self.quality(item, audits_by_candidate.get(item.candidate_id, []), history) for item in candidates]
        )
        if not np.any(np.isfinite(qualities)):
            raise ValueError("No candidate has a receiver audit")
        probabilities = softmax(qualities, temperature)
        if training:
            generator = rng or np.random.default_rng()
            index = int(generator.choice(len(candidates), p=probabilities))
        else:
            index = int(np.argmax(qualities))
        return candidates[index]


class CommunicationPerturbation:
    """Applies C1 factual corruption, C2 deviations, or C3 persistent reliability."""

    def __init__(self, config: dict[str, Any], seed: int):
        cfg = config["communication"]
        self.condition = str(cfg["condition"]).lower()
        self.factual_corruption_rate = float(cfg["factual_corruption_rate"])
        self.strategic_deviation_rate = float(cfg["strategic_deviation_rate"])
        self.persistent_reliability = [float(value) for value in cfg.get("persistent_reliability", [])]
        self.rng = np.random.default_rng(seed)

    def _agent_reliability(self, agent_id: str) -> float:
        if not agent_id.startswith("household_") or not self.persistent_reliability:
            return 1.0
        index = int(agent_id.rsplit("_", 1)[1])
        return self.persistent_reliability[index]

    def corrupt_claim(self, candidate: CandidateMessage) -> CandidateMessage:
        rate = self.factual_corruption_rate
        if self.condition == "c3":
            rate = 1.0 - self._agent_reliability(candidate.sender_id)
        if self.condition not in {"c1", "c3"} or self.rng.random() >= rate:
            return candidate
        opposite = {"increase": "decrease", "decrease": "increase"}
        original = candidate.claim.direction
        candidate.claim.direction = opposite.get(original, "increase")
        if original in candidate.text.lower():
            candidate.text = candidate.text.replace(original, candidate.claim.direction, 1)
        return candidate

    def maybe_deviate(
        self,
        agent_id: str,
        action: np.ndarray,
        declared_commitment: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        if self.condition == "c2":
            rate = self.strategic_deviation_rate
        elif self.condition == "c3":
            rate = 1.0 - self._agent_reliability(agent_id)
        else:
            rate = 0.0
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.rng.random() >= rate:
            return action_array, False
        if declared_commitment is not None and np.asarray(declared_commitment).size == action_array.size:
            deviated = -np.asarray(declared_commitment, dtype=np.float32).reshape(-1)
        else:
            deviated = -action_array
        return np.clip(deviated, -1.0, 1.0), True
