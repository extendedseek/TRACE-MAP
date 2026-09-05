"""End-to-end TRACE-MAP decentralized inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from trace_map.beliefs import OpponentBeliefModel, action_profile_likelihood
from trace_map.communication import CommunicationPerturbation, CounterfactualCredibility
from trace_map.envs.base import MultiAgentEconomy
from trace_map.fusion import FusedFeatures, PolicyFusion
from trace_map.language.base import LanguagePolicy, MessageRequest, ReasoningRequest, TextEncoder
from trace_map.memory import MemoryItem, MemorySelection, RegimeAwareMemoryBank
from trace_map.reasoning import ReasoningScheduler
from trace_map.types import (
    CandidateMessage,
    CounterfactualAudit,
    DecisionTrace,
    ObservationBundle,
    ReasoningMode,
    StepResult,
    to_jsonable,
)
from trace_map.utils import cosine_similarity


@dataclass
class PipelineOutput:
    observation: ObservationBundle
    mode: ReasoningMode
    features: dict[str, FusedFeatures]
    reasoning_text: dict[str, str]
    reasoning_embeddings: dict[str, np.ndarray]
    memory_selections: dict[str, MemorySelection]
    memory_embeddings: dict[str, np.ndarray]
    selected_messages: dict[str, CandidateMessage]
    audits: list[CounterfactualAudit]
    received: dict[str, list[tuple[CandidateMessage, CounterfactualAudit]]]
    provisional_belief_entropy: dict[str, float]
    actions: dict[str, np.ndarray] = field(default_factory=dict)
    deviations: dict[str, bool] = field(default_factory=dict)


RemovalValueFunction = Callable[[str, FusedFeatures, str | None], float]


DEFAULT_TEXTUAL_POLICY = (
    "Use only locally available and public evidence. Prefer regime-compatible memories; "
    "treat factual support as necessary but not sufficient; evaluate sender commitment, "
    "receiver loss under deviation, and observed behavior before changing a physical action."
)


class TraceMapPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        environment: MultiAgentEconomy,
        language_policy: LanguagePolicy,
        text_encoder: TextEncoder,
    ):
        self.config = config
        self.environment = environment
        self.language_policy = language_policy
        self.encoder = text_encoder
        configured_dimension = int(config["language"]["embedding_dim"])
        if text_encoder.dimension != configured_dimension:
            raise ValueError("Text encoder output dimension must equal language.embedding_dim")
        self.dimension = configured_dimension
        seed = int(config["run"]["seed"])
        self.scheduler = ReasoningScheduler(config)
        self.memory_banks = {
            agent_id: RegimeAwareMemoryBank(config, seed + index + 1)
            for index, agent_id in enumerate(environment.agent_ids)
        }
        self.beliefs = OpponentBeliefModel(environment.agent_ids, config)
        self.credibility = CounterfactualCredibility(config)
        self.perturbation = CommunicationPerturbation(config, seed + 2000)
        self.fusion = PolicyFusion(config, seed + 3000)
        self.rng = np.random.default_rng(seed + 4000)
        self.textual_policies = {
            agent_id: DEFAULT_TEXTUAL_POLICY for agent_id in environment.agent_ids
        }
        self.previous_public: dict[str, float] | None = None
        self.previous_actions = {
            agent_id: np.zeros(spec.dim, dtype=np.float32)
            for agent_id, spec in environment.action_specs.items()
        }
        self.message_history: dict[str, list[np.ndarray]] = {
            agent_id: [] for agent_id in environment.agent_ids
        }
        self.episode_trace: list[dict[str, Any]] = []
        self.global_step = 0

    def reset_episode(self) -> None:
        self.previous_public = None
        self.beliefs = OpponentBeliefModel(self.environment.agent_ids, self.config)
        self.previous_actions = {
            agent_id: np.zeros(spec.dim, dtype=np.float32)
            for agent_id, spec in self.environment.action_specs.items()
        }
        self.message_history = {agent_id: [] for agent_id in self.environment.agent_ids}
        self.episode_trace = []

    def _query_text(self, agent_id: str, observation: ObservationBundle) -> str:
        public = ", ".join(f"{key}={value:.6g}" for key, value in sorted(observation.public.items()))
        belief = self.beliefs.aggregate(agent_id)
        local = ",".join(f"{value:.5g}" for value in observation.local[agent_id])
        return (
            f"agent={agent_id}; regime={observation.regime.name}; public=[{public}]; "
            f"local=[{local}]; opponent_belief=[{','.join(f'{v:.4g}' for v in belief)}]"
        )

    def prepare(self, observation: ObservationBundle, training: bool = False) -> PipelineOutput:
        previous_public = self.previous_public or observation.public
        mode = self.scheduler.mode(observation.step, observation.public, self.previous_public)
        query_texts = [self._query_text(agent_id, observation) for agent_id in self.environment.agent_ids]
        query_embeddings = self.encoder.encode(query_texts)

        selections: dict[str, MemorySelection] = {}
        memory_embeddings: dict[str, np.ndarray] = {}
        reasoning_text: dict[str, str] = {}
        for index, agent_id in enumerate(self.environment.agent_ids):
            selection = self.memory_banks[agent_id].retrieve(
                query_embeddings[index], observation.regime, self.global_step, training
            )
            selections[agent_id] = selection
            memory_embeddings[agent_id] = self.memory_banks[agent_id].pool(
                selection, self.dimension
            )
            if mode is ReasoningMode.INACTIVE:
                reasoning_text[agent_id] = ""
            else:
                request = ReasoningRequest(
                    agent_id=agent_id,
                    mode=mode,
                    local_observation=observation.local[agent_id].copy(),
                    public=dict(observation.public),
                    previous_public=dict(previous_public),
                    regime=observation.regime,
                    memory_summaries=[item.event_summary for item in selection.items],
                    textual_policy=self.textual_policies[agent_id],
                    max_tokens=int(self.config["language"]["reasoning_max_tokens"]),
                )
                reasoning_text[agent_id] = self.language_policy.reason(request)
        reasoning_embeddings_array = self.encoder.encode(
            [reasoning_text[agent_id] for agent_id in self.environment.agent_ids]
        )
        reasoning_embeddings = {
            agent_id: reasoning_embeddings_array[index]
            for index, agent_id in enumerate(self.environment.agent_ids)
        }

        selected_messages: dict[str, CandidateMessage] = {}
        selected_audits: list[CounterfactualAudit] = []
        household_receivers = [
            agent_id for agent_id in self.environment.agent_ids if agent_id.startswith("household_")
        ]
        if self.scheduler.communication_active(mode):
            for sender_id in self.environment.agent_ids:
                receivers = [receiver for receiver in household_receivers if receiver != sender_id]
                if not receivers:
                    continue
                sender_dim = self.environment.action_specs[sender_id].dim
                request = MessageRequest(
                    sender_id=sender_id,
                    reasoning=reasoning_text[sender_id],
                    public=dict(observation.public),
                    previous_public=dict(previous_public),
                    regime=observation.regime,
                    sender_action_dim=sender_dim,
                    receiver_action_dim=2,
                    horizon=int(self.config["communication"]["counterfactual_horizon"]),
                    max_tokens=int(self.config["language"]["message_max_tokens"]),
                    temperature=float(self.config["language"]["temperature"]),
                    top_p=float(self.config["language"]["top_p"]),
                )
                generated = self.language_policy.generate_statements(
                    request, int(self.config["communication"]["message_candidates"])
                )
                embeddings = self.encoder.encode([statement.text for statement in generated])
                candidates = [
                    CandidateMessage(
                        candidate_id=f"s{self.global_step}:{sender_id}:c{candidate_index}",
                        sender_id=sender_id,
                        text=statement.text,
                        claim=statement.claim,
                        embedding=embeddings[candidate_index],
                    )
                    for candidate_index, statement in enumerate(generated)
                ]
                candidates = [self.perturbation.corrupt_claim(item) for item in candidates]
                audits_by_candidate: dict[str, list[CounterfactualAudit]] = {}
                for candidate in candidates:
                    candidate_audits: list[CounterfactualAudit] = []
                    for receiver_id in receivers:
                        similarity = cosine_similarity(
                            candidate.embedding, reasoning_embeddings[receiver_id]
                        )
                        relevance = float(np.clip(0.5 * (similarity + 1.0), 0.0, 1.0))
                        recommendation = candidate.claim.receiver_recommendation
                        influence = (
                            float(np.clip(np.mean(np.abs(recommendation)), 0.0, 1.0))
                            if recommendation is not None
                            else 0.0
                        )
                        audit = self.credibility.audit(
                            candidate=candidate,
                            receiver_id=receiver_id,
                            current_public=observation.public,
                            previous_public=previous_public,
                            predicted_sender_action=self.previous_actions[sender_id],
                            posterior_certainty=self.beliefs.certainty(receiver_id, sender_id),
                            relevance=relevance,
                            influence=influence,
                        )
                        candidate_audits.append(audit)
                    audits_by_candidate[candidate.candidate_id] = candidate_audits
                selected = self.credibility.select(
                    candidates,
                    audits_by_candidate,
                    history_embeddings=self.message_history[sender_id],
                    training=training,
                    rng=self.rng,
                )
                selected_messages[sender_id] = selected
                for audit in audits_by_candidate[selected.candidate_id]:
                    selected_audits.append(audit)
                    declared = selected.claim.sender_commitment
                    likelihood_action = (
                        self.previous_actions[sender_id]
                        if declared is None
                        else np.asarray(declared, dtype=np.float32)
                    )
                    likelihood = action_profile_likelihood(
                        likelihood_action, self.beliefs.profile_count
                    )
                    self.beliefs.update_from_message(
                        audit.receiver_id,
                        sender_id,
                        likelihood,
                        audit.trust_weight,
                    )

        received: dict[str, list[tuple[CandidateMessage, CounterfactualAudit]]] = {
            agent_id: [] for agent_id in self.environment.agent_ids
        }
        for audit in selected_audits:
            received[audit.receiver_id].append((selected_messages[audit.sender_id], audit))
        retained_count = int(self.config["communication"]["retained_senders"])
        for receiver_id in received:
            received[receiver_id].sort(
                key=lambda item: item[1].trust_weight * item[1].relevance, reverse=True
            )
            received[receiver_id] = received[receiver_id][:retained_count]

        features: dict[str, FusedFeatures] = {}
        belief_entropy: dict[str, float] = {}
        for agent_id in self.environment.agent_ids:
            belief_feature = self.beliefs.aggregate(agent_id)
            features[agent_id] = self.fusion.build(
                observation.local[agent_id],
                reasoning_embeddings[agent_id],
                memory_embeddings[agent_id],
                received[agent_id],
                belief_feature,
            )
            targets = [target for target in self.environment.agent_ids if target != agent_id]
            belief_entropy[agent_id] = float(
                np.mean([self.beliefs.uncertainty(agent_id, target) for target in targets])
            )
        return PipelineOutput(
            observation=observation,
            mode=mode,
            features=features,
            reasoning_text=reasoning_text,
            reasoning_embeddings=reasoning_embeddings,
            memory_selections=selections,
            memory_embeddings=memory_embeddings,
            selected_messages=selected_messages,
            audits=selected_audits,
            received=received,
            provisional_belief_entropy=belief_entropy,
        )

    def heuristic_actions(self, output: PipelineOutput) -> dict[str, np.ndarray]:
        """Deterministic smoke-only policy; full training uses PyTorch actors."""

        regime = output.observation.regime
        pressure = regime.depreciation_rate + regime.consumption_tax_rate + regime.interest_rate
        actions: dict[str, np.ndarray] = {}
        government = np.zeros(self.environment.action_specs["government"].dim, dtype=np.float32)
        government[0] = np.clip(2.0 * (pressure - 0.15), -1.0, 1.0)
        government[-1] = np.clip(1.5 * (pressure - 0.12), -1.0, 1.0)
        actions["government"] = government
        for agent_id in self.environment.agent_ids:
            if not agent_id.startswith("household_"):
                continue
            communication_signal = float(np.mean(output.features[agent_id].communication))
            action = np.asarray(
                [
                    np.clip(4.0 * (pressure - 0.16) + communication_signal, -1.0, 1.0),
                    np.clip(0.25 * (0.20 - pressure) - communication_signal, -1.0, 1.0),
                ],
                dtype=np.float32,
            )
            actions[agent_id] = action
        return actions

    def apply_strategic_deviations(
        self, actions: dict[str, np.ndarray], output: PipelineOutput
    ) -> dict[str, np.ndarray]:
        actual: dict[str, np.ndarray] = {}
        for agent_id, action in actions.items():
            selected = output.selected_messages.get(agent_id)
            commitment = selected.claim.sender_commitment if selected is not None else None
            deviated, changed = self.perturbation.maybe_deviate(agent_id, action, commitment)
            actual[agent_id] = self.environment.action_specs[agent_id].clip(deviated)
            output.deviations[agent_id] = changed
        output.actions = {agent_id: value.copy() for agent_id, value in actual.items()}
        return actual

    def finalize_step(
        self,
        output: PipelineOutput,
        step_result: StepResult,
        removal_value_fn: RemovalValueFunction | None = None,
    ) -> list[dict[str, Any]]:
        for observer in self.environment.agent_ids:
            for target, action in output.actions.items():
                if observer != target:
                    self.beliefs.refine_from_behavior(observer, target, action)

        records: list[dict[str, Any]] = []
        for agent_id in self.environment.agent_ids:
            selection = output.memory_selections[agent_id]
            if removal_value_fn is None:
                full_value = float(step_result.rewards[agent_id])
                removal_values = {
                    item.memory_id: full_value - 0.01 * float(np.tanh(score))
                    for item, score in zip(selection.items, selection.scores)
                }
            else:
                full_value = float(removal_value_fn(agent_id, output.features[agent_id], None))
                removal_values = {
                    item.memory_id: float(
                        removal_value_fn(agent_id, output.features[agent_id], item.memory_id)
                    )
                    for item in selection.items
                }
            attribution = self.memory_banks[agent_id].apply_removal_attribution(
                selection, full_value, removal_values
            )
            sent = output.selected_messages.get(agent_id)
            trace = DecisionTrace(
                step=output.observation.step,
                agent_id=agent_id,
                reasoning_mode=output.mode.value,
                reasoning_text=output.reasoning_text[agent_id],
                selected_memory_ids=selection.ids,
                memory_scores=selection.scores.tolist(),
                sent_candidate_id=None if sent is None else sent.candidate_id,
                received_audits=[audit for _, audit in output.received[agent_id]],
                belief_entropy=output.provisional_belief_entropy[agent_id],
                action=output.actions[agent_id].copy(),
                regime=output.observation.regime.name,
                attribution=attribution,
            )
            record = to_jsonable(trace)
            record["step_reward"] = float(step_result.rewards[agent_id])
            record["deviated_from_commitment"] = bool(output.deviations.get(agent_id, False))
            records.append(record)
            self.episode_trace.append(record)

        should_store = output.mode is ReasoningMode.LONG or step_result.terminated or step_result.truncated
        if should_store:
            for agent_id in self.environment.agent_ids:
                sent = output.selected_messages.get(agent_id)
                event = (
                    f"Regime {output.observation.regime.name} at step {output.observation.step}; "
                    f"reward={step_result.rewards[agent_id]:.5f}; mode={output.mode.value}."
                )
                memory_text = " ".join(
                    part
                    for part in (
                        event,
                        output.reasoning_text[agent_id],
                        "" if sent is None else sent.text,
                    )
                    if part
                )
                embedding = self.encoder.encode([memory_text])[0]
                self.memory_banks[agent_id].add(
                    MemoryItem(
                        memory_id=f"m{self.global_step}:{agent_id}",
                        event_summary=event,
                        regime=output.observation.regime,
                        reasoning=output.reasoning_text[agent_id],
                        communication="" if sent is None else sent.text,
                        action=output.actions[agent_id].copy(),
                        realized_return=float(step_result.rewards[agent_id]),
                        reliability=0.5,
                        stored_step=self.global_step,
                        embedding=embedding,
                    )
                )
        for sender_id, message in output.selected_messages.items():
            self.message_history[sender_id].append(message.embedding.copy())
            self.message_history[sender_id] = self.message_history[sender_id][-20:]
        self.previous_public = dict(output.observation.public)
        self.previous_actions = {
            agent_id: action.copy() for agent_id, action in output.actions.items()
        }
        self.global_step += 1
        return records

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "textual_policies": dict(self.textual_policies),
            "beliefs": self.beliefs.state_dict(),
            "memory_banks": {
                agent_id: bank.state_dict() for agent_id, bank in self.memory_banks.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.global_step = int(state["global_step"])
        self.textual_policies.update(state["textual_policies"])
        self.beliefs.load_state_dict(state["beliefs"])
        for agent_id, bank_state in state["memory_banks"].items():
            self.memory_banks[agent_id].load_state_dict(bank_state)
