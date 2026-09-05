"""Two-timescale TRACE-MAP trainer.

PyTorch is optional for the package but required by this module. The trainer
implements replay-based deterministic actor--critic updates, centralized
language-conditioned critics, local surrogate distillation, sampled unilateral
deviation regularization, reflective-memory attribution, and batched textual
policy revision.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from trace_map.config import checkpoint_digest, config_digest, save_config
from trace_map.envs import make_environment
from trace_map.language import make_language_components
from trace_map.language_credit import (
    CentralizedLanguageCritic,
    LanguageCredit,
    TextualPolicyReviser,
    TextualPolicyStore,
)
from trace_map.losses import actor_loss, deviation_regularizer, surrogate_loss, value_loss
from trace_map.metrics import (
    brier_score,
    empirical_exploitability,
    negative_log_likelihood,
    recovery_time,
    trusted_harmful_message_rate,
    useful_at_k,
)
from trace_map.models.networks import Actor, CentralizedCritic, LocalValueSurrogate, polyak_update
from trace_map.pipeline import FusedFeatures, PipelineOutput, TraceMapPipeline
from trace_map.replay import ReplayBuffer
from trace_map.types import Transition
from trace_map.utils import append_jsonl, atomic_write_json, run_metadata, seed_everything


class TraceMapTrainer:
    def __init__(self, config: dict[str, Any], output_directory: str | Path):
        self.config = config
        self.output = Path(output_directory)
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = int(config["run"]["seed"])
        seed_everything(self.seed, bool(config["run"].get("deterministic", True)))
        requested_device = str(config["run"].get("device", "cpu"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; set run.device=cpu deliberately")
        self.device = torch.device(requested_device)
        save_config(config, self.output / "resolved_config.yaml")
        atomic_write_json(
            self.output / "run_metadata.json",
            {**run_metadata(), "config_sha256": config_digest(config), "mode": "train"},
        )

        self.environment = make_environment(config)
        language_policy, encoder = make_language_components(config)
        self.pipeline = TraceMapPipeline(config, self.environment, language_policy, encoder)
        initial_observation = self.environment.reset(self.seed)
        self._build_networks(initial_observation)
        self.replay = ReplayBuffer(int(config["training"]["replay_capacity"]), self.seed + 100)
        self.credit_critic = CentralizedLanguageCritic()
        self.policy_store = TextualPolicyStore(dict(self.pipeline.textual_policies))
        self.policy_reviser = TextualPolicyReviser(language_policy, self.policy_store)
        self.completed_trajectories: list[list[dict[str, Any]]] = []
        self.completed_credits: list[LanguageCredit] = []

    def _build_networks(self, observation: Any) -> None:
        model_cfg = self.config["model"]
        embedding_dim = int(self.config["language"]["embedding_dim"])
        language_dim = 4 * embedding_dim
        agent_ids = self.environment.agent_ids
        self.agent_ids = agent_ids
        joint_action_dim = sum(self.environment.action_specs[agent].dim for agent in agent_ids)
        centralized_language_dim = len(agent_ids) * language_dim
        global_dim = int(np.asarray(observation.global_state).size)
        self.actors = nn.ModuleDict()
        self.target_actors = nn.ModuleDict()
        self.critics = nn.ModuleDict()
        self.target_critics = nn.ModuleDict()
        self.surrogates = nn.ModuleDict()
        for agent_id in agent_ids:
            input_dim = int(np.asarray(observation.local[agent_id]).size) + language_dim
            spec = self.environment.action_specs[agent_id]
            actor = Actor(
                input_dim,
                torch.from_numpy(spec.low),
                torch.from_numpy(spec.high),
                model_cfg["actor_hidden"],
                model_cfg.get("activation", "relu"),
            )
            critic = CentralizedCritic(
                global_dim,
                joint_action_dim,
                centralized_language_dim,
                model_cfg["critic_hidden"],
                model_cfg.get("activation", "relu"),
            )
            surrogate = LocalValueSurrogate(
                input_dim,
                spec.dim,
                model_cfg.get("auxiliary_hidden", [128, 64]),
            )
            self.actors[agent_id] = actor
            self.target_actors[agent_id] = copy.deepcopy(actor)
            self.critics[agent_id] = critic
            self.target_critics[agent_id] = copy.deepcopy(critic)
            self.surrogates[agent_id] = surrogate
        for module in (
            self.actors,
            self.target_actors,
            self.critics,
            self.target_critics,
            self.surrogates,
        ):
            module.to(self.device)
        training = self.config["training"]
        self.actor_optimizer = torch.optim.Adam(
            self.actors.parameters(), lr=float(training["actor_lr"])
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(), lr=float(training["critic_lr"])
        )
        self.surrogate_optimizer = torch.optim.Adam(
            self.surrogates.parameters(), lr=float(training["auxiliary_lr"])
        )

    def _actor_actions(self, prepared: PipelineOutput, explore: bool) -> dict[str, np.ndarray]:
        actions: dict[str, np.ndarray] = {}
        noise = float(self.config["model"].get("action_noise", 0.0)) if explore else 0.0
        for agent_id in self.agent_ids:
            feature = torch.as_tensor(
                prepared.features[agent_id].actor_input,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            with torch.no_grad():
                action = self.actors[agent_id](feature).cpu().numpy()[0]
            if noise > 0:
                action += np.random.normal(0.0, noise, size=action.shape)
            actions[agent_id] = self.environment.action_specs[agent_id].clip(action)
        return actions

    def _removal_value(self, agent_id: str, fused: FusedFeatures, memory_id: str | None) -> float:
        feature = fused.actor_input.copy()
        if memory_id is not None:
            local_dim = feature.size - 4 * int(self.config["language"]["embedding_dim"])
            memory_start = local_dim + int(self.config["language"]["embedding_dim"])
            memory_end = memory_start + int(self.config["language"]["embedding_dim"])
            feature[memory_start:memory_end] = 0.0
        tensor = torch.as_tensor(feature, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actors[agent_id](tensor)
            return float(self.surrogates[agent_id].value(tensor, action).item())

    def _transition(
        self,
        prepared: PipelineOutput,
        actions: dict[str, np.ndarray],
        rewards: dict[str, float],
        next_prepared: PipelineOutput,
        done: bool,
    ) -> Transition:
        return Transition(
            global_state=prepared.observation.global_state.copy(),
            local_features={agent: prepared.features[agent].actor_input.copy() for agent in self.agent_ids},
            language_features={
                agent: prepared.features[agent].language_context.copy() for agent in self.agent_ids
            },
            actions={agent: actions[agent].copy() for agent in self.agent_ids},
            rewards=dict(rewards),
            next_global_state=next_prepared.observation.global_state.copy(),
            next_local_features={
                agent: next_prepared.features[agent].actor_input.copy() for agent in self.agent_ids
            },
            next_language_features={
                agent: next_prepared.features[agent].language_context.copy() for agent in self.agent_ids
            },
            done=done,
        )

    def _stack(self, batch: list[Transition], attribute: str, agent: str | None = None) -> torch.Tensor:
        values = []
        for transition in batch:
            value = getattr(transition, attribute)
            if agent is not None:
                value = value[agent]
            values.append(np.asarray(value, dtype=np.float32))
        return torch.as_tensor(np.stack(values), dtype=torch.float32, device=self.device)

    def _joint_actions(self, batch: list[Transition]) -> torch.Tensor:
        return torch.cat([self._stack(batch, "actions", agent) for agent in self.agent_ids], dim=-1)

    def _central_language(self, batch: list[Transition], next_state: bool = False) -> torch.Tensor:
        attribute = "next_language_features" if next_state else "language_features"
        return torch.cat([self._stack(batch, attribute, agent) for agent in self.agent_ids], dim=-1)

    def update(self) -> dict[str, float]:
        training = self.config["training"]
        batch_size = int(training["batch_size"])
        batch = list(self.replay.sample(batch_size))
        global_state = self._stack(batch, "global_state")
        next_global = self._stack(batch, "next_global_state")
        joint_action = self._joint_actions(batch)
        language = self._central_language(batch)
        next_language = self._central_language(batch, next_state=True)
        done = torch.as_tensor(
            [[float(item.done)] for item in batch], dtype=torch.float32, device=self.device
        )
        discount = float(training["discount"])

        with torch.no_grad():
            next_actions = torch.cat(
                [
                    self.target_actors[agent](self._stack(batch, "next_local_features", agent))
                    for agent in self.agent_ids
                ],
                dim=-1,
            )
        critic_losses = []
        surrogate_losses = []
        for agent_id in self.agent_ids:
            predicted = self.critics[agent_id].value(global_state, joint_action, language)
            with torch.no_grad():
                target = self.target_critics[agent_id].value(next_global, next_actions, next_language)
                target = (1.0 - done) * target
            reward = self._stack(batch, "rewards", agent_id).reshape(-1, 1)
            critic_losses.append(value_loss(predicted, reward, discount, target))
            local = self._stack(batch, "local_features", agent_id)
            agent_actions = self._stack(batch, "actions", agent_id)
            surrogate_prediction = self.surrogates[agent_id].value(local, agent_actions)
            surrogate_losses.append(surrogate_loss(surrogate_prediction, predicted))

        critic_total = torch.stack(critic_losses).sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critics.parameters(), float(training["max_grad_norm"])
        )
        self.critic_optimizer.step()

        surrogate_total = torch.stack(surrogate_losses).sum()
        self.surrogate_optimizer.zero_grad(set_to_none=True)
        surrogate_total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.surrogates.parameters(), float(training["max_grad_norm"])
        )
        self.surrogate_optimizer.step()

        for parameter in self.critics.parameters():
            parameter.requires_grad_(False)
        actor_losses = []
        selected_values = []
        deviation_values = []
        deviation_agents = [agent for agent in self.agent_ids if agent.startswith("household_")]
        deviation_count = int(self.config["communication"]["sampled_deviations"])
        action_offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for agent in self.agent_ids:
            width = self.environment.action_specs[agent].dim
            action_offsets[agent] = (cursor, cursor + width)
            cursor += width

        for agent_id in self.agent_ids:
            start, end = action_offsets[agent_id]
            policy_action = self.actors[agent_id](self._stack(batch, "local_features", agent_id))
            policy_joint = joint_action.detach().clone()
            policy_joint[:, start:end] = policy_action
            q_value = self.critics[agent_id].value(global_state, policy_joint, language)
            actor_losses.append(actor_loss(q_value))
            if agent_id in deviation_agents:
                selected_values.append(q_value.squeeze(-1))
                per_deviation = []
                spec = self.environment.action_specs[agent_id]
                for _ in range(deviation_count):
                    random_action = torch.empty_like(policy_action).uniform_(-1.0, 1.0)
                    low = torch.as_tensor(spec.low, device=self.device)
                    high = torch.as_tensor(spec.high, device=self.device)
                    random_action = low + 0.5 * (random_action + 1.0) * (high - low)
                    deviated_joint = policy_joint.detach().clone()
                    deviated_joint[:, start:end] = random_action
                    per_deviation.append(
                        self.critics[agent_id]
                        .value(global_state, deviated_joint, language)
                        .squeeze(-1)
                    )
                deviation_values.append(torch.stack(per_deviation, dim=-1))
        actor_total = torch.stack(actor_losses).sum()
        if deviation_agents and self.config.get("ablation", {}).get("deviation_regularizer", True):
            selected_tensor = torch.stack(selected_values, dim=1)
            deviation_tensor = torch.stack(deviation_values, dim=1)
            dev_loss = deviation_regularizer(
                selected_tensor,
                deviation_tensor,
                float(self.config["opponent"]["deviation_tolerance"]),
            )
        else:
            dev_loss = torch.zeros((), device=self.device)
        weights = training["loss_weights"]
        optimized_actor = float(weights["actor"]) * actor_total + float(weights["deviation"]) * dev_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        optimized_actor.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actors.parameters(), float(training["max_grad_norm"])
        )
        self.actor_optimizer.step()
        for parameter in self.critics.parameters():
            parameter.requires_grad_(True)

        coefficient = float(training["polyak"])
        for agent_id in self.agent_ids:
            polyak_update(self.actors[agent_id], self.target_actors[agent_id], coefficient)
            polyak_update(self.critics[agent_id], self.target_critics[agent_id], coefficient)
        return {
            "critic_loss": float(critic_total.detach().cpu()),
            "actor_loss": float(actor_total.detach().cpu()),
            "surrogate_loss": float(surrogate_total.detach().cpu()),
            "deviation_loss": float(dev_loss.detach().cpu()),
        }

    def _revise_textual_policies(self) -> None:
        if not self.completed_credits:
            return
        updates = self.policy_reviser.revise_batch(
            self.completed_trajectories, self.completed_credits
        )
        self.pipeline.textual_policies.update(updates)
        atomic_write_json(self.output / "textual_policies.json", self.policy_store.policies)
        self.completed_trajectories.clear()
        self.completed_credits.clear()

    def train(self) -> dict[str, Any]:
        training = self.config["training"]
        total_steps = int(training["total_environment_steps"])
        exploratory_steps = int(training["exploratory_steps"])
        batch_size = int(training["batch_size"])
        checkpoint_every = int(training["checkpoint_every"])
        self.pipeline.reset_episode()
        observation = self.environment.reset(self.seed)
        prepared = self.pipeline.prepare(observation, training=True)
        episode = 0
        episode_returns = {agent: 0.0 for agent in self.agent_ids}
        latest_losses: dict[str, float] = {}
        try:
            for step in range(total_steps):
                if step < exploratory_steps:
                    proposed = self.environment.sample_actions(np.random.default_rng(self.seed + step))
                else:
                    proposed = self._actor_actions(prepared, explore=True)
                actual = self.pipeline.apply_strategic_deviations(proposed, prepared)
                result = self.environment.step(actual)
                records = self.pipeline.finalize_step(
                    prepared, result, removal_value_fn=self._removal_value
                )
                done = bool(result.terminated or result.truncated)
                next_prepared = self.pipeline.prepare(result.observation, training=True)
                self.replay.add(
                    self._transition(prepared, actual, result.rewards, next_prepared, done)
                )
                for agent, reward in result.rewards.items():
                    episode_returns[agent] += float(reward)
                for record in records:
                    append_jsonl(self.output / "decision_trace.jsonl", {"episode": episode, **record})
                if (
                    len(self.replay) >= batch_size
                    and step >= exploratory_steps
                    and step % int(training["update_every"]) == 0
                ):
                    for _ in range(int(training["gradient_steps"])):
                        latest_losses = self.update()
                    append_jsonl(self.output / "training.jsonl", {"step": step, **latest_losses})
                if done:
                    social_return = sum(
                        value for key, value in episode_returns.items() if key.startswith("household_")
                    )
                    for agent_id in self.agent_ids:
                        credit = self.credit_critic.assign(
                            agent_id,
                            self.pipeline.episode_trace,
                            episode_returns[agent_id],
                            social_return,
                        )
                        self.completed_trajectories.append(list(self.pipeline.episode_trace))
                        self.completed_credits.append(credit)
                    append_jsonl(
                        self.output / "episodes.jsonl",
                        {"episode": episode, "step": step, "returns": episode_returns},
                    )
                    episode += 1
                    if (
                        self.config.get("ablation", {}).get("textual_revision", True)
                        and episode % int(training["text_update_episodes"]) == 0
                    ):
                        self._revise_textual_policies()
                    self.pipeline.reset_episode()
                    observation = self.environment.reset(self.seed + episode)
                    prepared = self.pipeline.prepare(observation, training=True)
                    episode_returns = {agent: 0.0 for agent in self.agent_ids}
                else:
                    observation = result.observation
                    prepared = next_prepared
                if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                    self.save_checkpoint(step + 1)
        finally:
            self.environment.close()
        self.save_checkpoint(total_steps)
        summary = {"steps": total_steps, "episodes": episode, "latest_losses": latest_losses}
        atomic_write_json(self.output / "training_summary.json", summary)
        return summary

    def save_checkpoint(self, step: int) -> Path:
        path = self.output / "checkpoints" / f"step_{step:09d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "actors": self.actors.state_dict(),
                "target_actors": self.target_actors.state_dict(),
                "critics": self.critics.state_dict(),
                "target_critics": self.target_critics.state_dict(),
                "surrogates": self.surrogates.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "surrogate_optimizer": self.surrogate_optimizer.state_dict(),
                "pipeline": self.pipeline.state_dict(),
                "config_sha256": config_digest(self.config),
                "checkpoint_compatibility_sha256": checkpoint_digest(self.config),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        expected = checkpoint_digest(self.config)
        actual = checkpoint.get("checkpoint_compatibility_sha256")
        if actual != expected:
            raise RuntimeError(
                "Checkpoint architecture/training digest does not match the resolved config"
            )
        self.actors.load_state_dict(checkpoint["actors"])
        self.target_actors.load_state_dict(checkpoint["target_actors"])
        self.critics.load_state_dict(checkpoint["critics"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])
        self.surrogates.load_state_dict(checkpoint["surrogates"])
        self.pipeline.load_state_dict(checkpoint["pipeline"])
        return int(checkpoint["step"])

    def _deviation_gains(
        self, prepared: PipelineOutput, actions: dict[str, np.ndarray]
    ) -> list[float]:
        global_state = torch.as_tensor(
            prepared.observation.global_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        language = torch.as_tensor(
            np.concatenate(
                [prepared.features[agent].language_context for agent in self.agent_ids]
            ),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        joint = np.concatenate([actions[agent] for agent in self.agent_ids]).astype(np.float32)
        joint_tensor = torch.as_tensor(joint, device=self.device).unsqueeze(0)
        offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for agent in self.agent_ids:
            width = self.environment.action_specs[agent].dim
            offsets[agent] = (cursor, cursor + width)
            cursor += width
        gains: list[float] = []
        rng = np.random.default_rng(self.seed + self.pipeline.global_step)
        with torch.no_grad():
            for agent in self.agent_ids:
                if not agent.startswith("household_"):
                    continue
                critic = self.critics[agent]
                baseline = float(critic.value(global_state, joint_tensor, language).item())
                start, end = offsets[agent]
                best = baseline
                spec = self.environment.action_specs[agent]
                for _ in range(int(self.config["communication"]["sampled_deviations"])):
                    alternative = joint.copy()
                    alternative[start:end] = rng.uniform(spec.low, spec.high)
                    alternative_tensor = torch.as_tensor(
                        alternative, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    best = max(
                        best,
                        float(critic.value(global_state, alternative_tensor, language).item()),
                    )
                gains.append(best - baseline)
        return gains

    def evaluate(self, episodes: int | None = None) -> dict[str, Any]:
        """Evaluate frozen actors while allowing the paper's online memory/belief updates."""

        episode_count = int(episodes or self.config["evaluation"]["episodes"])
        for module in (self.actors, self.critics, self.surrogates):
            module.eval()
        audits = []
        attribution_values: list[float] = []
        deviation_gains: list[float] = []
        household_episode_returns: list[float] = []
        government_episode_returns: list[float] = []
        social_series: list[float] = []
        recovery_values: list[int] = []
        profile_probabilities: list[float] = []
        latency_ms: list[float] = []
        import time

        try:
            for episode in range(episode_count):
                self.pipeline.reset_episode()
                observation = self.environment.reset(self.seed + episode)
                returns = {agent: 0.0 for agent in self.agent_ids}
                episode_social: list[float] = []
                done = False
                while not done:
                    start = time.perf_counter()
                    prepared = self.pipeline.prepare(observation, training=False)
                    proposed = self._actor_actions(prepared, explore=False)
                    actual = self.pipeline.apply_strategic_deviations(proposed, prepared)
                    deviation_gains.extend(self._deviation_gains(prepared, actual))
                    result = self.environment.step(actual)
                    records = self.pipeline.finalize_step(
                        prepared, result, removal_value_fn=self._removal_value
                    )
                    latency_ms.append((time.perf_counter() - start) * 1000.0)
                    audits.extend(prepared.audits)
                    welfare = float(result.info.get("social_welfare", 0.0))
                    social_series.append(welfare)
                    episode_social.append(welfare)
                    for agent, reward in result.rewards.items():
                        returns[agent] += float(reward)
                    for record in records:
                        attribution_values.extend(
                            float(value) for value in record["attribution"].values()
                        )
                        append_jsonl(
                            self.output / "evaluation_trace.jsonl",
                            {"episode": episode, **record},
                        )
                    observation = result.observation
                    done = bool(result.terminated or result.truncated)
                household_episode_returns.append(
                    float(
                        np.mean(
                            [value for key, value in returns.items() if key.startswith("household_")]
                        )
                    )
                )
                government_episode_returns.append(returns["government"])
                shift = self.config["environment"].get("shift", {})
                if shift.get("enabled", False):
                    recovered = recovery_time(
                        episode_social,
                        int(shift["step"]),
                        int(self.config["evaluation"]["recovery_window"]),
                        float(self.config["evaluation"]["recovery_fraction"]),
                    )
                    if recovered is not None:
                        recovery_values.append(recovered)
                if self.config["communication"]["condition"] == "c3":
                    reliability = self.config["communication"]["persistent_reliability"]
                    for target in [agent for agent in self.agent_ids if agent.startswith("household_")]:
                        index = int(target.rsplit("_", 1)[1])
                        profile = min(
                            self.pipeline.beliefs.profile_count - 1,
                            int(float(reliability[index]) * self.pipeline.beliefs.profile_count),
                        )
                        for observer in self.agent_ids:
                            if observer != target:
                                profile_probabilities.append(
                                    float(self.pipeline.beliefs.posterior(observer, target)[profile])
                                )
        finally:
            self.environment.close()
        predicted_safe = [1.0 - audit.strategic_risk for audit in audits]
        safe_labels = [not audit.harmful for audit in audits]
        recovery = float(np.mean(recovery_values)) if recovery_values else None
        metrics: dict[str, Any] = {
            "mode": "learned_policy_evaluation",
            "seed": self.seed,
            "episodes": episode_count,
            "average_household_return": float(np.mean(household_episode_returns)),
            "government_return": float(np.mean(government_episode_returns)),
            "mean_social_welfare": float(np.mean(social_series)) if social_series else 0.0,
            "brier_score": brier_score(predicted_safe, safe_labels) if audits else None,
            "thmr": trusted_harmful_message_rate(
                audits, float(self.config["communication"]["trust_threshold"])
            ),
            "profile_nll": negative_log_likelihood(profile_probabilities)
            if profile_probabilities
            else None,
            "empirical_exploitability": empirical_exploitability(deviation_gains),
            "useful_at_k_critic": useful_at_k(attribution_values),
            "recovery_time": recovery,
            "mean_decision_latency_ms": float(np.mean(latency_ms)) if latency_ms else 0.0,
            "p95_decision_latency_ms": float(np.quantile(latency_ms, 0.95))
            if latency_ms
            else 0.0,
        }
        atomic_write_json(self.output / "metrics.json", metrics)
        return metrics
