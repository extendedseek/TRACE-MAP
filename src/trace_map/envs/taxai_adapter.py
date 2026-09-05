"""Adapter for the upstream TaxAI ``economic_society`` environment.

The import is lazy so dependency-light tests can run without TaxAI, PyTorch,
Gym, Pygame, or OmegaConf.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from trace_map.envs.base import MultiAgentEconomy
from trace_map.envs.mock_economy import REGIMES
from trace_map.types import ActionSpec, ObservationBundle, Regime, StepResult


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class TaxAIAdapter(MultiAgentEconomy):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        env_cfg = config["environment"]
        candidate = Path(env_cfg["taxai_repo"])
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        self.repo = candidate.resolve()
        if not (self.repo / "env" / "env_core.py").is_file():
            raise FileNotFoundError(
                f"TaxAI checkout not found at {self.repo}. Run: bash scripts/setup_taxai.sh"
            )
        self._check_revision(str(env_cfg.get("taxai_commit", "")))
        sys.path.insert(0, str(self.repo)) if str(self.repo) not in sys.path else None
        try:
            from omegaconf import OmegaConf
            from env.env_core import economic_society
        except ImportError as error:
            raise ImportError(
                "TaxAI dependencies are missing. Install with: pip install -e '.[taxai,train]'"
            ) from error

        config_path = self.repo / str(env_cfg.get("taxai_config", "cfg/default.yaml"))
        taxai_config = OmegaConf.load(config_path)
        households = int(env_cfg["households"])
        for entity in taxai_config.Environment.Entities:
            if entity.entity_name == "household":
                entity.entity_args.n = households
        taxai_config.Environment.env_core.env_args.depreciation_rate = float(
            env_cfg["depreciation_rate"]
        )
        taxai_config.Environment.env_core.env_args.consumption_tax_rate = float(
            env_cfg["consumption_tax_rate"]
        )
        taxai_config.Environment.env_core.env_args.interest_rate = float(env_cfg["interest_rate"])
        taxai_config.Environment.env_core.env_args.gov_task = str(env_cfg.get("task", "social_welfare"))
        with _working_directory(self.repo):
            self.env = economic_society(taxai_config.Environment)
        self.env.episode_length = int(env_cfg["episode_length"])
        self.households = households
        self._agent_ids = ["government"] + [f"household_{index}" for index in range(households)]
        gov_space = self.env.government.action_space
        household_space = self.env.households.action_space
        self._action_specs = {
            "government": ActionSpec(gov_space.low, gov_space.high),
            **{
                f"household_{index}": ActionSpec(household_space.low[index], household_space.high[index])
                for index in range(households)
            },
        }
        self._shift = dict(env_cfg.get("shift", {}))
        self.regime = Regime(
            str(env_cfg["condition"]),
            float(env_cfg["depreciation_rate"]),
            float(env_cfg["consumption_tax_rate"]),
            float(env_cfg["interest_rate"]),
        )
        self._step = 0

    def _check_revision(self, expected: str) -> None:
        if not expected:
            return
        result = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        actual = result.stdout.strip()
        if result.returncode == 0 and actual != expected:
            raise RuntimeError(
                f"TaxAI revision mismatch: expected {expected}, found {actual}. "
                "Run scripts/setup_taxai.sh or update the config deliberately."
            )

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agent_ids)

    @property
    def action_specs(self) -> dict[str, ActionSpec]:
        return dict(self._action_specs)

    def _apply_regime(self, regime: Regime) -> None:
        self.regime = regime
        self.env.depreciation_rate = regime.depreciation_rate
        self.env.consumption_tax_rate = regime.consumption_tax_rate
        self.env.interest_rate = regime.interest_rate

    def _activate_shift_if_needed(self) -> None:
        if self._shift.get("enabled", False) and self._step == int(self._shift.get("step", -1)):
            target = str(self._shift.get("target", "e3")).lower()
            if target not in REGIMES:
                raise ValueError(f"Unknown TaxAI target regime: {target}")
            self._apply_regime(REGIMES[target])

    def reset(self, seed: int | None = None) -> ObservationBundle:
        if seed is not None:
            np.random.seed(seed)
            try:
                import torch

                torch.manual_seed(seed)
            except ImportError:
                pass
        self._step = 0
        env_cfg = self.config["environment"]
        self._apply_regime(
            Regime(
                str(env_cfg["condition"]),
                float(env_cfg["depreciation_rate"]),
                float(env_cfg["consumption_tax_rate"]),
                float(env_cfg["interest_rate"]),
            )
        )
        with _working_directory(self.repo):
            global_obs, private_obs = self.env.reset()
        observation = self._bundle(global_obs, private_obs)
        observation.validate(self.agent_ids)
        return observation

    def step(self, actions: dict[str, np.ndarray]) -> StepResult:
        if set(actions) != set(self.agent_ids):
            raise ValueError("TaxAI action keys do not match configured agents")
        self._activate_shift_if_needed()
        government_action = self._action_specs["government"].clip(actions["government"])
        household_actions = np.stack(
            [
                self._action_specs[f"household_{index}"].clip(actions[f"household_{index}"])
                for index in range(self.households)
            ]
        )
        action_dict = {
            self.env.government.name: government_action,
            self.env.households.name: household_actions,
        }
        with _working_directory(self.repo):
            global_obs, private_obs, gov_reward, household_reward, done = self.env.step(action_dict)
        self._step += 1
        rewards = {"government": float(np.asarray(gov_reward).reshape(-1)[0])}
        household_reward = np.asarray(household_reward).reshape(-1)
        rewards.update(
            {
                f"household_{index}": float(household_reward[index])
                for index in range(self.households)
            }
        )
        observation = self._bundle(global_obs, private_obs)
        observation.validate(self.agent_ids)
        truncated = self._step >= int(self.config["environment"]["episode_length"])
        return StepResult(observation, rewards, bool(done and not truncated), truncated, self._info())

    def _bundle(self, global_obs: np.ndarray, private_obs: np.ndarray) -> ObservationBundle:
        global_state = np.asarray(global_obs, dtype=np.float32).reshape(-1)
        private = np.asarray(private_obs, dtype=np.float32)
        local: dict[str, np.ndarray] = {"government": global_state.copy()}
        for index in range(self.households):
            local[f"household_{index}"] = np.concatenate([global_state, private[index].reshape(-1)])
        return ObservationBundle(local, global_state, self._public(), self.regime, self._step)

    def _public(self) -> dict[str, float]:
        household_rewards = np.asarray(getattr(self.env, "households_reward", np.zeros(self.households)))
        gdp = float(getattr(self.env, "GDP", 0.0))
        spending = float(getattr(self.env, "Gt_prob", 0.0)) * gdp
        return {
            "wealth_inequality": float(getattr(self.env, "wealth_gini", 0.0)),
            "income_inequality": float(getattr(self.env, "income_gini", 0.0)),
            "social_welfare": float(np.sum(household_rewards)),
            "output": gdp,
            "wage": float(getattr(self.env, "WageRate", 0.0)),
            "public_debt": float(getattr(self.env, "Bt", 0.0)),
            "aggregate_expenditure": spending,
            "depreciation_rate": self.regime.depreciation_rate,
            "consumption_tax_rate": self.regime.consumption_tax_rate,
            "interest_rate": self.regime.interest_rate,
        }

    def _info(self) -> dict[str, Any]:
        public = self._public()
        return {
            "step": self._step,
            "regime": self.regime.name,
            "gdp": public["output"],
            "wealth_gini": public["wealth_inequality"],
            "income_gini": public["income_inequality"],
            "social_welfare": public["social_welfare"],
            "public_debt": public["public_debt"],
        }

    def close(self) -> None:
        self.env.close()
