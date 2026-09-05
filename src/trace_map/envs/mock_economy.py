"""A deterministic, dependency-light economy for tests and smoke runs.

This is not a substitute for TaxAI and is never used for paper-scale claims. It
preserves the observation/action roles and stress interventions needed to test
TRACE-MAP's information flow.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trace_map.envs.base import MultiAgentEconomy
from trace_map.types import ActionSpec, ObservationBundle, Regime, StepResult
from trace_map.utils import sigmoid


REGIMES = {
    "e1": Regime("e1", 0.055, 0.055, 0.035),
    "e2": Regime("e2", 0.080, 0.045, 0.070),
    "e3": Regime("e3", 0.110, 0.090, 0.085),
}


def gini(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return 0.0
    data = data - min(0.0, float(np.min(data))) + 1e-9
    total = float(np.sum(data))
    if total <= 1e-12:
        return 0.0
    sorted_values = np.sort(data)
    index = np.arange(1, data.size + 1, dtype=np.float64)
    return float(np.clip(np.sum((2 * index - data.size - 1) * sorted_values) / (data.size * total), 0, 1))


class MockEconomicSociety(MultiAgentEconomy):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        env_cfg = config["environment"]
        self.households = int(env_cfg["households"])
        self.episode_length = int(env_cfg["episode_length"])
        self.task = str(env_cfg.get("task", "social_welfare"))
        self._agent_ids = ["government"] + [f"household_{index}" for index in range(self.households)]
        self._action_specs = {
            "government": ActionSpec(-np.ones(5), np.ones(5)),
            **{
                f"household_{index}": ActionSpec(-np.ones(2), np.ones(2))
                for index in range(self.households)
            },
        }
        self._initial_regime = Regime(
            str(env_cfg["condition"]),
            float(env_cfg["depreciation_rate"]),
            float(env_cfg["consumption_tax_rate"]),
            float(env_cfg["interest_rate"]),
        )
        self._shift = dict(env_cfg.get("shift", {}))
        self.rng = np.random.default_rng(int(config["run"]["seed"]))
        self._step = 0
        self.regime = self._initial_regime
        self.assets = np.ones(self.households)
        self.productivity = np.ones(self.households)
        self.public_debt = 0.0
        self.last_consumption = np.ones(self.households)
        self.last_labor = np.ones(self.households) * 0.5
        self.last_rewards = np.zeros(self.households)
        self.last_output = 1.0
        self.last_wage = 1.0
        self.last_spending = 0.0

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agent_ids)

    @property
    def action_specs(self) -> dict[str, ActionSpec]:
        return dict(self._action_specs)

    def reset(self, seed: int | None = None) -> ObservationBundle:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step = 0
        self.regime = self._initial_regime
        self.assets = self.rng.lognormal(mean=2.0, sigma=0.35, size=self.households)
        self.productivity = self.rng.lognormal(mean=0.0, sigma=0.15, size=self.households)
        self.public_debt = float(np.sum(self.assets) * 0.08)
        self.last_consumption = np.maximum(self.assets * 0.08, 0.1)
        self.last_labor = np.full(self.households, 0.45)
        self.last_rewards = np.asarray(
            [sigmoid(np.log1p(c) - 0.5 * h**3) for c, h in zip(self.last_consumption, self.last_labor)]
        )
        self.last_output = float(np.sum(self.productivity * self.last_labor))
        self.last_wage = self.last_output / max(float(np.sum(self.last_labor)), 1e-6)
        self.last_spending = 0.0
        observation = self._observation()
        observation.validate(self.agent_ids)
        return observation

    def _activate_shift_if_needed(self) -> None:
        if not self._shift.get("enabled", False):
            return
        if self._step != int(self._shift.get("step", -1)):
            return
        target = str(self._shift.get("target", "e3")).lower()
        if target not in REGIMES:
            raise ValueError(f"Unknown mock-economy target regime: {target}")
        self.regime = REGIMES[target]

    def step(self, actions: dict[str, np.ndarray]) -> StepResult:
        if set(actions) != set(self.agent_ids):
            missing = sorted(set(self.agent_ids) - set(actions))
            extra = sorted(set(actions) - set(self.agent_ids))
            raise ValueError(f"Action keys mismatch; missing={missing}, extra={extra}")
        self._activate_shift_if_needed()
        clipped = {agent: self._action_specs[agent].clip(value) for agent, value in actions.items()}

        government = (clipped["government"] + 1.0) / 2.0
        income_tax = 0.02 + 0.23 * government[0]
        income_progressivity = 0.8 + 0.4 * government[1]
        wealth_tax = 0.04 * government[2]
        transfer_share = 0.12 * government[3]
        spending_share = 0.22 * government[4]

        household_actions = np.stack(
            [clipped[f"household_{index}"] for index in range(self.households)]
        )
        scaled = (household_actions + 1.0) / 2.0
        saving_rate = 0.50 + 0.45 * scaled[:, 0]
        labor = 0.15 + 0.85 * scaled[:, 1]

        effective_labor = float(np.sum(self.productivity * labor))
        capital = max(float(np.sum(self.assets)), 1e-6)
        output = capital ** (1.0 / 3.0) * max(effective_labor, 1e-6) ** (2.0 / 3.0)
        wage = (2.0 / 3.0) * output / max(effective_labor, 1e-6)
        labor_income = wage * self.productivity * labor
        mean_income = max(float(np.mean(labor_income)), 1e-6)
        progressive_base = np.power(np.maximum(labor_income / mean_income, 1e-6), income_progressivity)
        income_taxes = np.minimum(labor_income * 0.8, income_tax * mean_income * progressive_base)
        wealth_taxes = wealth_tax * self.assets
        pretax_revenue = float(np.sum(income_taxes + wealth_taxes))
        transfers = transfer_share * output / self.households
        spending = spending_share * output

        resources = (
            self.assets * (1.0 + self.regime.interest_rate - self.regime.depreciation_rate)
            + labor_income
            - income_taxes
            - wealth_taxes
            + transfers
        )
        consumption = np.maximum(
            (1.0 - saving_rate) * resources / (1.0 + self.regime.consumption_tax_rate),
            1e-5,
        )
        consumption_tax = self.regime.consumption_tax_rate * consumption
        next_assets = np.maximum(resources - consumption - consumption_tax, 1e-5)
        total_revenue = pretax_revenue + float(np.sum(consumption_tax))
        self.public_debt = (
            (1.0 + self.regime.interest_rate) * self.public_debt
            + spending
            + transfers * self.households
            - total_revenue
        )

        household_rewards = np.asarray(
            [sigmoid(np.log1p(c) - (h**3) / 3.0) for c, h in zip(consumption, labor)],
            dtype=np.float64,
        )
        social_welfare = float(np.sum(household_rewards))
        inequality = gini(next_assets)
        if self.task == "gdp":
            government_reward = (output - self.last_output) / max(abs(self.last_output), 1e-6)
        elif self.task == "gini":
            government_reward = -inequality
        elif self.task == "gdp_gini":
            growth = (output - self.last_output) / max(abs(self.last_output), 1e-6)
            government_reward = growth - inequality
        else:
            government_reward = social_welfare / self.households - 0.15 * inequality

        innovation = self.rng.normal(0.0, 0.025, size=self.households)
        self.productivity = np.exp(0.98 * np.log(np.maximum(self.productivity, 1e-6)) + innovation)
        self.assets = next_assets
        self.last_consumption = consumption
        self.last_labor = labor
        self.last_rewards = household_rewards
        self.last_output = float(output)
        self.last_wage = float(wage)
        self.last_spending = float(spending)
        self._step += 1

        finite = all(
            np.all(np.isfinite(value))
            for value in (self.assets, self.productivity, household_rewards)
        ) and np.isfinite(government_reward)
        terminated = not finite
        truncated = self._step >= self.episode_length
        rewards = {"government": float(government_reward)}
        rewards.update(
            {
                f"household_{index}": float(household_rewards[index])
                for index in range(self.households)
            }
        )
        info = self._info()
        observation = self._observation()
        observation.validate(self.agent_ids)
        return StepResult(observation, rewards, terminated, truncated, info)

    def _macro_vector(self) -> np.ndarray:
        wealth_gini = gini(self.assets)
        income_proxy = self.productivity * self.last_labor * self.last_wage
        income_gini = gini(income_proxy)
        welfare_per_household = float(np.mean(self.last_rewards))
        output_per_household = self.last_output / self.households
        debt_ratio = self.public_debt / max(self.last_output, 1e-6)
        spending_ratio = self.last_spending / max(self.last_output, 1e-6)
        return np.asarray(
            [
                wealth_gini,
                income_gini,
                welfare_per_household,
                np.log1p(max(output_per_household, 0.0)),
                np.log1p(max(self.last_wage, 0.0)),
                np.clip(debt_ratio, -10.0, 10.0),
                spending_ratio,
            ],
            dtype=np.float32,
        )

    def _public(self) -> dict[str, float]:
        return {
            "wealth_inequality": gini(self.assets),
            "income_inequality": gini(self.productivity * self.last_labor * self.last_wage),
            "social_welfare": float(np.sum(self.last_rewards)),
            "output": float(self.last_output),
            "wage": float(self.last_wage),
            "public_debt": float(self.public_debt),
            "aggregate_expenditure": float(self.last_spending),
            "depreciation_rate": self.regime.depreciation_rate,
            "consumption_tax_rate": self.regime.consumption_tax_rate,
            "interest_rate": self.regime.interest_rate,
        }

    def _observation(self) -> ObservationBundle:
        global_state = self._macro_vector()
        local: dict[str, np.ndarray] = {"government": global_state.copy()}
        assets_scale = max(float(np.mean(self.assets)), 1e-6)
        productivity_scale = max(float(np.mean(self.productivity)), 1e-6)
        for index in range(self.households):
            private = np.asarray(
                [self.assets[index] / assets_scale, self.productivity[index] / productivity_scale],
                dtype=np.float32,
            )
            local[f"household_{index}"] = np.concatenate([global_state, private])
        return ObservationBundle(local, global_state.copy(), self._public(), self.regime, self._step)

    def _info(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "regime": self.regime.name,
            "gdp": float(self.last_output),
            "wealth_gini": gini(self.assets),
            "income_gini": gini(self.productivity * self.last_labor * self.last_wage),
            "social_welfare": float(np.sum(self.last_rewards)),
            "public_debt": float(self.public_debt),
        }

    def close(self) -> None:
        return None
