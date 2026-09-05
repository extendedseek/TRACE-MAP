from __future__ import annotations

import unittest

import numpy as np

from trace_map.envs.mock_economy import MockEconomicSociety

from common import smoke_config


class EnvironmentTests(unittest.TestCase):
    def test_shapes_and_regime_shift(self):
        environment = MockEconomicSociety(smoke_config())
        observation = environment.reset(9)
        self.assertEqual(len(observation.local), 4)
        self.assertEqual(environment.action_specs["government"].dim, 5)
        self.assertEqual(environment.action_specs["household_0"].dim, 2)
        rng = np.random.default_rng(0)
        result = None
        for _ in range(7):
            result = environment.step(environment.sample_actions(rng))
        self.assertIsNotNone(result)
        self.assertEqual(result.observation.regime.name, "e3")
        self.assertTrue(all(np.isfinite(value) for value in result.rewards.values()))


if __name__ == "__main__":
    unittest.main()
