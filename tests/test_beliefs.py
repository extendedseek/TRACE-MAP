from __future__ import annotations

import unittest

import numpy as np

from trace_map.beliefs import OpponentBeliefModel, action_profile_likelihood, certainty

from common import smoke_config


class BeliefTests(unittest.TestCase):
    def setUp(self):
        self.model = OpponentBeliefModel(["a", "b"], smoke_config())

    def test_low_credibility_leaves_uniform_prior(self):
        likelihood = np.asarray([0.8, 0.05, 0.05, 0.05, 0.05])
        posterior = self.model.update_from_message("a", "b", likelihood, credibility=0.0)
        np.testing.assert_allclose(posterior, np.ones(5) / 5)

    def test_high_credibility_changes_posterior(self):
        likelihood = np.asarray([0.8, 0.05, 0.05, 0.05, 0.05])
        posterior = self.model.update_from_message("a", "b", likelihood, credibility=1.0)
        self.assertGreater(posterior[0], 0.7)
        self.assertGreater(certainty(posterior), 0.0)

    def test_behavior_likelihood_is_normalized(self):
        likelihood = action_profile_likelihood(np.asarray([0.8, 0.6]), 5)
        self.assertAlmostEqual(float(np.sum(likelihood)), 1.0)
        self.assertEqual(int(np.argmax(likelihood)), 4)


if __name__ == "__main__":
    unittest.main()
