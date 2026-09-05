from __future__ import annotations

import unittest

import numpy as np

from trace_map.communication import (
    CounterfactualCredibility,
    CounterfactualValues,
    FactualVerifier,
    commitment_consistency,
)
from trace_map.language.template_backend import HashingTextEncoder
from trace_map.types import CandidateMessage, StructuredClaim

from common import smoke_config


class CommunicationTests(unittest.TestCase):
    def setUp(self):
        self.config = smoke_config()
        self.encoder = HashingTextEncoder(self.config["language"]["embedding_dim"])
        self.claim = StructuredClaim(
            "output", "decrease", 3, np.zeros(2), np.asarray([0.8, 0.8]), 1.0
        )
        self.candidate = CandidateMessage(
            "c0",
            "household_0",
            "Output will decrease; reduce saving now.",
            self.claim,
            self.encoder.encode(["message"])[0],
        )

    def test_factual_support_tracks_observed_direction(self):
        current, previous = {"output": 0.8}, {"output": 1.0}
        self.assertAlmostEqual(FactualVerifier.support(self.claim, current, previous), 1.0)

    def test_commitment_consistency(self):
        self.assertAlmostEqual(commitment_consistency(np.zeros(2), np.zeros(2)), 1.0)
        self.assertLess(commitment_consistency(np.ones(2), -np.ones(2)), 0.1)

    def test_counterfactual_harm_is_labeled(self):
        verifier = CounterfactualCredibility(self.config)

        def evaluator(claim, predicted, receiver):
            return CounterfactualValues(0.0, 0.1, -0.6, 0.0, 0.5)

        audit = verifier.audit(
            self.candidate,
            "household_1",
            {"output": 0.8},
            {"output": 1.0},
            predicted_sender_action=np.ones(2),
            posterior_certainty=1.0,
            relevance=1.0,
            influence=1.0,
            evaluator=evaluator,
        )
        self.assertTrue(audit.harmful)
        self.assertGreater(audit.strategic_risk, 0.5)
        self.assertLess(audit.trust_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
