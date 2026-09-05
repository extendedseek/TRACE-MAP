from __future__ import annotations

import copy
import unittest

import numpy as np

from trace_map.language.template_backend import HashingTextEncoder
from trace_map.memory import MemoryItem, RegimeAwareMemoryBank, regime_compatibility
from trace_map.types import Regime

from common import smoke_config


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(smoke_config())
        self.config["memory"]["selected_count"] = 1
        self.encoder = HashingTextEncoder(self.config["language"]["embedding_dim"])
        self.e1 = Regime("e1", 0.055, 0.055, 0.035)
        self.e3 = Regime("e3", 0.110, 0.090, 0.085)

    def _item(self, name, regime, text, step=0):
        return MemoryItem(
            name,
            text,
            regime,
            "reasoning",
            "message",
            np.zeros(2),
            1.0,
            0.5,
            step,
            self.encoder.encode([text])[0],
        )

    def test_regime_compatibility_is_reflexive_and_discriminative(self):
        self.assertAlmostEqual(regime_compatibility(self.e1, self.e1), 1.0)
        self.assertLess(regime_compatibility(self.e1, self.e3), 0.5)

    def test_retrieval_prefers_matching_regime_for_equal_semantics(self):
        bank = RegimeAwareMemoryBank(self.config)
        embedding = self.encoder.encode(["liquidity buffer under output decline"])[0]
        first = self._item("e1", self.e1, "liquidity buffer under output decline")
        second = self._item("e3", self.e3, "liquidity buffer under output decline")
        # Avoid prospective merge so both regimes remain available.
        bank.add(first)
        bank.add(second)
        selected = bank.retrieve(embedding, self.e3, step=10)
        self.assertEqual(selected.ids, ["e3"])

    def test_removal_attribution_updates_reliability(self):
        bank = RegimeAwareMemoryBank(self.config)
        bank.add(self._item("useful", self.e1, "useful memory"))
        query = self.encoder.encode(["useful memory"])[0]
        selected = bank.retrieve(query, self.e1, step=1)
        before = selected.items[0].reliability
        attribution = bank.apply_removal_attribution(
            selected, full_value=1.0, removal_values={"useful": 0.7}
        )
        self.assertAlmostEqual(attribution["useful"], 0.3)
        self.assertGreater(selected.items[0].reliability, before)


if __name__ == "__main__":
    unittest.main()
