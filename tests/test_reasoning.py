from __future__ import annotations

import unittest

from trace_map.reasoning import ReasoningScheduler
from trace_map.types import ReasoningMode

from common import smoke_config


class ReasoningTests(unittest.TestCase):
    def test_long_checkpoint_has_precedence(self):
        scheduler = ReasoningScheduler(smoke_config())
        public = {
            "wealth_inequality": 0.2,
            "social_welfare": 2.0,
            "output": 3.0,
            "wage": 1.0,
            "public_debt": 0.2,
            "aggregate_expenditure": 0.1,
        }
        self.assertIs(scheduler.mode(4, public, public), ReasoningMode.LONG)

    def test_shock_and_inactive_modes(self):
        scheduler = ReasoningScheduler(smoke_config())
        previous = {key: 0.0 for key in (
            "wealth_inequality", "social_welfare", "output", "wage", "public_debt", "aggregate_expenditure"
        )}
        calm = dict(previous)
        shock = dict(previous)
        shock["output"] = 0.2
        self.assertIs(scheduler.mode(1, calm, previous), ReasoningMode.INACTIVE)
        self.assertIs(scheduler.mode(1, shock, previous), ReasoningMode.SHORT)


if __name__ == "__main__":
    unittest.main()
