from __future__ import annotations

import copy
import unittest

from trace_map.config import (
    ConfigError,
    checkpoint_digest,
    deep_merge,
    load_config,
    validate_config,
)

from common import ROOT, smoke_config


class ConfigTests(unittest.TestCase):
    def test_smoke_config_is_valid(self):
        config = smoke_config()
        self.assertEqual(config["environment"]["backend"], "mock")
        self.assertEqual(config["memory"]["selected_count"], 2)

    def test_yaml_and_dotted_overrides(self):
        config = load_config(
            ROOT / "configs" / "base.yaml",
            [ROOT / "configs" / "environment" / "e3.yaml"],
            ["run.seed=42", "environment.backend=mock"],
        )
        self.assertEqual(config["run"]["seed"], 42)
        self.assertEqual(config["environment"]["condition"], "e3")
        self.assertAlmostEqual(config["environment"]["interest_rate"], 0.085)

    def test_deep_merge_preserves_unrelated_keys(self):
        result = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"x": 3}})
        self.assertEqual(result, {"a": {"x": 3, "y": 2}})

    def test_invalid_memory_counts_fail(self):
        config = copy.deepcopy(smoke_config())
        config["memory"]["selected_count"] = 5
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_checkpoint_digest_allows_evaluation_regime_change(self):
        base = load_config(
            ROOT / "configs" / "base.yaml",
            assignments=["environment.backend=mock"],
        )
        stress = load_config(
            ROOT / "configs" / "base.yaml",
            [ROOT / "configs" / "environment" / "e3.yaml"],
            ["environment.backend=mock", "run.seed=4"],
        )
        self.assertEqual(checkpoint_digest(base), checkpoint_digest(stress))


if __name__ == "__main__":
    unittest.main()
