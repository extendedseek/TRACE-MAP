from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trace_map.envs import make_environment
from trace_map.evaluate import run_smoke
from trace_map.language import make_language_components
from trace_map.pipeline import TraceMapPipeline

from common import smoke_config


class PipelineTests(unittest.TestCase):
    def test_feature_dimensions_and_trace(self):
        config = smoke_config()
        environment = make_environment(config)
        policy, encoder = make_language_components(config)
        pipeline = TraceMapPipeline(config, environment, policy, encoder)
        observation = environment.reset(7)
        prepared = pipeline.prepare(observation)
        for agent_id in environment.agent_ids:
            expected = observation.local[agent_id].size + 4 * config["language"]["embedding_dim"]
            self.assertEqual(prepared.features[agent_id].actor_input.size, expected)
        actions = pipeline.apply_strategic_deviations(
            pipeline.heuristic_actions(prepared), prepared
        )
        result = environment.step(actions)
        records = pipeline.finalize_step(prepared, result)
        self.assertEqual(len(records), len(environment.agent_ids))
        self.assertTrue(all(len(bank) == 1 for bank in pipeline.memory_banks.values()))

    def test_smoke_writes_auditable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_smoke(smoke_config(), directory)
            self.assertEqual(metrics["steps"], 12)
            self.assertTrue((Path(directory) / "decision_trace.jsonl").is_file())
            self.assertTrue((Path(directory) / "pipeline_state.json").is_file())


if __name__ == "__main__":
    unittest.main()
