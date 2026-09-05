"""Environment adapters."""

from __future__ import annotations

from typing import Any

from trace_map.envs.base import MultiAgentEconomy


def make_environment(config: dict[str, Any]) -> MultiAgentEconomy:
    backend = config["environment"]["backend"]
    if backend == "mock":
        from trace_map.envs.mock_economy import MockEconomicSociety

        return MockEconomicSociety(config)
    if backend == "taxai":
        from trace_map.envs.taxai_adapter import TaxAIAdapter

        return TaxAIAdapter(config)
    raise ValueError(f"Unsupported environment backend: {backend}")


__all__ = ["MultiAgentEconomy", "make_environment"]
