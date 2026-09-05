"""Multi-timescale economic reasoning scheduler (manuscript Eq. 3)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from trace_map.types import ReasoningMode


MONITORED_KEYS = (
    "wealth_inequality",
    "social_welfare",
    "output",
    "wage",
    "public_debt",
    "aggregate_expenditure",
)


class ReasoningScheduler:
    def __init__(self, config: dict[str, Any]):
        cfg = config["reasoning"]
        self.long_interval = int(cfg["long_interval"])
        self.epsilon = float(cfg.get("epsilon", 1e-8))
        thresholds = cfg.get("shock_thresholds")
        self.thresholds = None if thresholds is None else np.asarray(thresholds, dtype=np.float64)
        if self.thresholds is not None and self.thresholds.size != len(MONITORED_KEYS):
            raise ValueError(f"Expected {len(MONITORED_KEYS)} shock thresholds")
        self.shock_quantile = float(cfg.get("shock_quantile", 0.90))
        self.communicate_on_short_shock = bool(cfg.get("communicate_on_short_shock", True))

    @staticmethod
    def vector(public: dict[str, float]) -> np.ndarray:
        return np.asarray([float(public.get(key, 0.0)) for key in MONITORED_KEYS], dtype=np.float64)

    def calibrate(self, public_history: Iterable[dict[str, float]]) -> np.ndarray:
        values = np.stack([self.vector(item) for item in public_history])
        if values.shape[0] < 2:
            raise ValueError("At least two observations are required to calibrate shock thresholds")
        standardized = (values - np.mean(values, axis=0)) / np.maximum(np.std(values, axis=0), self.epsilon)
        changes = np.abs(np.diff(standardized, axis=0))
        self.thresholds = np.maximum(
            np.quantile(changes, self.shock_quantile, axis=0), self.epsilon
        )
        return self.thresholds.copy()

    def mode(
        self,
        step: int,
        current_public: dict[str, float],
        previous_public: dict[str, float] | None,
    ) -> ReasoningMode:
        if step % self.long_interval == 0:
            return ReasoningMode.LONG
        if previous_public is None or self.thresholds is None:
            return ReasoningMode.INACTIVE
        change = np.abs(self.vector(current_public) - self.vector(previous_public))
        normalized = change / (self.thresholds + self.epsilon)
        return ReasoningMode.SHORT if float(np.max(normalized)) > 1.0 else ReasoningMode.INACTIVE

    def communication_active(self, mode: ReasoningMode) -> bool:
        return mode is ReasoningMode.LONG or (
            mode is ReasoningMode.SHORT and self.communicate_on_short_shock
        )
