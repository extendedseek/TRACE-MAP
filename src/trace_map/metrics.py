"""Evaluation metrics used by the manuscript."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from trace_map.types import CounterfactualAudit


def brier_score(probabilities: Sequence[float], labels: Sequence[int | bool]) -> float:
    predicted = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    if predicted.size == 0 or predicted.shape != target.shape:
        raise ValueError("Brier score requires non-empty equal-shaped inputs")
    return float(np.mean((predicted - target) ** 2))


def trusted_harmful_message_rate(
    audits: Iterable[CounterfactualAudit], threshold: float = 0.5
) -> float:
    harmful = [audit for audit in audits if audit.harmful]
    if not harmful:
        return 0.0
    trusted = sum(audit.trust_weight >= threshold for audit in harmful)
    return float(trusted / len(harmful))


def negative_log_likelihood(probability_of_true_profile: Sequence[float]) -> float:
    values = np.asarray(probability_of_true_profile, dtype=np.float64)
    if values.size == 0:
        raise ValueError("NLL requires at least one probability")
    return float(-np.mean(np.log(np.clip(values, 1e-12, 1.0))))


def empirical_exploitability(deviation_gains: Sequence[float]) -> float:
    values = np.asarray(deviation_gains, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(np.maximum(values, 0.0)))


def useful_at_k(counterfactual_return_changes: Sequence[float]) -> float:
    values = np.asarray(counterfactual_return_changes, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(values > 0.0))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(predicted: Sequence[float], observed: Sequence[float]) -> float:
    left = np.asarray(predicted, dtype=np.float64)
    right = np.asarray(observed, dtype=np.float64)
    if left.size < 2 or left.shape != right.shape:
        return 0.0
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def recovery_time(
    series: Sequence[float],
    shift_step: int,
    window: int = 10,
    recovery_fraction: float = 0.95,
) -> int | None:
    values = np.asarray(series, dtype=np.float64)
    if shift_step <= 0 or shift_step >= values.size or window < 1:
        return None
    baseline_start = max(0, shift_step - window)
    target = float(np.mean(values[baseline_start:shift_step])) * recovery_fraction
    for start in range(shift_step, values.size - window + 1):
        if float(np.mean(values[start : start + window])) >= target:
            return start - shift_step
    return None


def aggregate_seed_metrics(records: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                grouped[key].append(float(value))
    return {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values),
        }
        for key, values in sorted(grouped.items())
    }
