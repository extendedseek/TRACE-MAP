"""Configuration loading, deep overrides, provenance-preserving saves, and validation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    pass


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {source}")
    return payload


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts or any(not part for part in parts):
        raise ConfigError(f"Invalid override key: {dotted_key}")
    cursor = config
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            cursor[part] = {}
        elif not isinstance(existing, dict):
            raise ConfigError(f"Cannot descend into non-mapping key: {part}")
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_assignment(assignment: str) -> tuple[str, Any]:
    if "=" not in assignment:
        raise ConfigError(f"Override must use key=value syntax: {assignment}")
    key, raw_value = assignment.split("=", 1)
    return key.strip(), yaml.safe_load(raw_value)


def load_config(
    path: str | Path,
    overrides: Iterable[str | Path] = (),
    assignments: Iterable[str] = (),
) -> dict[str, Any]:
    config = _read_yaml(path)
    for override in overrides:
        config = deep_merge(config, _read_yaml(override))
    for assignment in assignments:
        key, value = parse_assignment(assignment)
        set_dotted(config, key, value)
    validate_config(config)
    return config


def require(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        cursor = cursor[part]
    return cursor


def validate_config(config: dict[str, Any]) -> None:
    required = [
        "run.seed",
        "environment.backend",
        "environment.households",
        "environment.episode_length",
        "language.embedding_dim",
        "memory.candidate_count",
        "memory.selected_count",
        "memory.capacity_per_agent",
        "reasoning.long_interval",
        "communication.message_candidates",
        "communication.retained_senders",
        "opponent.profile_count",
    ]
    for key in required:
        require(config, key)

    backend = require(config, "environment.backend")
    if backend not in {"mock", "taxai"}:
        raise ConfigError("environment.backend must be 'mock' or 'taxai'")
    if int(require(config, "environment.households")) < 1:
        raise ConfigError("At least one household is required")
    if int(require(config, "environment.episode_length")) < 1:
        raise ConfigError("episode_length must be positive")
    candidate_count = int(require(config, "memory.candidate_count"))
    selected_count = int(require(config, "memory.selected_count"))
    capacity = int(require(config, "memory.capacity_per_agent"))
    if not 1 <= selected_count <= candidate_count <= capacity:
        raise ConfigError("Memory counts must satisfy 1 <= selected <= candidates <= capacity")
    if int(require(config, "communication.retained_senders")) < 1:
        raise ConfigError("At least one sender must be retained")
    if int(require(config, "opponent.profile_count")) < 2:
        raise ConfigError("Opponent inference needs at least two profiles")

    for key in (
        "environment.depreciation_rate",
        "environment.consumption_tax_rate",
        "environment.interest_rate",
    ):
        value = float(require(config, key))
        if value < 0:
            raise ConfigError(f"{key} must be non-negative")

    for key in (
        "communication.factual_corruption_rate",
        "communication.strategic_deviation_rate",
        "communication.trust_threshold",
    ):
        value = float(require(config, key))
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{key} must lie in [0, 1]")

    reliability = require(config, "communication.persistent_reliability")
    if reliability and len(reliability) < int(require(config, "environment.households")):
        raise ConfigError("persistent_reliability must cover every household or be empty")
    if any(not 0.0 <= float(value) <= 1.0 for value in reliability):
        raise ConfigError("Persistent reliability values must lie in [0, 1]")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def config_digest(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def checkpoint_digest(config: dict[str, Any]) -> str:
    """Digest architecture/training semantics while allowing evaluation scenarios to vary."""

    compatible = deepcopy(config)
    compatible.pop("evaluation", None)
    compatible.get("run", {}).pop("seed", None)
    compatible.get("run", {}).pop("name", None)
    environment = compatible.get("environment", {})
    for key in (
        "condition",
        "depreciation_rate",
        "consumption_tax_rate",
        "interest_rate",
        "shift",
    ):
        environment.pop(key, None)
    communication = compatible.get("communication", {})
    for key in (
        "condition",
        "factual_corruption_rate",
        "strategic_deviation_rate",
        "persistent_reliability",
    ):
        communication.pop(key, None)
    return config_digest(compatible)
