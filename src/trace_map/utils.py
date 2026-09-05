"""Small dependency-light utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from trace_map.types import to_jsonable


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-value))


def softmax(values: Iterable[float], temperature: float = 1.0) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return array
    if temperature <= 0:
        result = np.zeros_like(array)
        result[int(np.argmax(array))] = 1.0
        return result
    shifted = array / temperature - np.max(array / temperature)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float64)


def cosine_similarity(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size:
        raise ValueError("Cosine similarity requires equal-dimensional vectors")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denom <= eps else float(np.dot(left, right) / denom)


def stable_hash(text: str, modulo: int = 2**32) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % modulo


def append_jsonl(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


def run_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        metadata["torch"] = None
    return metadata
