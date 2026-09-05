from __future__ import annotations

from pathlib import Path

from trace_map.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def smoke_config():
    return load_config(ROOT / "configs" / "smoke.yaml")
