"""YAML config loading with dotted CLI overrides (--set train.max_steps=10)."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.startswith("[") and value.endswith("]"):
        return yaml.safe_load(value)
    return value


def deep_update(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in other.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. A config may name `base: other.yaml` (relative path) to inherit from."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        base_path = os.path.join(os.path.dirname(path), base)
        cfg = deep_update(load_config(base_path), cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw)
    return cfg
