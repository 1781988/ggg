from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "candidate_pool": 50,
    "fallback_pool": 100,
    "top_k": 10,
    "mode": "fixed_mmr",
    "fixed_tokens": 112,
    "min_tokens": 32,
    "max_tokens": 128,
    "budget_multiple": 8,
    "complexity": {
        "query_length_weight": 0.35,
        "dispersion_weight": 0.30,
        "ambiguity_weight": 0.35,
        "length_reference": 24,
    },
    "selector": {
        "relevance_weight": 1.0,
        "redundancy_weight": 0.25,
        "layout_weight": 0.10,
        "layout_bins": 4,
        "prefilter_factor": 4.0,
    },
    "confidence": {
        "margin_weight": 0.40,
        "entropy_weight": 0.25,
        "coverage_weight": 0.35,
        "coverage_similarity": 0.25,
        "fallback_threshold": 0.48,
    },
    "fusion": {
        "dense_weight": 0.0,
        "lexical_weight": 0.0,
    },
    "fallback": {
        "enabled": True,
        "use_full_tokens": True,
    },
    "acceptance": {
        "max_ndcg5_drop": 0.01,
        "min_token_reduction": 0.45,
        "min_latency_reduction": 0.30,
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = _deep_merge(DEFAULT_CONFIG, loaded)
    return config


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
