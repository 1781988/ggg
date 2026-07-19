from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DocumentEmbedding:
    doc_id: str
    tokens: np.ndarray
    positions: np.ndarray | None = None
    dense: np.ndarray | None = None
    ocr_text: str = ""


@dataclass(frozen=True)
class QueryEmbedding:
    query_id: str
    text: str
    tokens: np.ndarray
    dense: np.ndarray | None = None


@dataclass(frozen=True)
class RankedDocument:
    doc_id: str
    score: float
    selected_tokens: int


@dataclass
class QueryTrace:
    query_id: str
    budget: int
    confidence: float
    fallback: bool
    latency_ms: float
    candidate_count: int
    processed_token_total: int = 0
    full_token_total: int = 0
    scored_document_operations: int = 0
    ranked: list[RankedDocument] = field(default_factory=list)


@dataclass
class ExperimentResult:
    metrics: dict[str, float]
    efficiency: dict[str, float]
    reliability: dict[str, float]
    per_query: dict[str, dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "efficiency": self.efficiency,
            "reliability": self.reliability,
            "per_query": self.per_query,
            "metadata": self.metadata,
        }
