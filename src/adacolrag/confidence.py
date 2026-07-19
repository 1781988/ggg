from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compression import normalized_entropy


@dataclass(frozen=True)
class ConfidenceEstimator:
    margin_weight: float
    entropy_weight: float
    coverage_weight: float
    coverage_similarity: float

    def estimate(self, scores: np.ndarray, query_tokens: np.ndarray, top_document_tokens: np.ndarray) -> float:
        if scores.size == 0:
            return 0.0
        sorted_scores = np.sort(scores)[::-1]
        margin = float(sorted_scores[0] - sorted_scores[1]) if sorted_scores.size >= 2 else 0.0
        margin_component = float(np.clip(margin / 0.20, 0.0, 1.0))
        entropy_component = 1.0 - normalized_entropy(scores)
        token_similarities = (query_tokens @ top_document_tokens.T).max(axis=1)
        coverage_component = float(np.mean(token_similarities >= self.coverage_similarity))
        total = self.margin_weight + self.entropy_weight + self.coverage_weight
        confidence = (
            self.margin_weight * margin_component
            + self.entropy_weight * entropy_component
            + self.coverage_weight * coverage_component
        ) / max(total, 1e-8)
        return float(np.clip(confidence, 0.0, 1.0))
