from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def late_interaction_score(query_tokens: np.ndarray, document_tokens: np.ndarray) -> float:
    if query_tokens.size == 0 or document_tokens.size == 0:
        return float("-inf")
    similarities = query_tokens @ document_tokens.T
    return float(similarities.max(axis=1).mean())


def mean_vector(tokens: np.ndarray) -> np.ndarray:
    vector = tokens.mean(axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def normalized_entropy(scores: np.ndarray) -> float:
    if scores.size <= 1:
        return 0.0
    shifted = scores - scores.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None))))
    return entropy / math.log(scores.size)


@dataclass(frozen=True)
class BudgetController:
    min_tokens: int
    max_tokens: int
    multiple: int
    length_weight: float
    dispersion_weight: float
    ambiguity_weight: float
    length_reference: float

    def estimate(self, query_tokens: np.ndarray, coarse_scores: np.ndarray) -> tuple[int, float]:
        query_length = min(query_tokens.shape[0] / max(self.length_reference, 1.0), 1.0)
        centroid = query_tokens.mean(axis=0, keepdims=True)
        dispersion = float(np.mean(1.0 - np.clip(query_tokens @ centroid.T, -1.0, 1.0)))
        dispersion = float(np.clip(dispersion, 0.0, 1.0))
        if coarse_scores.size >= 2:
            sorted_scores = np.sort(coarse_scores)[::-1]
            margin = float(sorted_scores[0] - sorted_scores[1])
            ambiguity = float(np.clip(1.0 - margin / 0.25, 0.0, 1.0))
        else:
            ambiguity = 1.0
        total_weight = self.length_weight + self.dispersion_weight + self.ambiguity_weight
        complexity = (
            self.length_weight * query_length
            + self.dispersion_weight * dispersion
            + self.ambiguity_weight * ambiguity
        ) / max(total_weight, 1e-8)
        raw_budget = self.min_tokens + complexity * (self.max_tokens - self.min_tokens)
        rounded = int(round(raw_budget / self.multiple) * self.multiple)
        budget = int(np.clip(rounded, self.min_tokens, self.max_tokens))
        return budget, float(complexity)


@dataclass(frozen=True)
class QueryAwareSelector:
    relevance_weight: float = 1.0
    redundancy_weight: float = 0.25
    layout_weight: float = 0.10
    layout_bins: int = 4

    def select(
        self,
        query_tokens: np.ndarray,
        document_tokens: np.ndarray,
        budget: int,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        token_count = document_tokens.shape[0]
        if budget >= token_count:
            return np.arange(token_count, dtype=np.int64)
        relevance = (query_tokens @ document_tokens.T).max(axis=0)
        selected: list[int] = [int(np.argmax(relevance))]
        available = np.ones(token_count, dtype=bool)
        available[selected[0]] = False
        coverage = np.zeros((self.layout_bins, self.layout_bins), dtype=np.int32)
        if positions is not None and positions.shape[0] == token_count:
            x, y = np.clip(positions[selected[0]], 0.0, 0.999999)
            coverage[int(y * self.layout_bins), int(x * self.layout_bins)] += 1
        while len(selected) < budget:
            candidates = np.flatnonzero(available)
            if candidates.size == 0:
                break
            redundancy = (document_tokens[candidates] @ document_tokens[selected].T).max(axis=1)
            utility = self.relevance_weight * relevance[candidates] - self.redundancy_weight * redundancy
            if positions is not None and positions.shape[0] == token_count:
                candidate_positions = np.clip(positions[candidates], 0.0, 0.999999)
                bx = (candidate_positions[:, 0] * self.layout_bins).astype(int)
                by = (candidate_positions[:, 1] * self.layout_bins).astype(int)
                layout_bonus = 1.0 / (1.0 + coverage[by, bx])
                utility += self.layout_weight * layout_bonus
            chosen = int(candidates[int(np.argmax(utility))])
            selected.append(chosen)
            available[chosen] = False
            if positions is not None and positions.shape[0] == token_count:
                x, y = np.clip(positions[chosen], 0.0, 0.999999)
                coverage[int(y * self.layout_bins), int(x * self.layout_bins)] += 1
        return np.asarray(selected, dtype=np.int64)


def fixed_relevance_select(query_tokens: np.ndarray, document_tokens: np.ndarray, budget: int) -> np.ndarray:
    if budget >= document_tokens.shape[0]:
        return np.arange(document_tokens.shape[0], dtype=np.int64)
    relevance = (query_tokens @ document_tokens.T).max(axis=0)
    return np.argpartition(relevance, -budget)[-budget:]
