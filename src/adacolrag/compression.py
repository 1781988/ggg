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


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Return deterministic descending Top-K indices without sorting the full array."""

    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= values.size:
        return np.argsort(values, kind="stable")[::-1].astype(np.int64)
    partition = np.argpartition(values, values.size - k)[-k:]
    order = np.argsort(values[partition], kind="stable")[::-1]
    return partition[order].astype(np.int64)


@dataclass(frozen=True)
class BudgetController:
    """Legacy adaptive-budget controller retained for archived diagnostics.

    The submission method uses a fixed budget selected on the development corpus.
    Keeping this class preserves backwards compatibility with archived experiments.
    """

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
    prefilter_factor: float = 4.0

    def select(
        self,
        query_tokens: np.ndarray,
        document_tokens: np.ndarray,
        budget: int,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select query-relevant, non-redundant and spatially distributed tokens.

        Exact greedy MMR over all 1,024 page tokens is unnecessarily expensive.
        When ``prefilter_factor`` is positive, relevance Top-(factor x budget)
        candidates are formed first and MMR is applied only inside that pool. A
        factor of zero reproduces the exact archived selector.
        """

        token_count = int(document_tokens.shape[0])
        budget = min(max(int(budget), 0), token_count)
        if budget == 0:
            return np.empty(0, dtype=np.int64)
        if budget >= token_count:
            return np.arange(token_count, dtype=np.int64)

        relevance = (query_tokens @ document_tokens.T).max(axis=0)
        valid_positions = positions is not None and positions.shape[0] == token_count
        use_redundancy = self.redundancy_weight > 0.0
        use_layout = self.layout_weight > 0.0 and valid_positions

        if not use_redundancy and not use_layout:
            return topk_indices(relevance, budget)

        pool_size = token_count
        if self.prefilter_factor > 0.0:
            pool_size = min(
                token_count,
                max(budget, int(math.ceil(float(budget) * self.prefilter_factor))),
            )
        pool_indices = (
            topk_indices(relevance, pool_size)
            if pool_size < token_count
            else np.arange(token_count, dtype=np.int64)
        )
        pool_tokens = document_tokens[pool_indices]
        pool_relevance = relevance[pool_indices]
        pool_positions = positions[pool_indices] if valid_positions else None

        selected_local: list[int] = [int(np.argmax(pool_relevance))]
        available = np.ones(pool_size, dtype=bool)
        available[selected_local[0]] = False

        max_redundancy: np.ndarray | None = None
        if use_redundancy:
            max_redundancy = pool_tokens @ pool_tokens[selected_local[0]]

        coverage = np.zeros((self.layout_bins, self.layout_bins), dtype=np.int32)
        if use_layout:
            assert pool_positions is not None
            x, y = np.clip(pool_positions[selected_local[0]], 0.0, 0.999999)
            coverage[int(y * self.layout_bins), int(x * self.layout_bins)] += 1

        while len(selected_local) < budget:
            candidates = np.flatnonzero(available)
            if candidates.size == 0:
                break
            utility = self.relevance_weight * pool_relevance[candidates]
            if use_redundancy:
                assert max_redundancy is not None
                utility = utility - self.redundancy_weight * max_redundancy[candidates]
            if use_layout:
                assert pool_positions is not None
                candidate_positions = np.clip(pool_positions[candidates], 0.0, 0.999999)
                bx = (candidate_positions[:, 0] * self.layout_bins).astype(int)
                by = (candidate_positions[:, 1] * self.layout_bins).astype(int)
                layout_bonus = 1.0 / (1.0 + coverage[by, bx])
                utility = utility + self.layout_weight * layout_bonus

            chosen = int(candidates[int(np.argmax(utility))])
            selected_local.append(chosen)
            available[chosen] = False

            if use_redundancy:
                assert max_redundancy is not None
                similarity_to_chosen = pool_tokens @ pool_tokens[chosen]
                np.maximum(max_redundancy, similarity_to_chosen, out=max_redundancy)
            if use_layout:
                assert pool_positions is not None
                x, y = np.clip(pool_positions[chosen], 0.0, 0.999999)
                coverage[int(y * self.layout_bins), int(x * self.layout_bins)] += 1

        return pool_indices[np.asarray(selected_local, dtype=np.int64)]


def fixed_relevance_select(query_tokens: np.ndarray, document_tokens: np.ndarray, budget: int) -> np.ndarray:
    if budget >= document_tokens.shape[0]:
        return np.arange(document_tokens.shape[0], dtype=np.int64)
    relevance = (query_tokens @ document_tokens.T).max(axis=0)
    return topk_indices(relevance, budget)
