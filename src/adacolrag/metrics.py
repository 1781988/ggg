from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def _dcg(relevances: Sequence[int], k: int) -> float:
    score = 0.0
    for rank, relevance in enumerate(relevances[:k], start=1):
        score += (2**relevance - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    gains = [int(qrels.get(doc_id, 0)) for doc_id in ranked[:k]]
    ideal = sorted((int(value) for value in qrels.values()), reverse=True)
    denominator = _dcg(ideal, k)
    return 0.0 if denominator == 0 else _dcg(gains, k) / denominator


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, relevance in qrels.items() if relevance > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked[:k])
    return len(relevant & retrieved) / len(relevant)


def reciprocal_rank_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def aggregate_retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, float]:
    values: dict[str, list[float]] = {
        "nDCG@5": [],
        "nDCG@10": [],
        "Recall@1": [],
        "Recall@5": [],
        "Recall@10": [],
        "MRR@10": [],
    }
    for query_id, relevant in qrels.items():
        ranked = rankings.get(query_id, [])
        values["nDCG@5"].append(ndcg_at_k(ranked, relevant, 5))
        values["nDCG@10"].append(ndcg_at_k(ranked, relevant, 10))
        values["Recall@1"].append(recall_at_k(ranked, relevant, 1))
        values["Recall@5"].append(recall_at_k(ranked, relevant, 5))
        values["Recall@10"].append(recall_at_k(ranked, relevant, 10))
        values["MRR@10"].append(reciprocal_rank_at_k(ranked, relevant, 10))
    return {name: float(np.mean(metric_values)) if metric_values else 0.0 for name, metric_values in values.items()}


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0
