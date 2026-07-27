from __future__ import annotations

import numpy as np

from .io import l2_normalize
from .types import DocumentEmbedding, QueryEmbedding


def build_synthetic_fixture(seed: int = 42) -> tuple[
    dict[str, DocumentEmbedding],
    dict[str, QueryEmbedding],
    dict[str, dict[str, int]],
]:
    rng = np.random.default_rng(seed)
    dimension = 32
    topics = l2_normalize(rng.normal(size=(6, dimension)))
    documents: dict[str, DocumentEmbedding] = {}
    positions_axis = np.linspace(0.02, 0.98, 16)
    positions = np.asarray([(x, y) for y in positions_axis for x in positions_axis], dtype=np.float32)
    for index, topic in enumerate(topics):
        relevant = topic + 0.08 * rng.normal(size=(80, dimension))
        distractors = rng.normal(size=(176, dimension))
        tokens = l2_normalize(np.vstack([relevant, distractors]))
        dense = l2_normalize(tokens[:80].mean(axis=0), axis=0)
        documents[f"doc-{index}"] = DocumentEmbedding(
            doc_id=f"doc-{index}",
            tokens=tokens,
            positions=positions,
            dense=dense,
            ocr_text=f"topic {index} revenue table visual evidence",
        )
    queries: dict[str, QueryEmbedding] = {}
    qrels: dict[str, dict[str, int]] = {}
    for index, topic in enumerate(topics):
        count = 6 + index
        query_tokens = l2_normalize(topic + 0.05 * rng.normal(size=(count, dimension)))
        queries[f"q-{index}"] = QueryEmbedding(
            query_id=f"q-{index}",
            text=f"find topic {index} revenue evidence",
            tokens=query_tokens,
            dense=l2_normalize(query_tokens.mean(axis=0), axis=0),
        )
        qrels[f"q-{index}"] = {f"doc-{index}": 1}
    return documents, queries, qrels
