from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def lexical_scores(query: str, documents: dict[str, str]) -> dict[str, float]:
    query_terms = tokenize(query)
    if not query_terms:
        return {doc_id: 0.0 for doc_id in documents}
    doc_tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
    document_frequency: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        document_frequency.update(set(tokens))
    total_documents = max(len(doc_tokens), 1)
    scores: dict[str, float] = {}
    for doc_id, tokens in doc_tokens.items():
        frequencies = Counter(tokens)
        length_norm = max(math.sqrt(len(tokens)), 1.0)
        score = 0.0
        for term in query_terms:
            inverse_frequency = math.log((total_documents + 1) / (document_frequency[term] + 1)) + 1.0
            score += frequencies[term] * inverse_frequency
        scores[doc_id] = score / length_norm
    return scores
