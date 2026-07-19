from __future__ import annotations

import platform
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from .compression import BudgetController, QueryAwareSelector, fixed_relevance_select, late_interaction_score, mean_vector
from .confidence import ConfidenceEstimator
from .config import config_hash
from .lexical import lexical_scores
from .metrics import aggregate_retrieval_metrics, percentile
from .types import DocumentEmbedding, ExperimentResult, QueryEmbedding, QueryTrace, RankedDocument


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    numbers = np.asarray(list(values.values()), dtype=np.float64)
    minimum = float(numbers.min())
    maximum = float(numbers.max())
    if maximum - minimum < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}


class AdaColRAGPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        complexity = config["complexity"]
        self.budget_controller = BudgetController(
            min_tokens=int(config["min_tokens"]),
            max_tokens=int(config["max_tokens"]),
            multiple=int(config["budget_multiple"]),
            length_weight=float(complexity["query_length_weight"]),
            dispersion_weight=float(complexity["dispersion_weight"]),
            ambiguity_weight=float(complexity["ambiguity_weight"]),
            length_reference=float(complexity["length_reference"]),
        )
        selector = config["selector"]
        self.selector = QueryAwareSelector(
            relevance_weight=float(selector["relevance_weight"]),
            redundancy_weight=float(selector["redundancy_weight"]),
            layout_weight=float(selector["layout_weight"]),
            layout_bins=int(selector["layout_bins"]),
        )
        confidence = config["confidence"]
        self.confidence_estimator = ConfidenceEstimator(
            margin_weight=float(confidence["margin_weight"]),
            entropy_weight=float(confidence["entropy_weight"]),
            coverage_weight=float(confidence["coverage_weight"]),
            coverage_similarity=float(confidence["coverage_similarity"]),
        )

    def _coarse_scores(
        self,
        query: QueryEmbedding,
        documents: dict[str, DocumentEmbedding],
    ) -> dict[str, float]:
        use_dense = query.dense is not None and all(doc.dense is not None for doc in documents.values())
        if use_dense:
            return {doc_id: float(query.dense @ doc.dense) for doc_id, doc in documents.items()}
        query_mean = mean_vector(query.tokens)
        return {doc_id: float(query_mean @ mean_vector(doc.tokens)) for doc_id, doc in documents.items()}

    def _select_tokens(self, query: QueryEmbedding, document: DocumentEmbedding, budget: int) -> np.ndarray:
        mode = str(self.config["mode"])
        if mode == "full":
            return np.arange(document.tokens.shape[0], dtype=np.int64)
        if mode == "fixed":
            return fixed_relevance_select(query.tokens, document.tokens, min(budget, document.tokens.shape[0]))
        return self.selector.select(
            query.tokens,
            document.tokens,
            min(budget, document.tokens.shape[0]),
            document.positions,
        )

    def search(self, query: QueryEmbedding, documents: dict[str, DocumentEmbedding]) -> QueryTrace:
        start = time.perf_counter()
        coarse = self._coarse_scores(query, documents)
        candidate_pool = min(int(self.config["candidate_pool"]), len(documents))
        candidate_ids = sorted(coarse, key=coarse.get, reverse=True)[:candidate_pool]
        coarse_values = np.asarray([coarse[doc_id] for doc_id in candidate_ids], dtype=np.float32)
        mode = str(self.config["mode"])
        if mode == "dense":
            ranked = [
                RankedDocument(doc_id=doc_id, score=float(coarse[doc_id]), selected_tokens=0)
                for doc_id in candidate_ids[: int(self.config["top_k"])]
            ]
            elapsed = (time.perf_counter() - start) * 1000
            return QueryTrace(
                query_id=query.query_id,
                budget=0,
                confidence=1.0,
                fallback=False,
                latency_ms=elapsed,
                candidate_count=candidate_pool,
                processed_token_total=0,
                full_token_total=0,
                scored_document_operations=0,
                ranked=ranked,
            )
        if mode == "full":
            budget = max(doc.tokens.shape[0] for doc in documents.values())
        elif mode == "fixed":
            budget = int(self.config["fixed_tokens"])
        else:
            budget, _ = self.budget_controller.estimate(query.tokens, coarse_values)
        visual_scores: dict[str, float] = {}
        selected_counts: dict[str, int] = {}
        selected_tokens_by_doc: dict[str, np.ndarray] = {}
        processed_token_total = 0
        full_token_total = 0
        scored_document_operations = 0
        for doc_id in candidate_ids:
            document = documents[doc_id]
            selected_indices = self._select_tokens(query, document, budget)
            selected_tokens = document.tokens[selected_indices]
            selected_tokens_by_doc[doc_id] = selected_tokens
            selected_counts[doc_id] = int(selected_indices.size)
            processed_token_total += int(selected_indices.size)
            full_token_total += int(document.tokens.shape[0])
            scored_document_operations += 1
            visual_scores[doc_id] = late_interaction_score(query.tokens, selected_tokens)
        fused_scores = dict(visual_scores)
        fusion = self.config["fusion"]
        dense_weight = float(fusion.get("dense_weight", 0.0))
        lexical_weight = float(fusion.get("lexical_weight", 0.0))
        if dense_weight > 0:
            dense_normalized = _minmax({doc_id: coarse[doc_id] for doc_id in candidate_ids})
            visual_normalized = _minmax(visual_scores)
            fused_scores = {
                doc_id: (1.0 - dense_weight) * visual_normalized[doc_id] + dense_weight * dense_normalized[doc_id]
                for doc_id in candidate_ids
            }
        if lexical_weight > 0:
            lexical = lexical_scores(query.text, {doc_id: documents[doc_id].ocr_text for doc_id in candidate_ids})
            lexical_normalized = _minmax(lexical)
            current_normalized = _minmax(fused_scores)
            fused_scores = {
                doc_id: (1.0 - lexical_weight) * current_normalized[doc_id]
                + lexical_weight * lexical_normalized[doc_id]
                for doc_id in candidate_ids
            }
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        score_array = np.asarray([fused_scores[doc_id] for doc_id in ranked_ids[:10]], dtype=np.float32)
        top_doc_id = ranked_ids[0]
        confidence = self.confidence_estimator.estimate(
            score_array,
            query.tokens,
            selected_tokens_by_doc[top_doc_id],
        )
        fallback_enabled = bool(self.config["fallback"]["enabled"])
        threshold = float(self.config["confidence"]["fallback_threshold"])
        fallback = fallback_enabled and confidence < threshold
        if fallback:
            fallback_pool = min(int(self.config["fallback_pool"]), len(documents))
            fallback_ids = sorted(coarse, key=coarse.get, reverse=True)[:fallback_pool]
            use_full = bool(self.config["fallback"].get("use_full_tokens", True))
            fallback_budget = None if use_full else int(self.config["max_tokens"])
            for doc_id in fallback_ids:
                document = documents[doc_id]
                if fallback_budget is None:
                    selected = document.tokens
                else:
                    indices = self.selector.select(
                        query.tokens,
                        document.tokens,
                        min(fallback_budget, document.tokens.shape[0]),
                        document.positions,
                    )
                    selected = document.tokens[indices]
                selected_counts[doc_id] = int(selected.shape[0])
                processed_token_total += int(selected.shape[0])
                full_token_total += int(document.tokens.shape[0])
                scored_document_operations += 1
                visual_scores[doc_id] = late_interaction_score(query.tokens, selected)
            fused_scores = dict(visual_scores)
            if dense_weight > 0:
                dense_normalized = _minmax({doc_id: coarse[doc_id] for doc_id in fallback_ids})
                visual_normalized = _minmax({doc_id: visual_scores[doc_id] for doc_id in fallback_ids})
                fused_scores = {
                    doc_id: (1.0 - dense_weight) * visual_normalized[doc_id]
                    + dense_weight * dense_normalized[doc_id]
                    for doc_id in fallback_ids
                }
            if lexical_weight > 0:
                lexical = lexical_scores(query.text, {doc_id: documents[doc_id].ocr_text for doc_id in fallback_ids})
                lexical_normalized = _minmax(lexical)
                current_normalized = _minmax(fused_scores)
                fused_scores = {
                    doc_id: (1.0 - lexical_weight) * current_normalized[doc_id]
                    + lexical_weight * lexical_normalized[doc_id]
                    for doc_id in fallback_ids
                }
            ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        top_k = int(self.config["top_k"])
        ranked = [
            RankedDocument(
                doc_id=doc_id,
                score=float(fused_scores[doc_id]),
                selected_tokens=int(selected_counts.get(doc_id, 0)),
            )
            for doc_id in ranked_ids[:top_k]
        ]
        elapsed = (time.perf_counter() - start) * 1000
        return QueryTrace(
            query_id=query.query_id,
            budget=budget,
            confidence=confidence,
            fallback=fallback,
            latency_ms=elapsed,
            candidate_count=candidate_pool,
            processed_token_total=processed_token_total,
            full_token_total=full_token_total,
            scored_document_operations=scored_document_operations,
            ranked=ranked,
        )

    def evaluate(
        self,
        documents: dict[str, DocumentEmbedding],
        queries: dict[str, QueryEmbedding],
        qrels: dict[str, dict[str, int]],
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentResult:
        traces = {query_id: self.search(query, documents) for query_id, query in queries.items()}
        rankings = {
            query_id: [ranked.doc_id for ranked in trace.ranked]
            for query_id, trace in traces.items()
        }
        metrics = aggregate_retrieval_metrics(rankings, qrels)
        latencies = [trace.latency_ms for trace in traces.values()]
        full_mean = float(np.mean([doc.tokens.shape[0] for doc in documents.values()])) if documents else 0.0
        processed_totals = [trace.processed_token_total for trace in traces.values()]
        full_totals = [trace.full_token_total for trace in traces.values()]
        operation_totals = [trace.scored_document_operations for trace in traces.values()]
        processed_sum = float(sum(processed_totals))
        full_sum = float(sum(full_totals))
        compression_ratio = 0.0 if full_sum == 0 else 1.0 - processed_sum / full_sum
        selected_per_operation = 0.0 if sum(operation_totals) == 0 else processed_sum / sum(operation_totals)
        efficiency = {
            "mean_selected_tokens": float(selected_per_operation),
            "mean_full_tokens": full_mean,
            "mean_visual_tokens_processed_per_query": float(np.mean(processed_totals)) if processed_totals else 0.0,
            "mean_scored_document_operations": float(np.mean(operation_totals)) if operation_totals else 0.0,
            "token_reduction": float(compression_ratio),
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
        }
        confidences = [trace.confidence for trace in traces.values()]
        fallbacks = [trace.fallback for trace in traces.values()]
        reliability = {
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "fallback_rate": float(np.mean(fallbacks)) if fallbacks else 0.0,
            "low_confidence_rate": float(
                np.mean([value < float(self.config["confidence"]["fallback_threshold"]) for value in confidences])
            )
            if confidences
            else 0.0,
        }
        per_query = {
            query_id: {
                "budget": trace.budget,
                "confidence": trace.confidence,
                "fallback": trace.fallback,
                "latency_ms": trace.latency_ms,
                "processed_token_total": trace.processed_token_total,
                "full_token_total": trace.full_token_total,
                "scored_document_operations": trace.scored_document_operations,
                "ranking": [asdict(item) for item in trace.ranked],
            }
            for query_id, trace in traces.items()
        }
        result_metadata = {
            "config_hash": config_hash(self.config),
            "mode": self.config["mode"],
            "seed": self.config["seed"],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "queries": len(queries),
            "documents": len(documents),
        }
        if metadata:
            result_metadata.update(metadata)
        return ExperimentResult(
            metrics=metrics,
            efficiency=efficiency,
            reliability=reliability,
            per_query=per_query,
            metadata=result_metadata,
        )
