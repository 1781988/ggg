from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .types import DocumentEmbedding, QueryEmbedding


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.clip(norm, 1e-8, None)


def _safe_name(identifier: str) -> str:
    return identifier.replace("/", "__").replace("\\", "__")


def load_dataset_metadata(dataset_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    root = Path(dataset_dir)
    corpus = read_jsonl(root / "corpus.jsonl")
    queries = read_jsonl(root / "queries.jsonl")
    qrels_records = read_jsonl(root / "qrels.jsonl")
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_records:
        qrels.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = int(row.get("relevance", 1))
    return corpus, queries, qrels


def load_colvision_embeddings(
    dataset_dir: str | Path,
    embedding_dir: str | Path,
) -> tuple[dict[str, DocumentEmbedding], dict[str, QueryEmbedding]]:
    corpus, queries, _ = load_dataset_metadata(dataset_dir)
    emb_root = Path(embedding_dir)
    documents: dict[str, DocumentEmbedding] = {}
    for row in corpus:
        doc_id = str(row["doc_id"])
        path = emb_root / "documents" / f"{_safe_name(doc_id)}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing document embedding: {path}")
        payload = np.load(path)
        tokens = l2_normalize(payload["tokens"])
        positions = payload["positions"] if "positions" in payload.files else None
        documents[doc_id] = DocumentEmbedding(
            doc_id=doc_id,
            tokens=tokens,
            positions=positions,
            ocr_text=str(row.get("ocr_text", "")),
        )
    query_embeddings: dict[str, QueryEmbedding] = {}
    for row in queries:
        query_id = str(row["query_id"])
        path = emb_root / "queries" / f"{_safe_name(query_id)}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing query embedding: {path}")
        tokens = l2_normalize(np.load(path))
        query_embeddings[query_id] = QueryEmbedding(
            query_id=query_id,
            text=str(row["text"]),
            tokens=tokens,
        )
    return documents, query_embeddings


def attach_dense_embeddings(
    documents: dict[str, DocumentEmbedding],
    queries: dict[str, QueryEmbedding],
    dense_dir: str | Path,
) -> tuple[dict[str, DocumentEmbedding], dict[str, QueryEmbedding]]:
    root = Path(dense_dir)
    updated_docs: dict[str, DocumentEmbedding] = {}
    for doc_id, doc in documents.items():
        path = root / "documents" / f"{_safe_name(doc_id)}.npy"
        dense = l2_normalize(np.load(path).reshape(-1), axis=0) if path.exists() else None
        updated_docs[doc_id] = DocumentEmbedding(
            doc_id=doc.doc_id,
            tokens=doc.tokens,
            positions=doc.positions,
            dense=dense,
            ocr_text=doc.ocr_text,
        )
    updated_queries: dict[str, QueryEmbedding] = {}
    for query_id, query in queries.items():
        path = root / "queries" / f"{_safe_name(query_id)}.npy"
        dense = l2_normalize(np.load(path).reshape(-1), axis=0) if path.exists() else None
        updated_queries[query_id] = QueryEmbedding(
            query_id=query.query_id,
            text=query.text,
            tokens=query.tokens,
            dense=dense,
        )
    return updated_docs, updated_queries


def dataset_checksum(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]
