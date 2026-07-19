#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload["data"]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize corpus/query/qrels manifests to AdaColRAG JSONL")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corpus = read_rows(Path(args.corpus))
    queries = read_rows(Path(args.queries))
    qrels = read_rows(Path(args.qrels))
    normalized_corpus = []
    image_dir = output / "images"
    if args.copy_images:
        image_dir.mkdir(exist_ok=True)
    for row in corpus:
        doc_id = str(row.get("doc_id", row.get("id")))
        image_path = Path(str(row.get("image_path", row.get("image"))))
        if args.copy_images:
            destination = image_dir / f"{doc_id}{image_path.suffix.lower()}"
            shutil.copy2(image_path, destination)
            image_value = str(destination.relative_to(output))
        else:
            image_value = str(image_path)
        normalized_corpus.append(
            {"doc_id": doc_id, "image_path": image_value, "ocr_text": str(row.get("ocr_text", ""))}
        )
    normalized_queries = [
        {"query_id": str(row.get("query_id", row.get("id"))), "text": str(row.get("text", row.get("query")))}
        for row in queries
    ]
    normalized_qrels = [
        {
            "query_id": str(row.get("query_id")),
            "doc_id": str(row.get("doc_id")),
            "relevance": int(row.get("relevance", row.get("score", 1))),
        }
        for row in qrels
    ]
    write_jsonl(output / "corpus.jsonl", normalized_corpus)
    write_jsonl(output / "queries.jsonl", normalized_queries)
    write_jsonl(output / "qrels.jsonl", normalized_qrels)


if __name__ == "__main__":
    main()
