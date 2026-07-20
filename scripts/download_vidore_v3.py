#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_stream_writer(path: Path):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    return partial, partial.open("w", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a ViDoRe V3 Hugging Face dataset and convert it to AdaColRAG manifests"
    )
    parser.add_argument("--dataset", default="vidore/vidore_v3_finance_en")
    parser.add_argument("--split", default="test")
    parser.add_argument("--language", default="english")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream parquet rows and save pages progressively instead of waiting for the full corpus download",
    )
    parser.add_argument("--max-documents", type=int, default=None, help="Optional debug limit")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional debug limit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {"split": args.split, "streaming": args.streaming}
    if args.revision:
        load_kwargs["revision"] = args.revision

    print(
        json.dumps(
            {
                "stage": "initializing",
                "dataset": args.dataset,
                "split": args.split,
                "streaming": args.streaming,
                "output_dir": str(output_dir.resolve()),
                "HF_HOME": os.environ.get("HF_HOME"),
                "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    print("Loading query metadata...", flush=True)
    queries_ds = load_dataset(args.dataset, "queries", **load_kwargs)
    query_rows: list[dict] = []
    valid_query_ids: set[str] = set()
    requested_language = args.language.strip().lower()
    for index, row in enumerate(tqdm(queries_ds, desc="queries", unit="query")):
        if args.max_queries is not None and index >= args.max_queries:
            break
        row_language = str(row.get("language", "")).strip().lower()
        if requested_language and row_language and row_language != requested_language:
            continue
        query_id = str(row["query_id"])
        query_rows.append({"query_id": query_id, "text": str(row["query"])})
        valid_query_ids.add(query_id)
    if not query_rows:
        raise RuntimeError("No queries remain after language filtering")
    write_jsonl(output_dir / "queries.jsonl", query_rows)
    print(f"Saved {len(query_rows)} queries", flush=True)

    print("Streaming corpus pages; images will now appear progressively...", flush=True)
    corpus_ds = load_dataset(args.dataset, "corpus", **load_kwargs)
    valid_doc_ids: set[str] = set()
    corpus_path = output_dir / "corpus.jsonl"
    partial_path, corpus_handle = atomic_stream_writer(corpus_path)
    document_count = 0
    try:
        for index, row in enumerate(tqdm(corpus_ds, desc="corpus pages", unit="page")):
            if args.max_documents is not None and index >= args.max_documents:
                break
            doc_id = str(row["corpus_id"])
            image_path = image_dir / f"{doc_id}.png"
            if not image_path.exists():
                row["image"].convert("RGB").save(image_path)
            corpus_handle.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "image_path": str(image_path.relative_to(output_dir)),
                        "ocr_text": str(row.get("markdown", "") or ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            document_count += 1
            valid_doc_ids.add(doc_id)
            if document_count % 25 == 0:
                corpus_handle.flush()
    finally:
        corpus_handle.close()
    partial_path.replace(corpus_path)
    if document_count == 0:
        raise RuntimeError("No corpus pages were downloaded")
    print(f"Saved {document_count} corpus pages", flush=True)

    print("Loading qrels...", flush=True)
    qrels_ds = load_dataset(args.dataset, "qrels", **load_kwargs)
    qrel_rows: list[dict] = []
    for row in tqdm(qrels_ds, desc="qrels", unit="pair"):
        query_id = str(row["query_id"])
        doc_id = str(row["corpus_id"])
        if query_id not in valid_query_ids or doc_id not in valid_doc_ids:
            continue
        qrel_rows.append(
            {
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": int(row.get("score", 1)),
            }
        )
    if not qrel_rows:
        raise RuntimeError("No qrels match the selected queries and corpus")
    write_jsonl(output_dir / "qrels.jsonl", qrel_rows)

    metadata = {
        "source_dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "language": args.language,
        "streaming": args.streaming,
        "documents": document_count,
        "queries": len(query_rows),
        "qrels": len(qrel_rows),
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
