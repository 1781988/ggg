#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a ViDoRe V3 Hugging Face dataset and convert it to AdaColRAG manifests"
    )
    parser.add_argument("--dataset", default="vidore/vidore_v3_finance_en")
    parser.add_argument("--split", default="test")
    parser.add_argument("--language", default="english")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {"split": args.split}
    if args.revision:
        load_kwargs["revision"] = args.revision

    corpus_ds = load_dataset(args.dataset, "corpus", **load_kwargs)
    queries_ds = load_dataset(args.dataset, "queries", **load_kwargs)
    qrels_ds = load_dataset(args.dataset, "qrels", **load_kwargs)

    corpus_rows: list[dict] = []
    valid_doc_ids: set[str] = set()
    for row in tqdm(corpus_ds, desc="saving corpus images"):
        doc_id = str(row["corpus_id"])
        image_path = image_dir / f"{doc_id}.png"
        if not image_path.exists():
            row["image"].convert("RGB").save(image_path)
        corpus_rows.append(
            {
                "doc_id": doc_id,
                "image_path": str(image_path.relative_to(output_dir)),
                "ocr_text": str(row.get("markdown", "") or ""),
            }
        )
        valid_doc_ids.add(doc_id)

    query_rows: list[dict] = []
    valid_query_ids: set[str] = set()
    requested_language = args.language.strip().lower()
    for row in queries_ds:
        row_language = str(row.get("language", "")).strip().lower()
        if requested_language and row_language and row_language != requested_language:
            continue
        query_id = str(row["query_id"])
        query_rows.append({"query_id": query_id, "text": str(row["query"])})
        valid_query_ids.add(query_id)

    qrel_rows: list[dict] = []
    for row in qrels_ds:
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

    if not corpus_rows:
        raise RuntimeError("No corpus pages were downloaded")
    if not query_rows:
        raise RuntimeError("No queries remain after language filtering")
    if not qrel_rows:
        raise RuntimeError("No qrels match the selected queries and corpus")

    write_jsonl(output_dir / "corpus.jsonl", corpus_rows)
    write_jsonl(output_dir / "queries.jsonl", query_rows)
    write_jsonl(output_dir / "qrels.jsonl", qrel_rows)

    metadata = {
        "source_dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "language": args.language,
        "documents": len(corpus_rows),
        "queries": len(query_rows),
        "qrels": len(qrel_rows),
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
