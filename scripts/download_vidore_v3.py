#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

CONFIGS = ("corpus", "queries", "qrels")
KNOWN_FILES: dict[str, dict[str, list[str]]] = {
    "vidore/vidore_v3_finance_en": {
        "corpus": [
            "corpus/test-00000-of-00003.parquet",
            "corpus/test-00001-of-00003.parquet",
            "corpus/test-00002-of-00003.parquet",
        ],
        "queries": ["queries/test-00000-of-00001.parquet"],
        "qrels": ["qrels/test-00000-of-00001.parquet"],
    }
}


def log(message: str) -> None:
    print(message, flush=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    count = 0
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    partial.replace(path)
    return count


def expected_files(dataset: str, split: str, revision: str | None) -> dict[str, list[str]]:
    if split == "test" and dataset in KNOWN_FILES:
        return KNOWN_FILES[dataset]

    from huggingface_hub import HfApi

    endpoint = os.environ.get("HF_ENDPOINT") or None
    api = HfApi(endpoint=endpoint)
    log(f"[network] listing repository files from {endpoint or 'https://huggingface.co'}")
    files = api.list_repo_files(repo_id=dataset, repo_type="dataset", revision=revision)
    selected: dict[str, list[str]] = {}
    for config in CONFIGS:
        prefix = f"{config}/{split}-"
        selected[config] = sorted(
            file_name for file_name in files if file_name.startswith(prefix) and file_name.endswith(".parquet")
        )
        if not selected[config]:
            raise RuntimeError(f"No parquet files found for config={config!r}, split={split!r}")
    return selected


def download_raw_files(
    dataset: str,
    revision: str | None,
    split: str,
    raw_dir: Path,
) -> dict[str, list[Path]]:
    from huggingface_hub import hf_hub_download

    endpoint = os.environ.get("HF_ENDPOINT") or None
    selected = expected_files(dataset, split, revision)
    local_files: dict[str, list[Path]] = {}
    for config, filenames in selected.items():
        local_files[config] = []
        log(f"[download] {config}: {len(filenames)} file(s)")
        for index, filename in enumerate(filenames, start=1):
            destination = raw_dir / filename
            if destination.exists() and destination.stat().st_size > 0:
                log(f"[download] reuse {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
                local_files[config].append(destination)
                continue
            log(f"[download] {index}/{len(filenames)} {filename}")
            downloaded = hf_hub_download(
                repo_id=dataset,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                local_dir=raw_dir,
                endpoint=endpoint,
                resume_download=True,
            )
            path = Path(downloaded)
            log(f"[download] saved {path} ({path.stat().st_size / 1024**2:.1f} MiB)")
            local_files[config].append(path)
    return local_files


def discover_local_files(raw_dir: Path, split: str) -> dict[str, list[Path]]:
    local_files: dict[str, list[Path]] = {}
    for config in CONFIGS:
        matches = sorted((raw_dir / config).glob(f"{split}-*.parquet"))
        if not matches:
            raise FileNotFoundError(
                f"No local parquet files found in {raw_dir / config}. "
                "Download the raw files first or remove --convert-only."
            )
        local_files[config] = matches
    return local_files


def load_local_parquet(paths: list[Path], split: str):
    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files={split: [str(path) for path in paths]},
        split=split,
        streaming=True,
    )


def is_complete(output_dir: Path) -> bool:
    required = [
        output_dir / "corpus.jsonl",
        output_dir / "queries.jsonl",
        output_dir / "qrels.jsonl",
        output_dir / "dataset_info.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def convert_raw_files(
    dataset: str,
    revision: str | None,
    split: str,
    language: str,
    raw_files: dict[str, list[Path]],
    output_dir: Path,
    max_documents: int | None,
    max_queries: int | None,
) -> None:
    from tqdm import tqdm

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    requested_language = language.strip().lower()
    queries_ds = load_local_parquet(raw_files["queries"], split)
    query_rows: list[dict] = []
    valid_query_ids: set[str] = set()
    log("[convert] reading queries from local parquet")
    for index, row in enumerate(tqdm(queries_ds, desc="queries", unit="query")):
        if max_queries is not None and index >= max_queries:
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

    corpus_ds = load_local_parquet(raw_files["corpus"], split)
    corpus_path = output_dir / "corpus.jsonl"
    partial_path = corpus_path.with_suffix(corpus_path.suffix + ".partial")
    valid_doc_ids: set[str] = set()
    document_count = 0
    log("[convert] extracting page images from local corpus parquet")
    with partial_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(tqdm(corpus_ds, desc="corpus pages", unit="page")):
            if max_documents is not None and index >= max_documents:
                break
            doc_id = str(row["corpus_id"])
            image_path = image_dir / f"{doc_id}.png"
            if not image_path.exists() or image_path.stat().st_size == 0:
                row["image"].convert("RGB").save(image_path)
            handle.write(
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
                handle.flush()
    partial_path.replace(corpus_path)
    if document_count == 0:
        raise RuntimeError("No corpus pages were converted")

    qrels_ds = load_local_parquet(raw_files["qrels"], split)
    qrel_rows: list[dict] = []
    log("[convert] reading qrels from local parquet")
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
        "source_dataset": dataset,
        "revision": revision,
        "split": split,
        "language": language,
        "raw_files": {key: [str(path) for path in value] for key, value in raw_files.items()},
        "documents": document_count,
        "queries": len(query_rows),
        "qrels": len(qrel_rows),
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumably download ViDoRe V3 parquet files, then convert them to AdaColRAG manifests"
    )
    parser.add_argument("--dataset", default="vidore/vidore_v3_finance_en")
    parser.add_argument("--split", default="test")
    parser.add_argument("--language", default="english")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--enable-xet", action="store_true")
    parser.add_argument("--endpoint", default=None, help="Optional HF endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    if args.download_only and args.convert_only:
        parser.error("--download-only and --convert-only cannot be used together")
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")
    if not args.enable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(
        json.dumps(
            {
                "stage": "initializing",
                "dataset": args.dataset,
                "split": args.split,
                "raw_dir": str(raw_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
                "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                "disable_xet": os.environ.get("HF_HUB_DISABLE_XET"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.convert_only:
        raw_files = discover_local_files(raw_dir, args.split)
    else:
        raw_files = download_raw_files(args.dataset, args.revision, args.split, raw_dir)

    if args.download_only:
        log("[done] raw parquet download completed")
        return

    if is_complete(output_dir) and not args.force_convert:
        log(f"[reuse] normalized dataset already exists in {output_dir}; use --force-convert to rebuild")
        return

    convert_raw_files(
        dataset=args.dataset,
        revision=args.revision,
        split=args.split,
        language=args.language,
        raw_files=raw_files,
        output_dir=output_dir,
        max_documents=args.max_documents,
        max_queries=args.max_queries,
    )
    log("[done] dataset download and conversion completed")


if __name__ == "__main__":
    main()
