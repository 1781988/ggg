#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Some server profiles globally enable the deprecated hf_transfer backend. It is
# incompatible with environments where hf_transfer is not installed and is not
# needed when using a Hugging Face mirror. This must run before importing
# transformers/huggingface_hub because Hub environment flags are read at import.
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from adacolrag.io import load_dataset_metadata


def safe_name(identifier: str) -> str:
    return identifier.replace("/", "__").replace("\\", "__")


def completed_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def weighted_mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weighted_mask = attention_mask * attention_mask.cumsum(dim=1)
    numerator = torch.sum(hidden * weighted_mask.unsqueeze(-1).float(), dim=1)
    denominator = weighted_mask.sum(dim=1, keepdim=True).float().clamp_min(1e-8)
    return numerator / denominator


def resolve_image(dataset_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_dir / path


def encode(model, tokenizer, values, is_text: bool) -> np.ndarray:
    inputs = {
        "text": values if is_text else [""] * len(values),
        "image": [None] * len(values) if is_text else values,
        "tokenizer": tokenizer,
    }
    with torch.inference_mode():
        outputs = model(**inputs)
        pooled = weighted_mean_pooling(outputs.last_hidden_state, outputs.attention_mask)
        return torch.nn.functional.normalize(pooled, p=2, dim=1).detach().float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="openbmb/VisRAG-Ret")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even when files already exist")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    document_dir = output_dir / "documents"
    query_dir = output_dir / "queries"
    document_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    corpus, queries, _ = load_dataset_metadata(dataset_dir)

    pending_corpus = [
        row
        for row in corpus
        if args.force or not completed_file(document_dir / f"{safe_name(str(row['doc_id']))}.npy")
    ]
    pending_queries = [
        row
        for row in queries
        if args.force or not completed_file(query_dir / f"{safe_name(str(row['query_id']))}.npy")
    ]

    if not pending_corpus and not pending_queries and completed_file(output_dir / "manifest.json"):
        print(
            f"[reuse] VisRAG embeddings already complete: {len(corpus)} documents, {len(queries)} queries",
            flush=True,
        )
        return

    print(
        json.dumps(
            {
                "model": args.model,
                "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                "hub_cache": os.environ.get("HF_HUB_CACHE"),
                "legacy_hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "unset"),
                "documents_total": len(corpus),
                "documents_pending": len(pending_corpus),
                "queries_total": len(queries),
                "queries_pending": len(pending_queries),
            },
            indent=2,
        ),
        flush=True,
    )

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(args.device).eval()

    for start in tqdm(range(0, len(pending_corpus), args.batch_size), desc="documents"):
        rows = pending_corpus[start : start + args.batch_size]
        images = []
        for row in rows:
            with Image.open(resolve_image(dataset_dir, str(row["image_path"]))) as image:
                images.append(image.convert("RGB"))
        embeddings = encode(model, tokenizer, images, is_text=False)
        for row, embedding in zip(rows, embeddings, strict=True):
            np.save(document_dir / f"{safe_name(str(row['doc_id']))}.npy", embedding.astype(np.float32))

    instruction = "Represent this query for retrieving relevant documents: "
    for start in tqdm(range(0, len(pending_queries), args.batch_size), desc="queries"):
        rows = pending_queries[start : start + args.batch_size]
        texts = [instruction + str(row["text"]) for row in rows]
        embeddings = encode(model, tokenizer, texts, is_text=True)
        for row, embedding in zip(rows, embeddings, strict=True):
            np.save(query_dir / f"{safe_name(str(row['query_id']))}.npy", embedding.astype(np.float32))

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "format_version": 1,
                "type": "visrag_dense",
                "model": args.model,
                "dtype": args.dtype,
                "documents": len(corpus),
                "queries": len(queries),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
