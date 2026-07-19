#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from adacolrag.io import load_dataset_metadata, l2_normalize


def safe_name(identifier: str) -> str:
    return identifier.replace("/", "__").replace("\\", "__")


def load_model(model_name: str, device: str, dtype: str):
    from colpali_engine import models as col_models

    lowered = model_name.lower()
    if "colqwen2.5" in lowered and hasattr(col_models, "ColQwen2_5"):
        model_cls = col_models.ColQwen2_5
        processor_cls = col_models.ColQwen2_5_Processor
    elif "colqwen" in lowered:
        model_cls = col_models.ColQwen2
        processor_cls = col_models.ColQwen2Processor
    else:
        model_cls = col_models.ColPali
        processor_cls = col_models.ColPaliProcessor
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = model_cls.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device).eval()
    processor = processor_cls.from_pretrained(model_name)
    return model, processor


def strip_mask(embedding: torch.Tensor, mask: torch.Tensor | None) -> np.ndarray:
    tensor = embedding.detach().float().cpu()
    if mask is not None:
        mask_cpu = mask.detach().bool().cpu().reshape(-1)
        if mask_cpu.numel() == tensor.shape[0]:
            tensor = tensor[mask_cpu]
    array = tensor.numpy()
    nonzero = np.linalg.norm(array, axis=1) > 1e-8
    return l2_normalize(array[nonzero])


def approximate_positions(token_count: int) -> np.ndarray:
    width = max(int(math.ceil(math.sqrt(token_count))), 1)
    height = int(math.ceil(token_count / width))
    positions = []
    for index in range(token_count):
        row, column = divmod(index, width)
        positions.append(((column + 0.5) / width, (row + 0.5) / height))
    return np.asarray(positions, dtype=np.float32)


def resolve_image(dataset_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_dir / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="vidore/colpali-v1.3")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    document_dir = output_dir / "documents"
    query_dir = output_dir / "queries"
    document_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    corpus, queries, _ = load_dataset_metadata(dataset_dir)
    model, processor = load_model(args.model, args.device, args.dtype)

    for start in tqdm(range(0, len(corpus), args.batch_size), desc="documents"):
        rows = corpus[start : start + args.batch_size]
        images = [Image.open(resolve_image(dataset_dir, str(row["image_path"]))).convert("RGB") for row in rows]
        batch = processor.process_images(images).to(model.device)
        with torch.inference_mode():
            embeddings = model(**batch)
        masks = None
        if hasattr(processor, "get_image_mask"):
            try:
                masks = processor.get_image_mask(batch)
            except Exception:
                masks = None
        for index, row in enumerate(rows):
            mask = masks[index] if masks is not None else None
            tokens = strip_mask(embeddings[index], mask)
            positions = approximate_positions(tokens.shape[0])
            np.savez_compressed(
                document_dir / f"{safe_name(str(row['doc_id']))}.npz",
                tokens=tokens.astype(np.float32),
                positions=positions,
            )

    for start in tqdm(range(0, len(queries), args.batch_size), desc="queries"):
        rows = queries[start : start + args.batch_size]
        batch = processor.process_queries([str(row["text"]) for row in rows]).to(model.device)
        with torch.inference_mode():
            embeddings = model(**batch)
        attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else getattr(batch, "attention_mask", None)
        for index, row in enumerate(rows):
            mask = attention_mask[index] if attention_mask is not None else None
            tokens = strip_mask(embeddings[index], mask)
            np.save(query_dir / f"{safe_name(str(row['query_id']))}.npy", tokens.astype(np.float32))

    manifest = {
        "format_version": 1,
        "type": "colvision_multi_vector",
        "model": args.model,
        "dtype": args.dtype,
        "documents": len(corpus),
        "queries": len(queries),
        "note": "positions are normalized approximate grid coordinates used only for the layout-coverage ablation",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
