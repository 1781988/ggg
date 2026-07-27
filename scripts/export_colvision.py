#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

# Clear the legacy transfer flag before colpali_engine imports Hugging Face Hub.
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from adacolrag.io import load_dataset_metadata, l2_normalize


def safe_name(identifier: str) -> str:
    return identifier.replace("/", "__").replace("\\", "__")


def completed_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def resolve_model_source(model_name: str) -> tuple[str, bool]:
    candidate = Path(model_name).expanduser()
    if candidate.is_dir():
        required = [
            candidate / "config.json",
            candidate / "model.safetensors.index.json",
            candidate / "model-00001-of-00002.safetensors",
            candidate / "model-00002-of-00002.safetensors",
            candidate / "tokenizer.json",
            candidate / "tokenizer_config.json",
            candidate / "preprocessor_config.json",
        ]
        missing = [str(path) for path in required if not completed_file(path)]
        if missing:
            raise FileNotFoundError(f"Incomplete local ColVision snapshot; missing files: {missing}")
        return str(candidate.resolve()), True
    return model_name, False


def load_model(model_name: str, device: str, dtype: str):
    from colpali_engine import models as col_models

    model_source, local_only = resolve_model_source(model_name)
    lowered = model_source.lower()
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
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device,
    }
    processor_kwargs = {}
    if local_only:
        # This is the central reliability guarantee for submission runs: once the
        # snapshot is prepared, Transformers must never fall back to the network.
        load_kwargs["local_files_only"] = True
        processor_kwargs["local_files_only"] = True
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model = model_cls.from_pretrained(model_source, **load_kwargs).eval()
    processor = processor_cls.from_pretrained(model_source, **processor_kwargs)
    return model, processor, model_source, local_only


def verify_checkpoint(model, model_source: str, local_only: bool) -> dict:
    """Record enough state to distinguish a merged checkpoint from a broken adapter load."""

    source_path = Path(model_source)
    merged = "merged" in model_source.lower()
    snapshot_info = None
    if local_only and completed_file(source_path / "snapshot_info.json"):
        snapshot_info = json.loads((source_path / "snapshot_info.json").read_text(encoding="utf-8"))
        merged = merged and bool(snapshot_info.get("verified", False))

    lora_parameters: list[dict] = []
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        detached = parameter.detach()
        lora_parameters.append(
            {
                "name": name,
                "shape": list(detached.shape),
                "nonzero": int(torch.count_nonzero(detached).item()),
                "max_abs": float(detached.abs().max().float().cpu()) if detached.numel() else 0.0,
            }
        )

    if merged:
        verified = True
        reason = "verified merged checkpoint; LoRA tensors are folded into base weights"
    else:
        lora_b = [item for item in lora_parameters if "lora_B" in item["name"]]
        verified = bool(lora_b) and any(item["nonzero"] > 0 and item["max_abs"] > 0 for item in lora_b)
        reason = (
            "non-zero LoRA-B tensors detected"
            if verified
            else "checkpoint has neither a verified merged snapshot nor a verifiably non-zero LoRA-B tensor"
        )

    return {
        "verified": verified,
        "reason": reason,
        "merged_checkpoint": merged,
        "local_files_only": local_only,
        "model_source": model_source,
        "snapshot_info": snapshot_info,
        "model_class": type(model).__name__,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "lora_parameter_count": len(lora_parameters),
        "lora_parameters": lora_parameters,
    }


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
    parser.add_argument("--model", default="vidore/colpali-v1.3-merged")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even when files already exist")
    parser.add_argument(
        "--allow-unverified-checkpoint",
        action="store_true",
        help="Continue when an unmerged adapter checkpoint cannot be verified. Not recommended for submission runs.",
    )
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
        if args.force or not completed_file(document_dir / f"{safe_name(str(row['doc_id']))}.npz")
    ]
    pending_queries = [
        row
        for row in queries
        if args.force or not completed_file(query_dir / f"{safe_name(str(row['query_id']))}.npy")
    ]

    if not pending_corpus and not pending_queries and completed_file(output_dir / "manifest.json"):
        print(
            f"[reuse] ColVision embeddings already complete: {len(corpus)} documents, {len(queries)} queries",
            flush=True,
        )
        return

    model_source, local_only = resolve_model_source(args.model)
    print(
        json.dumps(
            {
                "model_argument": args.model,
                "model_source": model_source,
                "local_files_only": local_only,
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

    model, processor, model_source, local_only = load_model(args.model, args.device, args.dtype)
    verification = verify_checkpoint(model, model_source, local_only)
    print("[checkpoint] " + json.dumps(verification, ensure_ascii=False, sort_keys=True), flush=True)
    if not verification["verified"] and not args.allow_unverified_checkpoint:
        raise RuntimeError(
            "ColVision checkpoint verification failed. Prepare the local merged snapshot with "
            "scripts/download_colpali_merged_model_curl.sh, or pass --allow-unverified-checkpoint only for diagnostics."
        )

    for start in tqdm(range(0, len(pending_corpus), args.batch_size), desc="documents"):
        rows = pending_corpus[start : start + args.batch_size]
        images = []
        for row in rows:
            with Image.open(resolve_image(dataset_dir, str(row["image_path"]))) as image:
                images.append(image.convert("RGB"))
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

    for start in tqdm(range(0, len(pending_queries), args.batch_size), desc="queries"):
        rows = pending_queries[start : start + args.batch_size]
        batch = processor.process_queries([str(row["text"]) for row in rows]).to(model.device)
        with torch.inference_mode():
            embeddings = model(**batch)
        attention_mask = batch.get("attention_mask") if isinstance(batch, dict) else getattr(batch, "attention_mask", None)
        for index, row in enumerate(rows):
            mask = attention_mask[index] if attention_mask is not None else None
            tokens = strip_mask(embeddings[index], mask)
            np.save(query_dir / f"{safe_name(str(row['query_id']))}.npy", tokens.astype(np.float32))

    manifest = {
        "format_version": 3,
        "type": "colvision_multi_vector",
        "model": args.model,
        "model_source": model_source,
        "dtype": args.dtype,
        "documents": len(corpus),
        "queries": len(queries),
        "checkpoint_verification": verification,
        "note": "positions are normalized approximate grid coordinates used only for the layout-coverage ablation",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
