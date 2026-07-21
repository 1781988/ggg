from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .io import (
    attach_dense_embeddings,
    dataset_checksum,
    load_colvision_embeddings,
    load_dataset_metadata,
    write_json,
)
from .pipeline import AdaColRAGPipeline
from .synthetic import build_synthetic_fixture


def _smoke(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    documents, queries, qrels = build_synthetic_fixture(seed=int(config["seed"]))
    result = AdaColRAGPipeline(config).evaluate(documents, queries, qrels, metadata={"dataset": "synthetic"})
    write_json(args.output, result.to_dict())
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    print(f"[stage] load config: {args.config}", flush=True)
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "mode": config["mode"],
                "candidate_pool": config["candidate_pool"],
                "top_k": config["top_k"],
                "fixed_tokens": config.get("fixed_tokens"),
                "min_tokens": config.get("min_tokens"),
                "max_tokens": config.get("max_tokens"),
                "selector_prefilter_factor": config["selector"].get("prefilter_factor", 0.0),
                "redundancy_weight": config["selector"].get("redundancy_weight", 0.0),
                "layout_weight": config["selector"].get("layout_weight", 0.0),
                "dense_weight": config["fusion"].get("dense_weight", 0.0),
                "lexical_weight": config["fusion"].get("lexical_weight", 0.0),
                "fallback": config["fallback"].get("enabled", False),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("[stage] load ColVision embeddings", flush=True)
    documents, queries = load_colvision_embeddings(args.dataset_dir, args.colvision_dir)
    print(f"[stage] loaded documents={len(documents)} queries={len(queries)}", flush=True)
    if args.dense_dir:
        print(f"[stage] attach dense embeddings: {args.dense_dir}", flush=True)
        documents, queries = attach_dense_embeddings(documents, queries, args.dense_dir)
    else:
        print("[stage] no dense embeddings attached", flush=True)
    print("[stage] load qrels and compute dataset checksum", flush=True)
    _, _, qrels = load_dataset_metadata(args.dataset_dir)
    checksum = dataset_checksum(
        [
            Path(args.dataset_dir) / "corpus.jsonl",
            Path(args.dataset_dir) / "queries.jsonl",
            Path(args.dataset_dir) / "qrels.jsonl",
        ]
    )
    print("[stage] start retrieval evaluation", flush=True)
    result = AdaColRAGPipeline(config).evaluate(
        documents,
        queries,
        qrels,
        metadata={
            "dataset_dir": str(args.dataset_dir),
            "dataset_checksum": checksum,
            "colvision_dir": str(args.colvision_dir),
            "dense_dir": str(args.dense_dir) if args.dense_dir else None,
        },
    )
    print(f"[stage] write result: {args.output}", flush=True)
    write_json(args.output, result.to_dict())
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adacolrag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="Run deterministic synthetic end-to-end evaluation")
    smoke.add_argument("--config", default="configs/smoke.yaml")
    smoke.add_argument("--output", default="results/smoke_actual.json")
    smoke.set_defaults(func=_smoke)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate exported embeddings")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--dataset-dir", required=True)
    evaluate.add_argument("--colvision-dir", required=True)
    evaluate.add_argument("--dense-dir")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(func=_evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(args.func(args))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
