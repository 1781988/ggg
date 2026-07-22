#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from adacolrag.config import load_config
from adacolrag.io import attach_dense_embeddings, load_colvision_embeddings, load_dataset_metadata
from adacolrag.pipeline import AdaColRAGPipeline


def parse_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    p25, p50, p75, p95 = np.percentile(array, [25, 50, 75, 95])
    return {
        "count": int(array.size),
        "mean": mean,
        "std": std,
        "cv": 0.0 if abs(mean) < 1e-12 else std / abs(mean),
        "min": float(array.min()),
        "max": float(array.max()),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "iqr": float(p75 - p25),
        "p95": float(p95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure retrieval latency in one process after loading embeddings once"
    )
    parser.add_argument("--matrix", default="configs/final_paper_matrix.yaml")
    parser.add_argument(
        "--only",
        default="visrag_dense,colpali_full,visrag_top50_full,adacolrag",
        help="Comma-separated experiment IDs",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--colvision-dir", required=True)
    parser.add_argument("--dense-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()

    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    if args.warmups < 1:
        raise SystemExit("--warmups must be at least 1 for steady-state timing")

    matrix_path = Path(args.matrix)
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))["experiments"]
    by_id = {item["id"]: item for item in matrix}
    selected = parse_ids(args.only)
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise SystemExit(f"unknown experiment IDs: {', '.join(unknown)}")

    print("[load] ColPali multi-vector embeddings", flush=True)
    base_documents, base_queries = load_colvision_embeddings(args.dataset_dir, args.colvision_dir)
    print("[load] dense embeddings", flush=True)
    dense_documents, dense_queries = attach_dense_embeddings(base_documents, base_queries, args.dense_dir)
    _, _, qrels = load_dataset_metadata(args.dataset_dir)
    print(
        f"[load] documents={len(base_documents)} queries={len(base_queries)} "
        f"warmups={args.warmups} repeats={args.repeats}",
        flush=True,
    )

    report: dict[str, dict] = {}
    for experiment_id in selected:
        experiment = by_id[experiment_id]
        config = load_config(experiment["config"])
        requires_dense = bool(experiment.get("requires_dense", False))
        documents = dense_documents if requires_dense else base_documents
        queries = dense_queries if requires_dense else base_queries
        pipeline = AdaColRAGPipeline(config)

        for warmup_index in range(args.warmups):
            print(
                f"[steady-timing] experiment={experiment_id} phase=warmup "
                f"run={warmup_index + 1}/{args.warmups}",
                flush=True,
            )
            pipeline.evaluate(documents, queries, qrels)

        runs: list[dict] = []
        flattened_query_latencies: list[float] = []
        for repeat_index in range(args.repeats):
            print(
                f"[steady-timing] experiment={experiment_id} phase=measured "
                f"run={repeat_index + 1}/{args.repeats}",
                flush=True,
            )
            start = time.perf_counter()
            result = pipeline.evaluate(documents, queries, qrels)
            wall_seconds = time.perf_counter() - start
            query_latencies = [float(item["latency_ms"]) for item in result.per_query.values()]
            flattened_query_latencies.extend(query_latencies)
            runs.append(
                {
                    "wall_seconds": wall_seconds,
                    "mean_latency_ms": float(result.efficiency["mean_latency_ms"]),
                    "p50_latency_ms": float(result.efficiency["p50_latency_ms"]),
                    "p95_latency_ms": float(result.efficiency["p95_latency_ms"]),
                    "nDCG@5": float(result.metrics["nDCG@5"]),
                    "nDCG@10": float(result.metrics["nDCG@10"]),
                }
            )

        ndcg5_values = [item["nDCG@5"] for item in runs]
        if max(ndcg5_values) - min(ndcg5_values) > 1e-12:
            raise RuntimeError(f"non-deterministic nDCG@5 for {experiment_id}: {ndcg5_values}")

        report[experiment_id] = {
            "config": experiment["config"],
            "requires_dense": requires_dense,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "runs": runs,
            "wall_seconds": summarize([item["wall_seconds"] for item in runs]),
            "mean_latency_ms": summarize([item["mean_latency_ms"] for item in runs]),
            "p50_latency_ms": summarize([item["p50_latency_ms"] for item in runs]),
            "p95_latency_ms": summarize([item["p95_latency_ms"] for item in runs]),
            "query_latency_ms": summarize(flattened_query_latencies),
            "nDCG@5": summarize(ndcg5_values),
            "nDCG@10": summarize([item["nDCG@10"] for item in runs]),
        }

    payload = {
        "timing_mode": "single_process_steady_state",
        "timing_boundary": "retrieval over preloaded embeddings; excludes model encoding and disk loading",
        "dataset_dir": args.dataset_dir,
        "colvision_dir": args.colvision_dir,
        "dense_dir": args.dense_dir,
        "matrix": str(matrix_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pid": os.getpid(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "experiments": report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
