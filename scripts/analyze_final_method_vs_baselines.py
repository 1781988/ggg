#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qrels.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = int(row.get("relevance", 1))
    return qrels


def dcg(relevances: Iterable[int], k: int) -> float:
    total = 0.0
    for rank, relevance in enumerate(list(relevances)[:k], start=1):
        total += (2 ** int(relevance) - 1) / math.log2(rank + 1)
    return total


def per_query_ndcg(result: dict, qrels: dict[str, dict[str, int]], k: int) -> dict[str, float]:
    values: dict[str, float] = {}
    traces = result["per_query"]
    for query_id, relevant in qrels.items():
        ranking = [item["doc_id"] for item in traces[query_id]["ranking"]]
        gains = [relevant.get(doc_id, 0) for doc_id in ranking[:k]]
        denominator = dcg(sorted(relevant.values(), reverse=True), k)
        values[query_id] = 0.0 if denominator == 0 else dcg(gains, k) / denominator
    return values


def bootstrap_difference(
    baseline: dict[str, float],
    candidate: dict[str, float],
    samples: int,
    seed: int,
) -> dict:
    query_ids = sorted(set(baseline) & set(candidate))
    differences = np.asarray([candidate[qid] - baseline[qid] for qid in query_ids], dtype=np.float64)
    if differences.size == 0:
        raise ValueError("No common queries for paired bootstrap")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        batch = min(1000, samples - start)
        indices = rng.integers(0, differences.size, size=(batch, differences.size))
        means[start : start + batch] = differences[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    probability_nonpositive = float(np.mean(means <= 0.0))
    probability_nonnegative = float(np.mean(means >= 0.0))
    return {
        "queries": len(query_ids),
        "mean_difference": float(differences.mean()),
        "ci95": [float(lower), float(upper)],
        "p_two_sided_bootstrap": float(min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))),
        "candidate_better_fraction": float(np.mean(differences > 0.0)),
        "candidate_equal_fraction": float(np.mean(differences == 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the frozen final AdaColRAG result with stored paper baselines")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--submission-runs", required=True)
    parser.add_argument("--targeted-runs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--final-id", default="final_redundancy_dense025_no_fallback")
    parser.add_argument(
        "--baselines",
        default="visrag_dense,colpali_full,visrag_top50_full,visrag_top50_fixed_112,visrag_top50_mmr_redundancy,adacolrag_no_fallback,adacolrag",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bootstrap_samples < 1000:
        raise SystemExit("--bootstrap-samples must be at least 1000")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_runs = Path(args.submission_runs)
    targeted_runs = Path(args.targeted_runs)
    qrels = read_qrels(Path(args.dataset_dir) / "qrels.jsonl")

    final_path = targeted_runs / f"{args.final_id}.json"
    if not final_path.is_file():
        raise FileNotFoundError(final_path)
    final_payload = read_json(final_path)
    final_checksum = str(final_payload.get("metadata", {}).get("dataset_checksum", ""))
    final_per_query = {
        5: per_query_ndcg(final_payload, qrels, 5),
        10: per_query_ndcg(final_payload, qrels, 10),
    }

    rows: list[dict] = []
    for index, baseline_id in enumerate(item.strip() for item in args.baselines.split(",") if item.strip()):
        baseline_path = submission_runs / f"{baseline_id}.json"
        if not baseline_path.is_file():
            raise FileNotFoundError(baseline_path)
        baseline_payload = read_json(baseline_path)
        baseline_checksum = str(baseline_payload.get("metadata", {}).get("dataset_checksum", ""))
        if final_checksum and baseline_checksum != final_checksum:
            raise ValueError(
                f"dataset checksum mismatch for {baseline_id}: {baseline_checksum!r} != {final_checksum!r}"
            )
        for k in (5, 10):
            comparison = bootstrap_difference(
                per_query_ndcg(baseline_payload, qrels, k),
                final_per_query[k],
                args.bootstrap_samples,
                args.seed + index * 2 + (0 if k == 5 else 1),
            )
            rows.append(
                {
                    "baseline": baseline_id,
                    "candidate": "adacolrag_final",
                    "metric": f"nDCG@{k}",
                    "baseline_score": float(baseline_payload["metrics"][f"nDCG@{k}"]),
                    "candidate_score": float(final_payload["metrics"][f"nDCG@{k}"]),
                    **comparison,
                }
            )

    payload = {
        "dataset_dir": str(Path(args.dataset_dir)),
        "submission_runs": str(submission_runs),
        "targeted_runs": str(targeted_runs),
        "final_source_id": args.final_id,
        "final_paper_id": "adacolrag_final",
        "dataset_checksum": final_checksum,
        "bootstrap_samples": args.bootstrap_samples,
        "final_metrics": final_payload["metrics"],
        "final_efficiency": final_payload["efficiency"],
        "rows": rows,
    }
    (output_dir / "final_vs_baselines.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fieldnames = list(rows[0].keys())
    with (output_dir / "final_vs_baselines.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Baseline | Metric | Baseline | Final | Mean Δ | 95% bootstrap CI | p (two-sided) | Final better queries |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {baseline} | {metric} | {base:.4f} | {final:.4f} | {delta:+.4f} | "
            "[{low:+.4f}, {high:+.4f}] | {p:.4f} | {better:.2f}% |".format(
                baseline=row["baseline"],
                metric=row["metric"],
                base=row["baseline_score"],
                final=row["candidate_score"],
                delta=row["mean_difference"],
                low=row["ci95"][0],
                high=row["ci95"][1],
                p=row["p_two_sided_bootstrap"],
                better=100.0 * row["candidate_better_fraction"],
            )
        )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "final_vs_baselines.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
