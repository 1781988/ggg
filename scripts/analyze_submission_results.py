#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


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
        total += (2**int(relevance) - 1) / math.log2(rank + 1)
    return total


def per_query_ndcg(result: dict, qrels: dict[str, dict[str, int]], k: int = 5) -> dict[str, float]:
    values: dict[str, float] = {}
    for query_id, relevant in qrels.items():
        ranking = [item["doc_id"] for item in result["per_query"][query_id]["ranking"]]
        gains = [relevant.get(doc_id, 0) for doc_id in ranking[:k]]
        ideal = sorted(relevant.values(), reverse=True)
        denominator = dcg(ideal, k)
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
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        batch = min(1000, samples - start)
        indices = rng.integers(0, differences.size, size=(batch, differences.size))
        means[start : start + batch] = differences[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    probability_nonpositive = float(np.mean(means <= 0.0))
    probability_nonnegative = float(np.mean(means >= 0.0))
    p_two_sided = min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
    return {
        "queries": len(query_ids),
        "mean_difference": float(differences.mean()),
        "ci95": [float(lower), float(upper)],
        "p_two_sided_bootstrap": float(p_two_sided),
        "candidate_better_fraction": float(np.mean(differences > 0.0)),
        "candidate_equal_fraction": float(np.mean(differences == 0.0)),
    }


def format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def mark_pareto(rows: list[dict]) -> None:
    for row in rows:
        quality = float(row["nDCG@5"])
        latency = float(row["mean_latency_ms"])
        row["pareto_quality_latency"] = not any(
            float(other["nDCG@5"]) >= quality
            and float(other["mean_latency_ms"]) <= latency
            and (
                float(other["nDCG@5"]) > quality
                or float(other["mean_latency_ms"]) < latency
            )
            for other in rows
            if other is not row
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and statistically analyze AdaColRAG results")
    parser.add_argument("--matrix", default="configs/submission_matrix.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--comparisons",
        default=None,
        help="Optional comma-separated baseline:candidate pairs; defaults to matrix comparisons",
    )
    args = parser.parse_args()

    with Path(args.matrix).open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    experiments = matrix["experiments"]
    if args.comparisons is None:
        comparison_items = [str(item) for item in matrix.get("comparisons", [])]
    else:
        comparison_items = [part.strip() for part in args.comparisons.split(",") if part.strip()]

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qrels = read_qrels(Path(args.dataset_dir) / "qrels.jsonl")

    rows: list[dict] = []
    ndcg_per_query: dict[str, dict[str, float]] = {}
    for experiment in experiments:
        experiment_id = experiment["id"]
        path = results_dir / f"{experiment_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        ndcg_per_query[experiment_id] = per_query_ndcg(payload, qrels, 5)
        metrics = payload["metrics"]
        efficiency = payload["efficiency"]
        reliability = payload["reliability"]
        documents = int(payload["metadata"]["documents"])
        full_tokens = float(efficiency["mean_full_tokens"])
        full_corpus_work = documents * full_tokens
        processed = float(efficiency["mean_visual_tokens_processed_per_query"])
        operations = float(efficiency["mean_scored_document_operations"])
        system_reduction = 0.0 if full_corpus_work <= 0 else 1.0 - processed / full_corpus_work
        operation_reduction = 0.0 if documents <= 0 else 1.0 - operations / documents
        rows.append(
            {
                "id": experiment_id,
                "group": experiment.get("group", ""),
                "description": experiment.get("description", ""),
                "nDCG@5": metrics["nDCG@5"],
                "nDCG@10": metrics["nDCG@10"],
                "Recall@5": metrics["Recall@5"],
                "Recall@10": metrics["Recall@10"],
                "MRR@10": metrics["MRR@10"],
                "tokens_per_operation": efficiency["mean_selected_tokens"],
                "local_token_reduction": efficiency["token_reduction"],
                "visual_tokens_per_query": processed,
                "scoring_operations_per_query": operations,
                "system_token_work_reduction": system_reduction,
                "operation_reduction": operation_reduction,
                "mean_latency_ms": efficiency["mean_latency_ms"],
                "p50_latency_ms": efficiency["p50_latency_ms"],
                "p95_latency_ms": efficiency["p95_latency_ms"],
                "fallback_rate": reliability["fallback_rate"],
            }
        )
    mark_pareto(rows)

    comparisons: dict[str, dict] = {}
    for index, item in enumerate(comparison_items):
        baseline_id, candidate_id = item.split(":", 1)
        if baseline_id not in ndcg_per_query or candidate_id not in ndcg_per_query:
            raise KeyError(f"unknown comparison: {item}")
        comparisons[f"{baseline_id}__vs__{candidate_id}"] = {
            "baseline": baseline_id,
            "candidate": candidate_id,
            "metric": "nDCG@5",
            **bootstrap_difference(
                ndcg_per_query[baseline_id],
                ndcg_per_query[candidate_id],
                args.bootstrap_samples,
                args.seed + index,
            ),
        }

    summary = {
        "dataset_dir": str(Path(args.dataset_dir)),
        "results_dir": str(results_dir),
        "matrix": str(args.matrix),
        "bootstrap_samples": args.bootstrap_samples,
        "rows": rows,
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fieldnames = list(rows[0].keys())
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table_lines = [
        "| Method | Group | nDCG@5 | nDCG@10 | R@5 | R@10 | MRR@10 | Tokens/op. | Local red. | System work red. | Mean ms | p95 ms | Fallback | Pareto |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            "| {id} | {group} | {n5} | {n10} | {r5} | {r10} | {mrr} | {tokens} | {local} | {system} | {mean} | {p95} | {fallback} | {pareto} |".format(
                id=row["id"],
                group=row["group"],
                n5=format_float(row["nDCG@5"]),
                n10=format_float(row["nDCG@10"]),
                r5=format_float(row["Recall@5"]),
                r10=format_float(row["Recall@10"]),
                mrr=format_float(row["MRR@10"]),
                tokens=format_float(row["tokens_per_operation"], 1),
                local=f"{100 * float(row['local_token_reduction']):.2f}%",
                system=f"{100 * float(row['system_token_work_reduction']):.2f}%",
                mean=format_float(row["mean_latency_ms"], 1),
                p95=format_float(row["p95_latency_ms"], 1),
                fallback=f"{100 * float(row['fallback_rate']):.2f}%",
                pareto="yes" if row["pareto_quality_latency"] else "no",
            )
        )
    (output_dir / "results_table.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    comparison_lines = [
        "| Baseline | Candidate | Mean ΔnDCG@5 | 95% bootstrap CI | p (two-sided) | Candidate better queries |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for comparison in comparisons.values():
        comparison_lines.append(
            "| {baseline} | {candidate} | {difference:.4f} | [{lower:.4f}, {upper:.4f}] | {p:.4f} | {better:.2f}% |".format(
                baseline=comparison["baseline"],
                candidate=comparison["candidate"],
                difference=comparison["mean_difference"],
                lower=comparison["ci95"][0],
                upper=comparison["ci95"][1],
                p=comparison["p_two_sided_bootstrap"],
                better=100 * comparison["candidate_better_fraction"],
            )
        )
    (output_dir / "significance_table.md").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
