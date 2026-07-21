#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-dataset AdaColRAG submission summaries")
    parser.add_argument("--submission-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.submission_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for summary_path in sorted(root.glob("*/analysis/summary.json")):
        dataset_slug = summary_path.parents[1].name
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            records.append({"dataset": dataset_slug, **row})
    if not records:
        raise SystemExit(f"No analysis summaries found below {root}")

    fieldnames = list(records[0].keys())
    with (output_dir / "cross_dataset_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    by_dataset: dict[str, dict[str, dict]] = {}
    for row in records:
        by_dataset.setdefault(row["dataset"], {})[row["id"]] = row

    primary_ids = [
        "visrag_dense",
        "colpali_full",
        "visrag_top50_full",
        "visrag_top50_fixed_112",
        "visrag_top50_adaptive_mmr",
        "visrag_top50_adaptive_fallback",
        "adacolrag",
    ]
    lines = [
        "| Dataset | Method | nDCG@5 | nDCG@10 | R@10 | MRR@10 | Local red. | System work red. | Mean ms | Fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, methods in by_dataset.items():
        for experiment_id in primary_ids:
            if experiment_id not in methods:
                continue
            row = methods[experiment_id]
            lines.append(
                "| {dataset} | {method} | {n5:.4f} | {n10:.4f} | {r10:.4f} | {mrr:.4f} | {local:.2f}% | {system:.2f}% | {latency:.1f} | {fallback:.2f}% |".format(
                    dataset=dataset,
                    method=experiment_id,
                    n5=float(row["nDCG@5"]),
                    n10=float(row["nDCG@10"]),
                    r10=float(row["Recall@10"]),
                    mrr=float(row["MRR@10"]),
                    local=100 * float(row["local_token_reduction"]),
                    system=100 * float(row["system_token_work_reduction"]),
                    latency=float(row["mean_latency_ms"]),
                    fallback=100 * float(row["fallback_rate"]),
                )
            )
    (output_dir / "cross_dataset_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "cross_dataset_results.json").write_text(
        json.dumps({"records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
