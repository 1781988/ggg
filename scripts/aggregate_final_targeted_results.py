#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate final targeted AdaColRAG candidate results")
    parser.add_argument("--root", default="results/final_targeted")
    parser.add_argument("--output-dir", default="results/final_targeted/_aggregate")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for summary_path in sorted(root.glob("*/analysis/summary.json")):
        dataset = summary_path.parents[1].name
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            records.append({"dataset": dataset, **row})

    if not records:
        raise SystemExit(f"No targeted summaries found below {root}")

    fieldnames = list(records[0].keys())
    with (output_dir / "targeted_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    methods = sorted({str(row["id"]) for row in records})
    datasets = sorted({str(row["dataset"]) for row in records})
    by_key = {(str(row["dataset"]), str(row["id"])): row for row in records}

    table = [
        "| Dataset | Method | nDCG@5 | nDCG@10 | R@10 | MRR@10 | Local red. | Tokens/query | Mean ms | p95 ms | Fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        for method in methods:
            row = by_key.get((dataset, method))
            if row is None:
                continue
            table.append(
                "| {dataset} | {method} | {n5:.4f} | {n10:.4f} | {r10:.4f} | {mrr:.4f} | "
                "{local:.2f}% | {tokens:.1f} | {mean:.1f} | {p95:.1f} | {fallback:.2f}% |".format(
                    dataset=dataset,
                    method=method,
                    n5=float(row["nDCG@5"]),
                    n10=float(row["nDCG@10"]),
                    r10=float(row["Recall@10"]),
                    mrr=float(row["MRR@10"]),
                    local=100.0 * float(row["local_token_reduction"]),
                    tokens=float(row["visual_tokens_per_query"]),
                    mean=float(row["mean_latency_ms"]),
                    p95=float(row["p95_latency_ms"]),
                    fallback=100.0 * float(row["fallback_rate"]),
                )
            )

    macro = [
        "| Method | Macro nDCG@5 | Macro nDCG@10 | Macro R@10 | Macro MRR@10 | Mean local red. | Mean latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    macro_records: list[dict] = []
    for method in methods:
        rows = [row for row in records if row["id"] == method]
        if not rows:
            continue
        record = {
            "id": method,
            "datasets": len(rows),
            "nDCG@5": sum(float(row["nDCG@5"]) for row in rows) / len(rows),
            "nDCG@10": sum(float(row["nDCG@10"]) for row in rows) / len(rows),
            "Recall@10": sum(float(row["Recall@10"]) for row in rows) / len(rows),
            "MRR@10": sum(float(row["MRR@10"]) for row in rows) / len(rows),
            "local_token_reduction": sum(float(row["local_token_reduction"]) for row in rows) / len(rows),
            "mean_latency_ms": sum(float(row["mean_latency_ms"]) for row in rows) / len(rows),
        }
        macro_records.append(record)
        macro.append(
            "| {id} | {n5:.4f} | {n10:.4f} | {r10:.4f} | {mrr:.4f} | {local:.2f}% | {latency:.1f} |".format(
                id=method,
                n5=record["nDCG@5"],
                n10=record["nDCG@10"],
                r10=record["Recall@10"],
                mrr=record["MRR@10"],
                local=100.0 * record["local_token_reduction"],
                latency=record["mean_latency_ms"],
            )
        )

    (output_dir / "targeted_results.md").write_text("\n".join(table) + "\n\n" + "\n".join(macro) + "\n", encoding="utf-8")
    (output_dir / "targeted_results.json").write_text(
        json.dumps({"records": records, "macro": macro_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n".join(macro))


if __name__ == "__main__":
    main()
