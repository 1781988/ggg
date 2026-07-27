#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = [
    "vidore_v3_finance_en",
    "vidore_v3_industrial",
    "vidore_v3_pharmaceuticals",
    "vidore_v3_finance_fr",
]
BASELINES = [
    "visrag_dense",
    "colpali_full",
    "visrag_top50_full",
    "visrag_top50_fixed_112",
    "visrag_top50_mmr_redundancy",
]
FINAL_SOURCE_ID = "final_redundancy_dense025_no_fallback"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate final AdaColRAG paper tables")
    parser.add_argument("--submission-root", default="results/submission")
    parser.add_argument("--targeted-root", default="results/final_targeted")
    parser.add_argument("--paper-final-root", default="results/paper_final")
    parser.add_argument("--output-dir", default="paper/generated")
    args = parser.parse_args()

    submission_root = Path(args.submission_root)
    targeted_root = Path(args.targeted_root)
    paper_final_root = Path(args.paper_final_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_records: list[dict] = []
    final_records: list[dict] = []
    significance_records: list[dict] = []
    timing_records: list[dict] = []
    selection_records: list[dict] = []

    method_ids = BASELINES + ["adacolrag_final"]
    for slug in DATASETS:
        submission_runs = submission_root / slug / "runs"
        targeted_runs = targeted_root / slug / "runs"
        payloads: dict[str, dict] = {
            method_id: read_json(submission_runs / f"{method_id}.json") for method_id in BASELINES
        }
        payloads["adacolrag_final"] = read_json(targeted_runs / f"{FINAL_SOURCE_ID}.json")

        for method_id in method_ids:
            payload = payloads[method_id]
            metrics = payload["metrics"]
            efficiency = payload["efficiency"]
            record = {
                "dataset": slug,
                "method": method_id,
                "nDCG@5": float(metrics["nDCG@5"]),
                "nDCG@10": float(metrics["nDCG@10"]),
                "Recall@10": float(metrics["Recall@10"]),
                "MRR@10": float(metrics["MRR@10"]),
                "mean_selected_tokens": float(efficiency["mean_selected_tokens"]),
                "local_token_reduction": float(efficiency["token_reduction"]),
                "visual_tokens_per_query": float(efficiency["mean_visual_tokens_processed_per_query"]),
                "scoring_operations_per_query": float(efficiency["mean_scored_document_operations"]),
                "mean_latency_ms_original_trace": float(efficiency["mean_latency_ms"]),
            }
            main_records.append(record)
            if method_id == "adacolrag_final":
                documents = int(payload["metadata"]["documents"])
                mean_full_tokens = float(efficiency["mean_full_tokens"])
                full_corpus_work = documents * mean_full_tokens
                record = {
                    **record,
                    "documents": documents,
                    "system_visual_token_work_reduction": (
                        0.0
                        if full_corpus_work <= 0
                        else 1.0 - record["visual_tokens_per_query"] / full_corpus_work
                    ),
                    "page_operation_reduction": 1.0 - record["scoring_operations_per_query"] / documents,
                }
                final_records.append(record)

        significance = read_json(
            paper_final_root / slug / "significance" / "final_vs_baselines.json"
        )
        for row in significance["rows"]:
            significance_records.append({"dataset": slug, **row})

        timing = read_json(paper_final_root / slug / "steady_state_timing.json")
        for method_id, result in timing["experiments"].items():
            timing_records.append(
                {
                    "dataset": slug,
                    "method": method_id,
                    "repeats": int(result["repeats"]),
                    "mean_of_run_means_ms": float(result["mean_latency_ms"]["mean"]),
                    "median_run_mean_ms": float(result["mean_latency_ms"]["p50"]),
                    "run_mean_iqr_ms": float(result["mean_latency_ms"]["iqr"]),
                    "pooled_query_median_ms": float(result["query_latency_ms"]["p50"]),
                    "pooled_query_p95_ms": float(result["query_latency_ms"]["p95"]),
                    "run_mean_cv": float(result["mean_latency_ms"]["cv"]),
                }
            )

        for candidate_id in (
            "final_redundancy_dense025_no_fallback",
            "final_redundancy_dense025_fallback",
            "final_mmr_dense025_no_fallback",
            "final_mmr_dense025_fallback",
        ):
            payload = read_json(targeted_runs / f"{candidate_id}.json")
            selection_records.append(
                {
                    "dataset": slug,
                    "candidate": candidate_id,
                    "nDCG@5": float(payload["metrics"]["nDCG@5"]),
                    "nDCG@10": float(payload["metrics"]["nDCG@10"]),
                    "local_token_reduction": float(payload["efficiency"]["token_reduction"]),
                    "fallback_rate": float(payload["reliability"]["fallback_rate"]),
                }
            )

    # Macro rows for the main table.
    for method_id in method_ids:
        rows = [row for row in main_records if row["method"] == method_id]
        main_records.append(
            {
                "dataset": "macro_average",
                "method": method_id,
                **{
                    key: sum(float(row[key]) for row in rows) / len(rows)
                    for key in (
                        "nDCG@5",
                        "nDCG@10",
                        "Recall@10",
                        "MRR@10",
                        "mean_selected_tokens",
                        "local_token_reduction",
                        "visual_tokens_per_query",
                        "scoring_operations_per_query",
                        "mean_latency_ms_original_trace",
                    )
                },
            }
        )

    # JSON and CSV machine-readable outputs.
    payload = {
        "main_records": main_records,
        "final_efficiency": final_records,
        "significance": significance_records,
        "steady_state_timing": timing_records,
        "final_selection": selection_records,
    }
    (output_dir / "final_paper_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, records in (
        ("final_main_results.csv", main_records),
        ("final_efficiency.csv", final_records),
        ("final_significance.csv", significance_records),
        ("final_steady_timing.csv", timing_records),
        ("final_selection.csv", selection_records),
    ):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    # Main retrieval table.
    lines = [
        "| Dataset | Method | nDCG@5 | nDCG@10 | R@10 | MRR@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in main_records:
        lines.append(
            "| {dataset} | {method} | {n5:.4f} | {n10:.4f} | {r10:.4f} | {mrr:.4f} |".format(
                dataset=row["dataset"], method=row["method"], n5=row["nDCG@5"],
                n10=row["nDCG@10"], r10=row["Recall@10"], mrr=row["MRR@10"]
            )
        )
    (output_dir / "final_main_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Final-method efficiency table.
    lines = [
        "| Dataset | Tokens/op. | Local reduction | Visual tokens/query | Page ops/query | System token-work reduction | Page-op reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_records:
        lines.append(
            "| {dataset} | {tokens:.1f} | {local} | {visual:.1f} | {ops:.1f} | {system} | {opred} |".format(
                dataset=row["dataset"], tokens=row["mean_selected_tokens"],
                local=format_pct(row["local_token_reduction"]), visual=row["visual_tokens_per_query"],
                ops=row["scoring_operations_per_query"],
                system=format_pct(row["system_visual_token_work_reduction"]),
                opred=format_pct(row["page_operation_reduction"]),
            )
        )
    (output_dir / "final_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Significance table: primary nDCG@5 comparisons only.
    lines = [
        "| Dataset | Baseline | Baseline | Final | ΔnDCG@5 | 95% CI | p | Final better queries |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in significance_records:
        if row["metric"] != "nDCG@5":
            continue
        lines.append(
            "| {dataset} | {baseline} | {base:.4f} | {final:.4f} | {delta:+.4f} | "
            "[{low:+.4f}, {high:+.4f}] | {p:.4f} | {better:.2f}% |".format(
                dataset=row["dataset"], baseline=row["baseline"], base=row["baseline_score"],
                final=row["candidate_score"], delta=row["mean_difference"],
                low=row["ci95"][0], high=row["ci95"][1],
                p=row["p_two_sided_bootstrap"], better=100.0 * row["candidate_better_fraction"],
            )
        )
    (output_dir / "final_significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Steady-state timing table.
    lines = [
        "| Dataset | Method | Repeats | Median run mean (ms) | Run IQR (ms) | Query median (ms) | Query p95 (ms) | CV |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in timing_records:
        lines.append(
            "| {dataset} | {method} | {repeats} | {median:.2f} | {iqr:.2f} | {q50:.2f} | {q95:.2f} | {cv:.3f} |".format(
                dataset=row["dataset"], method=row["method"], repeats=row["repeats"],
                median=row["median_run_mean_ms"], iqr=row["run_mean_iqr_ms"],
                q50=row["pooled_query_median_ms"], q95=row["pooled_query_p95_ms"], cv=row["run_mean_cv"],
            )
        )
    (output_dir / "final_steady_timing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Final-candidate selection table with macro average.
    candidates = sorted({row["candidate"] for row in selection_records})
    selection_lines = [
        "| Candidate | Macro nDCG@5 | Macro nDCG@10 | Macro local reduction | Macro fallback rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        rows = [row for row in selection_records if row["candidate"] == candidate]
        selection_lines.append(
            "| {candidate} | {n5:.4f} | {n10:.4f} | {local} | {fallback} |".format(
                candidate=candidate,
                n5=sum(row["nDCG@5"] for row in rows) / len(rows),
                n10=sum(row["nDCG@10"] for row in rows) / len(rows),
                local=format_pct(sum(row["local_token_reduction"] for row in rows) / len(rows)),
                fallback=format_pct(sum(row["fallback_rate"] for row in rows) / len(rows)),
            )
        )
    (output_dir / "final_selection.md").write_text(
        "\n".join(selection_lines) + "\n", encoding="utf-8"
    )

    print(f"Wrote final paper tables to {output_dir}")


if __name__ == "__main__":
    main()
