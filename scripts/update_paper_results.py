#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def replace_block(text: str, name: str, content: str) -> str:
    begin = f"<!-- BEGIN AUTO:{name} -->"
    end = f"<!-- END AUTO:{name} -->"
    if begin not in text or end not in text:
        raise ValueError(f"paper is missing marker pair for {name}")
    prefix, remainder = text.split(begin, 1)
    _, suffix = remainder.split(end, 1)
    return prefix + begin + "\n\n" + content.strip() + "\n\n" + end + suffix


def timing_table(path: Path) -> str:
    if not path.is_file():
        return "_Repeated timing has not been generated yet._"
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "| Method | Repeats | Mean of mean latency (ms) | Std. (ms) | Mean wall time (s) | Wall-time std. (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for experiment_id, result in payload["experiments"].items():
        lines.append(
            "| {method} | {repeats} | {mean:.2f} | {std:.2f} | {wall:.2f} | {wall_std:.2f} |".format(
                method=experiment_id,
                repeats=result["repeats"],
                mean=result["mean_latency_ms"]["mean"],
                std=result["mean_latency_ms"]["std"],
                wall=result["wall_seconds"]["mean"],
                wall_std=result["wall_seconds"]["std"],
            )
        )
    return "\n".join(lines)


def read_or_placeholder(path: Path, placeholder: str) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else placeholder


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject generated experiment tables into paper/draft.md")
    parser.add_argument("--paper", default="paper/draft.md")
    parser.add_argument("--generated-dir", default="paper/generated")
    parser.add_argument("--submission-root", default="results/submission")
    parser.add_argument("--primary-dataset", default="vidore_v3_finance_en")
    args = parser.parse_args()

    paper_path = Path(args.paper)
    generated_dir = Path(args.generated_dir)
    primary_root = Path(args.submission_root) / args.primary_dataset
    text = paper_path.read_text(encoding="utf-8")
    replacements = {
        "CROSS_DATASET_RESULTS": read_or_placeholder(
            generated_dir / "cross_dataset_results.md",
            "_Run `scripts/run_submission_experiments.sh` to generate the cross-dataset table._",
        ),
        "PRIMARY_RESULTS": read_or_placeholder(
            primary_root / "analysis" / "results_table.md",
            "_The primary controlled-matrix table will be inserted after the submission rerun._",
        ),
        "PRIMARY_SIGNIFICANCE": read_or_placeholder(
            primary_root / "analysis" / "significance_table.md",
            "_Paired bootstrap intervals will be inserted after the submission rerun._",
        ),
        "REPEATED_TIMING": timing_table(primary_root / "repeated_timing.json"),
    }
    for name, content in replacements.items():
        text = replace_block(text, name, content)
    paper_path.write_text(text, encoding="utf-8")
    print(f"Updated {paper_path} from generated submission results")


if __name__ == "__main__":
    main()
