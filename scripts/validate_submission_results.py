#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a submission-grade AdaColRAG experiment bundle")
    parser.add_argument("--matrix", default="configs/submission_matrix.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--colvision-dir", required=True)
    parser.add_argument("--dense-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-nonmerged-colvision", action="store_true")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    dataset_dir = Path(args.dataset_dir)
    colvision_dir = Path(args.colvision_dir)
    dense_dir = Path(args.dense_dir)
    results_dir = Path(args.results_dir)
    output = Path(args.output)

    with matrix_path.open("r", encoding="utf-8") as handle:
        experiments = yaml.safe_load(handle)["experiments"]

    errors: list[str] = []
    warnings: list[str] = []
    required_dataset_files = ["corpus.jsonl", "queries.jsonl", "qrels.jsonl", "dataset_info.json"]
    for name in required_dataset_files:
        path = dataset_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing dataset file: {path}")

    col_manifest_path = colvision_dir / "manifest.json"
    dense_manifest_path = dense_dir / "manifest.json"
    if not col_manifest_path.is_file():
        errors.append(f"missing ColVision manifest: {col_manifest_path}")
        col_manifest = {}
    else:
        col_manifest = read_json(col_manifest_path)
    if not dense_manifest_path.is_file():
        errors.append(f"missing dense manifest: {dense_manifest_path}")
        dense_manifest = {}
    else:
        dense_manifest = read_json(dense_manifest_path)

    verification = col_manifest.get("checkpoint_verification", {})
    if not verification.get("verified", False):
        errors.append("ColVision checkpoint manifest is not verified")
    model_name = str(col_manifest.get("model", ""))
    if "merged" not in model_name.lower() and not args.allow_nonmerged_colvision:
        errors.append(f"submission run must use a merged ColVision checkpoint, found: {model_name!r}")

    expected_documents = col_manifest.get("documents")
    expected_queries = col_manifest.get("queries")
    if expected_documents != dense_manifest.get("documents"):
        errors.append("ColVision and dense document counts differ")
    if expected_queries != dense_manifest.get("queries"):
        errors.append("ColVision and dense query counts differ")

    checksums: set[str] = set()
    config_hashes: dict[str, str] = {}
    result_summary: dict[str, dict] = {}
    for experiment in experiments:
        experiment_id = experiment["id"]
        result_path = results_dir / f"{experiment_id}.json"
        if not result_path.is_file() or result_path.stat().st_size == 0:
            errors.append(f"missing result: {result_path}")
            continue
        payload = read_json(result_path)
        metadata = payload.get("metadata", {})
        metrics = payload.get("metrics", {})
        efficiency = payload.get("efficiency", {})
        per_query = payload.get("per_query", {})
        if metadata.get("documents") != expected_documents:
            errors.append(f"{experiment_id}: document count mismatch")
        if metadata.get("queries") != expected_queries:
            errors.append(f"{experiment_id}: query count mismatch")
        if len(per_query) != expected_queries:
            errors.append(f"{experiment_id}: per-query trace count mismatch")
        checksum = str(metadata.get("dataset_checksum", ""))
        if not checksum:
            errors.append(f"{experiment_id}: missing dataset checksum")
        else:
            checksums.add(checksum)
        requires_dense = bool(experiment.get("requires_dense", False))
        has_dense = bool(metadata.get("dense_dir"))
        if requires_dense != has_dense:
            errors.append(
                f"{experiment_id}: dense attachment mismatch; requires_dense={requires_dense}, metadata={has_dense}"
            )
        config_hash = str(metadata.get("config_hash", ""))
        if not config_hash:
            errors.append(f"{experiment_id}: missing config hash")
        elif config_hash in config_hashes:
            warnings.append(
                f"{experiment_id} and {config_hashes[config_hash]} share config hash {config_hash}; "
                "this is valid only when candidate sources differ through dense attachment"
            )
        else:
            config_hashes[config_hash] = experiment_id
        for name, value in {**metrics, **efficiency}.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                errors.append(f"{experiment_id}: non-finite value {name}={value}")
        result_summary[experiment_id] = {
            "nDCG@5": metrics.get("nDCG@5"),
            "nDCG@10": metrics.get("nDCG@10"),
            "MRR@10": metrics.get("MRR@10"),
            "token_reduction": efficiency.get("token_reduction"),
            "mean_latency_ms": efficiency.get("mean_latency_ms"),
            "fallback_rate": payload.get("reliability", {}).get("fallback_rate"),
        }

    if len(checksums) > 1:
        errors.append(f"results contain multiple dataset checksums: {sorted(checksums)}")

    report = {
        "valid": not errors,
        "matrix": str(matrix_path),
        "dataset_dir": str(dataset_dir),
        "colvision_manifest": col_manifest,
        "dense_manifest": dense_manifest,
        "dataset_checksums": sorted(checksums),
        "experiments_expected": [item["id"] for item in experiments],
        "result_summary": result_summary,
        "warnings": warnings,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
