#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


def parse_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat selected retrieval experiments for timing stability")
    parser.add_argument("--matrix", default="configs/submission_matrix.yaml")
    parser.add_argument("--only", required=True, help="Comma-separated experiment IDs")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--colvision-dir", required=True)
    parser.add_argument("--dense-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()

    if args.repeats < 1 or args.warmups < 0:
        raise SystemExit("--repeats must be >=1 and --warmups must be >=0")

    with Path(args.matrix).open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)["experiments"]
    by_id = {item["id"]: item for item in matrix}
    selected = parse_ids(args.only)
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise SystemExit(f"unknown experiment IDs: {', '.join(unknown)}")

    report: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="adacolrag_timing_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for experiment_id in selected:
            experiment = by_id[experiment_id]
            runs: list[dict] = []
            total_runs = args.warmups + args.repeats
            for run_index in range(total_runs):
                output_path = temp_dir / f"{experiment_id}_{run_index}.json"
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "adacolrag.cli",
                    "evaluate",
                    "--config",
                    experiment["config"],
                    "--dataset-dir",
                    args.dataset_dir,
                    "--colvision-dir",
                    args.colvision_dir,
                    "--output",
                    str(output_path),
                ]
                if experiment.get("requires_dense", False):
                    command.extend(["--dense-dir", args.dense_dir])
                phase = "warmup" if run_index < args.warmups else "measured"
                print(
                    f"[timing] experiment={experiment_id} phase={phase} "
                    f"run={run_index + 1}/{total_runs}",
                    flush=True,
                )
                wall_start = time.perf_counter()
                subprocess.run(command, check=True)
                wall_seconds = time.perf_counter() - wall_start
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                if phase == "measured":
                    runs.append(
                        {
                            "wall_seconds": wall_seconds,
                            "mean_latency_ms": float(payload["efficiency"]["mean_latency_ms"]),
                            "p50_latency_ms": float(payload["efficiency"]["p50_latency_ms"]),
                            "p95_latency_ms": float(payload["efficiency"]["p95_latency_ms"]),
                            "nDCG@5": float(payload["metrics"]["nDCG@5"]),
                        }
                    )
            report[experiment_id] = {
                "config": experiment["config"],
                "requires_dense": bool(experiment.get("requires_dense", False)),
                "warmups": args.warmups,
                "repeats": args.repeats,
                "runs": runs,
                "wall_seconds": summarize([item["wall_seconds"] for item in runs]),
                "mean_latency_ms": summarize([item["mean_latency_ms"] for item in runs]),
                "p50_latency_ms": summarize([item["p50_latency_ms"] for item in runs]),
                "p95_latency_ms": summarize([item["p95_latency_ms"] for item in runs]),
                "nDCG@5": summarize([item["nDCG@5"] for item in runs]),
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_dir": args.dataset_dir,
        "colvision_dir": args.colvision_dir,
        "dense_dir": args.dense_dir,
        "experiments": report,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
