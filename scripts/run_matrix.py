#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_only(values: list[str] | None) -> set[str]:
    selected: set[str] = set()
    for value in values or []:
        selected.update(item.strip() for item in value.split(",") if item.strip())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/matrix.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--colvision-dir", required=True)
    parser.add_argument("--dense-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--only",
        action="append",
        help="Run only selected experiment IDs; repeat or use comma-separated IDs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments whose non-empty output JSON already exists",
    )
    args = parser.parse_args()

    with Path(args.matrix).open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    experiments = matrix["experiments"]
    selected_ids = parse_only(args.only)
    if selected_ids:
        known_ids = {experiment["id"] for experiment in experiments}
        unknown = sorted(selected_ids - known_ids)
        if unknown:
            raise SystemExit(f"unknown experiment IDs: {', '.join(unknown)}")
        experiments = [experiment for experiment in experiments if experiment["id"] in selected_ids]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    total = len(experiments)

    for index, experiment in enumerate(experiments, start=1):
        experiment_id = experiment["id"]
        output_path = output_dir / f"{experiment_id}.json"
        requires_dense = bool(experiment.get("requires_dense", False))
        description = experiment.get("description", "")
        print(
            f"\n[matrix {index}/{total}] id={experiment_id} "
            f"requires_dense={requires_dense} description={description}",
            flush=True,
        )

        if requires_dense and not args.dense_dir:
            print(f"skip {experiment_id}: --dense-dir is required", flush=True)
            failures.append(experiment_id)
            continue
        if args.resume and output_path.exists() and output_path.stat().st_size > 0:
            print(f"reuse {output_path}", flush=True)
            continue

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
        # Dense embeddings are attached only to experiments that explicitly
        # require them. This prevents VisRAG coarse retrieval from leaking into
        # ColPali-only and ablation baselines.
        if requires_dense:
            command.extend(["--dense-dir", args.dense_dir])
        print("run:", " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures.append(experiment_id)
            print(f"failed: {experiment_id} returncode={completed.returncode}", flush=True)
        else:
            print(f"completed: {experiment_id} -> {output_path}", flush=True)

    if failures:
        raise SystemExit(f"failed experiments: {', '.join(failures)}")


if __name__ == "__main__":
    main()
