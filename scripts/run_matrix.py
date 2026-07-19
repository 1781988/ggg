#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/matrix.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--colvision-dir", required=True)
    parser.add_argument("--dense-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with Path(args.matrix).open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for experiment in matrix["experiments"]:
        if experiment.get("requires_dense", False) and not args.dense_dir:
            print(f"skip {experiment['id']}: --dense-dir is required")
            continue
        command = [
            sys.executable,
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
            str(output_dir / f"{experiment['id']}.json"),
        ]
        if args.dense_dir:
            command.extend(["--dense-dir", args.dense_dir])
        print("run:", " ".join(command))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures.append(experiment["id"])
    if failures:
        raise SystemExit(f"failed experiments: {', '.join(failures)}")


if __name__ == "__main__":
    main()
