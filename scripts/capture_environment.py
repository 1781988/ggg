#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture hardware, threading, and package metadata")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected_environment = {
        name: os.environ.get(name)
        for name in [
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "HF_ENDPOINT",
            "HF_HOME",
            "HF_HUB_CACHE",
            "TOKENIZERS_PARALLELISM",
        ]
    }
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
        "packages": {
            name: package_version(name)
            for name in [
                "numpy",
                "PyYAML",
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "colpali-engine",
                "peft",
                "sentencepiece",
                "timm",
                "decord",
                "datasets",
                "huggingface-hub",
            ]
        },
        "environment": selected_environment,
        "lscpu": command_output(["lscpu"]),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
