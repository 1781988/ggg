#!/usr/bin/env python
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
from pathlib import Path


def cache_dir_from_env() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def owner_pids(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["fuser", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "unknown (fuser is not installed)"
    output = f"{completed.stdout} {completed.stderr}".strip()
    return output or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect active and stale Hugging Face cache lock files")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--clean-stale", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else cache_dir_from_env()
    lock_root = cache_dir / ".locks"
    print(f"[hf-cache] cache_dir={cache_dir}", flush=True)
    print(f"[hf-cache] lock_root={lock_root}", flush=True)

    if not lock_root.exists():
        print("[hf-cache] no lock directory; preflight passed", flush=True)
        return

    active: list[Path] = []
    stale: list[Path] = []
    handles = []
    for lock_path in sorted(lock_root.rglob("*.lock")):
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            active.append(lock_path)
            handle.close()
        else:
            stale.append(lock_path)
            handles.append((lock_path, handle))

    if active:
        print("[hf-cache] active download locks detected:", flush=True)
        for lock_path in active:
            print(f"  ACTIVE {lock_path} owners={owner_pids(lock_path)}", flush=True)
        print(
            "[hf-cache] stop the existing model-download/export process before starting another experiment.",
            flush=True,
        )
        for _, handle in handles:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        raise SystemExit(3)

    if stale:
        print(f"[hf-cache] {len(stale)} inactive lock file(s) found", flush=True)
    for lock_path, handle in handles:
        if args.clean_stale:
            try:
                lock_path.unlink(missing_ok=True)
                print(f"  REMOVED {lock_path}", flush=True)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        else:
            print(f"  INACTIVE {lock_path}", flush=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    print("[hf-cache] preflight passed", flush=True)


if __name__ == "__main__":
    main()
