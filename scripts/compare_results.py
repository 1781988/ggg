#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_smoke(reference: dict, candidate: dict) -> list[str]:
    failures: list[str] = []
    minimum_ndcg = float(reference["invariants"]["min_ndcg5"])
    minimum_reduction = float(reference["invariants"]["min_token_reduction"])
    maximum_reduction = float(reference["invariants"]["max_token_reduction"])
    ndcg = float(candidate["metrics"]["nDCG@5"])
    reduction = float(candidate["efficiency"]["token_reduction"])
    if ndcg < minimum_ndcg:
        failures.append(f"nDCG@5 {ndcg:.6f} < {minimum_ndcg:.6f}")
    if not minimum_reduction <= reduction <= maximum_reduction:
        failures.append(
            f"token_reduction {reduction:.6f} outside [{minimum_reduction:.6f}, {maximum_reduction:.6f}]"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--max-ndcg5-drop", type=float, default=0.01)
    parser.add_argument("--min-token-reduction", type=float, default=0.50)
    parser.add_argument("--min-latency-reduction", type=float, default=0.30)
    parser.add_argument("--reference-smoke")
    parser.add_argument("--candidate-smoke")
    args = parser.parse_args()

    if args.reference_smoke or args.candidate_smoke:
        if not (args.reference_smoke and args.candidate_smoke):
            parser.error("both --reference-smoke and --candidate-smoke are required")
        failures = compare_smoke(load(args.reference_smoke), load(args.candidate_smoke))
    else:
        if not (args.baseline and args.candidate):
            parser.error("both --baseline and --candidate are required")
        baseline = load(args.baseline)
        candidate = load(args.candidate)
        baseline_ndcg = float(baseline["metrics"]["nDCG@5"])
        candidate_ndcg = float(candidate["metrics"]["nDCG@5"])
        baseline_latency = float(baseline["efficiency"]["mean_latency_ms"])
        candidate_latency = float(candidate["efficiency"]["mean_latency_ms"])
        token_reduction = float(candidate["efficiency"]["token_reduction"])
        ndcg_drop = baseline_ndcg - candidate_ndcg
        latency_reduction = 0.0 if baseline_latency <= 0 else 1.0 - candidate_latency / baseline_latency
        failures = []
        if ndcg_drop > args.max_ndcg5_drop:
            failures.append(f"nDCG@5 drop {ndcg_drop:.6f} > {args.max_ndcg5_drop:.6f}")
        if token_reduction < args.min_token_reduction:
            failures.append(f"token reduction {token_reduction:.6f} < {args.min_token_reduction:.6f}")
        if latency_reduction < args.min_latency_reduction:
            failures.append(f"latency reduction {latency_reduction:.6f} < {args.min_latency_reduction:.6f}")
        print(
            json.dumps(
                {
                    "ndcg5_drop": ndcg_drop,
                    "token_reduction": token_reduction,
                    "latency_reduction": latency_reduction,
                },
                indent=2,
            )
        )
    if failures:
        for failure in failures:
            print("FAIL:", failure)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
