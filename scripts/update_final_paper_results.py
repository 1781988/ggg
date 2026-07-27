#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


MARKERS = {
    "FINAL_SELECTION": "final_selection.md",
    "FINAL_MAIN_RESULTS": "final_main_results.md",
    "FINAL_SIGNIFICANCE": "final_significance.md",
    "FINAL_EFFICIENCY": "final_efficiency.md",
    "FINAL_STEADY_TIMING": "final_steady_timing.md",
}


def replace_block(text: str, name: str, content: str) -> str:
    begin = f"<!-- BEGIN AUTO:{name} -->"
    end = f"<!-- END AUTO:{name} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"paper must contain exactly one marker pair for {name}")
    prefix, remainder = text.split(begin, 1)
    _, suffix = remainder.split(end, 1)
    return prefix + begin + "\n\n" + content.strip() + "\n\n" + end + suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject final AdaColRAG evidence into paper/draft.md")
    parser.add_argument("--paper", default="paper/draft.md")
    parser.add_argument("--generated-dir", default="paper/generated")
    args = parser.parse_args()

    paper_path = Path(args.paper)
    generated_dir = Path(args.generated_dir)
    text = paper_path.read_text(encoding="utf-8")
    if any(marker in text for marker in ("<<<<<<<", ">>>>>>>")):
        raise ValueError(f"{paper_path} contains unresolved merge-conflict markers")

    for marker, filename in MARKERS.items():
        path = generated_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        text = replace_block(text, marker, path.read_text(encoding="utf-8"))

    paper_path.write_text(text, encoding="utf-8")
    print(f"Updated {paper_path} from final generated evidence")


if __name__ == "__main__":
    main()
