#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
OUTPUT="${OUTPUT:-AdaColRAG_final_method_selection_bundle.tar.gz}"
DATASETS=(
  vidore_v3_finance_en
  vidore_v3_industrial
  vidore_v3_pharmaceuticals
  vidore_v3_finance_fr
)

cd "$PROJECT_ROOT"
mkdir -p results/final_targeted/_meta

git rev-parse HEAD > results/final_targeted/_meta/git_head.txt
git status --short > results/final_targeted/_meta/git_status.txt

if grep -RInE '^(<<<<<<<|=======|>>>>>>>)' paper/draft.md >/tmp/adacolrag_conflicts.txt; then
  echo "paper/draft.md still contains merge-conflict markers:" >&2
  cat /tmp/adacolrag_conflicts.txt >&2
  exit 2
fi

python - <<'PY'
import json
from pathlib import Path

for root_name in ("results/submission", "results/final_targeted"):
    root = Path(root_name)
    if not root.is_dir():
        raise SystemExit(f"missing result root: {root}")

for slug in (
    "vidore_v3_finance_en",
    "vidore_v3_industrial",
    "vidore_v3_pharmaceuticals",
    "vidore_v3_finance_fr",
):
    for root_name, expected in (("results/submission", 13), ("results/final_targeted", 4)):
        root = Path(root_name) / slug
        validation_path = root / "validation.json"
        if not validation_path.is_file():
            raise SystemExit(f"missing validation: {validation_path}")
        data = json.loads(validation_path.read_text(encoding="utf-8"))
        if not data.get("valid") or data.get("errors"):
            raise SystemExit(f"invalid result bundle: {validation_path}: {data.get('errors')}")
        runs = list((root / "runs").glob("*.json"))
        if len(runs) != expected:
            raise SystemExit(f"unexpected run count below {root}: {len(runs)} != {expected}")
        if not (root / "repeated_timing.json").is_file():
            raise SystemExit(f"missing repeated timing: {root / 'repeated_timing.json'}")

print("all submission and targeted validation bundles: OK")
PY

manifest_list=/tmp/adacolrag_final_manifest_files.txt
: > "$manifest_list"
for slug in "${DATASETS[@]}"; do
  printf '%s\n' \
    "artifacts/$slug/colpali_v13_merged/manifest.json" \
    "artifacts/$slug/visrag_ret/manifest.json" \
    "data/$slug/dataset_info.json" \
    >> "$manifest_list"
done

for path in $(cat "$manifest_list"); do
  if [[ ! -s "$path" ]]; then
    echo "Missing required provenance file: $path" >&2
    exit 2
  fi
done

config_list=/tmp/adacolrag_final_config_files.txt
find configs -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) -print | sort > "$config_list"

script_list=/tmp/adacolrag_final_script_files.txt
printf '%s\n' \
  scripts/run_final_targeted_experiments.sh \
  scripts/package_final_experiment_review.sh \
  scripts/run_matrix.py \
  scripts/analyze_submission_results.py \
  scripts/repeat_benchmarks.py \
  scripts/validate_submission_results.py \
  scripts/aggregate_final_targeted_results.py \
  scripts/aggregate_submission_results.py \
  scripts/update_paper_results.py \
  > "$script_list"

archive_paths=(
  results/submission
  results/final_targeted
  paper/generated
  paper/draft.md
  paper/EXPERIMENT_PROTOCOL.md
  docs/SUBMISSION_RUNBOOK.md
  models/colpali-v1.3-merged/snapshot_info.json
  logs/final_targeted
)

while IFS= read -r path; do archive_paths+=("$path"); done < "$manifest_list"
while IFS= read -r path; do archive_paths+=("$path"); done < "$config_list"
while IFS= read -r path; do archive_paths+=("$path"); done < "$script_list"

tar -czf "$OUTPUT" "${archive_paths[@]}"

printf 'Created %s\n' "$PROJECT_ROOT/$OUTPUT"
ls -lh "$OUTPUT"
printf '\nKey archive contents:\n'
tar -tzf "$OUTPUT" | grep -E 'validation.json|summary.json|significance_table.md|repeated_timing.json|targeted_results|manifest.json|dataset_info.json|draft.md' | head -200
