#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
OUTPUT="${OUTPUT:-AdaColRAG_paper_finalization_review_bundle.tar.gz}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
DATASETS=(
  vidore_v3_finance_en
  vidore_v3_industrial
  vidore_v3_pharmaceuticals
  vidore_v3_finance_fr
)

cd "$PROJECT_ROOT"
mkdir -p results/paper_final/_meta

if [[ "$ALLOW_DIRTY" != "1" ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Tracked repository files are modified. Commit or restore them before packaging." >&2
    git status --short >&2
    exit 2
  fi
fi

if grep -RInE '^(<<<<<<<|=======|>>>>>>>)' paper/draft.md >/tmp/adacolrag_paper_conflicts.txt; then
  echo "paper/draft.md contains merge-conflict markers:" >&2
  cat /tmp/adacolrag_paper_conflicts.txt >&2
  exit 2
fi

git rev-parse HEAD > results/paper_final/_meta/git_head.txt
git status --short > results/paper_final/_meta/git_status.txt

python - <<'PY'
import json
from pathlib import Path

datasets = (
    "vidore_v3_finance_en",
    "vidore_v3_industrial",
    "vidore_v3_pharmaceuticals",
    "vidore_v3_finance_fr",
)
for slug in datasets:
    for root_name, expected in (("results/submission", 13), ("results/final_targeted", 4)):
        root = Path(root_name) / slug
        validation = root / "validation.json"
        if not validation.is_file():
            raise SystemExit(f"missing validation: {validation}")
        payload = json.loads(validation.read_text(encoding="utf-8"))
        if not payload.get("valid") or payload.get("errors"):
            raise SystemExit(f"invalid result bundle: {validation}: {payload.get('errors')}")
        runs = list((root / "runs").glob("*.json"))
        if len(runs) != expected:
            raise SystemExit(f"unexpected run count below {root}: {len(runs)} != {expected}")

    paper_root = Path("results/paper_final") / slug
    for required in (
        paper_root / "significance" / "final_vs_baselines.json",
        paper_root / "significance" / "final_vs_baselines.csv",
        paper_root / "significance" / "final_vs_baselines.md",
        paper_root / "steady_state_timing.json",
    ):
        if not required.is_file() or required.stat().st_size == 0:
            raise SystemExit(f"missing paper-final evidence: {required}")

for required in (
    Path("paper/generated/final_paper_results.json"),
    Path("paper/generated/final_main_results.md"),
    Path("paper/generated/final_significance.md"),
    Path("paper/generated/final_efficiency.md"),
    Path("paper/generated/final_steady_timing.md"),
    Path("paper/generated/final_selection.md"),
):
    if not required.is_file() or required.stat().st_size == 0:
        raise SystemExit(f"missing generated final table: {required}")

print("all final manuscript evidence: OK")
PY

manifest_list=/tmp/adacolrag_paper_final_manifests.txt
: > "$manifest_list"
for slug in "${DATASETS[@]}"; do
  printf '%s\n' \
    "artifacts/$slug/colpali_v13_merged/manifest.json" \
    "artifacts/$slug/visrag_ret/manifest.json" \
    "data/$slug/dataset_info.json" \
    >> "$manifest_list"
done

while IFS= read -r path; do
  if [[ ! -s "$path" ]]; then
    echo "Missing provenance file: $path" >&2
    exit 2
  fi
done < "$manifest_list"

config_list=/tmp/adacolrag_paper_final_configs.txt
find configs -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) -print | sort > "$config_list"

script_list=/tmp/adacolrag_paper_final_scripts.txt
printf '%s\n' \
  scripts/analyze_final_method_vs_baselines.py \
  scripts/steady_state_benchmarks.py \
  scripts/aggregate_final_paper_results.py \
  scripts/update_final_paper_results.py \
  scripts/run_paper_finalization.sh \
  scripts/package_paper_finalization_review.sh \
  scripts/run_matrix.py \
  scripts/analyze_submission_results.py \
  scripts/validate_submission_results.py \
  > "$script_list"

archive_paths=(
  results/submission
  results/final_targeted
  results/paper_final
  paper/generated
  paper/draft.md
  paper/EXPERIMENT_PROTOCOL.md
  docs/SUBMISSION_RUNBOOK.md
  README.md
  models/colpali-v1.3-merged/snapshot_info.json
  logs/paper_final
)
while IFS= read -r path; do archive_paths+=("$path"); done < "$manifest_list"
while IFS= read -r path; do archive_paths+=("$path"); done < "$config_list"
while IFS= read -r path; do archive_paths+=("$path"); done < "$script_list"

tar -czf "$OUTPUT" "${archive_paths[@]}"

printf 'Created %s/%s\n' "$PROJECT_ROOT" "$OUTPUT"
ls -lh "$OUTPUT"
printf '\nKey archive contents:\n'
tar -tzf "$OUTPUT" \
  | grep -E 'paper_final|final_(main|significance|efficiency|steady|selection)|validation.json|manifest.json|dataset_info.json|draft.md' \
  | head -300
