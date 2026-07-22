#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
DATASETS="${FINAL_PAPER_DATASETS:-vidore_v3_finance_en vidore_v3_industrial vidore_v3_pharmaceuticals vidore_v3_finance_fr}"
FINAL_MATRIX="${FINAL_MATRIX:-configs/final_paper_matrix.yaml}"
COLVISION_SLUG="${COLVISION_SLUG:-colpali_v13_merged}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
STEADY_REPEATS="${STEADY_REPEATS:-7}"
STEADY_WARMUPS="${STEADY_WARMUPS:-1}"
STEADY_METHODS="${STEADY_METHODS:-visrag_dense,colpali_full,visrag_top50_full,adacolrag}"
CPU_AFFINITY="${CPU_AFFINITY:-0-7}"
USE_TASKSET="${USE_TASKSET:-1}"

cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

mkdir -p results/paper_final/_meta logs/paper_final paper/generated

run_core() {
  if [[ "$USE_TASKSET" == "1" ]] && command -v taskset >/dev/null 2>&1; then
    taskset -c "$CPU_AFFINITY" conda run --no-capture-output -n adacolrag-core "$@"
  else
    conda run --no-capture-output -n adacolrag-core "$@"
  fi
}

printf '[preflight] git=%s\n' "$(git rev-parse HEAD)"
printf '[preflight] matrix=%s bootstrap=%s steady_repeats=%s steady_warmups=%s affinity=%s\n' \
  "$FINAL_MATRIX" "$BOOTSTRAP_SAMPLES" "$STEADY_REPEATS" "$STEADY_WARMUPS" "$CPU_AFFINITY"

run_core python - <<'PY'
from pathlib import Path
import yaml
from adacolrag.config import load_config

matrix_path = Path("configs/final_paper_matrix.yaml")
data = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
ids = [item["id"] for item in data["experiments"]]
assert len(ids) == len(set(ids))
assert "adacolrag" in ids
config = load_config("configs/adacolrag.yaml")
assert config["mode"] == "fixed_mmr"
assert int(config["fixed_tokens"]) == 112
assert float(config["selector"]["redundancy_weight"]) == 0.25
assert float(config["selector"]["layout_weight"]) == 0.0
assert float(config["selector"]["prefilter_factor"]) == 4.0
assert float(config["fusion"]["dense_weight"]) == 0.25
assert config["fallback"]["enabled"] is False
print("final paper configuration: OK")
PY

for slug in $DATASETS; do
  dataset_dir="data/$slug"
  colvision_dir="artifacts/$slug/$COLVISION_SLUG"
  dense_dir="artifacts/$slug/visrag_ret"
  submission_runs="results/submission/$slug/runs"
  targeted_runs="results/final_targeted/$slug/runs"
  output_root="results/paper_final/$slug"
  log_dir="logs/paper_final/$slug"
  mkdir -p "$output_root/significance" "$log_dir"

  printf '\n============================================================\n'
  printf 'Paper finalization: %s\n' "$slug"
  printf '============================================================\n'

  for required in \
    "$dataset_dir/qrels.jsonl" \
    "$colvision_dir/manifest.json" \
    "$dense_dir/manifest.json" \
    "$submission_runs/visrag_dense.json" \
    "$submission_runs/colpali_full.json" \
    "$submission_runs/visrag_top50_full.json" \
    "$targeted_runs/final_redundancy_dense025_no_fallback.json"
  do
    if [[ ! -s "$required" ]]; then
      echo "Missing required existing evidence: $required" >&2
      exit 2
    fi
  done

  printf '[1/2] Final method versus stored baselines bootstrap\n'
  run_core python -u scripts/analyze_final_method_vs_baselines.py \
    --dataset-dir "$dataset_dir" \
    --submission-runs "$submission_runs" \
    --targeted-runs "$targeted_runs" \
    --output-dir "$output_root/significance" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    2>&1 | tee "$log_dir/01_final_significance.log"

  printf '[2/2] Single-process steady-state retrieval timing\n'
  run_core python -u scripts/steady_state_benchmarks.py \
    --matrix "$FINAL_MATRIX" \
    --only "$STEADY_METHODS" \
    --dataset-dir "$dataset_dir" \
    --colvision-dir "$colvision_dir" \
    --dense-dir "$dense_dir" \
    --warmups "$STEADY_WARMUPS" \
    --repeats "$STEADY_REPEATS" \
    --output "$output_root/steady_state_timing.json" \
    2>&1 | tee "$log_dir/02_steady_state_timing.log"
done

printf '\n[aggregate] Build final manuscript tables\n'
run_core python -u scripts/aggregate_final_paper_results.py \
  --submission-root results/submission \
  --targeted-root results/final_targeted \
  --paper-final-root results/paper_final \
  --output-dir paper/generated \
  2>&1 | tee logs/paper_final/03_aggregate.log

printf '\n[paper] Inject final evidence tables\n'
run_core python -u scripts/update_final_paper_results.py \
  --paper paper/draft.md \
  --generated-dir paper/generated \
  2>&1 | tee logs/paper_final/04_update_paper.log

git rev-parse HEAD > results/paper_final/_meta/git_head.txt
git status --short > results/paper_final/_meta/git_status.txt
env | grep -E '^(OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS|CUDA_VISIBLE_DEVICES)=' \
  | sort > results/paper_final/_meta/runtime_environment.txt || true

printf '\nPaper finalization evidence completed.\n'
printf 'Results: %s/results/paper_final/\n' "$PROJECT_ROOT"
printf 'Generated tables: %s/paper/generated/final_*.md\n' "$PROJECT_ROOT"
printf 'Updated manuscript: %s/paper/draft.md\n' "$PROJECT_ROOT"
