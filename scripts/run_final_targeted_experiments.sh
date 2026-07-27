#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
MATRIX="${MATRIX:-configs/final_targeted_matrix.yaml}"
DATASETS="${FINAL_DATASETS:-vidore_v3_finance_en vidore_v3_industrial vidore_v3_pharmaceuticals vidore_v3_finance_fr}"
COLVISION_SLUG="${COLVISION_SLUG:-colpali_v13_merged}"
REPEATS="${FINAL_TIMING_REPEATS:-7}"
WARMUPS="${FINAL_TIMING_WARMUPS:-1}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
FRESH_TARGETED="${FRESH_TARGETED:-1}"
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

mkdir -p results/final_targeted logs/final_targeted

run_core() {
  if [[ "$USE_TASKSET" == "1" ]] && command -v taskset >/dev/null 2>&1; then
    taskset -c "$CPU_AFFINITY" conda run --no-capture-output -n adacolrag-core "$@"
  else
    conda run --no-capture-output -n adacolrag-core "$@"
  fi
}

printf '[preflight] git=%s\n' "$(git rev-parse HEAD)"
printf '[preflight] matrix=%s datasets=%s repeats=%s warmups=%s affinity=%s\n' \
  "$MATRIX" "$DATASETS" "$REPEATS" "$WARMUPS" "$CPU_AFFINITY"

run_core python - <<'PY'
from pathlib import Path
import yaml
from adacolrag.config import load_config

matrix_path = Path("configs/final_targeted_matrix.yaml")
data = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
assert len(data["experiments"]) == 4
for item in data["experiments"]:
    path = Path(item["config"])
    assert path.is_file(), path
    config = load_config(path)
    assert config["mode"] == "fixed_mmr"
    assert int(config["fixed_tokens"]) == 112
    assert abs(float(config["fusion"]["dense_weight"]) - 0.25) < 1e-12
print("final targeted matrix: OK")
PY

for slug in $DATASETS; do
  dataset_dir="data/$slug"
  colvision_dir="artifacts/$slug/$COLVISION_SLUG"
  dense_dir="artifacts/$slug/visrag_ret"
  result_root="results/final_targeted/$slug"
  run_dir="$result_root/runs"
  analysis_dir="$result_root/analysis"
  log_dir="logs/final_targeted/$slug"

  printf '\n============================================================\n'
  printf 'Final targeted experiments: %s\n' "$slug"
  printf '============================================================\n'

  for required in \
    "$dataset_dir/corpus.jsonl" \
    "$dataset_dir/queries.jsonl" \
    "$dataset_dir/qrels.jsonl" \
    "$dataset_dir/dataset_info.json" \
    "$colvision_dir/manifest.json" \
    "$dense_dir/manifest.json"
  do
    if [[ ! -s "$required" ]]; then
      echo "Missing required local file: $required" >&2
      exit 2
    fi
  done

  if [[ "$FRESH_TARGETED" == "1" ]]; then
    rm -rf "$result_root" "$log_dir"
  fi
  mkdir -p "$run_dir" "$analysis_dir" "$log_dir"

  printf '[1/4] Run four final candidates\n'
  run_core python -u scripts/run_matrix.py \
    --matrix "$MATRIX" \
    --dataset-dir "$dataset_dir" \
    --colvision-dir "$colvision_dir" \
    --dense-dir "$dense_dir" \
    --output-dir "$run_dir" \
    2>&1 | tee "$log_dir/01_matrix.log"

  printf '[2/4] Validate final candidate results\n'
  run_core python -u scripts/validate_submission_results.py \
    --matrix "$MATRIX" \
    --dataset-dir "$dataset_dir" \
    --colvision-dir "$colvision_dir" \
    --dense-dir "$dense_dir" \
    --results-dir "$run_dir" \
    --output "$result_root/validation.json" \
    2>&1 | tee "$log_dir/02_validation.log"

  printf '[3/4] Paired query-level bootstrap analysis\n'
  run_core python -u scripts/analyze_submission_results.py \
    --matrix "$MATRIX" \
    --dataset-dir "$dataset_dir" \
    --results-dir "$run_dir" \
    --output-dir "$analysis_dir" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    2>&1 | tee "$log_dir/03_analysis.log"

  printf '[4/4] Stable repeated timing\n'
  run_core python -u scripts/repeat_benchmarks.py \
    --matrix "$MATRIX" \
    --only "final_redundancy_dense025_no_fallback,final_redundancy_dense025_fallback,final_mmr_dense025_no_fallback,final_mmr_dense025_fallback" \
    --dataset-dir "$dataset_dir" \
    --colvision-dir "$colvision_dir" \
    --dense-dir "$dense_dir" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS" \
    --output "$result_root/repeated_timing.json" \
    2>&1 | tee "$log_dir/04_timing.log"

done

printf '\n[aggregate] Final candidate cross-dataset table\n'
run_core python -u scripts/aggregate_final_targeted_results.py \
  --root results/final_targeted \
  --output-dir results/final_targeted/_aggregate \
  2>&1 | tee logs/final_targeted/05_aggregate.log

printf '\nFinal targeted experiment workflow completed.\n'
printf 'Results: %s/results/final_targeted/\n' "$PROJECT_ROOT"
