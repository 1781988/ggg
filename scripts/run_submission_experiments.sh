#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
MAIN_MATRIX="${MAIN_MATRIX:-configs/submission_matrix.yaml}"
DEVELOPMENT_MATRIX="${DEVELOPMENT_MATRIX:-configs/development_matrix.yaml}"
SUBMISSION_DATASETS="${SUBMISSION_DATASETS:-vidore/vidore_v3_finance_en|vidore_v3_finance_en|english|0fe6508053e8aa31b1f1eaec4553f79e3974f59a;vidore/vidore_v3_industrial|vidore_v3_industrial|english|233d20721f4deb392a09a86cca01761adbc91157;vidore/vidore_v3_pharmaceuticals|vidore_v3_pharmaceuticals|english|262e203d7c59c55947f7042812e0bcc1eee190b1;vidore/vidore_v3_finance_fr|vidore_v3_finance_fr|french|d1178382ea47039073ad08e5e2c97ef51d6b629e}"
DEVELOPMENT_DATASET_SLUG="${DEVELOPMENT_DATASET_SLUG:-vidore_v3_finance_en}"
RUN_DEVELOPMENT="${RUN_DEVELOPMENT:-1}"
QUICK_MODE="${QUICK_MODE:-0}"
COLVISION_REMOTE_MODEL="${COLVISION_REMOTE_MODEL:-vidore/colpali-v1.3-merged}"
COLVISION_MODEL_REVISION="${COLVISION_MODEL_REVISION:-5b955e3415a7c5468ab33119d98d6d45c3a5b2c3}"
COLVISION_LOCAL_MODEL_DIR="${COLVISION_LOCAL_MODEL_DIR:-$PROJECT_ROOT/models/colpali-v1.3-merged}"
COLVISION_DOWNLOAD_ENDPOINTS="${COLVISION_DOWNLOAD_ENDPOINTS:-https://hf-mirror.com https://huggingface.co}"
DOWNLOAD_COLVISION_MODEL="${DOWNLOAD_COLVISION_MODEL:-1}"
COLVISION_MODEL="$COLVISION_LOCAL_MODEL_DIR"
COLVISION_SLUG="${COLVISION_SLUG:-colpali_v13_merged}"
COLVISION_BATCH_SIZE="${COLVISION_BATCH_SIZE:-1}"
VISRAG_MODEL="${VISRAG_MODEL:-openbmb/VisRAG-Ret}"
VISRAG_LOCAL_MODEL_DIR="${VISRAG_LOCAL_MODEL_DIR:-}"
VISRAG_BATCH_SIZE="${VISRAG_BATCH_SIZE:-1}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
REBUILD_EMBEDDINGS="${REBUILD_EMBEDDINGS:-0}"
FRESH_RESULTS="${FRESH_RESULTS:-1}"
RUN_TIMING="${RUN_TIMING:-1}"
RUN_DEVELOPMENT_TIMING="${RUN_DEVELOPMENT_TIMING:-1}"
TIMING_REPEATS="${TIMING_REPEATS:-3}"
TIMING_WARMUPS="${TIMING_WARMUPS:-1}"
TIMING_EXPERIMENTS="${TIMING_EXPERIMENTS:-visrag_dense,colpali_full,visrag_top50_full,visrag_top50_fixed_112,visrag_top50_mmr,adacolrag_no_fallback,adacolrag}"
DEVELOPMENT_TIMING_EXPERIMENTS="${DEVELOPMENT_TIMING_EXPERIMENTS:-dev_mmr_factor2,dev_mmr_factor4,dev_mmr_exact}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
MIN_FREE_GB="${MIN_FREE_GB:-80}"

if [[ "$QUICK_MODE" == "1" ]]; then
  SUBMISSION_DATASETS="vidore/vidore_v3_finance_en|vidore_v3_finance_en|english|0fe6508053e8aa31b1f1eaec4553f79e3974f59a"
  TIMING_REPEATS="${QUICK_TIMING_REPEATS:-1}"
  TIMING_WARMUPS="${QUICK_TIMING_WARMUPS:-0}"
  MIN_FREE_GB="${QUICK_MIN_FREE_GB:-20}"
fi

cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
unset HF_HUB_ENABLE_HF_TRANSFER || true

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

if [[ -n "$VISRAG_LOCAL_MODEL_DIR" ]]; then
  if [[ ! -s "$VISRAG_LOCAL_MODEL_DIR/config.json" ]] || \
     [[ ! -s "$VISRAG_LOCAL_MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "VISRAG_LOCAL_MODEL_DIR is incomplete: $VISRAG_LOCAL_MODEL_DIR" >&2
    exit 2
  fi
  VISRAG_MODEL="$VISRAG_LOCAL_MODEL_DIR"
fi

available_kb=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Insufficient free disk space. Required ${MIN_FREE_GB} GiB below $PROJECT_ROOT." >&2
  df -h "$PROJECT_ROOT"
  exit 2
fi

META_DIR="results/submission/_meta"
mkdir -p "$META_DIR" "logs/submission"

printf '\n[preflight] repository, matrices, and environments\n'
git status --short
git rev-parse HEAD
conda run --no-capture-output -n adacolrag-core python -c 'import adacolrag, numpy, yaml; print("core environment: OK")'
conda run --no-capture-output -n adacolrag-colpali python -c 'import torch, colpali_engine; print("colpali environment: OK", torch.cuda.is_available())'
conda run --no-capture-output -n adacolrag-visrag python -c 'import torch, transformers, sentencepiece, timm, decord; print("visrag environment: OK", torch.cuda.is_available())'
conda run --no-capture-output -n adacolrag-core python - <<'PY'
import yaml
for path in ("configs/development_matrix.yaml", "configs/submission_matrix.yaml"):
    data = yaml.safe_load(open(path, encoding="utf-8"))
    print(path, "experiments=", len(data["experiments"]), "comparisons=", len(data.get("comparisons", [])))
PY
conda run --no-capture-output -n adacolrag-core python -u scripts/check_hf_cache_locks.py \
  --cache-dir "$HF_HUB_CACHE" --clean-stale

printf '\n[model snapshot] Prepare merged ColPali locally; Python loading remains offline\n'
if [[ "$DOWNLOAD_COLVISION_MODEL" == "1" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT" \
  MODEL_REPO="$COLVISION_REMOTE_MODEL" \
  MODEL_REVISION="$COLVISION_MODEL_REVISION" \
  TARGET_DIR="$COLVISION_LOCAL_MODEL_DIR" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  COLPALI_DOWNLOAD_ENDPOINTS="$COLVISION_DOWNLOAD_ENDPOINTS" \
  bash scripts/download_colpali_merged_model_curl.sh \
    2>&1 | tee logs/submission/00_colpali_model.log
fi

for required_file in \
  config.json \
  model.safetensors.index.json \
  model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors \
  preprocessor_config.json \
  tokenizer.json \
  tokenizer_config.json \
  snapshot_info.json
do
  if [[ ! -s "$COLVISION_LOCAL_MODEL_DIR/$required_file" ]]; then
    echo "Local ColPali snapshot is incomplete: $COLVISION_LOCAL_MODEL_DIR/$required_file" >&2
    exit 2
  fi
done
COLVISION_MODEL="$COLVISION_LOCAL_MODEL_DIR"
echo "COLVISION_MODEL=$COLVISION_MODEL"

conda run --no-capture-output -n adacolrag-core python -u scripts/capture_environment.py \
  --output "$META_DIR/environment_core.json"
conda run --no-capture-output -n adacolrag-colpali python -u scripts/capture_environment.py \
  --output "$META_DIR/environment_colpali.json"
conda run --no-capture-output -n adacolrag-visrag python -u scripts/capture_environment.py \
  --output "$META_DIR/environment_visrag.json"

IFS=';' read -r -a DATASET_ITEMS <<< "$SUBMISSION_DATASETS"
for DATASET_SPEC in "${DATASET_ITEMS[@]}"; do
  IFS='|' read -r DATASET_NAME DATASET_SLUG DATASET_LANGUAGE DATASET_REVISION <<< "$DATASET_SPEC"
  if [[ -z "$DATASET_NAME" || -z "$DATASET_SLUG" || -z "$DATASET_LANGUAGE" || -z "$DATASET_REVISION" ]]; then
    echo "Invalid dataset spec: $DATASET_SPEC" >&2
    exit 2
  fi

  RAW_DATA_DIR="data/raw/$DATASET_SLUG"
  DATASET_DIR="data/$DATASET_SLUG"
  COLVISION_DIR="artifacts/$DATASET_SLUG/$COLVISION_SLUG"
  DENSE_DIR="artifacts/$DATASET_SLUG/visrag_ret"
  RESULT_ROOT="results/submission/$DATASET_SLUG"
  RUN_DIR="$RESULT_ROOT/runs"
  ANALYSIS_DIR="$RESULT_ROOT/analysis"
  DEVELOPMENT_ROOT="$RESULT_ROOT/development"
  LOG_DIR="logs/submission/$DATASET_SLUG"
  mkdir -p "$RAW_DATA_DIR" "$DATASET_DIR" "$LOG_DIR"

  if [[ "$FRESH_RESULTS" == "1" ]]; then
    rm -rf "$RESULT_ROOT"
  fi
  mkdir -p "$RUN_DIR" "$ANALYSIS_DIR"

  if [[ "$REBUILD_EMBEDDINGS" == "1" ]]; then
    rm -rf "$COLVISION_DIR" "$DENSE_DIR"
  fi
  mkdir -p "$COLVISION_DIR" "$DENSE_DIR"

  printf '\n============================================================\n'
  printf 'Dataset: %s (%s, language=%s, revision=%s)\n' \
    "$DATASET_NAME" "$DATASET_SLUG" "$DATASET_LANGUAGE" "$DATASET_REVISION"
  printf '============================================================\n'

  printf '\n[1/9] Download or reuse pinned dataset\n'
  conda run --no-capture-output -n adacolrag-core python -u scripts/download_submission_dataset.py \
    --dataset "$DATASET_NAME" \
    --revision "$DATASET_REVISION" \
    --split test \
    --language "$DATASET_LANGUAGE" \
    --raw-dir "$RAW_DATA_DIR" \
    --output-dir "$DATASET_DIR" \
    2>&1 | tee "$LOG_DIR/01_dataset.log"

  printf '\n[2/9] Export or reuse verified merged ColPali embeddings\n'
  conda run --no-capture-output -n adacolrag-colpali python -u scripts/export_colvision.py \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$COLVISION_DIR" \
    --model "$COLVISION_MODEL" \
    --batch-size "$COLVISION_BATCH_SIZE" \
    --dtype "$DTYPE" \
    --device "$GPU_DEVICE" \
    2>&1 | tee "$LOG_DIR/02_colvision.log"

  printf '\n[3/9] Export or reuse VisRAG dense embeddings\n'
  conda run --no-capture-output -n adacolrag-visrag python -u scripts/export_visrag.py \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$DENSE_DIR" \
    --model "$VISRAG_MODEL" \
    --batch-size "$VISRAG_BATCH_SIZE" \
    --dtype "$DTYPE" \
    --device "$GPU_DEVICE" \
    2>&1 | tee "$LOG_DIR/03_visrag.log"

  if [[ "$RUN_DEVELOPMENT" == "1" && "$DATASET_SLUG" == "$DEVELOPMENT_DATASET_SLUG" ]]; then
    printf '\n[4/9] Run Finance EN development sensitivity matrix\n'
    mkdir -p "$DEVELOPMENT_ROOT/runs" "$DEVELOPMENT_ROOT/analysis"
    conda run --no-capture-output -n adacolrag-core python -u scripts/run_matrix.py \
      --matrix "$DEVELOPMENT_MATRIX" \
      --dataset-dir "$DATASET_DIR" \
      --colvision-dir "$COLVISION_DIR" \
      --dense-dir "$DENSE_DIR" \
      --output-dir "$DEVELOPMENT_ROOT/runs" \
      2>&1 | tee "$LOG_DIR/04_development_matrix.log"
    conda run --no-capture-output -n adacolrag-core python -u scripts/validate_submission_results.py \
      --matrix "$DEVELOPMENT_MATRIX" \
      --dataset-dir "$DATASET_DIR" \
      --colvision-dir "$COLVISION_DIR" \
      --dense-dir "$DENSE_DIR" \
      --results-dir "$DEVELOPMENT_ROOT/runs" \
      --output "$DEVELOPMENT_ROOT/validation.json" \
      2>&1 | tee "$LOG_DIR/04_development_validation.log"
    conda run --no-capture-output -n adacolrag-core python -u scripts/analyze_submission_results.py \
      --matrix "$DEVELOPMENT_MATRIX" \
      --dataset-dir "$DATASET_DIR" \
      --results-dir "$DEVELOPMENT_ROOT/runs" \
      --output-dir "$DEVELOPMENT_ROOT/analysis" \
      --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
      2>&1 | tee "$LOG_DIR/04_development_analysis.log"
    if [[ "$RUN_DEVELOPMENT_TIMING" == "1" ]]; then
      conda run --no-capture-output -n adacolrag-core python -u scripts/repeat_benchmarks.py \
        --matrix "$DEVELOPMENT_MATRIX" \
        --only "$DEVELOPMENT_TIMING_EXPERIMENTS" \
        --dataset-dir "$DATASET_DIR" \
        --colvision-dir "$COLVISION_DIR" \
        --dense-dir "$DENSE_DIR" \
        --repeats "$TIMING_REPEATS" \
        --warmups "$TIMING_WARMUPS" \
        --output "$DEVELOPMENT_ROOT/repeated_timing.json" \
        2>&1 | tee "$LOG_DIR/04_development_timing.log"
    fi
  else
    printf '\n[4/9] Development sensitivity matrix skipped for %s\n' "$DATASET_SLUG"
  fi

  printf '\n[5/9] Run the 13-experiment controlled main matrix\n'
  conda run --no-capture-output -n adacolrag-core python -u scripts/run_matrix.py \
    --matrix "$MAIN_MATRIX" \
    --dataset-dir "$DATASET_DIR" \
    --colvision-dir "$COLVISION_DIR" \
    --dense-dir "$DENSE_DIR" \
    --output-dir "$RUN_DIR" \
    2>&1 | tee "$LOG_DIR/05_main_matrix.log"

  printf '\n[6/9] Validate main results and checkpoint provenance\n'
  conda run --no-capture-output -n adacolrag-core python -u scripts/validate_submission_results.py \
    --matrix "$MAIN_MATRIX" \
    --dataset-dir "$DATASET_DIR" \
    --colvision-dir "$COLVISION_DIR" \
    --dense-dir "$DENSE_DIR" \
    --results-dir "$RUN_DIR" \
    --output "$RESULT_ROOT/validation.json" \
    2>&1 | tee "$LOG_DIR/06_validation.log"

  printf '\n[7/9] Generate main tables and paired bootstrap analysis\n'
  conda run --no-capture-output -n adacolrag-core python -u scripts/analyze_submission_results.py \
    --matrix "$MAIN_MATRIX" \
    --dataset-dir "$DATASET_DIR" \
    --results-dir "$RUN_DIR" \
    --output-dir "$ANALYSIS_DIR" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    2>&1 | tee "$LOG_DIR/07_analysis.log"

  if [[ "$RUN_TIMING" == "1" ]]; then
    printf '\n[8/9] Repeat selected main timing experiments\n'
    conda run --no-capture-output -n adacolrag-core python -u scripts/repeat_benchmarks.py \
      --matrix "$MAIN_MATRIX" \
      --only "$TIMING_EXPERIMENTS" \
      --dataset-dir "$DATASET_DIR" \
      --colvision-dir "$COLVISION_DIR" \
      --dense-dir "$DENSE_DIR" \
      --repeats "$TIMING_REPEATS" \
      --warmups "$TIMING_WARMUPS" \
      --output "$RESULT_ROOT/repeated_timing.json" \
      2>&1 | tee "$LOG_DIR/08_timing.log"
  else
    printf '\n[8/9] Repeated timing disabled (RUN_TIMING=0)\n'
  fi

  printf '\n[9/9] Dataset result inventory\n'
  find "$RESULT_ROOT" -maxdepth 4 -type f -printf '%P\n' | sort

done

printf '\n[aggregate] Cross-dataset main tables\n'
conda run --no-capture-output -n adacolrag-core python -u scripts/aggregate_submission_results.py \
  --submission-root results/submission \
  --output-dir paper/generated \
  2>&1 | tee logs/submission/09_cross_dataset.log

printf '\nSubmission experiment workflow completed.\n'
printf 'Main results: %s/results/submission/<dataset>/runs/\n' "$PROJECT_ROOT"
printf 'Finance development results: %s/results/submission/%s/development/\n' "$PROJECT_ROOT" "$DEVELOPMENT_DATASET_SLUG"
printf 'Generated paper tables: %s/paper/generated/\n' "$PROJECT_ROOT"
