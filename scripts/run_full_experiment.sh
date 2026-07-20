#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
DATASET_NAME="${DATASET_NAME:-vidore/vidore_v3_finance_en}"
DATASET_SLUG="${DATASET_SLUG:-vidore_v3_finance_en}"
DATASET_LANGUAGE="${DATASET_LANGUAGE:-english}"
DATASET_REVISION="${DATASET_REVISION:-main}"
DATASET_DOWNLOAD_MODE="${DATASET_DOWNLOAD_MODE:-auto}"
COLVISION_MODEL="${COLVISION_MODEL:-vidore/colpali-v1.3}"
COLVISION_SLUG="${COLVISION_SLUG:-colpali_v13}"
VISRAG_MODEL="${VISRAG_MODEL:-openbmb/VisRAG-Ret}"
VISRAG_LOCAL_MODEL_DIR="${VISRAG_LOCAL_MODEL_DIR:-}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
COLVISION_BATCH_SIZE="${COLVISION_BATCH_SIZE:-1}"
VISRAG_BATCH_SIZE="${VISRAG_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
USE_PROJECT_HF_CACHE="${USE_PROJECT_HF_CACHE:-0}"
CLEAN_STALE_HF_LOCKS="${CLEAN_STALE_HF_LOCKS:-1}"

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_VERBOSITY="${HF_HUB_VERBOSITY:-info}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# The legacy hf_transfer backend is commonly left enabled in shell profiles.
# It fails when the optional hf_transfer package is absent and is unnecessary
# for mirror-based downloads. Export scripts clear it again before importing
# transformers/huggingface_hub.
unset HF_HUB_ENABLE_HF_TRANSFER || true

if [[ "$USE_PROJECT_HF_CACHE" == "1" ]]; then
  export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
else
  export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
fi
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

cd "$PROJECT_ROOT"

if [[ -n "$VISRAG_LOCAL_MODEL_DIR" ]]; then
  if [[ ! -s "$VISRAG_LOCAL_MODEL_DIR/config.json" ]] || \
     [[ ! -s "$VISRAG_LOCAL_MODEL_DIR/model.safetensors.index.json" ]]; then
    echo "VISRAG_LOCAL_MODEL_DIR is incomplete: $VISRAG_LOCAL_MODEL_DIR" >&2
    exit 2
  fi
  VISRAG_MODEL="$VISRAG_LOCAL_MODEL_DIR"
fi

RAW_DATA_DIR="data/raw/$DATASET_SLUG"
DATASET_DIR="data/$DATASET_SLUG"
COLVISION_DIR="artifacts/$DATASET_SLUG/$COLVISION_SLUG"
DENSE_DIR="artifacts/$DATASET_SLUG/visrag_ret"
RESULT_DIR="results/real/$DATASET_SLUG"
LOG_DIR="logs/$DATASET_SLUG"
mkdir -p "$RAW_DATA_DIR" "$DATASET_DIR" "$RESULT_DIR" "$LOG_DIR"

printf '\n[0/6] Hugging Face cache preflight\n'
echo "HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}"
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-<unset>}"
echo "VISRAG_MODEL=$VISRAG_MODEL"
LOCK_ARGS=(--cache-dir "$HF_HUB_CACHE")
if [[ "$CLEAN_STALE_HF_LOCKS" == "1" ]]; then
  LOCK_ARGS+=(--clean-stale)
fi
conda run --no-capture-output -n adacolrag-core python -u scripts/check_hf_cache_locks.py \
  "${LOCK_ARGS[@]}"

printf '\n[1/6] Prepare ViDoRe dataset\n'
echo "HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
DOWNLOAD_ARGS=(
  --dataset "$DATASET_NAME"
  --revision "$DATASET_REVISION"
  --language "$DATASET_LANGUAGE"
  --raw-dir "$RAW_DATA_DIR"
  --output-dir "$DATASET_DIR"
)
if [[ "$DATASET_DOWNLOAD_MODE" == "convert-only" ]]; then
  DOWNLOAD_ARGS+=(--convert-only)
elif [[ "$DATASET_DOWNLOAD_MODE" == "download-only" ]]; then
  DOWNLOAD_ARGS+=(--download-only)
elif [[ "$DATASET_DOWNLOAD_MODE" != "auto" ]]; then
  echo "Unsupported DATASET_DOWNLOAD_MODE=$DATASET_DOWNLOAD_MODE (use auto, download-only, or convert-only)" >&2
  exit 2
fi
conda run --no-capture-output -n adacolrag-core python -u scripts/download_vidore_v3.py \
  "${DOWNLOAD_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/01_download_dataset.log"

if [[ "$DATASET_DOWNLOAD_MODE" == "download-only" ]]; then
  echo "Raw parquet download completed. Re-run with DATASET_DOWNLOAD_MODE=convert-only to continue."
  exit 0
fi

printf '\n[2/6] Export ColVision multi-vector embeddings\n'
conda run --no-capture-output -n adacolrag-colpali python -u scripts/export_colvision.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$COLVISION_DIR" \
  --model "$COLVISION_MODEL" \
  --batch-size "$COLVISION_BATCH_SIZE" \
  --dtype "$DTYPE" \
  --device "$GPU_DEVICE" \
  2>&1 | tee "$LOG_DIR/02_export_colvision.log"

printf '\n[3/6] Export VisRAG dense embeddings\n'
conda run --no-capture-output -n adacolrag-visrag python -c \
  'import sentencepiece, timm, decord; print(f"[visrag-env] sentencepiece={sentencepiece.__version__} timm={timm.__version__} decord={decord.__version__}")'
conda run --no-capture-output -n adacolrag-visrag python -u scripts/export_visrag.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$DENSE_DIR" \
  --model "$VISRAG_MODEL" \
  --batch-size "$VISRAG_BATCH_SIZE" \
  --dtype "$DTYPE" \
  --device "$GPU_DEVICE" \
  2>&1 | tee "$LOG_DIR/03_export_visrag.log"

printf '\n[4/6] Run all baselines and proposed variants\n'
conda run --no-capture-output -n adacolrag-core python -u scripts/run_matrix.py \
  --dataset-dir "$DATASET_DIR" \
  --colvision-dir "$COLVISION_DIR" \
  --dense-dir "$DENSE_DIR" \
  --output-dir "$RESULT_DIR" \
  2>&1 | tee "$LOG_DIR/04_run_matrix.log"

printf '\n[5/6] Check the primary acceptance criteria\n'
set +e
conda run --no-capture-output -n adacolrag-core python -u scripts/compare_results.py \
  --baseline "$RESULT_DIR/colpali_full.json" \
  --candidate "$RESULT_DIR/adacolrag.json" \
  --max-ndcg5-drop 0.01 \
  --min-token-reduction 0.50 \
  --min-latency-reduction 0.30 \
  2>&1 | tee "$LOG_DIR/05_compare_results.log"
COMPARE_STATUS=${PIPESTATUS[0]}
set -e

printf '\n[6/6] Result inventory\n'
find "$RESULT_DIR" -maxdepth 1 -type f -name '*.json' -printf '%f\n' | sort

if [[ "$COMPARE_STATUS" -ne 0 ]]; then
  echo "The experiment completed, but AdaColRAG did not pass every configured acceptance threshold."
  exit "$COMPARE_STATUS"
fi

echo "The experiment completed and passed all configured acceptance thresholds."
