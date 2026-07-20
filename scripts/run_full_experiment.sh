#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
DATASET_NAME="${DATASET_NAME:-vidore/vidore_v3_finance_en}"
DATASET_SLUG="${DATASET_SLUG:-vidore_v3_finance_en}"
DATASET_LANGUAGE="${DATASET_LANGUAGE:-english}"
COLVISION_MODEL="${COLVISION_MODEL:-vidore/colpali-v1.3}"
COLVISION_SLUG="${COLVISION_SLUG:-colpali_v13}"
VISRAG_MODEL="${VISRAG_MODEL:-openbmb/VisRAG-Ret}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
COLVISION_BATCH_SIZE="${COLVISION_BATCH_SIZE:-2}"
VISRAG_BATCH_SIZE="${VISRAG_BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"

export PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_VERBOSITY="${HF_HUB_VERBOSITY:-info}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$PROJECT_ROOT"

DATASET_DIR="data/$DATASET_SLUG"
COLVISION_DIR="artifacts/$DATASET_SLUG/$COLVISION_SLUG"
DENSE_DIR="artifacts/$DATASET_SLUG/visrag_ret"
RESULT_DIR="results/real/$DATASET_SLUG"
LOG_DIR="logs/$DATASET_SLUG"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

printf '\n[1/6] Download and normalize dataset\n'
echo "HF_HOME=${HF_HOME:-<default>}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-0}"
echo "HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
conda run --no-capture-output -n adacolrag-core python scripts/download_vidore_v3.py \
  --dataset "$DATASET_NAME" \
  --language "$DATASET_LANGUAGE" \
  --streaming \
  --output-dir "$DATASET_DIR" \
  2>&1 | tee "$LOG_DIR/01_download_dataset.log"

printf '\n[2/6] Export ColVision multi-vector embeddings\n'
conda run --no-capture-output -n adacolrag-colpali python scripts/export_colvision.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$COLVISION_DIR" \
  --model "$COLVISION_MODEL" \
  --batch-size "$COLVISION_BATCH_SIZE" \
  --dtype "$DTYPE" \
  --device "$GPU_DEVICE" \
  2>&1 | tee "$LOG_DIR/02_export_colvision.log"

printf '\n[3/6] Export VisRAG dense embeddings\n'
conda run --no-capture-output -n adacolrag-visrag python scripts/export_visrag.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$DENSE_DIR" \
  --model "$VISRAG_MODEL" \
  --batch-size "$VISRAG_BATCH_SIZE" \
  --dtype "$DTYPE" \
  --device "$GPU_DEVICE" \
  2>&1 | tee "$LOG_DIR/03_export_visrag.log"

printf '\n[4/6] Run all baselines and proposed variants\n'
conda run --no-capture-output -n adacolrag-core python scripts/run_matrix.py \
  --dataset-dir "$DATASET_DIR" \
  --colvision-dir "$COLVISION_DIR" \
  --dense-dir "$DENSE_DIR" \
  --output-dir "$RESULT_DIR" \
  2>&1 | tee "$LOG_DIR/04_run_matrix.log"

printf '\n[5/6] Check the primary acceptance criteria\n'
set +e
conda run --no-capture-output -n adacolrag-core python scripts/compare_results.py \
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
