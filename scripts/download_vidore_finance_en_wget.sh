#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
HF_BASE_URL="${HF_BASE_URL:-https://huggingface.co}"
REVISION="${DATASET_REVISION:-main}"
RAW_DIR="${RAW_DIR:-$PROJECT_ROOT/data/raw/vidore_v3_finance_en}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/vidore_v3_finance_en}"
REPO_PATH="datasets/vidore/vidore_v3_finance_en"

cd "$PROJECT_ROOT"
mkdir -p "$RAW_DIR/corpus" "$RAW_DIR/queries" "$RAW_DIR/qrels" "$OUTPUT_DIR"

FILES=(
  "corpus/test-00000-of-00003.parquet"
  "corpus/test-00001-of-00003.parquet"
  "corpus/test-00002-of-00003.parquet"
  "queries/test-00000-of-00001.parquet"
  "qrels/test-00000-of-00001.parquet"
)

for relative in "${FILES[@]}"; do
  destination="$RAW_DIR/$relative"
  partial="$destination.part"
  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]]; then
    echo "[reuse] $destination"
    continue
  fi
  url="$HF_BASE_URL/$REPO_PATH/resolve/$REVISION/$relative?download=true"
  echo "[wget] $relative"
  wget \
    --continue \
    --tries=0 \
    --timeout=60 \
    --read-timeout=60 \
    --retry-connrefused \
    --show-progress \
    --output-document="$partial" \
    "$url"
  mv "$partial" "$destination"
done

echo "[convert] raw parquet files are complete; converting to AdaColRAG format"
conda run --no-capture-output -n adacolrag-core python -u scripts/download_vidore_v3.py \
  --dataset vidore/vidore_v3_finance_en \
  --revision "$REVISION" \
  --split test \
  --language english \
  --raw-dir "$RAW_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --convert-only

echo "[done] dataset is available at $OUTPUT_DIR"
