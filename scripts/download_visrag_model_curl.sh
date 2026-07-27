#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
HF_BASE_URL="${HF_BASE_URL:-https://hf-mirror.com}"
MODEL_REPO="${MODEL_REPO:-openbmb/VisRAG-Ret}"
MODEL_REVISION="${MODEL_REVISION:-main}"
TARGET_DIR="${TARGET_DIR:-$PROJECT_ROOT/models/VisRAG-Ret}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

mkdir -p "$TARGET_DIR"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 2
fi

available_kb=$(df -Pk "$TARGET_DIR" | awk 'NR==2 {print $4}')
required_kb=$((9 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "At least 9 GiB of free disk space is required in $TARGET_DIR" >&2
  df -h "$TARGET_DIR"
  exit 2
fi

small_files=(
  config.json
  configuration_minicpm.py
  model.safetensors.index.json
  modeling_minicpm.py
  modeling_minicpmv.py
  modeling_visrag_ret.py
  resampler.py
  special_tokens_map.json
  tokenizer.json
  tokenizer.model
  tokenizer.py
  tokenizer_config.json
)

large_files=(
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
)

declare -A expected_size=(
  [model-00001-of-00002.safetensors]=4993235928
  [model-00002-of-00002.safetensors]=1876772888
)

declare -A expected_sha=(
  [model-00001-of-00002.safetensors]=2f2f1f0b353cd125705111c0363dd45baa26a586972152b2d55858bb7e8e53a2
  [model-00002-of-00002.safetensors]=85c5084077ee3ac7d0bae9bd746f78364ade0299b4fd074e245988e5e9b407f5
)

url_for() {
  local file="$1"
  printf '%s/%s/resolve/%s/%s?download=true' \
    "${HF_BASE_URL%/}" "$MODEL_REPO" "$MODEL_REVISION" "$file"
}

download_small() {
  local file="$1"
  local target="$TARGET_DIR/$file"
  local tmp="$target.tmp"
  echo "[small] $file"
  curl \
    --fail \
    --location \
    --retry 20 \
    --retry-all-errors \
    --retry-delay 3 \
    --connect-timeout 30 \
    --speed-time 120 \
    --speed-limit 1024 \
    --output "$tmp" \
    "$(url_for "$file")"
  test -s "$tmp"
  mv "$tmp" "$target"
}

seed_from_hf_cache() {
  local file="$1"
  local sha="${expected_sha[$file]}"
  local target_part="$TARGET_DIR/$file.part"
  local complete_blob="$HF_HUB_CACHE/models--openbmb--VisRAG-Ret/blobs/$sha"
  local incomplete_blob="$complete_blob.incomplete"

  if [[ -e "$TARGET_DIR/$file" || -e "$target_part" ]]; then
    return
  fi
  if [[ -s "$complete_blob" ]]; then
    echo "[seed] copying complete cached blob for $file"
    cp --reflink=auto "$complete_blob" "$TARGET_DIR/$file"
  elif [[ -s "$incomplete_blob" ]]; then
    echo "[seed] reusing cached partial download for $file"
    cp --reflink=auto "$incomplete_blob" "$target_part"
  fi
}

download_large() {
  local file="$1"
  local target="$TARGET_DIR/$file"
  local part="$target.part"
  local size="${expected_size[$file]}"
  local sha="${expected_sha[$file]}"

  seed_from_hf_cache "$file"

  if [[ -s "$target" ]] && [[ "$(stat -c %s "$target")" == "$size" ]]; then
    echo "[reuse] $file already has expected size"
  else
    rm -f "$target"
    touch "$part"
    echo "[large] $file resume_at=$(stat -c %s "$part") expected=$size"
    curl \
      --fail \
      --location \
      --continue-at - \
      --retry 100 \
      --retry-all-errors \
      --retry-delay 5 \
      --connect-timeout 30 \
      --speed-time 120 \
      --speed-limit 1024 \
      --output "$part" \
      "$(url_for "$file")"
    actual_size=$(stat -c %s "$part")
    if [[ "$actual_size" != "$size" ]]; then
      echo "Unexpected size for $file: $actual_size != $size" >&2
      exit 3
    fi
    mv "$part" "$target"
  fi

  echo "$sha  $target" | sha256sum --check --status || {
    echo "SHA256 verification failed for $file" >&2
    exit 4
  }
  echo "[verified] $file"
}

printf 'Downloading %s to %s\n' "$MODEL_REPO" "$TARGET_DIR"
printf 'Endpoint: %s\n' "$HF_BASE_URL"
printf 'Existing HF cache: %s\n' "$HF_HUB_CACHE"

for file in "${small_files[@]}"; do
  if [[ -s "$TARGET_DIR/$file" ]]; then
    echo "[reuse] $file"
  else
    download_small "$file"
  fi
done

for file in "${large_files[@]}"; do
  download_large "$file"
done

python - "$TARGET_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = [
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer_config.json",
    "tokenizer.model",
    "tokenizer.py",
    "modeling_visrag_ret.py",
]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"Missing model files: {missing}")
index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
print(json.dumps({"model_dir": str(root.resolve()), "shards": shards}, indent=2))
PY

echo "VisRAG model download completed: $TARGET_DIR"
