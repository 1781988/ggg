#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
MODEL_REPO="${MODEL_REPO:-vidore/colpali-v1.3-merged}"
# Pin the snapshot that added the tokenizer files while retaining the verified merged weights.
MODEL_REVISION="${MODEL_REVISION:-5b955e3415a7c5468ab33119d98d6d45c3a5b2c3}"
TARGET_DIR="${TARGET_DIR:-$PROJECT_ROOT/models/colpali-v1.3-merged}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
DOWNLOAD_ENDPOINTS="${COLPALI_DOWNLOAD_ENDPOINTS:-https://hf-mirror.com https://huggingface.co}"

mkdir -p "$TARGET_DIR"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 2
fi

available_kb=$(df -Pk "$TARGET_DIR" | awk 'NR==2 {print $4}')
required_kb=$((8 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "At least 8 GiB of free disk space is required in $TARGET_DIR" >&2
  df -h "$TARGET_DIR"
  exit 2
fi

small_files=(
  config.json
  model.safetensors.index.json
  preprocessor_config.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
)

large_files=(
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
)

# These are the actual serialized safetensors file sizes, including file headers.
# model.safetensors.index.json metadata.total_size is tensor payload size and must
# not be used as an individual shard file-size check.
declare -A expected_size=(
  [model-00001-of-00002.safetensors]=4986817288
  [model-00002-of-00002.safetensors]=862495528
)

declare -A expected_sha=(
  [model-00001-of-00002.safetensors]=ea254a039c48511ab7a7154a2c5567a3cd2acd78296632302a10291c2e3f2e3c
  [model-00002-of-00002.safetensors]=aa4a0c14309cbd2f1962386b484fd152f0f133210b9c1b4f5f919b18ebaa8111
)

url_for() {
  local endpoint="$1"
  local file="$2"
  printf '%s/%s/resolve/%s/%s?download=true' \
    "${endpoint%/}" "$MODEL_REPO" "$MODEL_REVISION" "$file"
}

run_curl() {
  local profile="$1"
  local url="$2"
  local output="$3"
  local -a common=(
    --fail
    --location
    --continue-at -
    --retry 12
    --retry-all-errors
    --retry-delay 4
    --connect-timeout 30
    --speed-time 120
    --speed-limit 1024
    --output "$output"
  )

  case "$profile" in
    direct)
      env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
        curl "${common[@]}" "$url"
      ;;
    http)
      local proxy="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
      [[ -n "$proxy" ]] || return 97
      env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
        curl --proxy "$proxy" "${common[@]}" "$url"
      ;;
    socks)
      local proxy="${ALL_PROXY:-${all_proxy:-}}"
      [[ -n "$proxy" ]] || return 98
      env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
        curl --proxy "$proxy" "${common[@]}" "$url"
      ;;
    *)
      return 99
      ;;
  esac
}

download_with_fallback() {
  local file="$1"
  local output="$2"
  local endpoint profile url

  touch "$output"
  for endpoint in $DOWNLOAD_ENDPOINTS; do
    url=$(url_for "$endpoint" "$file")
    for profile in direct http socks; do
      echo "[download] file=$file endpoint=$endpoint transport=$profile resume_at=$(stat -c %s "$output")"
      if run_curl "$profile" "$url" "$output"; then
        echo "[download] completed file=$file endpoint=$endpoint transport=$profile"
        return 0
      fi
      echo "[download] attempt failed; switching transport or endpoint" >&2
    done
  done

  echo "All download transports failed for $file" >&2
  return 1
}

seed_from_hf_cache() {
  local file="$1"
  local sha="${expected_sha[$file]}"
  local target="$TARGET_DIR/$file"
  local part="$target.part"
  local complete_blob="$HF_HUB_CACHE/models--vidore--colpali-v1.3-merged/blobs/$sha"
  local incomplete_blob="$complete_blob.incomplete"

  if [[ -s "$target" || -s "$part" ]]; then
    return
  fi
  if [[ -s "$complete_blob" ]]; then
    echo "[seed] copying complete Hugging Face cache blob for $file"
    cp --reflink=auto "$complete_blob" "$target"
  elif [[ -s "$incomplete_blob" ]]; then
    echo "[seed] reusing partial Hugging Face cache blob for $file"
    cp --reflink=auto "$incomplete_blob" "$part"
  fi
}

download_small() {
  local file="$1"
  local target="$TARGET_DIR/$file"
  local part="$target.part"
  if [[ -s "$target" ]]; then
    echo "[reuse] $file"
    return
  fi
  download_with_fallback "$file" "$part"
  test -s "$part"
  mv "$part" "$target"
}

verify_sha() {
  local file="$1"
  local path="$2"
  local sha="${expected_sha[$file]}"
  echo "$sha  $path" | sha256sum --check --status
}

download_large() {
  local file="$1"
  local target="$TARGET_DIR/$file"
  local part="$target.part"
  local size="${expected_size[$file]}"

  seed_from_hf_cache "$file"

  if [[ -s "$target" ]] && verify_sha "$file" "$target"; then
    local actual_size
    actual_size=$(stat -c %s "$target")
    if [[ "$actual_size" != "$size" ]]; then
      echo "[warning] $file has verified SHA256 but size $actual_size differs from expected $size" >&2
    fi
    echo "[reuse] $file passed SHA256 verification"
    return
  fi

  if [[ -e "$target" ]]; then
    mv "$target" "$target.invalid.$(date +%s)"
  fi

  if [[ -s "$part" ]] && verify_sha "$file" "$part"; then
    echo "[recover] existing .part file already has the official SHA256"
  else
    download_with_fallback "$file" "$part"
  fi

  local actual_size
  actual_size=$(stat -c %s "$part")
  if [[ "$actual_size" != "$size" ]]; then
    echo "[warning] downloaded size for $file is $actual_size; expected serialized size is $size" >&2
  fi

  if ! verify_sha "$file" "$part"; then
    echo "SHA256 verification failed for $file" >&2
    echo "The file is retained at $part for diagnosis or resumable retry." >&2
    exit 4
  fi

  mv "$part" "$target"
  echo "[verified] $file size=$(stat -c %s "$target")"
}

printf 'Preparing local snapshot %s at %s\n' "$MODEL_REPO" "$TARGET_DIR"
printf 'Pinned revision: %s\n' "$MODEL_REVISION"
printf 'Endpoint order: %s\n' "$DOWNLOAD_ENDPOINTS"
printf 'Existing HF cache: %s\n' "$HF_HUB_CACHE"

for file in "${small_files[@]}"; do
  download_small "$file"
done
for file in "${large_files[@]}"; do
  download_large "$file"
done

python - "$TARGET_DIR" "$MODEL_REPO" "$MODEL_REVISION" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
repo = sys.argv[2]
revision = sys.argv[3]
required = [
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
missing = [name for name in required if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"Missing model files: {missing}")
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
if shards != ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    raise SystemExit(f"Unexpected shard inventory: {shards}")
metadata = {
    "repo_id": repo,
    "revision": revision,
    "model_dir": str(root.resolve()),
    "architectures": config.get("architectures"),
    "tensor_payload_bytes": index.get("metadata", {}).get("total_size"),
    "serialized_shard_bytes": sum((root / shard).stat().st_size for shard in shards),
    "shards": shards,
    "verified": True,
}
(root / "snapshot_info.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY

echo "ColPali merged local snapshot is ready: $TARGET_DIR"
