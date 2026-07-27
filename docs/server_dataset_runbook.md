# Server dataset and experiment runbook

This runbook assumes the repository is checked out at `~/GMY/AdaColRAG`.

## 1. Pull the latest research branch

```bash
cd ~/GMY/AdaColRAG
git fetch origin
git switch agent/adacolrag-wsdm-framework
git pull --ff-only origin agent/adacolrag-wsdm-framework
```

## 2. Verify Hugging Face connectivity

```bash
curl -I --max-time 20 https://huggingface.co
curl -I --max-time 20 'https://huggingface.co/datasets/vidore/vidore_v3_finance_en/resolve/main/queries/test-00000-of-00001.parquet?download=true'
```

When direct Hugging Face access is unavailable, either configure the server proxy or temporarily use a compatible mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 3. Recommended two-stage dataset preparation

Set stable download options:

```bash
cd ~/GMY/AdaColRAG
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60
export PYTHONUNBUFFERED=1
mkdir -p data/raw/vidore_v3_finance_en data/vidore_v3_finance_en
```

Download the five raw parquet files with resume support:

```bash
conda run --no-capture-output -n adacolrag-core python -u scripts/download_vidore_v3.py \
  --dataset vidore/vidore_v3_finance_en \
  --revision main \
  --split test \
  --language english \
  --raw-dir data/raw/vidore_v3_finance_en \
  --output-dir data/vidore_v3_finance_en \
  --download-only
```

Confirm the raw files:

```bash
find data/raw/vidore_v3_finance_en -type f -name '*.parquet' -printf '%p %k KB\n' | sort
```

Convert local parquet into page PNG files and AdaColRAG manifests:

```bash
conda run --no-capture-output -n adacolrag-core python -u scripts/download_vidore_v3.py \
  --dataset vidore/vidore_v3_finance_en \
  --revision main \
  --split test \
  --language english \
  --raw-dir data/raw/vidore_v3_finance_en \
  --output-dir data/vidore_v3_finance_en \
  --convert-only
```

Expected normalized files:

```text
data/vidore_v3_finance_en/
├── images/
├── corpus.jsonl
├── queries.jsonl
├── qrels.jsonl
└── dataset_info.json
```

## 4. Wget fallback

If the Python/Hugging Face downloader still cannot transfer files, use the direct resumable fallback:

```bash
cd ~/GMY/AdaColRAG
chmod +x scripts/download_vidore_finance_en_wget.sh
PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
HF_BASE_URL="https://huggingface.co" \
bash scripts/download_vidore_finance_en_wget.sh
```

Mirror fallback:

```bash
PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
HF_BASE_URL="https://hf-mirror.com" \
bash scripts/download_vidore_finance_en_wget.sh
```

The script downloads three corpus shards, one query shard, and one qrels shard, then converts them locally.

## 5. Run the full experiment after the dataset is ready

```bash
cd ~/GMY/AdaColRAG
export CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
DATASET_NAME="vidore/vidore_v3_finance_en" \
DATASET_SLUG="vidore_v3_finance_en" \
DATASET_LANGUAGE="english" \
DATASET_REVISION="main" \
DATASET_DOWNLOAD_MODE="convert-only" \
COLVISION_MODEL="vidore/colpali-v1.3" \
COLVISION_SLUG="colpali_v13" \
VISRAG_MODEL="openbmb/VisRAG-Ret" \
GPU_DEVICE="cuda:0" \
COLVISION_BATCH_SIZE="1" \
VISRAG_BATCH_SIZE="1" \
DTYPE="bfloat16" \
bash scripts/run_full_experiment.sh
```

Use `DATASET_DOWNLOAD_MODE=convert-only` after raw parquet files have been downloaded. Use `auto` when the script should both download and convert.

## 6. Inspect progress and results

```bash
tail -f logs/vidore_v3_finance_en/01_download_dataset.log
tail -f logs/vidore_v3_finance_en/02_export_colvision.log
tail -f logs/vidore_v3_finance_en/03_export_visrag.log
tail -f logs/vidore_v3_finance_en/04_run_matrix.log
cat logs/vidore_v3_finance_en/05_compare_results.log
```

```bash
find results/real/vidore_v3_finance_en -maxdepth 1 -type f -name '*.json' -printf '%f\n' | sort
```
