# Server Runbook: Final AdaColRAG Experiments

## 1. Pull the updated branch

```bash
cd ~/GMY/AdaColRAG

git status -sb
git fetch origin
git switch agent/adacolrag-wsdm-framework
git pull --ff-only origin agent/adacolrag-wsdm-framework

git log -1 --oneline
```

Do not restore older local copies of the experiment scripts over the updated versions.

## 2. Update environments and editable installs

```bash
cd ~/GMY/AdaColRAG

conda env update -f environment/core.yml --prune
conda env update -f environment/colpali.yml --prune
conda env update -f environment/visrag.yml --prune

conda run -n adacolrag-core python -m pip install -e .
conda run -n adacolrag-colpali python -m pip install -e .
conda run -n adacolrag-visrag python -m pip install -e .
```

## 3. Verify code before GPU work

```bash
cd ~/GMY/AdaColRAG

conda run -n adacolrag-core pytest -q
conda run -n adacolrag-core python -m compileall -q src scripts tests
bash -n scripts/run_submission_experiments.sh
bash -n scripts/run_all_required_experiments.sh

conda run --no-capture-output -n adacolrag-core python - <<'PY'
import yaml
for path in ("configs/development_matrix.yaml", "configs/submission_matrix.yaml"):
    data = yaml.safe_load(open(path, encoding="utf-8"))
    print(path, "experiments:", len(data["experiments"]), "comparisons:", len(data.get("comparisons", [])))
PY
```

Expected matrix counts:

```text
configs/development_matrix.yaml experiments: 13
configs/submission_matrix.yaml experiments: 13
```

## 4. Confirm local model snapshots

```bash
test -s models/colpali-v1.3-merged/snapshot_info.json && echo "ColPali snapshot: OK"
test -s models/VisRAG-Ret/config.json && echo "VisRAG snapshot: OK"

python -m json.tool models/colpali-v1.3-merged/snapshot_info.json
```

The ColPali snapshot must report `verified: true`.

## 5. Finance EN framework validation

This command reuses the already verified Finance EN embeddings, runs the new 13-point development matrix and 13-system main matrix, validates both result bundles, performs bootstrap analysis, repeats selected timing, aggregates tables, and updates the manuscript.

```bash
cd ~/GMY/AdaColRAG
chmod +x scripts/run_submission_experiments.sh scripts/run_all_required_experiments.sh

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/user/models/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
QUICK_MODE=1 \
DOWNLOAD_COLVISION_MODEL=0 \
COLVISION_LOCAL_MODEL_DIR="$HOME/GMY/AdaColRAG/models/colpali-v1.3-merged" \
VISRAG_LOCAL_MODEL_DIR="$HOME/GMY/AdaColRAG/models/VisRAG-Ret" \
COLVISION_BATCH_SIZE=1 \
VISRAG_BATCH_SIZE=1 \
REBUILD_EMBEDDINGS=0 \
FRESH_RESULTS=1 \
TIMING_REPEATS=3 \
TIMING_WARMUPS=1 \
BOOTSTRAP_SAMPLES=10000 \
bash scripts/run_all_required_experiments.sh
```

Expected Finance EN outputs:

```text
results/submission/vidore_v3_finance_en/development/runs/       13 JSON files
results/submission/vidore_v3_finance_en/runs/                   13 JSON files
results/submission/vidore_v3_finance_en/development/validation.json
results/submission/vidore_v3_finance_en/validation.json
results/submission/vidore_v3_finance_en/development/analysis/
results/submission/vidore_v3_finance_en/analysis/
results/submission/vidore_v3_finance_en/development/repeated_timing.json
results/submission/vidore_v3_finance_en/repeated_timing.json
```

Both validation files must contain `"valid": true`.

## 6. Full four-dataset one-click run

Run inside `tmux`. The command reuses Finance EN embeddings, exports only missing embeddings for Industrial, Pharmaceuticals, and Finance FR, runs the frozen main matrix on all four datasets, and updates `paper/draft.md`.

```bash
tmux new -s adacolrag-final

cd ~/GMY/AdaColRAG

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/user/models/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
DOWNLOAD_COLVISION_MODEL=0 \
COLVISION_LOCAL_MODEL_DIR="$HOME/GMY/AdaColRAG/models/colpali-v1.3-merged" \
VISRAG_LOCAL_MODEL_DIR="$HOME/GMY/AdaColRAG/models/VisRAG-Ret" \
COLVISION_BATCH_SIZE=1 \
VISRAG_BATCH_SIZE=1 \
GPU_DEVICE="cuda:0" \
DTYPE="bfloat16" \
REBUILD_EMBEDDINGS=0 \
FRESH_RESULTS=1 \
RUN_DEVELOPMENT=1 \
RUN_TIMING=1 \
RUN_DEVELOPMENT_TIMING=1 \
TIMING_REPEATS=3 \
TIMING_WARMUPS=1 \
BOOTSTRAP_SAMPLES=10000 \
MIN_FREE_GB=80 \
bash scripts/run_all_required_experiments.sh
```

Detach with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t adacolrag-final
```

## 7. Monitor progress

```bash
tail -f logs/submission/vidore_v3_finance_en/04_development_matrix.log
tail -f logs/submission/vidore_v3_finance_en/05_main_matrix.log
tail -f logs/submission/vidore_v3_industrial/02_colvision.log
tail -f logs/submission/vidore_v3_industrial/03_visrag.log
tail -f logs/submission/vidore_v3_industrial/05_main_matrix.log
```

GPU monitoring:

```bash
watch -n 2 nvidia-smi
```

## 8. Important output paths

```text
results/submission/_meta/
results/submission/<dataset>/runs/
results/submission/<dataset>/analysis/
results/submission/<dataset>/validation.json
results/submission/<dataset>/repeated_timing.json
results/submission/vidore_v3_finance_en/development/
logs/submission/<dataset>/
paper/generated/cross_dataset_results.md
paper/generated/cross_dataset_deltas.md
paper/generated/cross_dataset_results.csv
paper/generated/cross_dataset_results.json
paper/draft.md
```

## 9. Resume and reuse policy

- `REBUILD_EMBEDDINGS=0`: reuse complete embeddings and create only missing ones.
- `FRESH_RESULTS=1`: delete old result JSONs and rerun the current matrices.
- `FRESH_RESULTS=0`: preserve result directories; use only when configs and code are unchanged.
- Do not delete verified local model snapshots or Finance EN embeddings.

The primary method no longer uses adaptive budgeting. Old adaptive result files are historical artifacts and are not consumed by the final matrices.
