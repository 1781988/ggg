# Server Runbook: Full AdaColRAG Submission Rerun

## 1. Pull the updated branch

```bash
cd ~/GMY/AdaColRAG
git fetch origin
git switch agent/adacolrag-wsdm-framework
git pull --ff-only origin agent/adacolrag-wsdm-framework
```

## 2. Update environments

```bash
conda env update -f environment/core.yml --prune
conda env update -f environment/colpali.yml --prune
conda env update -f environment/visrag.yml --prune

conda run -n adacolrag-core python -m pip install -e .
conda run -n adacolrag-colpali python -m pip install -e .
conda run -n adacolrag-visrag python -m pip install -e .
```

## 3. Verify before the expensive run

```bash
conda run -n adacolrag-core pytest -q
python -m compileall -q src scripts tests
bash -n scripts/run_submission_experiments.sh
bash -n scripts/run_all_required_experiments.sh
```

## 4. Quick Finance EN rerun

Use this to validate the merged checkpoint and all 17 systems before downloading every domain.

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
VISRAG_LOCAL_MODEL_DIR="$HOME/GMY/AdaColRAG/models/VisRAG-Ret" \
COLVISION_BATCH_SIZE=1 \
VISRAG_BATCH_SIZE=1 \
REBUILD_EMBEDDINGS=1 \
FRESH_RESULTS=1 \
bash scripts/run_all_required_experiments.sh
```

Expected critical checks:

- ColPali log reports `merged_checkpoint: true` and `verified: true`;
- 17 JSON files appear in `results/submission/vidore_v3_finance_en/runs/`;
- `validation.json` contains `"valid": true`;
- generated result and significance tables appear under the dataset `analysis/` directory;
- the generated blocks in `paper/draft.md` are replaced.

## 5. Full four-dataset rerun

Run inside `tmux` and reserve substantial disk space.

```bash
tmux new -s adacolrag-submission

cd ~/GMY/AdaColRAG

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/user/models/.hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
VISRAG_LOCAL_MODEL_DIR="$HOME/GMY/AdaColRAG/models/VisRAG-Ret" \
COLVISION_BATCH_SIZE=1 \
VISRAG_BATCH_SIZE=1 \
TIMING_REPEATS=3 \
TIMING_WARMUPS=1 \
BOOTSTRAP_SAMPLES=10000 \
REBUILD_EMBEDDINGS=1 \
FRESH_RESULTS=1 \
MIN_FREE_GB=80 \
bash scripts/run_all_required_experiments.sh
```

Detach from tmux with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t adacolrag-submission
```

## 6. Important output paths

```text
results/submission/_meta/
results/submission/<dataset>/runs/
results/submission/<dataset>/analysis/
results/submission/<dataset>/validation.json
results/submission/<dataset>/repeated_timing.json
logs/submission/<dataset>/
paper/generated/
paper/draft.md
```

## 7. Resume policy

The full submission script is designed as a fresh rerun by default. To reuse verified embeddings and regenerate only experiment results:

```bash
REBUILD_EMBEDDINGS=0 FRESH_RESULTS=1 bash scripts/run_all_required_experiments.sh
```

To preserve current result directories as well, set `FRESH_RESULTS=0`; however, use this only when the matrix and checkpoint manifests are unchanged.
