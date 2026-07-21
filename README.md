# AdaColRAG

**Fast query-aware visual-token selection with confidence-aware recovery for efficient visual document retrieval.**

AdaColRAG combines frozen VisRAG dense candidate generation with selected-token ColPali late interaction. The primary method uses a fixed development-selected token budget, relevance-prefiltered MMR, dense/late-interaction fusion, and confidence-triggered full-token recovery.

The earlier query-complexity budget controller is retained only for archived reproducibility. Verified Finance EN experiments found no significant advantage over a matched fixed budget, so adaptive budgeting is not part of the final method or main matrix.

## Final experiment structure

The repository contains two distinct matrices:

- `configs/development_matrix.yaml`: 13 Finance EN sensitivity experiments;
- `configs/submission_matrix.yaml`: 13 controlled main systems for four datasets.

Finance EN is the development collection. Industrial, Pharmaceuticals, and Finance FR are held-out evaluations.

### Effective modules under test

1. VisRAG Top-50 candidate generation;
2. fixed 112-token relevance baseline;
3. fast MMR redundancy suppression;
4. approximate layout coverage;
5. dense/late-interaction score fusion;
6. confidence-triggered Top-100 full-token recovery;
7. optional OCR lexical fusion.

### Controlled main systems

```text
visrag_dense
colpali_full
colpali_mean_top50_full
visrag_top50_full
colpali_mean_top50_fixed_112
visrag_top50_fixed_112
visrag_top50_mmr_redundancy
visrag_top50_mmr_layout
visrag_top50_mmr
visrag_top50_mmr_fallback
adacolrag_no_fallback
adacolrag
adacolrag_ocr
```

All selector ablations use identical VisRAG Top-50 candidates and a fixed 112-token budget.

## Repository layout

```text
configs/development_matrix.yaml         Finance EN calibration matrix
configs/submission_matrix.yaml          13-system held-out main matrix
configs/adacolrag.yaml                  primary fixed-budget fast-MMR method
environment/                             isolated core/ColPali/VisRAG environments
scripts/download_submission_dataset.py deterministic pinned dataset downloader
scripts/export_colvision.py             verified merged ColPali export
scripts/export_visrag.py                VisRAG dense export
scripts/run_matrix.py                   progress-aware matrix runner
scripts/validate_submission_results.py integrity and provenance checks
scripts/analyze_submission_results.py  tables, Pareto flags, paired bootstrap
scripts/repeat_benchmarks.py            repeated retrieval-stage timing
scripts/aggregate_submission_results.py cross-dataset tables and deltas
scripts/update_paper_results.py         inject generated tables into paper
scripts/run_submission_experiments.sh   complete experiment workflow
scripts/run_all_required_experiments.sh one-click workflow plus paper update
paper/EXPERIMENT_PROTOCOL.md            scientific experiment protocol
paper/draft.md                          WSDM-oriented manuscript source
docs/SUBMISSION_RUNBOOK.md              server commands and output paths
src/adacolrag/                           retrieval framework
```

## Installation

```bash
cd ~/GMY/AdaColRAG

conda env update -f environment/core.yml --prune
conda env update -f environment/colpali.yml --prune
conda env update -f environment/visrag.yml --prune

conda run -n adacolrag-core python -m pip install -e .
conda run -n adacolrag-colpali python -m pip install -e .
conda run -n adacolrag-visrag python -m pip install -e .

conda run -n adacolrag-core pytest -q
```

Submission runs require:

```text
models/colpali-v1.3-merged/   verified local merged checkpoint
models/VisRAG-Ret/            complete local VisRAG snapshot
```

## One-click four-dataset run

Use `tmux` and reuse existing verified Finance EN embeddings:

```bash
cd ~/GMY/AdaColRAG
chmod +x scripts/run_submission_experiments.sh scripts/run_all_required_experiments.sh

tmux new -s adacolrag-final

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
REBUILD_EMBEDDINGS=0 \
FRESH_RESULTS=1 \
RUN_DEVELOPMENT=1 \
RUN_TIMING=1 \
TIMING_REPEATS=3 \
TIMING_WARMUPS=1 \
BOOTSTRAP_SAMPLES=10000 \
MIN_FREE_GB=80 \
bash scripts/run_all_required_experiments.sh
```

The script performs:

1. repository/environment/model preflight;
2. pinned dataset download and conversion;
3. embedding reuse or export;
4. Finance EN development sensitivity;
5. 13-system main matrix on all four datasets;
6. result/checkpoint validation;
7. paired query-level bootstrap analysis;
8. repeated timing;
9. cross-dataset aggregation;
10. automatic `paper/draft.md` table update.

## Finance EN-only framework check

```bash
PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
QUICK_MODE=1 \
DOWNLOAD_COLVISION_MODEL=0 \
COLVISION_LOCAL_MODEL_DIR="$HOME/GMY/AdaColRAG/models/colpali-v1.3-merged" \
VISRAG_LOCAL_MODEL_DIR="$HOME/GMY/AdaColRAG/models/VisRAG-Ret" \
REBUILD_EMBEDDINGS=0 \
FRESH_RESULTS=1 \
TIMING_REPEATS=3 \
TIMING_WARMUPS=1 \
BOOTSTRAP_SAMPLES=10000 \
bash scripts/run_all_required_experiments.sh
```

This runs both Finance EN matrices. It is a framework validation and development analysis, not the final cross-domain evaluation.

## Outputs

```text
results/submission/_meta/
results/submission/vidore_v3_finance_en/development/
results/submission/<dataset>/runs/
results/submission/<dataset>/analysis/
results/submission/<dataset>/validation.json
results/submission/<dataset>/repeated_timing.json
logs/submission/<dataset>/
paper/generated/cross_dataset_results.md
paper/generated/cross_dataset_deltas.md
paper/generated/cross_dataset_results.csv
paper/generated/cross_dataset_results.json
paper/draft.md
```

Each result JSON preserves query-level rankings, selected-token budgets, confidence, fallback decisions, latency, token work, and scoring-operation counts.

## Integrity and interpretation policy

The validator fails if results are missing, dataset checksums differ, embedding counts disagree, dense signals leak into non-dense controls, checkpoint provenance is unverified, or metrics are non-finite.

The workflow reports local token reduction and full-corpus system work separately. System work reduction is not index-storage reduction. The old 50% local-token target is not a publication gate. Claims must jointly consider effectiveness, uncertainty, token work, repeated latency, and controlled attribution.

Full operating instructions are in `docs/SUBMISSION_RUNBOOK.md`; paper requirements are in `paper/EXPERIMENT_PROTOCOL.md`.
