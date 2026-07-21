# AdaColRAG

**Query-adaptive visual-token selection with confidence-aware recovery for efficient visual document retrieval.**

AdaColRAG separates frozen-model embedding export from CPU retrieval experiments. ColPali supplies page/query multi-vectors, VisRAG-Ret supplies dense candidate vectors, and the core framework evaluates candidate generation, token budgeting, relevance--diversity--coverage selection, score fusion, and confidence-triggered full-token recovery.

## Current submission framework

The repository contains two experiment levels:

- `configs/matrix.yaml`: compact development matrix;
- `configs/submission_matrix.yaml`: 17-system controlled submission matrix.

The submission workflow addresses the main validity gaps found in the first Finance EN pilot:

1. it defaults to the official merged checkpoint `vidore/colpali-v1.3-merged`;
2. ColPali export records and enforces checkpoint verification;
3. identical VisRAG Top-50 candidate controls isolate candidate pruning, fixed budgets, adaptive budgets, MMR, fallback, and fusion;
4. four ViDoRe V3 datasets are run by default: Finance EN, Industrial, Pharmaceuticals, and Finance FR;
5. query-level paired bootstrap intervals are generated;
6. hardware, package versions, Git commit, BLAS threads, and repeated timing are recorded;
7. result integrity is checked before manuscript tables are generated;
8. `paper/draft.md` is automatically updated from generated Markdown tables.

The complete server procedure is in `docs/SUBMISSION_RUNBOOK.md`; the paper-specific requirements are in `paper/EXPERIMENT_PROTOCOL.md`.

## Repository layout

```text
configs/submission_matrix.yaml          17 controlled systems
configs/visrag_top50_*.yaml             identical-candidate controls
environment/                            isolated core/ColPali/VisRAG environments
scripts/export_colvision.py             verified merged-checkpoint multi-vector export
scripts/export_visrag.py                dense VisRAG export
scripts/run_matrix.py                    progress-aware matrix runner
scripts/validate_submission_results.py  result/checkpoint integrity checks
scripts/analyze_submission_results.py   tables and paired bootstrap analysis
scripts/repeat_benchmarks.py             repeated retrieval timing
scripts/capture_environment.py           hardware/software metadata
scripts/aggregate_submission_results.py cross-dataset aggregation
scripts/update_paper_results.py          inject generated tables into the paper
scripts/run_submission_experiments.sh    full multi-domain workflow
scripts/run_all_required_experiments.sh  one-click workflow plus paper update
paper/draft.md                           WSDM-oriented manuscript source
paper/generated/                         generated cross-domain tables
src/adacolrag/                           retrieval framework
```

## Installation

From `~/GMY/AdaColRAG`:

```bash
conda env update -f environment/core.yml --prune
conda env update -f environment/colpali.yml --prune
conda env update -f environment/visrag.yml --prune

conda run -n adacolrag-core python -m pip install -e .
conda run -n adacolrag-colpali python -m pip install -e .
conda run -n adacolrag-visrag python -m pip install -e .

conda run -n adacolrag-core pytest -q
```

The primary ColPali checkpoint is:

```text
vidore/colpali-v1.3-merged
```

The merged checkpoint is required for submission runs because it avoids ambiguous LoRA key remapping. Unmerged adapter checkpoints are rejected unless `--allow-unverified-checkpoint` is passed explicitly for diagnostics.

## One-click submission run

The default workflow reruns embeddings, 17 experiments, bootstrap analysis, repeated timing, and four datasets. Reserve substantial disk space and use `tmux`.

```bash
cd ~/GMY/AdaColRAG
chmod +x scripts/run_submission_experiments.sh
chmod +x scripts/run_all_required_experiments.sh

tmux new -s adacolrag-submission

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
REBUILD_EMBEDDINGS=1 \
FRESH_RESULTS=1 \
bash scripts/run_all_required_experiments.sh
```

When `VISRAG_LOCAL_MODEL_DIR` is omitted, the model is loaded from `openbmb/VisRAG-Ret` through the configured Hugging Face endpoint/cache.

### Quick Finance EN verification

Use this before the full multi-domain run:

```bash
PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
QUICK_MODE=1 \
VISRAG_LOCAL_MODEL_DIR="$HOME/GMY/AdaColRAG/models/VisRAG-Ret" \
COLVISION_BATCH_SIZE=1 \
VISRAG_BATCH_SIZE=1 \
REBUILD_EMBEDDINGS=1 \
FRESH_RESULTS=1 \
bash scripts/run_all_required_experiments.sh
```

`QUICK_MODE=1` runs only Finance EN and one measured timing pass. It still runs the full 17-system controlled matrix and all integrity/statistical analysis.

## Default dataset specification

The full script uses a semicolon-separated environment variable:

```text
vidore/vidore_v3_finance_en|vidore_v3_finance_en|english;
vidore/vidore_v3_industrial|vidore_v3_industrial|english;
vidore/vidore_v3_pharmaceuticals|vidore_v3_pharmaceuticals|english;
vidore/vidore_v3_finance_fr|vidore_v3_finance_fr|french
```

Override it with `SUBMISSION_DATASETS` to add or remove datasets.

## Submission matrix

The 17 systems are grouped as follows.

### External and exhaustive baselines

- `visrag_dense`
- `colpali_full`

### Candidate-generation controls

- `colpali_mean_top50_full`
- `visrag_top50_full`

### ColPali-mean candidate token baselines

- `fixed_32`
- `fixed_64`
- `fixed_128`
- `adaptive_budget`
- `adaptive_mmr`
- `adaptive_fallback`

### Identical VisRAG Top-50 candidate controls

- `visrag_top50_fixed_112`
- `visrag_top50_fixed_128`
- `visrag_top50_adaptive_budget`
- `visrag_top50_adaptive_mmr`
- `visrag_top50_adaptive_fallback`

### Complete systems

- `adacolrag`
- `adacolrag_ocr`

The identical-candidate chain is the principal ablation evidence. It avoids attributing a VisRAG candidate-recall gain to token selection.

## Outputs

For each dataset:

```text
results/submission/<dataset>/
├── runs/                       17 query-level result JSON files
├── analysis/
│   ├── summary.json
│   ├── summary.csv
│   ├── results_table.md
│   └── significance_table.md
├── validation.json
└── repeated_timing.json
```

Global metadata and paper outputs:

```text
results/submission/_meta/
paper/generated/cross_dataset_results.md
paper/generated/cross_dataset_results.csv
paper/generated/cross_dataset_results.json
paper/draft.md
```

## Integrity policy

`validate_submission_results.py` fails when:

- a matrix result or query trace is missing;
- dataset checksums differ across systems;
- ColPali and VisRAG document/query counts differ;
- a ColPali-only experiment accidentally contains dense embeddings;
- the ColPali checkpoint is not verified and merged;
- metric or efficiency values are non-finite.

The workflow does not enforce the old 50% local-token engineering target. It reports local token reduction, candidate-operation reduction, full-corpus visual-token work reduction, effectiveness, confidence, and latency separately.

## Timing boundary

The core timing covers retrieval over preloaded embeddings:

- candidate scoring;
- token selection;
- ColPali late interaction;
- fusion;
- fallback;
- sorting.

It excludes model encoding, disk loading, network transfer, indexing, and downstream generation. Repeated timings are therefore retrieval-stage implementation comparisons, not end-to-end serving measurements.

## Paper workflow

`paper/draft.md` contains four generated table blocks. After all experiments finish, `scripts/update_paper_results.py` inserts:

- the full Finance EN controlled matrix;
- cross-domain primary results;
- paired bootstrap intervals;
- repeated timing statistics.

Figure placeholders remain textual and should be converted to publication graphics after the rerun.
