# AdaColRAG

**Redundancy-aware visual-token selection and dense–late interaction fusion for efficient visual document retrieval.**

AdaColRAG combines frozen VisRAG dense candidate generation with selected-token ColPali late interaction. The final method uses:

- VisRAG Top-50 candidate generation;
- a fixed 112-token budget per candidate;
- relevance Top-448 prefiltering;
- redundancy-aware MMR with weight 0.25;
- dense/late-interaction fusion with dense weight 0.25;
- no layout term;
- no confidence fallback;
- no OCR fusion.

Adaptive budgeting, approximate layout coverage, fallback, and OCR remain in the repository only for archived or negative ablations.

## Evidence status

The repository contains validated experiments on four ViDoRe V3 collections:

- Finance EN: 2,942 pages and 309 queries;
- Industrial: 5,244 pages and 283 queries;
- Pharmaceuticals: 2,313 pages and 364 queries;
- Finance FR: 2,384 pages and 320 queries.

The selected method reaches 0.4656 macro nDCG@5, compared with 0.4246 for exhaustive ColPali and 0.4143 for dense VisRAG. It scores 112 of 1,024 visual tokens on 50 candidate pages, corresponding to 89.06% local token reduction and approximately 99.81% full-corpus visual-token work reduction.

## Final repository structure

```text
configs/adacolrag.yaml                       frozen final method
configs/final_paper_matrix.yaml              final paper systems and comparisons
configs/development_matrix.yaml              Finance EN sensitivity experiments
configs/submission_matrix.yaml               archived 13-system main experiment matrix
configs/final_targeted_matrix.yaml           four final-method candidates

scripts/analyze_final_method_vs_baselines.py final paired bootstrap postprocessing
scripts/steady_state_benchmarks.py           single-process steady-state timing
scripts/aggregate_final_paper_results.py     final manuscript table generation
scripts/update_final_paper_results.py        final table injection
scripts/run_paper_finalization.sh            one-click statistical/timing finalization
scripts/package_paper_finalization_review.sh lightweight final evidence bundle

results/submission/                           validated original baselines/ablations
results/final_targeted/                       validated final-method candidates
results/paper_final/                          final significance and steady timing
paper/generated/final_*.md                   final manuscript tables
paper/draft.md                                rewritten manuscript source
paper/EXPERIMENT_PROTOCOL.md                  final evidence protocol
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

## Final paper evidence workflow

The full effectiveness experiments and final-candidate experiments must already exist locally. The finalization workflow does not download datasets, load model checkpoints, or regenerate embeddings. It only performs missing paired statistics and single-process timing.

```bash
cd ~/GMY/AdaColRAG
chmod +x scripts/run_paper_finalization.sh scripts/package_paper_finalization_review.sh

PROJECT_ROOT="$HOME/GMY/AdaColRAG" \
BOOTSTRAP_SAMPLES=10000 \
STEADY_REPEATS=7 \
STEADY_WARMUPS=1 \
CPU_AFFINITY="0-7" \
USE_TASKSET=1 \
OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
bash scripts/run_paper_finalization.sh
```

This command:

1. compares the final method with stored major baselines using paired query-level nDCG@5/10 bootstrap;
2. loads each dataset's embeddings once;
3. performs in-process warm-up and seven measured timing runs;
4. generates final main, significance, efficiency, selection, and timing tables;
5. injects the generated tables into `paper/draft.md`.

## Final output paths

```text
results/paper_final/<dataset>/significance/final_vs_baselines.json
results/paper_final/<dataset>/steady_state_timing.json
paper/generated/final_main_results.md
paper/generated/final_significance.md
paper/generated/final_efficiency.md
paper/generated/final_steady_timing.md
paper/generated/final_selection.md
paper/generated/final_paper_results.json
paper/draft.md
```

## Package the evidence for review

```bash
OUTPUT="AdaColRAG_paper_finalization_review_bundle.tar.gz" \
bash scripts/package_paper_finalization_review.sh
```

The archive excludes model weights, page images, parquet data, and embedding arrays. It includes validated result JSON files, final statistics, steady-state timing, manifests, configurations, scripts, generated tables, and the current manuscript.

## Interpretation policy

- System visual-token work reduction is not index-storage reduction.
- Measured timing excludes image/query encoding and disk loading.
- Finance EN is the development collection; held-out generalization claims rely on Industrial, Pharmaceuticals, and Finance FR.
- Layout coverage, fallback, OCR, and adaptive budgeting are negative or optional ablations, not final-method contributions.
- Dense VisRAG remains the minimum-latency baseline; AdaColRAG targets a stronger quality–cost operating point.
