#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
PRIMARY_DATASET="${PRIMARY_DATASET:-vidore_v3_finance_en}"

cd "$PROJECT_ROOT"

bash scripts/run_submission_experiments.sh

printf '\n[paper] Inject generated tables into paper/draft.md\n'
conda run --no-capture-output -n adacolrag-core python -u scripts/update_paper_results.py \
  --paper paper/draft.md \
  --generated-dir paper/generated \
  --submission-root results/submission \
  --primary-dataset "$PRIMARY_DATASET"

printf '\nAll required experiments, validation, analysis, timing, aggregation, and paper-table updates completed.\n'
printf 'Manuscript: %s/paper/draft.md\n' "$PROJECT_ROOT"
