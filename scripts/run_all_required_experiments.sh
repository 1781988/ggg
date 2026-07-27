#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/GMY/AdaColRAG}"
PRIMARY_DATASET="${PRIMARY_DATASET:-vidore_v3_finance_en}"

cd "$PROJECT_ROOT"

printf '\n[proxy preflight] Active proxy variables\n'
for proxy_name in ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy NO_PROXY no_proxy; do
  proxy_value="${!proxy_name:-}"
  if [[ -n "$proxy_value" ]]; then
    printf '%s=%s\n' "$proxy_name" "$proxy_value"
  fi
done

if env | grep -Eiq '^(ALL_PROXY|all_proxy|HTTPS_PROXY|https_proxy|HTTP_PROXY|http_proxy)=socks'; then
  printf '[proxy preflight] SOCKS proxy detected; verifying socksio in model environments\n'
  conda run --no-capture-output -n adacolrag-colpali python -c \
    'import socksio; print("colpali SOCKS support: OK", getattr(socksio, "__version__", "installed"))'
  conda run --no-capture-output -n adacolrag-visrag python -c \
    'import socksio; print("visrag SOCKS support: OK", getattr(socksio, "__version__", "installed"))'
else
  printf '[proxy preflight] No SOCKS proxy detected\n'
fi

bash scripts/run_submission_experiments.sh

printf '\n[paper] Inject development, held-out, significance, and timing tables into paper/draft.md\n'
conda run --no-capture-output -n adacolrag-core python -u scripts/update_paper_results.py \
  --paper paper/draft.md \
  --generated-dir paper/generated \
  --submission-root results/submission \
  --primary-dataset "$PRIMARY_DATASET"

printf '\nAll required development and held-out experiments completed.\n'
printf 'Manuscript: %s/paper/draft.md\n' "$PROJECT_ROOT"
printf 'Finance development bundle: %s/results/submission/%s/development/\n' "$PROJECT_ROOT" "$PRIMARY_DATASET"
printf 'Per-dataset main bundles: %s/results/submission/<dataset>/\n' "$PROJECT_ROOT"
printf 'Cross-dataset tables: %s/paper/generated/\n' "$PROJECT_ROOT"
