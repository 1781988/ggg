# AdaColRAG Submission Experiment Protocol

This file defines the mandatory rerun used to replace the archived Finance EN pilot.

## Required model provenance

- ColPali: `vidore/colpali-v1.3-merged`
- VisRAG: `openbmb/VisRAG-Ret` or a byte-identical local snapshot
- ColPali manifest must contain `checkpoint_verification.verified = true`
- Unmerged adapter runs are diagnostic only and must not populate submission tables

## Required datasets

1. ViDoRe V3 Finance EN
2. ViDoRe V3 Industrial
3. ViDoRe V3 Pharmaceuticals
4. ViDoRe V3 Finance FR

## Required systems

The source of truth is `configs/submission_matrix.yaml`. It contains 17 systems covering:

- dense and exhaustive external baselines;
- ColPali-mean and VisRAG candidate controls;
- fixed 32/64/112/128-token controls;
- adaptive relevance-only and adaptive MMR selection;
- confidence-aware fallback without fusion;
- complete AdaColRAG and the OCR extension.

## Required analyses

- nDCG@5/10, Recall@1/5/10, MRR@10;
- local token reduction;
- visual tokens and page-scoring operations per query;
- system work reduction versus exhaustive full-token ColPali;
- mean/p50/p95 retrieval-stage latency;
- fallback rate and fallback/non-fallback decomposition;
- paired query-level bootstrap intervals with 10,000 samples;
- one warm-up and three measured timing runs for key systems;
- hardware, package, thread, cache, and Git metadata;
- result and checkpoint integrity validation.

## Required control logic

The principal causal chain uses the same VisRAG Top-50 candidates:

1. full-token ColPali;
2. Fixed-112;
3. Fixed-128;
4. adaptive relevance-only budget;
5. adaptive MMR;
6. adaptive MMR plus full-token fallback;
7. complete AdaColRAG with dense score fusion.

This chain separates candidate pruning from token selection and fusion.

## One-click command

```bash
bash scripts/run_all_required_experiments.sh
```

The script creates per-dataset result bundles under `results/submission/`, cross-dataset tables under `paper/generated/`, and injects generated tables into `paper/draft.md`.

## Interpretation rule

The original 50% local-token target is not an acceptance gate for publication. It must still be reported exactly. Paper claims should be based on effectiveness, local compression, system work, latency, uncertainty, and controlled attribution together.
