# AdaColRAG Final Experiment Protocol

This protocol separates **development calibration** from **held-out evaluation** and removes adaptive budgeting from the primary method because it did not significantly outperform a matched fixed budget on verified Finance EN results.

## 1. Frozen model provenance

- Fine retriever: verified local snapshot of `vidore/colpali-v1.3-merged`.
- Dense retriever: `openbmb/VisRAG-Ret` or a byte-identical verified local snapshot.
- ColPali manifest must contain:
  - `checkpoint_verification.verified = true`;
  - `checkpoint_verification.merged_checkpoint = true`;
  - `checkpoint_verification.local_files_only = true`.
- Unmerged or unverified checkpoints are diagnostic only and must not populate paper tables.

## 2. Dataset roles

| Dataset | Language | Role |
|---|---|---|
| ViDoRe V3 Finance EN | English | development sensitivity plus main-matrix continuity |
| ViDoRe V3 Industrial | English | held-out domain transfer |
| ViDoRe V3 Pharmaceuticals | English | held-out scientific/table transfer |
| ViDoRe V3 Finance FR | French | held-out multilingual transfer |

The workflow pins dataset revisions and known parquet shard inventories. Only Finance EN may influence hyperparameters.

## 3. Primary method

The frozen primary configuration is:

- VisRAG Top-50 candidate generation;
- fixed 112-token budget per initially scored candidate;
- relevance prefilter of four times the final budget;
- redundancy weight `0.25`;
- approximate layout coverage weight `0.10`;
- dense fusion weight `0.15`;
- confidence threshold `0.48`;
- Top-100 full-token recovery for low-confidence queries.

The legacy query-complexity budget controller remains in code only to reproduce archived experiments. It is not part of the primary method or main claims.

## 4. Finance EN development matrix

`configs/development_matrix.yaml` evaluates:

- fixed budgets: 64, 96, 112, 128;
- MMR prefilter factors: 2, 4, exact all-token MMR;
- recovery thresholds: 0.44, 0.48, 0.52;
- dense fusion weights: 0.05, 0.15, 0.25.

This matrix answers whether the frozen setting is a reasonable quality--latency operating point. Development results must be reported separately from held-out results.

## 5. Main 13-system matrix

The source of truth is `configs/submission_matrix.yaml`.

1. `visrag_dense`: dense-only full-corpus baseline.
2. `colpali_full`: exhaustive full-token ColPali.
3. `colpali_mean_top50_full`: mean-vector candidate control with full tokens.
4. `visrag_top50_full`: VisRAG candidate control with full tokens.
5. `colpali_mean_top50_fixed_112`: candidate-source control at fixed token work.
6. `visrag_top50_fixed_112`: relevance-only matched token baseline.
7. `visrag_top50_mmr_redundancy`: redundancy-only selector.
8. `visrag_top50_mmr_layout`: layout-only selector.
9. `visrag_top50_mmr`: combined fast selector.
10. `visrag_top50_mmr_fallback`: selector plus recovery without dense fusion.
11. `adacolrag_no_fallback`: selector plus dense fusion without recovery.
12. `adacolrag`: complete method.
13. `adacolrag_ocr`: optional OCR extension.

All selector comparisons use identical VisRAG Top-50 candidates and a fixed 112-token budget.

## 6. Planned statistical comparisons

Matrix-level comparisons isolate:

- AdaColRAG versus exhaustive ColPali and VisRAG dense;
- candidate source under full-token and fixed-token scoring;
- redundancy-only, layout-only, and combined selection versus relevance-only;
- recovery without fusion;
- dense fusion without recovery;
- recovery after fusion;
- optional OCR contribution.

Every comparison uses paired query-level nDCG@5 with 10,000 bootstrap resamples.

## 7. Required outputs

For each dataset:

- nDCG@5/10, Recall@1/5/10, MRR@10;
- local token reduction;
- visual tokens and page-scoring operations per query;
- system work reduction versus exhaustive full-token ColPali;
- mean/P50/P95 retrieval-stage latency;
- fallback rate and query-level confidence traces;
- paired bootstrap intervals and two-sided bootstrap probability;
- one warm-up and three measured timing runs for key systems;
- result and checkpoint validation;
- hardware, package, thread, cache, dataset revision, and Git metadata.

## 8. One-click execution

```bash
bash scripts/run_all_required_experiments.sh
```

The command runs model and dataset preflight, Finance EN development diagnostics, the four-dataset main matrix, validation, bootstrap analysis, repeated timing, cross-dataset aggregation, and paper-table injection.

## 9. Interpretation rules

- Adaptive budgeting is a reported negative result, not a contribution.
- The 50% local-token target is not a publication gate; the exact achieved value must be reported.
- System work reduction must not be described as index-storage reduction.
- AdaColRAG must not be described as universally superior to VisRAG unless held-out significance supports that statement.
- Primary cross-domain claims use only the frozen configuration on Industrial, Pharmaceuticals, and Finance FR.
