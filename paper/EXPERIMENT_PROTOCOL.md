# AdaColRAG Final Experiment Protocol

This protocol separates **development calibration**, **held-out evaluation**, and the final targeted method-selection stage. Adaptive budgeting is excluded from the primary method because it did not significantly outperform a matched fixed budget on verified Finance EN results.

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

## 3. Evidence-supported method components

The full completed matrix supports the following components:

- VisRAG Top-50 candidate generation;
- fixed 112-token online scoring budget;
- relevance prefilter of four times the final budget;
- redundancy-aware fast MMR with weight `0.25`;
- dense/late-interaction score fusion;
- optional confidence-triggered Top-100 full-token recovery.

The following components are not primary claims:

- adaptive budgeting: negative result and archived reproducibility only;
- approximate layout coverage: no stable quality gain and substantial latency cost;
- OCR fusion: optional extension only;
- fallback: reliability operating mode unless final targeted results justify making it the default.

## 4. Finance EN development matrix

`configs/development_matrix.yaml` evaluates:

- fixed budgets: 64, 96, 112, 128;
- MMR prefilter factors: 2, 4, exact all-token MMR;
- recovery thresholds: 0.44, 0.48, 0.52;
- dense fusion weights: 0.05, 0.15, 0.25.

The verified development results select fixed 112 tokens and prefilter factor 4. Dense weight `0.25` outperformed `0.15` on Finance EN and therefore requires held-out confirmation before the final manuscript is frozen.

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

## 6. Final targeted method-selection matrix

The source of truth is `configs/final_targeted_matrix.yaml`. It contains four candidates, evaluated on all four datasets using the already exported embeddings:

1. redundancy-only MMR, dense weight `0.25`, no fallback;
2. redundancy-only MMR, dense weight `0.25`, fallback;
3. redundancy plus layout MMR, dense weight `0.25`, no fallback;
4. redundancy plus layout MMR, dense weight `0.25`, fallback.

This stage answers the two remaining method-selection questions:

- whether layout should be deleted from the final system;
- whether fallback should be the default or an optional reliability mode.

Every pair is analyzed with 10,000 paired query-level bootstrap resamples. Each candidate also receives one warm-up and seven measured timing runs under fixed thread and CPU-affinity settings.

## 7. Final selection rule

The default method should be the simplest candidate satisfying all of the following:

- highest or statistically indistinguishable macro nDCG@5;
- no systematic held-out degradation;
- materially lower latency when quality is statistically tied;
- higher local token reduction when fallback quality gains are not significant;
- evidence-supported components only.

Fallback may remain an optional reliability mode when its quality gain is concentrated in one domain and its computation cost is substantial.

## 8. Required outputs

For each dataset and experiment stage:

- nDCG@5/10, Recall@1/5/10, MRR@10;
- local token reduction;
- visual tokens and page-scoring operations per query;
- system work reduction versus exhaustive full-token ColPali;
- mean/P50/P95 retrieval-stage latency;
- fallback rate and query-level confidence traces;
- paired bootstrap intervals and two-sided bootstrap probability;
- repeated timing with warm-up;
- result and checkpoint validation;
- hardware, package, thread, cache, dataset revision, and Git metadata.

## 9. Execution

Completed main workflow:

```bash
bash scripts/run_all_required_experiments.sh
```

Final targeted workflow:

```bash
bash scripts/run_final_targeted_experiments.sh
```

Final review bundle:

```bash
bash scripts/package_final_experiment_review.sh
```

## 10. Interpretation rules

- Adaptive budgeting is a reported negative result, not a contribution.
- Approximate layout coverage is removed unless final targeted results show a reproducible gain.
- The 50% local-token target is not a publication gate; the exact achieved value must be reported.
- System work reduction must not be described as index-storage reduction.
- AdaColRAG must not be described as universally superior to VisRAG unless held-out significance supports that statement.
- Primary cross-domain claims use frozen configurations on Industrial, Pharmaceuticals, and Finance FR.
- Numerical claims in the final manuscript must be generated from validated result bundles, not copied from archived pilot runs.
