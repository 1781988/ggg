# AdaColRAG Final Paper Protocol

This protocol defines the evidence that may populate the final manuscript. It supersedes the earlier adaptive-budget and confidence-recovery framing.

## 1. Frozen final method

The primary configuration is `configs/adacolrag.yaml`:

- VisRAG Top-50 dense candidate generation;
- fixed 112-token budget per candidate;
- relevance prefilter factor 4, yielding at most 448 MMR candidates;
- redundancy weight 0.25;
- layout weight 0.0;
- dense fusion weight 0.25;
- no OCR fusion;
- no confidence-triggered fallback.

Adaptive budgeting, approximate layout coverage, fallback, and OCR remain available only for archived or negative ablations.

## 2. Model provenance

- ColPali: verified local snapshot of `vidore/colpali-v1.3-merged`.
- VisRAG: `openbmb/VisRAG-Ret` or a byte-identical local snapshot.
- ColPali manifests must report `verified=true`, `merged_checkpoint=true`, and `local_files_only=true`.
- All compared runs for a dataset must have the same dataset checksum and query count.

## 3. Dataset roles

| Dataset | Language | Role |
|---|---|---|
| Finance EN | English | development sensitivity and continuity |
| Industrial | English | held-out domain transfer |
| Pharmaceuticals | English | held-out scientific/table transfer |
| Finance FR | French | held-out multilingual transfer |

Only Finance EN may influence the token budget, MMR prefilter factor, or fusion weight.

## 4. Final evidence sources

The final manuscript combines already validated result bundles:

- `results/submission/<dataset>/runs/`: external baselines and controlled ablations;
- `results/final_targeted/<dataset>/runs/`: four final-method candidates;
- `results/paper_final/<dataset>/significance/`: final method versus primary baselines;
- `results/paper_final/<dataset>/steady_state_timing.json`: single-process timing;
- `paper/generated/final_*.md`: automatically generated manuscript tables.

The final result source is:

```text
final_redundancy_dense025_no_fallback
```

It is exposed as `adacolrag` through `configs/adacolrag.yaml` and `configs/final_paper_matrix.yaml`.

## 5. Required primary comparisons

The final method must be compared with:

1. `visrag_dense`;
2. `colpali_full`;
3. `visrag_top50_full`;
4. `visrag_top50_fixed_112`;
5. `visrag_top50_mmr_redundancy`;
6. the archived α=0.15 no-fallback system;
7. the archived complete α=0.15 system.

Each comparison uses paired query-level nDCG@5 and nDCG@10 with 10,000 bootstrap samples.

## 6. Timing protocol

Absolute timing must use `scripts/steady_state_benchmarks.py` rather than the archived subprocess-based timing script.

- Load ColPali and VisRAG embeddings once per dataset.
- Run at least one full warm-up evaluation in the same process.
- Run at least seven measured evaluations in the same process.
- Fix CPU affinity and BLAS thread counts.
- Report median run mean, run-level IQR, pooled query median, pooled query P95, and coefficient of variation.
- Exclude image/query encoding, disk loading, network transfer, and downstream generation.

The deterministic efficiency claims remain:

- 112 selected tokens per scored page;
- 50 scored pages per query;
- 5,600 visual tokens per query;
- 89.06% local token reduction;
- approximately 99.81% full-corpus visual-token work reduction.

## 7. Interpretation rules

- Do not claim index-storage reduction.
- Do not claim universal superiority over VisRAG when a paired interval crosses zero.
- Treat layout coverage, fallback, OCR, and adaptive budgeting as negative or optional ablations.
- Use Industrial, Pharmaceuticals, and Finance FR for held-out generalization claims.
- Keep measured latency separate from analytical token-work reduction.

## 8. One-click finalization

After the validated main and targeted experiments exist locally:

```bash
bash scripts/run_paper_finalization.sh
```

This command performs only statistical postprocessing and single-process timing. It does not download data, run model encoding, or rerun the full effectiveness matrix.
