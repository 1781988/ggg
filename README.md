# AdaColRAG

**Query-adaptive visual token compression for efficient multimodal document retrieval and visual RAG.**

AdaColRAG is a research scaffold for a WSDM 2027-oriented project built around three compatible ideas:

1. **ColPali/ColVision late interaction** for page-image retrieval;
2. **VisRAG-Ret dense retrieval** as a complementary coarse retriever;
3. **query-adaptive visual-token selection** with confidence-triggered fallback and optional OCR fusion.

The repository deliberately separates **model-specific embedding export** from the **core retrieval experiment**. This avoids dependency conflicts between current ColPali and VisRAG stacks, permits repeated ablations without re-encoding pages, and makes the main experiments runnable on a single 48 GB GPU or from precomputed embeddings on CPU.

> Status: the code framework, synthetic regression test, experiment matrix, and paper draft are implemented. No real ViDoRe score is claimed until the commands below are executed and the produced JSON files are checked into `results/real/`.

### Current validated implementation status

The repository was locally validated before publication with four unit/integration tests and a deterministic synthetic retrieval fixture. The synthetic check produced `nDCG@5 = 1.000` and `82.29%` visual-token reduction with the default configuration. These numbers verify only implementation behavior on controlled vectors; they are **not** ViDoRe or paper results. Real-model comparison remains `TBD` until embeddings are exported and the experiment matrix is run.

## 1. Proposed method

For each query, AdaColRAG performs:

1. **coarse candidate generation** using page-level mean vectors or VisRAG-Ret dense vectors;
2. **query complexity estimation** from query-token count, embedding dispersion, and coarse-score ambiguity;
3. **dynamic token budgeting** between `min_tokens` and `max_tokens`;
4. **query-aware MMR token selection** balancing relevance and non-redundancy;
5. **ColPali late-interaction reranking** on compressed visual tokens;
6. **confidence estimation** from score margin, ranking entropy, and visual evidence coverage;
7. **fallback reranking** with a larger/full token budget when confidence is low;
8. optional **OCR/BM25 rank fusion** and/or **VisRAG dense fusion**.

This design tests whether token reduction can improve the latency/index-compute trade-off without materially reducing retrieval quality, while avoiding overconfident failures on difficult queries.

## 2. Repository layout

```text
configs/                         experiment and baseline configurations
environment/                     isolated conda environments
paper/draft.md                   paper draft with method and experiment plan
results/                         metric schema, synthetic reference, real-result policy
scripts/export_colvision.py      export ColPali/ColQwen multi-vector embeddings
scripts/export_visrag.py         export VisRAG-Ret dense embeddings
scripts/run_matrix.py            run all configured baselines/ablations
scripts/compare_results.py       detect quality or efficiency regressions
src/adacolrag/                   core framework
tests/                           unit and end-to-end synthetic tests
```

## 3. Installation

### 3.1 Core experiment environment

The core environment runs from exported embeddings and therefore does not require a GPU.

```bash
conda env create -f environment/core.yml
conda activate adacolrag-core
pip install -e .
pytest -q
adacolrag smoke --output results/smoke_actual.json
```

Expected smoke-test properties are stored in `results/smoke_reference.json`. The exact latency is not compared because it is hardware-dependent; quality and token-count invariants are checked.

### 3.2 ColPali/ColVision export environment

```bash
conda env create -f environment/colpali.yml
conda activate adacolrag-colpali
pip install -e .
```

Default primary retriever:

```text
vidore/colpali-v1.3
```

Low-resource secondary retriever:

```text
vidore/colqwen2-v1.0
```

### 3.3 VisRAG-Ret export environment

VisRAG uses `trust_remote_code=True` and has a dependency stack that can conflict with the newest ColPali environment. Use the isolated environment:

```bash
conda env create -f environment/visrag.yml
conda activate adacolrag-visrag
pip install -e .
```

Model:

```text
openbmb/VisRAG-Ret
```

The VisRAG model license must be reviewed separately before non-academic use.

## 4. Data format

Prepare one dataset directory:

```text
data/<dataset>/
├── corpus.jsonl
├── queries.jsonl
└── qrels.jsonl
```

`corpus.jsonl`:

```json
{"doc_id":"doc-0001","image_path":"/absolute/or/relative/page.png","ocr_text":"optional OCR text"}
```

`queries.jsonl`:

```json
{"query_id":"q-0001","text":"What was operating revenue in 2025?"}
```

`qrels.jsonl`:

```json
{"query_id":"q-0001","doc_id":"doc-0001","relevance":1}
```

Relative image paths are resolved from the dataset directory.

## 5. Export model embeddings

### 5.1 ColPali full multi-vector embeddings

```bash
conda activate adacolrag-colpali
python scripts/export_colvision.py \
  --dataset-dir data/vidore_v3_finance_en \
  --output-dir artifacts/vidore_v3_finance_en/colpali_v13 \
  --model vidore/colpali-v1.3 \
  --batch-size 2 \
  --dtype bfloat16 \
  --device cuda:0
```

The exporter writes:

```text
artifacts/.../colpali_v13/
├── documents/<doc_id>.npz   # visual token matrix and optional patch positions
├── queries/<query_id>.npy   # query token matrix
└── manifest.json
```

For the lower-memory model:

```bash
python scripts/export_colvision.py \
  --dataset-dir data/vidore_v3_finance_en \
  --output-dir artifacts/vidore_v3_finance_en/colqwen2_v10 \
  --model vidore/colqwen2-v1.0 \
  --batch-size 2 \
  --dtype bfloat16 \
  --device cuda:0
```

### 5.2 VisRAG-Ret dense embeddings

```bash
conda activate adacolrag-visrag
python scripts/export_visrag.py \
  --dataset-dir data/vidore_v3_finance_en \
  --output-dir artifacts/vidore_v3_finance_en/visrag_ret \
  --model openbmb/VisRAG-Ret \
  --batch-size 4 \
  --dtype bfloat16 \
  --device cuda:0
```

## 6. Run one experiment

```bash
conda activate adacolrag-core
adacolrag evaluate \
  --config configs/adacolrag.yaml \
  --dataset-dir data/vidore_v3_finance_en \
  --colvision-dir artifacts/vidore_v3_finance_en/colpali_v13 \
  --dense-dir artifacts/vidore_v3_finance_en/visrag_ret \
  --output results/real/vidore_v3_finance_en/adacolrag.json
```

Run the full baseline and ablation matrix:

```bash
python scripts/run_matrix.py \
  --dataset-dir data/vidore_v3_finance_en \
  --colvision-dir artifacts/vidore_v3_finance_en/colpali_v13 \
  --dense-dir artifacts/vidore_v3_finance_en/visrag_ret \
  --output-dir results/real/vidore_v3_finance_en
```

## 7. Baselines and ablations

The matrix is defined in `configs/matrix.yaml`.

| ID | System | Purpose |
|---|---|---|
| B0 | VisRAG dense | parsing-free single-vector baseline |
| B1 | ColPali full | quality upper reference for late interaction |
| B2 | ColPali fixed-32 | aggressive fixed compression |
| B3 | ColPali fixed-64 | medium fixed compression |
| B4 | ColPali fixed-128 | conservative fixed compression |
| B5 | adaptive budget | query complexity only |
| B6 | adaptive + MMR | proposed token selector |
| B7 | adaptive + MMR + fallback | proposed confidence-aware system |
| B8 | proposed + dense fusion | ColPali/VisRAG fusion |
| B9 | proposed + OCR fusion | optional OCR hybrid |

Ablations remove one component at a time: query complexity, MMR diversity, evidence coverage, fallback, dense fusion, and OCR fusion.

## 8. Metrics

Each output JSON reports:

- retrieval: `nDCG@5`, `nDCG@10`, `Recall@1/5/10`, `MRR@10`;
- efficiency: mean selected tokens, compression ratio, mean latency, p50/p95 latency;
- reliability: fallback rate, low-confidence rate, mean confidence, and quality on fallback/non-fallback subsets;
- reproducibility: config hash, seed, model manifests, dataset checksum, and runtime metadata.

The primary WSDM-style success criterion in `configs/adacolrag.yaml` is:

```text
nDCG@5 drop versus full ColPali <= 0.01 absolute
AND visual-token reduction >= 50%
AND reranking latency reduction >= 30%
```

These are **acceptance thresholds**, not claimed results.

## 9. Compare against the full baseline

```bash
python scripts/compare_results.py \
  --baseline results/real/vidore_v3_finance_en/colpali_full.json \
  --candidate results/real/vidore_v3_finance_en/adacolrag.json \
  --max-ndcg5-drop 0.01 \
  --min-token-reduction 0.50 \
  --min-latency-reduction 0.30
```

Exit code `0` means all thresholds passed; non-zero means at least one quality or efficiency requirement failed. This makes the comparison usable in CI.

## 10. Recommended real experiment sequence

1. Run B0–B4 on one ViDoRe V2 or V3 dataset to validate all exported embeddings.
2. Tune only on a designated development split; freeze the configuration before test evaluation.
3. Run B5–B9 and all ablations with three seeds where stochasticity exists.
4. Repeat the final configuration on at least four domains and one multilingual split.
5. Report a paired query-level bootstrap confidence interval for nDCG and latency.
6. Store every produced JSON under `results/real/`; do not manually copy numbers into the paper.
7. Generate paper tables from result JSON files to avoid transcription errors.

## 11. Hardware guidance

- Exporting `vidore/colpali-v1.3`: one A6000 48 GB is sufficient for inference with small batches.
- Core reranking from saved embeddings: CPU is supported; CUDA acceleration is optional.
- VisRAG-Ret export: use its isolated Python 3.10/CUDA-compatible environment.
- No full VLM pretraining is required. Optional training should be limited to the small budget controller or fusion calibrator.

## 12. Reproducibility and result integrity

- The code never inserts unpublished benchmark numbers.
- `results/smoke_reference.json` validates implementation invariants only.
- Real metrics must be generated from model outputs and include the config hash.
- A failed acceptance test is retained as a result, not hidden.
- Any tuning on test queries must be reported as such and is not considered a clean test result.

## 13. Research paper

The current manuscript skeleton is in [`paper/draft.md`](paper/draft.md). It includes the problem definition, equations, hypotheses, experiment table, ablations, threats to validity, and a result-population protocol. All numeric result cells remain `TBD` until generated by the code.

## 14. Upstream projects

This repository is an independent research implementation and does not vendor upstream code. It interfaces with:

- `illuin-tech/colpali` / `colpali-engine`;
- `illuin-tech/vidore-benchmark` and MTEB/ViDoRe datasets;
- `OpenBMB/VisRAG` and `openbmb/VisRAG-Ret`.

Cite the original ColPali, ViDoRe, and VisRAG papers when publishing results.
