# AdaColRAG: Confidence-Aware Query-Adaptive Visual Token Compression for Efficient Multimodal Document Retrieval

> Target venue: WSDM 2027 Main Track  
> Status: method and experiment draft; all empirical values marked `TBD` must be populated from generated result JSON files.

## Abstract

Visually rich document retrieval models such as ColPali directly encode page images into multi-vector representations and apply late interaction with textual queries. This preserves layout, tables, figures, and other non-textual evidence, but its computation and storage grow with the number of visual tokens retained per page. Existing fixed-ratio token compression methods cannot distinguish simple queries that require a small local region from ambiguous or compositional queries that require broader page evidence. We propose **AdaColRAG**, a confidence-aware visual retrieval framework that allocates a query-dependent token budget, selects query-relevant yet non-redundant visual tokens, and invokes an expanded-token fallback only when the compressed ranking is uncertain. A dense VisRAG-Ret representation is used for coarse candidate generation and optional rank fusion, while ColPali multi-vector embeddings provide fine-grained late-interaction reranking. The method requires no full multimodal-model retraining and can operate over precomputed embeddings. We evaluate retrieval quality, latency, visual-token usage, and fallback reliability on ViDoRe V2/V3 and document-VQA retrieval tasks. The principal hypothesis is that AdaColRAG preserves full ColPali retrieval quality within a small absolute nDCG@5 tolerance while substantially reducing reranking tokens and latency. Results are `TBD`.

## 1. Introduction

Real documents communicate through more than plain text. Tables, plots, page layout, font hierarchy, diagrams, and spatial grouping often determine whether a page is relevant to an information need. OCR-first retrieval pipelines can lose or distort these signals. Visual document retrievers address this limitation by embedding page images directly. ColPali is a representative late-interaction architecture: each query token matches its strongest visual page token and the token-wise maxima are aggregated into a page score.

This design is accurate and interpretable, but multi-vector matching is more expensive than single-vector retrieval. A document page may contain hundreds of visual tokens, many of which are irrelevant to a specific query or redundant with neighboring patches. A fixed global compression ratio is therefore inefficient: it over-allocates tokens to easy queries and can discard crucial evidence for hard queries.

We study the following question:

> Can a visual document retriever allocate and select page tokens according to each query, while retaining a reliable fallback path for difficult or uncertain cases?

AdaColRAG combines four mechanisms. First, a dense page representation, obtained from VisRAG-Ret or by mean pooling ColPali tokens, produces a small candidate set. Second, a budget controller estimates query complexity from query length, query-token dispersion, and coarse-ranking ambiguity. Third, a query-aware MMR selector retains tokens that are relevant to the query, diverse with respect to already selected tokens, and spatially distributed across the page. Fourth, a confidence estimator uses ranking margin, score entropy, and query-evidence coverage to decide whether the compressed ranking is trustworthy. Low-confidence cases are reranked with a larger or full token budget.

The intended contributions are:

1. **Query-adaptive visual-token budgeting.** We formulate token allocation as a function of textual complexity and retrieval ambiguity rather than a fixed compression ratio.
2. **Relevance-diversity-layout token selection.** We introduce a lightweight MMR-style selector for ColPali page embeddings, requiring no VLM retraining.
3. **Confidence-triggered recovery.** We estimate whether compressed retrieval has sufficient evidence and selectively recover full-token matching.
4. **A reproducible efficiency evaluation protocol.** We jointly report retrieval metrics, visual-token reduction, latency distributions, fallback behavior, and query-level traces.

## 2. Related Work

### 2.1 Visual document retrieval

ColPali [1] encodes document page images with a vision-language model and performs multi-vector late interaction with query embeddings. ViDoRe [1–3] provides benchmarks spanning visually rich domains and multilingual settings. Later ColVision models extend the architecture with different multimodal backbones. These systems avoid brittle document parsing, but matching cost scales with query length, candidate count, and page-token count.

### 2.2 Visual retrieval-augmented generation

VisRAG [4] directly embeds document images for retrieval and conditions a vision-language generator on retrieved pages. EVisRAG [5] further studies evidence-guided multi-image reasoning. Our work focuses primarily on the retrieval stage and treats generation as an optional downstream evaluation. Unlike end-to-end VLM fine-tuning, AdaColRAG operates on exported embeddings and modifies candidate generation, token selection, and fallback logic.

### 2.3 Token pruning and efficient multimodal inference

Visual-token pruning has been studied for reducing VLM inference cost. Many methods prune tokens inside a generative VLM or use a fixed token budget. Document retrieval differs in two respects: relevance is explicitly conditioned on the search query, and a compressed error can change which documents enter the RAG context. We therefore optimize query-document matching and include retrieval confidence as a first-class signal.

### 2.4 Selective prediction and retrieval confidence

Selective systems abstain or invoke a more expensive procedure when confidence is insufficient. Retrieval confidence can be estimated from score margins, entropy, or agreement across retrievers. AdaColRAG extends this idea with visual evidence coverage: a ranking is less reliable when many query tokens lack a sufficiently similar visual token in the top page.

## 3. Problem Formulation

Let a query be encoded as

\[
Q = [q_1, \ldots, q_m] \in \mathbb{R}^{m \times d},
\]

and a document page as

\[
D = [d_1, \ldots, d_n] \in \mathbb{R}^{n \times d},
\]

where all token vectors are L2-normalized. Full late interaction computes

\[
s_{\mathrm{LI}}(Q,D)=\frac{1}{m}\sum_{i=1}^{m}\max_{1\le j\le n} q_i^\top d_j.
\]

For candidate set size \(C\), the matching cost is proportional to \(O(Cmnd)\). We seek a query-dependent subset \(S_Q(D)\) with budget \(K_Q\ll n\), yielding

\[
\tilde{s}_{\mathrm{LI}}(Q,D)=\frac{1}{m}\sum_{i=1}^{m}\max_{j\in S_Q(D)}q_i^\top d_j.
\]

The target quality-efficiency constraint is

\[
\Delta \mathrm{nDCG@5} \le \epsilon,
\qquad
1-\frac{\mathbb{E}[K_Q]}{n} \ge \rho,
\]

where \(\epsilon\) is the maximum acceptable quality drop and \(\rho\) is the desired token reduction.

## 4. Method

### 4.1 Two-stage candidate generation and reranking

A coarse vector is produced either by VisRAG-Ret or by mean pooling normalized ColPali tokens:

\[
\bar{d}=\mathrm{norm}\left(\frac{1}{n}\sum_{j=1}^{n}d_j\right),
\qquad
\bar{q}=\mathrm{norm}\left(\frac{1}{m}\sum_{i=1}^{m}q_i\right).
\]

Cosine similarity retrieves the top \(C\) candidates. Query-adaptive multi-vector matching is then applied only to this set.

### 4.2 Query-complexity estimation

The budget controller combines three normalized features.

**Length:**

\[
f_{\mathrm{len}}=\min\left(\frac{m}{m_0},1\right).
\]

**Embedding dispersion:**

\[
f_{\mathrm{disp}}=\frac{1}{m}\sum_i \left(1-q_i^\top \bar{q}\right).
\]

**Coarse ambiguity:** with the two largest coarse scores \(c_1\ge c_2\),

\[
f_{\mathrm{amb}}=\mathrm{clip}\left(1-\frac{c_1-c_2}{\tau_m},0,1\right).
\]

The aggregate complexity is

\[
z_Q=\frac{w_l f_{\mathrm{len}}+w_d f_{\mathrm{disp}}+w_a f_{\mathrm{amb}}}{w_l+w_d+w_a}.
\]

The token budget is quantized for efficient batching:

\[
K_Q=\mathrm{RoundMultiple}\left(K_{\min}+z_Q(K_{\max}-K_{\min}), b\right).
\]

### 4.3 Query-aware MMR token selection

For each document token, relevance is

\[
r_j=\max_i q_i^\top d_j.
\]

Starting from the highest-relevance token, we greedily select

\[
j^*=\arg\max_{j\notin S}
\left[
\lambda_r r_j
-\lambda_c \max_{k\in S}d_j^\top d_k
+\lambda_l g(j,S)
\right],
\]

where the second term penalizes redundant content. The layout term \(g(j,S)\) rewards underrepresented spatial bins, reducing collapse into a single page region.

### 4.4 Dense and lexical fusion

When VisRAG dense scores are available, normalized scores are fused as

\[
s_{\mathrm{fuse}}=(1-\alpha)s_{\mathrm{LI}}+\alpha s_{\mathrm{dense}}.
\]

An optional OCR lexical score can be added with weight \(\beta\). OCR is not required by the main method and is evaluated separately.

### 4.5 Confidence and fallback

We use margin confidence, low-entropy confidence, and visual evidence coverage:

\[
c_{\mathrm{cov}}=\frac{1}{m}\sum_i
\mathbb{I}\left[\max_{j\in S_Q(D_1)}q_i^\top d_j \ge \tau_s\right].
\]

The total confidence is

\[
c_Q=w_m c_{\mathrm{margin}}+w_e c_{\mathrm{entropy}}+w_v c_{\mathrm{cov}}.
\]

If \(c_Q<\tau_c\), AdaColRAG reranks an expanded candidate set using \(K_{\max}\) or all document tokens.

## 5. Experimental Design

### 5.1 Research questions

- **RQ1:** Does query-adaptive budgeting preserve full ColPali quality better than fixed budgets at the same mean token count?
- **RQ2:** Does MMR-style diversity improve fine-grained and multi-region queries?
- **RQ3:** Does confidence-triggered fallback recover errors efficiently?
- **RQ4:** Are gains consistent across domains, languages, query types, and visual evidence types?
- **RQ5:** Does VisRAG dense fusion improve candidate recall and final ranking?

### 5.2 Datasets

Primary evaluation should include:

- ViDoRe V2 challenging retrieval subsets;
- ViDoRe V3 English datasets from multiple professional domains;
- at least one multilingual ViDoRe split;
- optional page-retrieval conversions of ChartQA, InfographicVQA, MP-DocVQA, and SlideVQA.

Hyperparameters are selected on development data only.

### 5.3 Models

| Role | Model | Use |
|---|---|---|
| Fine retriever | `vidore/colpali-v1.3` | primary multi-vector baseline and proposed method |
| Low-resource alternative | `vidore/colqwen2-v1.0` | backbone sensitivity |
| Dense retriever | `openbmb/VisRAG-Ret` | coarse retrieval and fusion baseline |
| Optional generator | Qwen2.5-VL-3B-Instruct or EVisRAG-3B | end-to-end answer evaluation only |

No full retriever pretraining is required for the principal experiments.

### 5.4 Baselines

| ID | Method | Candidate stage | Fine stage | Fallback |
|---|---|---|---|---|
| B0 | VisRAG-Ret | dense | none | no |
| B1 | ColPali full | mean/dense | all tokens | no |
| B2 | Fixed-32 | mean/dense | top-32 relevant tokens | no |
| B3 | Fixed-64 | mean/dense | top-64 relevant tokens | no |
| B4 | Fixed-128 | mean/dense | top-128 relevant tokens | no |
| B5 | Adaptive budget | mean/dense | adaptive relevance-only | no |
| B6 | Adaptive MMR | mean/dense | relevance/diversity/layout | no |
| B7 | Adaptive MMR + fallback | mean/dense | adaptive, then expanded | yes |
| B8 | AdaColRAG | VisRAG dense | adaptive MMR + fusion | yes |
| B9 | AdaColRAG + OCR | hybrid | adaptive MMR + hybrid fusion | yes |

### 5.5 Ablations

1. fixed mean budget instead of query-adaptive budget;
2. remove query-length feature;
3. remove query-dispersion feature;
4. remove coarse-ambiguity feature;
5. remove redundancy penalty;
6. remove layout coverage;
7. remove evidence-coverage confidence;
8. remove fallback;
9. replace full fallback with `K_max` fallback;
10. remove VisRAG dense fusion;
11. remove OCR lexical fusion.

### 5.6 Metrics

**Retrieval quality:** nDCG@5, nDCG@10, Recall@1/5/10, MRR@10.

**Efficiency:** mean and p95 query latency, mean selected visual tokens, total visual-token operations, candidate reranking operations, accelerator memory, and index storage.

**Reliability:** fallback rate, precision of fallback triggering, quality on fallback/non-fallback subsets, and error recovery rate.

**Statistical analysis:** paired query-level bootstrap confidence intervals and paired randomization/permutation tests.

### 5.7 Primary acceptance criterion

The preregistered engineering target is:

- absolute nDCG@5 drop versus full ColPali no larger than `0.01`;
- at least `50%` mean visual-token reduction;
- at least `30%` reranking latency reduction;
- fallback rate low enough that net computation remains below full ColPali.

These are targets, not results.

## 6. Implementation Details

The repository uses separate environments for ColPali and VisRAG embedding export. This prevents dependency conflicts and allows all reranking ablations to reuse identical embeddings. Embeddings are L2-normalized and stored in float32 for deterministic core experiments.

Default hyperparameters are:

| Hyperparameter | Value |
|---|---:|
| candidate pool | 50 |
| fallback pool | 100 |
| minimum tokens | 32 |
| maximum tokens | 128 |
| budget multiple | 8 |
| relevance weight | 1.00 |
| redundancy weight | 0.25 |
| layout weight | 0.10 |
| confidence threshold | 0.48 |
| dense fusion weight | 0.15 |

The final paper must report sensitivity curves rather than only the selected operating point.

## 7. Results

### 7.1 Main retrieval and efficiency results

| Method | nDCG@5 | Recall@5 | MRR@10 | Mean tokens | Token reduction | Mean latency | p95 latency | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VisRAG-Ret | TBD | TBD | TBD | 0 | n/a | TBD | TBD | 0 |
| ColPali full | TBD | TBD | TBD | TBD | 0% | TBD | TBD | 0 |
| Fixed-32 | TBD | TBD | TBD | 32 | TBD | TBD | TBD | 0 |
| Fixed-64 | TBD | TBD | TBD | 64 | TBD | TBD | TBD | 0 |
| Fixed-128 | TBD | TBD | TBD | 128 | TBD | TBD | TBD | 0 |
| AdaColRAG | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 7.2 Ablation results

| Variant | nDCG@5 | Mean tokens | Mean latency | Fallback rate |
|---|---:|---:|---:|---:|
| Full method | TBD | TBD | TBD | TBD |
| − adaptive budget | TBD | TBD | TBD | TBD |
| − redundancy | TBD | TBD | TBD | TBD |
| − layout coverage | TBD | TBD | TBD | TBD |
| − confidence fallback | TBD | TBD | TBD | 0 |
| − dense fusion | TBD | TBD | TBD | TBD |

### 7.3 Query-type and failure analysis

Queries should be grouped by length, coarse ambiguity, evidence type, and whether relevant evidence is local or distributed. At least 100 errors should be classified into candidate miss, small-text evidence, chart/table structure, multi-region evidence, multilingual mismatch, over-compression, confidence false negative, confidence false positive, and annotation ambiguity.

## 8. Threats to Validity

1. Approximate patch coordinates may not perfectly match the vision encoder grid.
2. Latency is hardware-, batch-, and implementation-dependent; hardware and synchronization policy must be reported.
3. Candidate recall can dominate final performance; controlled comparisons must use identical candidate sets.
4. Hyperparameter tuning on test queries invalidates a clean generalization claim.
5. OCR fusion can weaken the parsing-free claim and is therefore optional.
6. ViDoRe domains may not represent all enterprise documents.

## 9. Reproducibility Checklist

- [ ] Release all configuration files.
- [ ] Pin model identifiers and exporter versions.
- [ ] Record dataset checksums.
- [ ] Preserve query-level rankings and traces.
- [ ] Report all seeds and confidence intervals.
- [ ] Report failed acceptance tests.
- [ ] Generate tables from JSON outputs.
- [ ] Include hardware and latency protocol.
- [ ] Confirm licenses for datasets and model weights.

## 10. Conclusion

AdaColRAG reframes visual-token compression as a query-dependent and confidence-aware retrieval policy. It combines low-cost dense candidate generation, fine-grained ColPali reranking, adaptive token allocation, non-redundant token selection, and selective fallback. The empirical conclusion remains `TBD` until the complete benchmark matrix is executed.

## References

[1] M. Faysse et al. “ColPali: Efficient Document Retrieval with Vision Language Models.” 2024.  
[2] Q. Macé et al. “ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval.” 2025.  
[3] A. Loison et al. “ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios.” 2026.  
[4] S. Yu et al. “VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents.” 2024.  
[5] Y. Sun et al. “VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation.” 2025.
