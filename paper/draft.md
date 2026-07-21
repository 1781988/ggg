# AdaColRAG: Query-Adaptive Visual Token Selection with Confidence-Aware Fallback for Efficient Document Retrieval

> **Anonymous WSDM 2027 manuscript draft.** This Markdown source follows the structure of an ACM `sigconf` research paper. The submission version must be converted to `\documentclass[sigconf,anonymous,review]{acmart}` and compressed to the WSDM 2027 limit. Numerical results below are populated only from the archived JSON outputs for ViDoRe V3 Finance EN. Statements that require additional validation are identified explicitly in Section 8.

## Abstract

Visual document retrievers preserve page layout, tables, charts, and figures by encoding document screenshots directly. Multi-vector models such as ColPali are particularly effective because each query token can match its strongest visual page token, but full late interaction scales with both the number of candidate pages and the number of visual tokens per page. Existing storage-oriented compression is usually query independent and therefore cannot know which page regions will matter for a future request. We present **AdaColRAG**, a training-free retrieval framework that performs query-time visual-token selection. A dense retriever first produces a candidate set; a query-complexity controller allocates 32--128 visual tokens per candidate; a relevance--diversity selector removes redundant evidence while encouraging spatial coverage; and a confidence estimator selectively invokes full-token recovery for uncertain queries. Experiments on the English financial subset of ViDoRe V3, containing 2,942 pages and 309 queries, show that AdaColRAG reaches 0.4827 nDCG@5, compared with 0.4374 for exhaustive full-token ColPali and 0.4744 for VisRAG-Ret. Relative to exhaustive ColPali, AdaColRAG reduces mean retrieval-stage CPU latency from 14.43 s to 1.02 s and reduces total visual-token scoring work by 98.31%. Within the pages that are actually scored, it removes 47.25% of visual tokens; this is slightly below a preregistered 50% engineering target but does not alter the observed quality and latency gains. Ablations show that diversity-aware selection improves nDCG@5 from 0.3273 to 0.3472 at the same token budget, while dense candidate generation and confidence-aware recovery account for most of the final effectiveness. The current study establishes a promising operating point, while cross-domain evaluation, repeated timing, controlled candidate-set baselines, and checkpoint verification remain necessary before making a broad generalization claim.

**Keywords:** visual document retrieval; multimodal retrieval; late interaction; visual-token selection; selective computation; retrieval-augmented generation

## 1 Introduction

Enterprise search increasingly operates over visually rich reports, presentations, invoices, forms, and scientific documents. Relevant evidence may be expressed through text, but it may also depend on table structure, chart geometry, typography, or the spatial association between labels and values. OCR-centric pipelines can discard such evidence or propagate parsing errors. Document-image retrieval therefore treats the rendered page as the primary retrieval unit, an approach explored by screenshot embedding systems [4, 6], multimodal document retrievers [1, 7], and visual RAG pipelines [2, 8].

ColPali [1] adapts ColBERT-style late interaction [11] to page images. A page is represented by many visual embeddings, and each query embedding retrieves its strongest matching page token. This architecture preserves fine-grained evidence, but it also makes scoring expensive. In our Finance EN evaluation, every ColPali page contains 1,024 stored visual tokens. Exhaustively scoring 2,942 pages therefore requires approximately 3.01 million visual-token operations per query before accounting for query-token dimensionality.

The key observation behind this work is that **visual evidence requirements are query dependent**. A simple lookup may depend on a compact table cell, whereas an ambiguous or compositional question may require multiple regions. Query-independent token removal cannot anticipate this difference. Light-ColPali [5] provides strong evidence that storage-oriented pruning without query information can be brittle and that token merging is preferable when the query is unavailable. AdaColRAG addresses a complementary setting: it retains the complete offline index but selects a small query-conditioned subset only for candidates that are scored at search time.

AdaColRAG combines dense candidate generation, query-adaptive budgeting, relevance--diversity token selection, and selective fallback. It requires no retriever fine-tuning. The dense stage uses VisRAG-Ret [2] to identify a small candidate set. The fine stage allocates a per-query token budget and performs ColPali late interaction over selected page tokens. A confidence estimator detects uncertain compressed rankings and selectively reruns a larger candidate set with complete page representations.

Our contributions are:

1. **Query-time adaptive visual-token selection.** We formulate page-token reduction as a query-conditioned retrieval decision rather than a permanent, query-independent index transformation.
2. **A relevance--diversity--coverage selector for visual late interaction.** The selector balances MaxSim relevance, redundancy suppression inspired by maximal marginal relevance [18], and approximate page coverage.
3. **Confidence-aware recovery.** The system trades computation for reliability by invoking full-token reranking only on low-confidence queries, connecting efficient retrieval with selective prediction [19].
4. **An auditable efficiency decomposition.** We report retrieval quality, candidate operations, within-candidate token reduction, full-corpus visual-token work, latency distributions, confidence, and query-level traces rather than collapsing efficiency into a single number.
5. **An empirical analysis on ViDoRe V3 Finance EN.** AdaColRAG improves early-rank quality over exhaustive ColPali while sharply reducing retrieval-stage CPU work. We also identify where the gains originate and which claims remain unsupported by the current single-domain experiment.

**Figure 1 placeholder -- System overview.** Draw a left-to-right pipeline. The left panel contains a textual query and a collection of page screenshots. The first stage shows VisRAG-Ret mapping the query and all pages to single dense vectors and returning the top 50 candidates. The second stage shows the query-complexity controller outputting a budget in the range 32--128. The third stage depicts each candidate page as a 32-by-32 grid of 1,024 visual tokens, with a query-dependent subset highlighted across multiple page regions. The fourth stage performs ColPali MaxSim late interaction and dense-score fusion. A confidence gauge branches either to the final ranking or to a red fallback path that reranks the top 100 dense candidates with full tokens. Annotate that encoders are frozen and page embeddings are precomputed.

## 2 Related Work

### 2.1 Visual document retrieval

Visual document retrieval avoids committing to a single OCR and layout parser. Document Screenshot Embedding represents rendered pages with a single visual embedding [4], while ColPali uses multi-vector late interaction to retain fine-grained evidence [1]. VisRAG extends visual retrieval to retrieval-augmented generation and reports strong retrieval from direct page-image representations [2]. UniSE studies unified screenshot representations across modalities [6], and Unveil combines visual and textual signals before distilling them into a parsing-free retriever [7]. Recent benchmarks and systems broaden the task to long documents, layouts, and iterative reasoning [3, 8--10].

The closest efficiency study is Light-ColPali [5], which evaluates query-independent patch pruning and merging for reducing index storage. Its finding that pruning is difficult without query information motivates our query-time setting. AdaColRAG does not compress the stored index. Instead, it reduces the number of token vectors used during candidate scoring after the query becomes available. The two approaches are therefore complementary: storage compression targets index footprint, whereas AdaColRAG targets online scoring work and selective recovery.

### 2.2 Dense and late-interaction retrieval

Dense dual encoders enable efficient candidate generation by independently encoding queries and documents [13, 16]. DPR [13], RocketQA [28], and related systems establish dense retrieval as a standard first-stage architecture. ColBERT introduces late interaction, combining independent encoding with token-level MaxSim [11], and ColBERTv2 improves its effectiveness and storage efficiency [12]. BEIR [14] and MTEB [15] emphasize heterogeneous evaluation because retriever rankings can vary substantially across domains.

AdaColRAG follows a classic retrieve-then-rerank structure but uses two different visual representations: a dense VisRAG vector for candidate generation and ColPali token vectors for fine scoring. This makes the candidate stage inexpensive while preserving token-level evidence for the pages most likely to be relevant.

### 2.3 Efficient visual tokens

Vision transformers contain redundant tokens, motivating dynamic token pruning [24], token reorganization [25], learned token pooling [26], and token merging [27]. These methods primarily accelerate image classification or transformer inference inside a model. Visual document retrieval differs in two ways. First, the query is available at retrieval time and provides a direct relevance signal for page regions. Second, token removal can alter which page enters the downstream context, so an aggressive error has retrieval-level consequences. Our method therefore applies token selection to exported document embeddings and includes an explicit recovery mechanism.

### 2.4 Document understanding and layout

Layout-aware document models jointly encode text, visual features, and spatial coordinates [20--22], while Donut demonstrates the viability of OCR-free document understanding [23]. These approaches motivate retaining spatially distributed evidence. AdaColRAG uses a lightweight layout-coverage term during token selection. In the current implementation, positions are approximate normalized grid coordinates reconstructed from token order rather than exact vision-encoder patch coordinates; this limitation is discussed in Section 8.

### 2.5 Selective computation and retrieval confidence

Selective prediction allows a system to reject uncertain instances or allocate additional computation [19]. Related retrieval systems use hierarchical or multi-stage computation to balance efficiency and quality [29, 30]. AdaColRAG estimates confidence from ranking margin, score entropy, and query-evidence coverage. Unlike abstention, fallback returns a ranking for every query but increases the candidate and token budget when the compressed result is judged unreliable.

## 3 Problem Formulation

Let a query be represented by normalized token embeddings

\[
Q=[q_1,\ldots,q_m]\in\mathbb{R}^{m\times d},
\]

and a page by normalized visual embeddings

\[
D=[d_1,\ldots,d_n]\in\mathbb{R}^{n\times d}.
\]

Full ColPali-style late interaction computes

\[
s_{\mathrm{LI}}(Q,D)=\frac{1}{m}\sum_{i=1}^{m}\max_{1\le j\le n}q_i^\top d_j.
\]

For \(N\) pages, exhaustive scoring requires work proportional to \(O(Nmnd)\). A two-stage retriever first selects a candidate set \(\mathcal{C}_Q\) of size \(C\ll N\), then chooses a query-dependent subset \(S_Q(D)\subseteq\{1,\ldots,n\}\) for each candidate:

\[
\tilde{s}_{\mathrm{LI}}(Q,D)
=\frac{1}{m}\sum_{i=1}^{m}\max_{j\in S_Q(D)}q_i^\top d_j.
\]

The online policy should maximize ranking quality while reducing both candidate operations and selected visual tokens. We distinguish two efficiency measures.

**Within-scored-page token reduction**

\[
R_{\mathrm{local}}
=1-\frac{\sum_{Q,D\in\mathrm{scored}(Q)}|S_Q(D)|}
{\sum_{Q,D\in\mathrm{scored}(Q)}|D|}.
\]

This is the `token_reduction` reported by the implementation and includes both initial and fallback scoring operations.

**Full-corpus visual-token work reduction**

\[
R_{\mathrm{system}}
=1-\frac{\sum_Q\sum_{D\in\mathrm{scored}(Q)}|S_Q(D)|}
{\sum_Q\sum_{D\in\mathcal{D}}|D|}.
\]

The second measure captures candidate pruning and token selection jointly. It must not be confused with index compression: all full page embeddings remain stored.

## 4 AdaColRAG

### 4.1 Dense candidate generation

VisRAG-Ret encodes each page and query as a normalized dense vector. Cosine similarity returns the top \(C=50\) candidates. When dense embeddings are unavailable in ablations, the fallback coarse representation is the normalized mean of ColPali tokens:

\[
\bar d=\mathrm{norm}\left(\frac{1}{n}\sum_jd_j\right),\qquad
\bar q=\mathrm{norm}\left(\frac{1}{m}\sum_iq_i\right).
\]

Only experiments explicitly marked as dense attach VisRAG embeddings. The experiment runner prevents dense-retrieval information from leaking into ColPali-only ablations.

### 4.2 Query-complexity budget

A budget controller combines three normalized signals.

\[
f_{\mathrm{len}}=\min(m/m_0,1)
\]

measures query length. Query-token dispersion is

\[
f_{\mathrm{disp}}=\frac{1}{m}\sum_i(1-q_i^\top\bar q).
\]

If \(c_1\ge c_2\) are the two largest coarse scores, ranking ambiguity is

\[
f_{\mathrm{amb}}=\mathrm{clip}\left(1-\frac{c_1-c_2}{\tau_m},0,1\right).
\]

The aggregate complexity is

\[
z_Q=\frac{w_lf_{\mathrm{len}}+w_df_{\mathrm{disp}}+w_af_{\mathrm{amb}}}
{w_l+w_d+w_a}.
\]

The page-token budget is quantized to a multiple of eight:

\[
K_Q=\mathrm{RoundMultiple}
\left(K_{\min}+z_Q(K_{\max}-K_{\min}),8\right),
\]

with \(K_{\min}=32\) and \(K_{\max}=128\).

### 4.3 Query-aware token selection

For document token \(d_j\), query relevance is

\[
r_j=\max_i q_i^\top d_j.
\]

Starting with the most relevant token, the selector greedily chooses

\[
j^*=\arg\max_{j\notin S}
\left[
\lambda_r r_j
-\lambda_c\max_{k\in S}d_j^\top d_k
+\lambda_l g(j,S)
\right].
\]

The redundancy term discourages selecting near-duplicate visual tokens. The coverage bonus \(g(j,S)\) favors approximate spatial bins that are currently underrepresented. For relevance-only ablations where \(\lambda_c=\lambda_l=0\), the implementation uses an equivalent top-\(K\) fast path.

**Figure 2 placeholder -- Query-adaptive token selection.** Show the same financial-report page under three queries: a local value lookup, a table comparison, and a multi-region question. Overlay the approximate 32-by-32 token grid. Highlight a compact cluster for the local query, two separated table regions for the comparison query, and a broader distributed subset for the multi-region query. A side bar should show budgets of 104, 112, and 120 tokens. Beneath the page, illustrate relevance, redundancy penalty, and coverage bonus as three score components.

### 4.4 Score fusion

For the selected tokens, visual late-interaction scores are min--max normalized over the candidate set. When dense fusion is enabled,

\[
s_{\mathrm{fuse}}(Q,D)
=(1-\alpha)\hat s_{\mathrm{LI}}(Q,D)
+\alpha\hat s_{\mathrm{dense}}(Q,D),
\]

with \(\alpha=0.15\) in the main configuration. The OCR extension additionally includes normalized lexical overlap with weight 0.15; it is not part of the primary method.

### 4.5 Confidence and fallback

Confidence combines top-rank margin, inverse normalized entropy, and query-token evidence coverage. For the top-ranked page,

\[
c_{\mathrm{cov}}=\frac{1}{m}\sum_i
\mathbf{1}
\left[
\max_{j\in S_Q(D_1)}q_i^\top d_j\ge\tau_s
\right].
\]

The final confidence is

\[
c_Q=w_mc_{\mathrm{margin}}
+w_ec_{\mathrm{entropy}}
+w_vc_{\mathrm{cov}}.
\]

If \(c_Q<0.48\), the main configuration reranks the top 100 dense candidates with complete page-token representations. Initial compressed work is retained in the accounting, so fallback queries include both passes.

**Figure 3 placeholder -- Confidence-aware fallback.** Plot a two-dimensional schematic with confidence on the horizontal axis and retrieval cost on the vertical axis. High-confidence queries remain on the compressed Top-50 path. Low-confidence queries move to a Top-100 full-token path. Include two query examples with their top-score margin, entropy, evidence coverage, composite confidence, and final decision. The caption should emphasize that fallback is a reliability policy rather than a separate trained model.

## 5 Experimental Setup

### 5.1 Dataset

We evaluate on **ViDoRe V3 Finance EN** [3]. The corpus consists of six 2024 10-K annual reports from major U.S. financial institutions. The English evaluation contains 2,942 page images and 309 unique English queries, with an average of 4.7 relevant pages per query. Relevance annotations are graded. All ten result files share dataset checksum `4c2c0f3687668c5d`, 2,942 document identifiers, and 309 query identifiers.

| Property | Value |
|---|---:|
| Domain | financial annual reports |
| Documents | 6 |
| Pages | 2,942 |
| English queries | 309 |
| Mean relevant pages per query | 4.7 |
| Page visual tokens | 1,024 |
| Dataset checksum | `4c2c0f3687668c5d` |

### 5.2 Encoders and exported representations

The fine retriever is `vidore/colpali-v1.3`; each page is stored as a matrix of normalized visual token embeddings, and each query as normalized token embeddings. The dense retriever is `openbmb/VisRAG-Ret`; each page and query has one normalized vector. Model inference is performed once during export. All matrix experiments reuse the same float32 embeddings, eliminating encoder variation across ablations.

The ColPali exporter reconstructs approximate two-dimensional token positions by placing exported token indices on a normalized square grid. These coordinates support the layout-coverage ablation but are not exact PaliGemma patch coordinates.

### 5.3 Compared systems

| ID | System | Candidate generation | Fine scoring | Recovery |
|---|---|---|---|---|
| B0 | VisRAG-Ret | dense, all pages | none | no |
| B1 | ColPali full | all pages | all 1,024 tokens | no |
| B2--B4 | Fixed-32/64/128 | ColPali mean, Top-50 | fixed relevant tokens | no |
| A1 | Adaptive budget | ColPali mean, Top-50 | adaptive relevance-only | no |
| A2 | Adaptive MMR | ColPali mean, Top-50 | adaptive relevance/diversity/coverage | no |
| A3 | Adaptive fallback | ColPali mean, Top-50 | adaptive MMR | Top-100 full |
| Ours | AdaColRAG | VisRAG, Top-50 | adaptive MMR + dense fusion | Top-100 full |
| Ours+OCR | AdaColRAG-OCR | VisRAG, Top-50 | adaptive MMR + dense/OCR fusion | Top-100 full |

The revised matrix runner attaches dense embeddings only to B0, AdaColRAG, and AdaColRAG-OCR. Therefore the fixed-budget and adaptive ablations are ColPali-only experiments.

### 5.4 Metrics and timing

We report nDCG@5, nDCG@10, Recall@1/5/10, and MRR@10 using graded qrels for nDCG and positive relevance for recall and MRR. Efficiency measures include tokens per scoring operation, within-scored-page token reduction, visual-token work per query, scoring operations per query, mean latency, p50 latency, and p95 latency.

Latency is measured with `time.perf_counter()` around retrieval over preloaded embeddings in a single Python process using NumPy. It includes dense or mean-vector coarse scoring, token selection, late interaction, fusion, fallback, and sorting. It excludes image encoding, query model encoding, disk loading, network transfer, and downstream generation. The archived JSON files do not record CPU model, thread count, or repeated-run variance; latency should therefore be interpreted as a within-run implementation comparison rather than a portable serving benchmark.

### 5.5 Hyperparameters

| Parameter | Value |
|---|---:|
| Initial candidate pool | 50 |
| Fallback pool | 100 |
| Minimum tokens | 32 |
| Maximum tokens | 128 |
| Budget multiple | 8 |
| Relevance weight | 1.00 |
| Redundancy weight | 0.25 |
| Coverage weight | 0.10 |
| Confidence threshold | 0.48 |
| Dense fusion weight | 0.15 |
| Top-\(k\) returned | 10 |

No retriever is trained on the Finance EN evaluation data. However, the operating point was not selected on a separate development split, so the current experiment should not be used to claim out-of-domain hyperparameter generalization.

## 6 Results

### 6.1 Complete retrieval matrix

| Method | nDCG@5 | nDCG@10 | R@5 | R@10 | MRR@10 | Tokens/op. | Local token red. | Mean latency (ms) | p95 (ms) | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VisRAG-Ret | 0.4744 | 0.5099 | 0.4559 | **0.5755** | **0.6501** | 0.0 | n/a | **3.6** | **3.7** | 0% |
| ColPali full | 0.4374 | 0.4579 | 0.4084 | 0.4978 | 0.5813 | 1024.0 | 0.00% | 14426.4 | 77743.7 | 0% |
| Fixed-32 | 0.2487 | 0.2676 | 0.2379 | 0.3063 | 0.3586 | 32.0 | 96.88% | 277.4 | 1517.3 | 0% |
| Fixed-64 | 0.2977 | 0.3146 | 0.2943 | 0.3597 | 0.4105 | 64.0 | 93.75% | 328.1 | 1665.9 | 0% |
| Fixed-128 | 0.3310 | 0.3513 | 0.3095 | 0.3854 | 0.4545 | 128.0 | 87.50% | 375.5 | 1705.3 | 0% |
| Adaptive budget | 0.3273 | 0.3458 | 0.3072 | 0.3783 | 0.4489 | 114.2 | 88.85% | 402.8 | 1728.3 | 0% |
| Adaptive MMR | 0.3472 | 0.3642 | 0.3245 | 0.3919 | 0.4770 | 114.2 | 88.85% | 784.0 | 2036.1 | 0% |
| Adaptive fallback | 0.3962 | 0.4133 | 0.3690 | 0.4462 | 0.5396 | 702.7 | 31.38% | 1254.9 | 4822.6 | 91.59% |
| **AdaColRAG** | **0.4827** | 0.5108 | **0.4528** | 0.5587 | 0.6386 | 540.2 | 47.25% | 1024.0 | 3557.7 | 44.01% |
| AdaColRAG + OCR | 0.4817 | **0.5141** | 0.4462 | 0.5700 | 0.6379 | 559.4 | 45.37% | 1056.8 | 3757.8 | 47.90% |

AdaColRAG improves nDCG@5 over exhaustive full-token ColPali by 0.0453 absolute, or 10.35% relative. Mean retrieval-stage latency falls from 14,426 ms to 1,024 ms, a 92.90% reduction. The local token-reduction metric is 47.25%, below the original 50% engineering threshold by 2.75 percentage points. This threshold failure is an operating-point outcome, not evidence that the retrieval result is invalid.

### 6.2 Candidate and token work

Exhaustive ColPali scores 2,942 pages and processes 3,012,608 page tokens per query. AdaColRAG performs 94.01 scoring operations and processes 50,785.76 tokens per query on average. Thus, relative to exhaustive ColPali:

\[
R_{\mathrm{operations}}=96.80\%,
\qquad
R_{\mathrm{system}}=98.31\%.
\]

These system-level reductions combine candidate pruning and token selection. They do not represent storage savings because the complete ColPali index is retained.

| Method | Scoring operations/query | Visual tokens/query | Reduction vs. full corpus |
|---|---:|---:|---:|
| ColPali full | 2942.00 | 3,012,608 | 0.00% |
| Fixed-128 | 50.00 | 6,400 | 99.79% |
| Adaptive MMR | 50.00 | 5,710 | 99.81% |
| AdaColRAG | 94.01 | 50,786 | 98.31% |
| AdaColRAG + OCR | 97.90 | 54,762 | 98.18% |

**Figure 4 placeholder -- Quality-efficiency frontier.** Create a scatter plot with mean retrieval-stage latency on a logarithmic horizontal axis and nDCG@5 on the vertical axis. Plot all ten systems. Use different markers for dense-only, exhaustive multi-vector, fixed budgets, adaptive ablations, and complete methods. Annotate VisRAG-Ret at the extreme low-latency point, ColPali full at the extreme high-latency point, and AdaColRAG near the upper-middle region. Add a second panel with full-corpus visual tokens per query on a logarithmic axis.

### 6.3 What each component contributes

**Fixed compression is efficient but loses quality.** Increasing a fixed budget from 32 to 128 improves nDCG@5 from 0.2487 to 0.3310, but remains 0.1065 below full ColPali. This confirms that aggressive query-conditioned relevance selection alone cannot recover exhaustive retrieval when candidate generation is weak.

**The current budget controller does not yet outperform a matched fixed budget.** Adaptive budget uses 114.2 tokens per scoring operation and obtains 0.3273 nDCG@5, slightly below Fixed-128 at 0.3310. Its budgets are concentrated at the high end: 1 query receives 96 tokens, 18 receive 104, 180 receive 112, and 110 receive 120. The controller therefore uses only a narrow part of its nominal 32--128 range. A stronger paper claim should focus on query-aware selection and selective recovery rather than asserting that the present complexity estimator alone is superior.

**Diversity and coverage help at constant token use.** Adaptive MMR and Adaptive budget both use 114.2 tokens per operation and have identical 88.85% local reduction, but MMR raises nDCG@5 from 0.3273 to 0.3472 and MRR@10 from 0.4489 to 0.4770. The additional CPU selection cost increases mean latency from 402.8 to 784.0 ms.

**Fallback recovers effectiveness but can dominate cost.** Without VisRAG, 91.59% of queries trigger fallback, reducing local token savings to 31.38%. With dense candidate generation and fusion, the fallback rate drops to 44.01% and nDCG@5 rises to 0.4827.

### 6.4 Fallback behavior

AdaColRAG triggers fallback for 136 of 309 queries.

| Query group | Queries | Mean budget | Operations/query | Tokens/query | Local token reduction | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| No fallback | 173 | 113.7 | 50 | 5,683 | 88.90% | 788.5 ms |
| Fallback | 136 | 115.2 | 150 | 108,159 | 29.58% | 1323.5 ms |

The 44.01% fallback rate explains why the aggregate local reduction is 47.25% even though the initial compressed pass removes almost 89% of tokens. A threshold of 0.48 is conservative. Threshold sensitivity should be evaluated on development data rather than adjusted after inspecting test outcomes.

**Figure 5 placeholder -- Fallback cost decomposition.** Use two stacked bars for non-fallback and fallback queries. Split each bar into initial Top-50 compressed tokens and Top-100 full-token recovery. Add labels for query count, local reduction, and mean latency. A small inset should show the confidence distribution with the threshold at 0.48.

### 6.5 Comparison with VisRAG-Ret

VisRAG-Ret is a strong baseline. AdaColRAG improves nDCG@5 by 0.0083, but VisRAG has higher MRR@10 and Recall@10 and is much faster in the current CPU embedding-stage implementation. The complete method therefore does not dominate dense retrieval on every metric. Its current advantage is improved early-rank nDCG together with token-level evidence selection and a recovery path. A broad superiority claim would require additional domains and downstream answer-generation evaluation.

### 6.6 OCR fusion

Adding OCR lexical fusion changes nDCG@5 from 0.4827 to 0.4817, nDCG@10 from 0.5108 to 0.5141, and Recall@10 from 0.5587 to 0.5700. It also increases fallback from 44.01% to 47.90% and latency from 1,024 to 1,057 ms. OCR is therefore best treated as an optional depth-recall extension, not the primary method.

## 7 Reproducibility and Code Audit

We conducted a static audit of the retrieval and evaluation code and a consistency audit of the ten archived result files.

### 7.1 Checks that passed

1. All result files contain 309 queries, 2,942 documents, and the same dataset checksum.
2. The metric implementation uses graded DCG, positive-relevance recall, and reciprocal rank in their standard forms.
3. Document and query embeddings are L2-normalized on load.
4. The revised matrix runner attaches VisRAG dense embeddings only to experiments that explicitly require them; dense signals are not present in the fixed-budget or ColPali-only adaptive result metadata.
5. Exhaustive ColPali scores all pages, while fixed and adaptive configurations use the declared candidate pools.
6. Fallback accounting includes both the initial compressed pass and the recovery pass, preventing the reported local token reduction from hiding recomputation.
7. Query-level rankings, budgets, confidence values, latency, operation counts, and fallback decisions are retained in every result.

### 7.2 Unresolved validation items

**ColPali checkpoint loading.** During export, the adapter checkpoint emitted `MISSING` and `UNEXPECTED` keys for `custom_text_proj` LoRA parameters. The official model card permits loading `vidore/colpali-v1.3` with the same API used by the exporter, so the warning may be a library key-remapping issue rather than a failed model. Nevertheless, the final baseline should be verified by checking that the projection LoRA weights are present and nonzero or by reproducing the experiment with the official merged checkpoint `vidore/colpali-v1.3-merged`. Until then, numerical comparisons involving ColPali should be considered implementation results rather than independently certified benchmark scores.

**Approximate layout positions.** The selector's spatial coordinates are reconstructed from token order on a square grid. This is sufficient for a coverage heuristic but does not justify claims about exact semantic regions or true encoder patch geometry.

**Timing metadata.** Hardware, NumPy thread count, warm-up, and repeated trials are absent from the JSON metadata. The relative latency differences are large, but portable systems claims require a controlled timing protocol.

**Single-domain evaluation.** Finance EN is one of several ViDoRe V3 domains. Generalization to industrial, pharmaceutical, scientific, human-resources, energy, physics, and French financial documents remains unknown.

**Candidate-stage confounding.** AdaColRAG changes both candidate generation and token scoring relative to full ColPali. Additional controlled systems using identical VisRAG Top-50 candidates with full tokens, fixed-128 tokens, adaptive relevance, and adaptive MMR are needed to isolate candidate pruning, compression, fusion, and fallback.

**No uncertainty estimates.** Each configuration was evaluated once. Paired bootstrap confidence intervals or randomization tests require per-query relevance scores and repeated timing measurements.

## 8 Limitations and Threats to Validity

The current results support the narrower claim that query-time token selection with dense candidate generation and selective fallback can produce a strong quality--cost operating point on one financial benchmark. They do not yet establish universal superiority.

First, a single benchmark may overrepresent recurring report structures. Second, the same test collection was used to report the chosen operating point, so threshold and fusion-weight sensitivity should be moved to a development split. Third, exhaustive ColPali and AdaColRAG use different candidate-generation strategies; the measured 92.90% latency reduction is a pipeline result rather than a pure token-selection result. Fourth, dense-only VisRAG is competitive and superior on some metrics. Fifth, current latency is CPU NumPy time over precomputed embeddings and excludes model encoding and production indexing. Sixth, full embeddings remain stored, so AdaColRAG does not reduce index storage. Seventh, the ColPali adapter warning must be resolved before publication. Finally, approximate token positions may limit the interpretation of the layout term.

These limitations affect the strength and breadth of the claim, not the internal consistency of the archived JSON results.

## 9 Ethical Considerations

The study evaluates retrieval over publicly released financial reports and benchmark queries. The method does not generate financial advice and should not be deployed as a decision-making system without source verification. Retrieval errors may omit qualifying disclosures or surface misleading pages; downstream systems should preserve provenance and allow users to inspect original documents. The benchmark and model licenses must be respected. Query traces can contain sensitive text in private deployments and should be protected through access control and retention policies. Efficiency claims should include the energy and hardware assumptions under which they were measured.

## 10 Conclusion

AdaColRAG treats visual-token use as a query-time retrieval policy. Dense candidate generation identifies a small page set, adaptive selection retains query-relevant and nonredundant evidence, and confidence-aware fallback recovers uncertain cases. On ViDoRe V3 Finance EN, the current implementation reaches 0.4827 nDCG@5, exceeding exhaustive ColPali by 0.0453 and VisRAG-Ret by 0.0083. It reduces retrieval-stage mean CPU latency by 92.90% relative to exhaustive ColPali and reduces full-corpus visual-token work by 98.31%, while removing 47.25% of tokens within the scoring operations actually performed.

The result is sufficiently strong to motivate a paper, but the scientifically defensible novelty is the **query-conditioned scoring policy and confidence-aware recovery**, not merely passing a predefined 50% compression threshold. Before a WSDM main-track submission, the most important next steps are checkpoint verification, controlled candidate-set baselines, cross-domain replication, and statistical uncertainty. The current manuscript reports the available evidence without treating those missing validations as completed.

## References

[1] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2025. **ColPali: Efficient Document Retrieval with Vision Language Models.** In *International Conference on Learning Representations (ICLR)*.

[2] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. **VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents.** In *International Conference on Learning Representations (ICLR)*.

[3] António Loison, Quentin Macé, Antoine Edy, Victor Xing, Tom Balough, Gabriel de Souza P. Moreira, Bo Liu, Manuel Faysse, Céline Hudelot, and Gautier Viaud. 2026. **ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios.** In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL), Volume 1: Long Papers*.

[4] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. 2024. **Unifying Multimodal Retrieval via Document Screenshot Embedding.** In *Proceedings of EMNLP*, 6492--6505. DOI: 10.18653/v1/2024.emnlp-main.373.

[5] Yubo Ma, Jinsong Li, Yuhang Zang, Xiaobao Wu, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2025. **Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings.** In *Findings of ACL*, 19568--19580. DOI: 10.18653/v1/2025.findings-acl.1003.

[6] Zheng Liu, Ze Liu, Zhengyang Liang, Junjie Zhou, Shitao Xiao, Chao Gao, Chen Jason Zhang, and Defu Lian. 2025. **Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval.** In *Proceedings of ACL*, 19238--19261. DOI: 10.18653/v1/2025.acl-long.943.

[7] Hao Sun, Yingyan Hou, Jiayan Guo, Bo Wang, Chunyu Yang, Jinsong Ni, and Yan Zhang. 2025. **Unveil: Unified Visual-Textual Integration and Distillation for Multi-modal Document Retrieval.** In *Proceedings of ACL*, 23935--23945. DOI: 10.18653/v1/2025.acl-long.1166.

[8] Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. 2025. **ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents.** In *Proceedings of EMNLP*, 9113--9134. DOI: 10.18653/v1/2025.emnlp-main.464.

[9] Kuicai Dong, Yujing Chang, Derrick Goh Xin Deik, Dexun Li, Ruiming Tang, and Yong Liu. 2025. **MMDocIR: Benchmarking Multimodal Retrieval for Long Documents.** In *Proceedings of EMNLP*, 30971--31005. DOI: 10.18653/v1/2025.emnlp-main.1576.

[10] Thong Nguyen, Yibin Lei, Jia-Huei Ju, and Andrew Yates. 2025. **SERVAL: Surprisingly Effective Zero-Shot Visual Document Retrieval Powered by Large Vision and Language Models.** In *Proceedings of EMNLP*, 30807--30822. DOI: 10.18653/v1/2025.emnlp-main.1568.

[11] Omar Khattab and Matei Zaharia. 2020. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.** In *Proceedings of SIGIR*, 39--48. DOI: 10.1145/3397271.3401075.

[12] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.** In *Proceedings of NAACL-HLT*, 3715--3734. DOI: 10.18653/v1/2022.naacl-main.272.

[13] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. **Dense Passage Retrieval for Open-Domain Question Answering.** In *Proceedings of EMNLP*, 6769--6781. DOI: 10.18653/v1/2020.emnlp-main.550.

[14] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.** In *NeurIPS Datasets and Benchmarks Track*.

[15] Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. **MTEB: Massive Text Embedding Benchmark.** In *Proceedings of EACL*, 2014--2037. DOI: 10.18653/v1/2023.eacl-main.148.

[16] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. **Retrieval Augmented Language Model Pre-Training.** In *Proceedings of ICML*, 3929--3938.

[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** In *Advances in Neural Information Processing Systems 33*.

[18] Jaime G. Carbonell and Jade Goldstein. 1998. **The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.** In *Proceedings of SIGIR*, 335--336. DOI: 10.1145/290941.291025.

[19] Yonatan Geifman and Ran El-Yaniv. 2017. **Selective Classification for Deep Neural Networks.** In *Advances in Neural Information Processing Systems 30*, 4878--4887.

[20] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. **LayoutLM: Pre-training of Text and Layout for Document Image Understanding.** In *Proceedings of KDD*, 1192--1200.

[21] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. 2021. **LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding.** In *Proceedings of ACL-IJCNLP*, 2579--2591. DOI: 10.18653/v1/2021.acl-long.201.

[22] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. 2021. **DocFormer: End-to-End Transformer for Document Understanding.** In *Proceedings of ICCV*, 993--1003.

[23] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2022. **OCR-Free Document Understanding Transformer.** In *Proceedings of ECCV*.

[24] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. 2021. **DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification.** In *Advances in Neural Information Processing Systems 34*.

[25] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. 2022. **Not All Patches Are What You Need: Expediting Vision Transformers via Token Reorganizations.** In *International Conference on Learning Representations (ICLR)*.

[26] Michael S. Ryoo, A. J. Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. 2021. **TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?** In *Advances in Neural Information Processing Systems 34*.

[27] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2023. **Token Merging: Your ViT but Faster.** In *International Conference on Learning Representations (ICLR)*.

[28] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. **RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.** In *Proceedings of NAACL-HLT*, 5835--5847. DOI: 10.18653/v1/2021.naacl-main.466.

[29] Ikuya Yamada, Akari Asai, and Hannaneh Hajishirzi. 2021. **Efficient Passage Retrieval with Hashing for Open-domain Question Answering.** In *Proceedings of ACL-IJCNLP*, 979--986. DOI: 10.18653/v1/2021.acl-short.123.

[30] Ye Liu, Kazuma Hashimoto, Yingbo Zhou, Semih Yavuz, Caiming Xiong, and Philip S. Yu. 2021. **Dense Hierarchical Retrieval for Open-domain Question Answering.** In *Findings of EMNLP*, 188--200. DOI: 10.18653/v1/2021.findings-emnlp.19.
