# AdaColRAG: Redundancy-Aware Visual Token Selection and Dense–Late Interaction Fusion for Efficient Document Retrieval

> **Anonymous WSDM manuscript source.** Finance EN is used only for development sensitivity analysis. Industrial, Pharmaceuticals, and Finance FR are held-out evaluations. Tables between `AUTO` markers are generated from validated result JSON files and must not be edited manually.

## Abstract

Visual document retrieval preserves tables, charts, typography, and spatial relations by searching rendered pages directly. Multi-vector retrievers such as ColPali retain fine-grained page evidence through token-level late interaction, but exhaustive scoring requires comparing every query with every visual token in the corpus. We present **AdaColRAG**, a training-free two-stage retriever that combines dense page retrieval, redundancy-aware query-time visual-token selection, and dense–late interaction score fusion. VisRAG first retrieves 50 candidate pages. For each candidate, query relevance prefilters 448 of 1,024 stored visual tokens, after which a maximal-marginal-relevance selector retains 112 tokens while suppressing redundant evidence. ColPali scores the selected tokens, and its late-interaction score is fused with the dense candidate score using a development-selected weight of 0.25. The full visual index is retained; selection changes only online scoring. Across Finance EN, Industrial, Pharmaceuticals, and Finance FR, the final method reaches 0.4656 macro nDCG@5, compared with 0.4246 for exhaustive ColPali and 0.4143 for dense VisRAG, while reducing local page-token scoring by 89.06% and full-corpus visual-token work by approximately 99.81%. Controlled experiments show that dense candidate generation, redundancy suppression, and dense–late interaction fusion are the supported components. Adaptive budgeting does not outperform a matched fixed budget, approximate layout coverage can degrade retrieval, and confidence-triggered full-token recovery provides no consistent gain. These negative findings lead to a simpler final method with stronger effectiveness and efficiency.

**Keywords:** visual document retrieval; multimodal retrieval; late interaction; visual-token selection; maximal marginal relevance; dense retrieval; retrieval-augmented generation

## 1 Introduction

Enterprise retrieval increasingly operates over annual reports, technical manuals, pharmaceutical documents, presentations, invoices, and forms. Relevant evidence can depend on table structure, chart geometry, typography, and the association between distant page regions. OCR-first pipelines may discard these signals or propagate parsing errors. Visual document retrieval therefore treats the rendered page as the primary search unit [1–10].

ColPali [1] adapts ColBERT-style late interaction [11, 12] to document images. Each page is represented by many visual embeddings, and each query token retrieves its strongest matching page token. This preserves local evidence but makes exhaustive search expensive: the exported index used in this work contains 1,024 visual tokens per page, so searching thousands of pages requires millions of token comparisons for every query.

Existing visual-index compression methods remove or merge page tokens before the query is known [5]. Such methods reduce storage but risk deleting evidence needed by future queries. We study a complementary online setting. AdaColRAG retains the complete frozen multi-vector index and changes only the query-time scoring policy. A dense retriever first restricts search to a small page set; the fine retriever then scores a query-conditioned subset of page tokens.

The final design follows an evidence-driven simplification process. A verified Finance EN development study showed that a query-complexity budget controller did not significantly outperform a fixed budget with comparable token use. Redundancy-aware token selection improved over relevance-only selection, while dense-score fusion provided the most stable component gain. Subsequent four-domain experiments showed that approximate layout coverage was neutral or harmful and that confidence-triggered full-token recovery consumed substantially more tokens without consistent effectiveness gains. We therefore remove adaptive budgeting, layout coverage, and fallback from the primary method.

AdaColRAG contains three main stages. First, frozen VisRAG-Ret [2] retrieves the Top-50 pages. Second, relevance-prefiltered MMR selects 112 nonredundant visual tokens from each candidate. Third, frozen ColPali late interaction reranks the selected evidence and is fused with the dense score. The method requires no retraining, does not alter stored embeddings, and exposes all candidate, token, score, and latency traces.

Our contributions are:

1. **Redundancy-aware query-time visual-token selection.** We introduce a training-free selector that retains query-relevant but nonduplicative page evidence under a fixed, controlled token budget.
2. **Efficient relevance-prefiltered MMR.** The selector runs on a relevance Top-448 pool rather than all 1,024 visual tokens, matching exact MMR effectiveness on the development collection while reducing selector work.
3. **Dense–late interaction fusion.** We show that global dense similarity and local ColPali MaxSim evidence are consistently complementary across English financial, industrial, pharmaceutical, and French financial collections.
4. **Controlled evidence-driven ablation.** Identical-candidate and identical-budget experiments isolate candidate generation, redundancy suppression, layout coverage, score fusion, fallback, and OCR; unsupported components are reported as negative results rather than retained in the method.
5. **A reproducible cross-domain evaluation.** The repository verifies merged checkpoint provenance, pins dataset revisions, preserves query-level traces, computes paired bootstrap intervals, and provides single-process steady-state timing over preloaded embeddings.

**Figure 1 placeholder — Final system overview.** Draw a left-to-right pipeline: query and page corpus → frozen VisRAG dense retrieval → Top-50 candidate pages → query-to-token relevance scoring → Top-448 token prefilter → redundancy-aware MMR selecting 112 tokens → frozen ColPali MaxSim → min–max normalized dense/late-interaction fusion with α=0.25 → final ranking. Mark the full page index as unchanged and all encoders as frozen. Do not include adaptive budgeting, layout coverage, OCR, or fallback in the primary path.

## 2 Related Work

### 2.1 Visual document retrieval

Document Screenshot Embedding [4] represents rendered pages with a single vector, whereas ColPali [1] uses multi-vector late interaction to retain local evidence. VisRAG [2] integrates visual retrieval into multimodal RAG. UniSE [6], Unveil [7], ViDoRAG [8], MMDocIR [9], and SERVAL [10] explore unified visual search, distillation, iterative reasoning, long-document retrieval, and zero-shot retrieval. ViDoRe V3 [3] evaluates retrieval on professional domains and multilingual queries.

Light-ColPali [5] studies query-independent patch reduction and token merging to reduce index storage. AdaColRAG instead preserves the complete index and reduces online scoring after observing the query. The two approaches target different costs and are potentially complementary.

### 2.2 Dense and late-interaction retrieval

Dense retrieval [13–17, 28–30] maps each query and document to one vector and enables efficient candidate generation. Late-interaction systems such as ColBERT and ColBERTv2 [11, 12] preserve multiple contextualized vectors and improve local matching at higher scoring cost. AdaColRAG explicitly combines these representations: dense vectors provide global candidate recall, while selected ColPali vectors provide fine-grained evidence.

### 2.3 Token reduction and query-aware selection

DynamicViT [24], token reorganization [25], TokenLearner [26], and Token Merging [27] reduce computation inside vision models. These methods typically operate during encoding and are not conditioned on a retrieval query. AdaColRAG leaves the encoder and stored index unchanged. It selects exported page tokens according to the current query. The selection objective is related to maximal marginal relevance [18], but operates over visual page embeddings rather than retrieved documents.

## 3 Problem Formulation

Let a query be represented by normalized token embeddings

\[
Q=[q_1,\ldots,q_m]\in\mathbb{R}^{m\times d},
\]

and a page by normalized visual embeddings

\[
D=[d_1,\ldots,d_n]\in\mathbb{R}^{n\times d}.
\]

Full late interaction computes

\[
s_{\mathrm{LI}}(Q,D)=\frac{1}{m}\sum_{i=1}^{m}\max_{1\leq j\leq n}q_i^\top d_j.
\]

For a corpus of \(N\) pages, exhaustive scoring has work proportional to \(O(Nmnd)\). A two-stage retriever first selects \(C\ll N\) candidate pages. AdaColRAG then selects a query-dependent subset \(S_Q(D)\) with fixed size \(K\ll n\):

\[
\tilde{s}_{\mathrm{LI}}(Q,D)=\frac{1}{m}\sum_{i=1}^{m}\max_{j\in S_Q(D)}q_i^\top d_j.
\]

We distinguish two efficiency measures. **Local token reduction** measures compression within scored page operations:

\[
R_{\mathrm{local}}=1-\frac{\sum_q T_q}{\sum_q T_q^{\mathrm{full}}}.
\]

**System visual-token work reduction** compares actual scoring work with exhaustive full-token ColPali:

\[
R_{\mathrm{system}}=1-\frac{\mathbb{E}_q[T_q]}{N\bar n},
\]

where \(ar n\) is the mean stored token count per page. This quantity combines candidate pruning and token selection. It is not an index-storage reduction.

## 4 Method

### 4.1 Dense candidate generation

VisRAG-Ret supplies normalized dense representations \(u_Q\) and \(u_D\). Cosine similarity retrieves \(C=50\) candidate pages. Candidate-source controls replace VisRAG with normalized mean-pooled ColPali tokens:

\[
\bar q=\operatorname{norm}\left(\frac{1}{m}\sum_iq_i\right),\qquad
\bar d=\operatorname{norm}\left(\frac{1}{n}\sum_jd_j\right).
\]

These controls quantify how much improvement originates from dense candidate recall rather than fine scoring.

### 4.2 Fixed token budget

The final method uses \(K=112\) page tokens per candidate. Finance EN development experiments compare 64, 96, 112, and 128 tokens. Fixed-112 significantly improves over Fixed-96, while Fixed-128 provides no significant additional gain. A previous adaptive controller produces almost the same average budget but does not significantly outperform Fixed-112, so adaptive budgeting is excluded from the final method.

### 4.3 Relevance prefiltering

For visual token \(d_j\), query relevance is

\[
r_j=\max_i q_i^\top d_j.
\]

Exact greedy MMR over all 1,024 page tokens is expensive. We retain the \(P\) most relevant tokens,

\[
P=\min(n,\gamma K),
\]

with \(\gamma=4\), yielding 448 prefiltered tokens. Finance EN experiments show that \(\gamma=4\) significantly outperforms \(\gamma=2\) and is statistically indistinguishable from exact all-token MMR.

### 4.4 Redundancy-aware token selection

Selection begins with the highest-relevance token. At each subsequent step, AdaColRAG chooses

\[
j^*=\arg\max_{j\notin S}
\left[
\lambda_r r_j-\lambda_c\max_{k\in S}d_j^\top d_k
\right].
\]

The first term retains query evidence. The second suppresses near-duplicate visual tokens, with \(\lambda_r=1\) and \(\lambda_c=0.25\). Maximum redundancy is updated incrementally after each selected token. The final method does not use approximate spatial coverage: four-domain experiments show no stable benefit, and Pharmaceuticals exhibits a statistically significant degradation when the layout term is added.

**Figure 2 placeholder — Redundancy-aware selection.** Panel A shows query-to-token relevance over a 32×32 page-token grid. Panel B keeps the Top-448 tokens. Panel C selects 112 tokens, comparing relevance-only selection with redundancy-aware selection. Highlight repeated table cells or repeated visual patterns that receive high relevance but are suppressed as redundant. Add a small complexity comparison between exact MMR and relevance-prefiltered MMR.

### 4.5 Dense–late interaction fusion

Selected-token late-interaction and dense scores are min–max normalized over the candidate set:

\[
s_{\mathrm{fuse}}=(1-\alpha)\hat{s}_{\mathrm{LI}}+\alpha\hat{s}_{\mathrm{dense}}.
\]

Finance EN development experiments compare \(\alpha\in\{0.05,0.15,0.25\}\). The value \(\alpha=0.25\) significantly outperforms 0.15 and subsequently improves over 0.15 on all three held-out collections. The final method therefore uses \(\alpha=0.25\).

## 5 Experimental Protocol

### 5.1 Datasets and split roles

We use four ViDoRe V3 collections [3]. Finance EN is the only development collection. Industrial, Pharmaceuticals, and Finance FR are held-out evaluations.

| Dataset | Domain | Language | Role | Pages | Queries |
|---|---|---|---|---:|---:|
| Finance EN | annual financial reports | English | development and continuity | 2,942 | 309 |
| Industrial | industrial technical documents | English | held-out domain transfer | 5,244 | 283 |
| Pharmaceuticals | pharmaceutical/scientific documents | English | held-out table and scientific transfer | 2,313 | 364 |
| Finance FR | annual financial reports | French | held-out multilingual transfer | 2,384 | 320 |

Dataset revisions and parquet inventories are pinned. Generated manifests record data checksums, page counts, query counts, and qrel counts.

### 5.2 Encoders and provenance

The fine retriever is the verified merged checkpoint `vidore/colpali-v1.3-merged`, loaded from a local snapshot with `local_files_only=True`. The dense retriever is `openbmb/VisRAG-Ret`. ColPali exports normalized multi-vector page/query arrays; VisRAG exports normalized single vectors. All experiments reuse identical embeddings within a dataset.

The workflow records Git commit, CPU, GPU, package versions, BLAS thread settings, checkpoint manifests, dataset checksums, query-level rankings, selected-token counts, and latency traces.

### 5.3 Final controlled systems

The final paper matrix contains external baselines, candidate controls, token controls, supported components, and negative ablations.

| System | Candidate stage | Fine scoring | Fusion | Purpose |
|---|---|---|---|---|
| `visrag_dense` | full-corpus VisRAG | none | dense only | single-vector baseline |
| `colpali_full` | full corpus | all 1,024 tokens | none | exhaustive multi-vector baseline |
| `colpali_mean_top50_full` | ColPali mean Top-50 | all tokens | none | candidate-source control |
| `visrag_top50_full` | VisRAG Top-50 | all tokens | none | dense candidate control |
| `colpali_mean_top50_fixed_112` | ColPali mean Top-50 | relevance Top-112 | none | candidate control at fixed work |
| `visrag_top50_fixed_112` | VisRAG Top-50 | relevance Top-112 | none | matched token baseline |
| `visrag_top50_mmr_redundancy` | VisRAG Top-50 | redundancy-aware 112 | none | selector contribution |
| `visrag_top50_mmr_layout` | VisRAG Top-50 | layout-only 112 | none | negative layout ablation |
| `adacolrag_layout` | VisRAG Top-50 | redundancy + layout 112 | dense, α=0.25 | negative combined-layout ablation |
| `adacolrag_fallback` | VisRAG Top-50 | redundancy-aware 112, optional recovery | dense, α=0.25 | optional recovery ablation |
| `adacolrag` | VisRAG Top-50 | redundancy-aware 112 | dense, α=0.25 | final method |

Adaptive budgeting and OCR are retained only in archived experiments.

### 5.4 Metrics and statistics

Retrieval metrics are nDCG@5, nDCG@10, Recall@1/5/10, and MRR@10. Efficiency metrics are local token reduction, visual tokens processed per query, page-scoring operations, system token-work reduction, and retrieval-stage latency over preloaded embeddings.

Every primary effectiveness comparison uses paired query-level bootstrap resampling with 10,000 samples. We report the mean difference, 95% interval, two-sided bootstrap probability, and fraction of improved queries.

Absolute timing is measured in a single process. Embeddings are loaded once, each method receives at least one full warm-up evaluation, and 7 measured evaluations run in the same process with fixed CPU affinity and BLAS thread counts. We report the median run mean, run-level interquartile range, pooled query median, and pooled query P95. Model encoding, disk loading, network transfer, and downstream generation are excluded.

## 6 Results

### 6.1 Development choices and final-method selection

Finance EN development supports Fixed-112 over Fixed-96, prefilter factor 4 over factor 2, and dense fusion weight 0.25 over 0.15. Exact MMR does not improve over factor 4. Four-domain targeted experiments then compare redundancy-only versus redundancy-plus-layout selection and recovery versus no recovery.

<!-- BEGIN AUTO:FINAL_SELECTION -->

| Candidate | Macro nDCG@5 | Macro nDCG@10 | Macro local reduction | Macro fallback rate |
|---|---:|---:|---:|---:|
| final_mmr_dense025_fallback | 0.4643 | 0.4895 | 46.29% | 46.61% |
| final_mmr_dense025_no_fallback | 0.4636 | 0.4867 | 89.06% | 0.00% |
| final_redundancy_dense025_fallback | 0.4633 | 0.4890 | 45.91% | 47.44% |
| final_redundancy_dense025_no_fallback | 0.4656 | 0.4874 | 89.06% | 0.00% |

<!-- END AUTO:FINAL_SELECTION -->

The selected method is redundancy-only selection with dense weight 0.25 and no recovery. It has the highest macro nDCG@5, retains 89.06% local token reduction, and avoids the high token cost of recovery. Layout coverage is removed because it does not improve macro effectiveness and significantly degrades Pharmaceuticals. Recovery is removed because no dataset shows a stable significant gain under the final selector and fusion setting.

### 6.2 Main cross-domain effectiveness

<!-- BEGIN AUTO:FINAL_MAIN_RESULTS -->

| Dataset | Method | nDCG@5 | nDCG@10 | R@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| vidore_v3_finance_en | visrag_dense | 0.4744 | 0.5099 | 0.5755 | 0.6501 |
| vidore_v3_finance_en | colpali_full | 0.4387 | 0.4568 | 0.4891 | 0.5855 |
| vidore_v3_finance_en | visrag_top50_full | 0.4546 | 0.4858 | 0.5354 | 0.6067 |
| vidore_v3_finance_en | visrag_top50_fixed_112 | 0.4211 | 0.4490 | 0.5147 | 0.5590 |
| vidore_v3_finance_en | visrag_top50_mmr_redundancy | 0.4328 | 0.4686 | 0.5296 | 0.5794 |
| vidore_v3_finance_en | adacolrag_final | 0.4966 | 0.5254 | 0.5730 | 0.6615 |
| vidore_v3_industrial | visrag_dense | 0.3859 | 0.4075 | 0.4666 | 0.4982 |
| vidore_v3_industrial | colpali_full | 0.4584 | 0.4695 | 0.4832 | 0.5834 |
| vidore_v3_industrial | visrag_top50_full | 0.4619 | 0.4766 | 0.4944 | 0.5840 |
| vidore_v3_industrial | visrag_top50_fixed_112 | 0.4088 | 0.4304 | 0.4651 | 0.5234 |
| vidore_v3_industrial | visrag_top50_mmr_redundancy | 0.4437 | 0.4644 | 0.4915 | 0.5701 |
| vidore_v3_industrial | adacolrag_final | 0.4688 | 0.4819 | 0.5041 | 0.5886 |
| vidore_v3_pharmaceuticals | visrag_dense | 0.5635 | 0.5833 | 0.6390 | 0.6796 |
| vidore_v3_pharmaceuticals | colpali_full | 0.5599 | 0.5848 | 0.6381 | 0.6744 |
| vidore_v3_pharmaceuticals | visrag_top50_full | 0.5639 | 0.5935 | 0.6551 | 0.6830 |
| vidore_v3_pharmaceuticals | visrag_top50_fixed_112 | 0.5311 | 0.5588 | 0.6094 | 0.6467 |
| vidore_v3_pharmaceuticals | visrag_top50_mmr_redundancy | 0.5598 | 0.5833 | 0.6324 | 0.6806 |
| vidore_v3_pharmaceuticals | adacolrag_final | 0.6006 | 0.6240 | 0.6721 | 0.7220 |
| vidore_v3_finance_fr | visrag_dense | 0.2335 | 0.2651 | 0.3270 | 0.3400 |
| vidore_v3_finance_fr | colpali_full | 0.2414 | 0.2672 | 0.3221 | 0.3409 |
| vidore_v3_finance_fr | visrag_top50_full | 0.2641 | 0.2893 | 0.3556 | 0.3624 |
| vidore_v3_finance_fr | visrag_top50_fixed_112 | 0.2505 | 0.2792 | 0.3427 | 0.3478 |
| vidore_v3_finance_fr | visrag_top50_mmr_redundancy | 0.2583 | 0.2906 | 0.3633 | 0.3599 |
| vidore_v3_finance_fr | adacolrag_final | 0.2962 | 0.3183 | 0.3839 | 0.3861 |
| macro_average | visrag_dense | 0.4143 | 0.4414 | 0.5020 | 0.5420 |
| macro_average | colpali_full | 0.4246 | 0.4446 | 0.4831 | 0.5461 |
| macro_average | visrag_top50_full | 0.4361 | 0.4613 | 0.5101 | 0.5591 |
| macro_average | visrag_top50_fixed_112 | 0.4029 | 0.4293 | 0.4830 | 0.5192 |
| macro_average | visrag_top50_mmr_redundancy | 0.4236 | 0.4517 | 0.5042 | 0.5475 |
| macro_average | adacolrag_final | 0.4656 | 0.4874 | 0.5333 | 0.5896 |

<!-- END AUTO:FINAL_MAIN_RESULTS -->

The final method reaches 0.4966, 0.4688, 0.6006, and 0.2962 nDCG@5 on Finance EN, Industrial, Pharmaceuticals, and Finance FR, respectively, for a macro average of 0.4656. Its macro improvement is 0.0409 over exhaustive ColPali, 0.0512 over dense VisRAG, and 0.0294 over VisRAG Top-50 followed by full-token ColPali.

### 6.3 Paired significance against major baselines

<!-- BEGIN AUTO:FINAL_SIGNIFICANCE -->

| Dataset | Baseline | Baseline | Final | ΔnDCG@5 | 95% CI | p | Final better queries |
|---|---|---:|---:|---:|---:|---:|---:|
| vidore_v3_finance_en | visrag_dense | 0.4744 | 0.4966 | +0.0223 | [-0.0072, +0.0532] | 0.1478 | 33.01% |
| vidore_v3_finance_en | colpali_full | 0.4387 | 0.4966 | +0.0579 | [+0.0330, +0.0828] | 0.0000 | 38.51% |
| vidore_v3_finance_en | visrag_top50_full | 0.4546 | 0.4966 | +0.0421 | [+0.0217, +0.0631] | 0.0000 | 35.92% |
| vidore_v3_finance_en | visrag_top50_fixed_112 | 0.4211 | 0.4966 | +0.0755 | [+0.0543, +0.0984] | 0.0000 | 41.75% |
| vidore_v3_finance_en | visrag_top50_mmr_redundancy | 0.4328 | 0.4966 | +0.0639 | [+0.0450, +0.0837] | 0.0000 | 39.81% |
| vidore_v3_finance_en | adacolrag_no_fallback | 0.4796 | 0.4966 | +0.0170 | [+0.0058, +0.0282] | 0.0016 | 20.39% |
| vidore_v3_finance_en | adacolrag | 0.4813 | 0.4966 | +0.0153 | [+0.0011, +0.0297] | 0.0354 | 22.65% |
| vidore_v3_industrial | visrag_dense | 0.3859 | 0.4688 | +0.0829 | [+0.0526, +0.1137] | 0.0000 | 36.75% |
| vidore_v3_industrial | colpali_full | 0.4584 | 0.4688 | +0.0103 | [-0.0102, +0.0310] | 0.3382 | 27.56% |
| vidore_v3_industrial | visrag_top50_full | 0.4619 | 0.4688 | +0.0068 | [-0.0117, +0.0262] | 0.4800 | 24.38% |
| vidore_v3_industrial | visrag_top50_fixed_112 | 0.4088 | 0.4688 | +0.0599 | [+0.0327, +0.0886] | 0.0000 | 30.74% |
| vidore_v3_industrial | visrag_top50_mmr_redundancy | 0.4437 | 0.4688 | +0.0250 | [+0.0073, +0.0423] | 0.0056 | 27.56% |
| vidore_v3_industrial | adacolrag_no_fallback | 0.4578 | 0.4688 | +0.0109 | [-0.0013, +0.0231] | 0.0808 | 18.37% |
| vidore_v3_industrial | adacolrag | 0.4717 | 0.4688 | -0.0029 | [-0.0158, +0.0095] | 0.6434 | 17.67% |
| vidore_v3_pharmaceuticals | visrag_dense | 0.5635 | 0.6006 | +0.0371 | [+0.0155, +0.0592] | 0.0014 | 36.54% |
| vidore_v3_pharmaceuticals | colpali_full | 0.5599 | 0.6006 | +0.0408 | [+0.0212, +0.0609] | 0.0000 | 31.04% |
| vidore_v3_pharmaceuticals | visrag_top50_full | 0.5639 | 0.6006 | +0.0367 | [+0.0185, +0.0550] | 0.0000 | 30.77% |
| vidore_v3_pharmaceuticals | visrag_top50_fixed_112 | 0.5311 | 0.6006 | +0.0695 | [+0.0469, +0.0925] | 0.0000 | 35.71% |
| vidore_v3_pharmaceuticals | visrag_top50_mmr_redundancy | 0.5598 | 0.6006 | +0.0409 | [+0.0246, +0.0579] | 0.0000 | 30.77% |
| vidore_v3_pharmaceuticals | adacolrag_no_fallback | 0.5851 | 0.6006 | +0.0155 | [+0.0059, +0.0257] | 0.0016 | 17.86% |
| vidore_v3_pharmaceuticals | adacolrag | 0.5867 | 0.6006 | +0.0139 | [+0.0008, +0.0274] | 0.0384 | 22.25% |
| vidore_v3_finance_fr | visrag_dense | 0.2335 | 0.2962 | +0.0627 | [+0.0368, +0.0892] | 0.0000 | 34.38% |
| vidore_v3_finance_fr | colpali_full | 0.2414 | 0.2962 | +0.0548 | [+0.0304, +0.0799] | 0.0000 | 34.69% |
| vidore_v3_finance_fr | visrag_top50_full | 0.2641 | 0.2962 | +0.0321 | [+0.0134, +0.0509] | 0.0008 | 27.50% |
| vidore_v3_finance_fr | visrag_top50_fixed_112 | 0.2505 | 0.2962 | +0.0457 | [+0.0239, +0.0684] | 0.0000 | 28.12% |
| vidore_v3_finance_fr | visrag_top50_mmr_redundancy | 0.2583 | 0.2962 | +0.0379 | [+0.0191, +0.0570] | 0.0004 | 28.75% |
| vidore_v3_finance_fr | adacolrag_no_fallback | 0.2813 | 0.2962 | +0.0149 | [+0.0031, +0.0269] | 0.0146 | 20.31% |
| vidore_v3_finance_fr | adacolrag | 0.2855 | 0.2962 | +0.0107 | [-0.0028, +0.0239] | 0.1214 | 21.25% |

<!-- END AUTO:FINAL_SIGNIFICANCE -->

Primary claims are based on nDCG@5 intervals for the final method against VisRAG, exhaustive ColPali, VisRAG Top-50 full-token scoring, the matched Fixed-112 baseline, and redundancy-aware selection without fusion. We do not describe the final method as universally superior unless the paired interval for the specific dataset and baseline excludes zero.

### 6.4 Efficiency

<!-- BEGIN AUTO:FINAL_EFFICIENCY -->

| Dataset | Tokens/op. | Local reduction | Visual tokens/query | Page ops/query | System token-work reduction | Page-op reduction |
|---|---:|---:|---:|---:|---:|---:|
| vidore_v3_finance_en | 112.0 | 89.06% | 5600.0 | 50.0 | 99.81% | 98.30% |
| vidore_v3_industrial | 112.0 | 89.06% | 5600.0 | 50.0 | 99.90% | 99.05% |
| vidore_v3_pharmaceuticals | 112.0 | 89.06% | 5600.0 | 50.0 | 99.76% | 97.84% |
| vidore_v3_finance_fr | 112.0 | 89.06% | 5600.0 | 50.0 | 99.77% | 97.90% |

<!-- END AUTO:FINAL_EFFICIENCY -->

The final method always scores 50 candidates with 112 selected visual tokens, or 5,600 visual tokens per query. This yields 89.06% local token reduction relative to full-token scoring of the same pages and approximately 99.81% system visual-token work reduction relative to exhaustive full-corpus ColPali. The complete multi-vector index remains stored.

### 6.5 Single-process steady-state timing

<!-- BEGIN AUTO:FINAL_STEADY_TIMING -->

| Dataset | Method | Repeats | Median run mean (ms) | Run IQR (ms) | Query median (ms) | Query p95 (ms) | CV |
|---|---|---:|---:|---:|---:|---:|---:|
| vidore_v3_finance_en | adacolrag | 7 | 69.82 | 0.21 | 69.79 | 71.04 | 0.002 |
| vidore_v3_finance_en | colpali_full | 7 | 774.50 | 9.93 | 768.62 | 799.94 | 0.023 |
| vidore_v3_finance_en | visrag_dense | 7 | 26.01 | 1.74 | 27.85 | 39.57 | 0.038 |
| vidore_v3_finance_en | visrag_top50_full | 7 | 8.77 | 0.03 | 8.76 | 9.40 | 0.004 |
| vidore_v3_industrial | adacolrag | 7 | 74.95 | 0.21 | 74.90 | 76.16 | 0.003 |
| vidore_v3_industrial | colpali_full | 7 | 88060.51 | 34712.63 | 81432.15 | 202842.06 | 0.338 |
| vidore_v3_industrial | visrag_dense | 7 | 14.64 | 0.28 | 15.93 | 18.40 | 0.015 |
| vidore_v3_industrial | visrag_top50_full | 7 | 18.14 | 0.30 | 18.09 | 1382.59 | 2.372 |
| vidore_v3_pharmaceuticals | adacolrag | 7 | 69.44 | 0.45 | 69.42 | 70.77 | 0.007 |
| vidore_v3_pharmaceuticals | colpali_full | 7 | 622.79 | 3.89 | 620.67 | 660.76 | 0.014 |
| vidore_v3_pharmaceuticals | visrag_dense | 7 | 14.66 | 12.73 | 11.52 | 16.08 | 0.765 |
| vidore_v3_pharmaceuticals | visrag_top50_full | 7 | 8.59 | 0.03 | 8.58 | 9.25 | 0.010 |
| vidore_v3_finance_fr | adacolrag | 7 | 68.78 | 0.65 | 68.60 | 76.20 | 0.019 |
| vidore_v3_finance_fr | colpali_full | 7 | 659.20 | 3.44 | 656.51 | 696.53 | 0.008 |
| vidore_v3_finance_fr | visrag_dense | 7 | 15.68 | 0.11 | 15.97 | 16.05 | 0.004 |
| vidore_v3_finance_fr | visrag_top50_full | 7 | 8.80 | 0.95 | 8.84 | 13.16 | 0.071 |

<!-- END AUTO:FINAL_STEADY_TIMING -->

**Figure 3 placeholder — Quality–cost comparison.** Panel A plots nDCG@5 against deterministic visual-token work per query. Panel B plots nDCG@5 against single-process steady-state query P95 latency. Include VisRAG, full-corpus ColPali, VisRAG Top-50 full-token scoring, relevance Top-112, redundancy-aware selection without fusion, and final AdaColRAG.

### 6.6 Component analysis

Candidate controls show that VisRAG Top-50 substantially improves over ColPali mean-vector Top-50 under both full-token and fixed-token fine scoring. Under identical VisRAG candidates and fixed token count, redundancy suppression improves the matched relevance-only baseline, particularly on Industrial and Pharmaceuticals. Dense fusion provides the most stable component improvement. Approximate layout coverage and fallback are negative or optional ablations, not primary contributions.

**Figure 4 placeholder — Component attribution.** Show a waterfall from ColPali mean candidates to VisRAG candidates, relevance Top-112, redundancy-aware selection, and dense fusion. Plot dataset-specific nDCG@5 differences with paired 95% intervals. Place layout and fallback in a separate gray “unsupported additions” panel.

## 7 Reproducibility and Implementation

The repository separates model export environments from a NumPy retrieval environment. The final workflow:

1. validates stored main and targeted results;
2. computes final-method paired bootstrap comparisons against stored baselines;
3. loads each dataset’s embeddings once and performs in-process warm-up and repeated timing;
4. aggregates final retrieval, significance, efficiency, selection, and timing tables;
5. injects generated tables into this manuscript;
6. packages the complete lightweight evidence bundle without model weights, page images, parquet files, or embedding arrays.

Relevance-prefiltered MMR is deterministic. The legacy adaptive-budget controller, layout selector, fallback path, and OCR fusion remain available to reproduce reported ablations but are not used by the final method.

## 8 Limitations and Threats to Validity

First, Finance EN is used for development; unbiased generalization claims rely on Industrial, Pharmaceuticals, and Finance FR. Second, the method retains the full multi-vector index and therefore reduces online computation rather than storage. Third, dense VisRAG remains substantially faster and may be preferable when minimum latency is more important than fine-grained ranking quality. Fourth, the selector is implemented in NumPy and does not represent an optimized GPU or production serving kernel. Fifth, the study evaluates page retrieval rather than downstream answer generation. Sixth, the fixed 112-token budget and fusion weight are selected from a limited grid. Seventh, results apply to the evaluated ColPali and VisRAG checkpoints and may not transfer unchanged to other encoders. Finally, system token-work reduction is an analytical operation count and must be reported separately from measured latency.

## 9 Ethical Considerations

The benchmarks contain publicly released professional documents and retrieval annotations. The method does not generate financial, industrial, or pharmaceutical advice. Retrieval failures can omit qualifying evidence or surface misleading pages, so downstream systems should retain page provenance and allow inspection of original documents. Private deployments must protect queries, traces, and retrieved pages through access controls and retention policies. Dataset and model licenses must be respected.

## 10 Conclusion

AdaColRAG combines dense candidate generation, fixed-budget redundancy-aware visual-token selection, and dense–late interaction fusion. Evidence-driven ablation removes three initially plausible but unsupported mechanisms: adaptive budgeting, approximate layout coverage, and confidence-triggered recovery. The resulting method is simpler, more efficient, and more effective. Across four professional-document collections, it improves macro nDCG@5 over both exhaustive ColPali and dense VisRAG while scoring only 112 of 1,024 tokens on 50 candidate pages. The final system requires no retraining and preserves the complete visual index, providing a reproducible quality–cost operating point for visual document retrieval.

## References

[1] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2025. **ColPali: Efficient Document Retrieval with Vision Language Models.** In *International Conference on Learning Representations (ICLR)*.

[2] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. **VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents.** In *International Conference on Learning Representations (ICLR)*.

[3] António Loison, Quentin Macé, Antoine Edy, Victor Xing, Tom Balough, Gabriel de Souza P. Moreira, Bo Liu, Manuel Faysse, Céline Hudelot, and Gautier Viaud. 2026. **ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios.** In *Proceedings of ACL*.

[4] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. 2024. **Unifying Multimodal Retrieval via Document Screenshot Embedding.** In *Proceedings of EMNLP*, 6492–6505.

[5] Yubo Ma, Jinsong Li, Yuhang Zang, Xiaobao Wu, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2025. **Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings.** In *Findings of ACL*, 19568–19580.

[6] Zheng Liu, Ze Liu, Zhengyang Liang, Junjie Zhou, Shitao Xiao, Chao Gao, Chen Jason Zhang, and Defu Lian. 2025. **Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval.** In *Proceedings of ACL*, 19238–19261.

[7] Hao Sun, Yingyan Hou, Jiayan Guo, Bo Wang, Chunyu Yang, Jinsong Ni, and Yan Zhang. 2025. **Unveil: Unified Visual-Textual Integration and Distillation for Multi-modal Document Retrieval.** In *Proceedings of ACL*, 23935–23945.

[8] Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. 2025. **ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents.** In *Proceedings of EMNLP*, 9113–9134.

[9] Kuicai Dong, Yujing Chang, Derrick Goh Xin Deik, Dexun Li, Ruiming Tang, and Yong Liu. 2025. **MMDocIR: Benchmarking Multimodal Retrieval for Long Documents.** In *Proceedings of EMNLP*, 30971–31005.

[10] Thong Nguyen, Yibin Lei, Jia-Huei Ju, and Andrew Yates. 2025. **SERVAL: Surprisingly Effective Zero-Shot Visual Document Retrieval Powered by Large Vision and Language Models.** In *Proceedings of EMNLP*, 30807–30822.

[11] Omar Khattab and Matei Zaharia. 2020. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.** In *Proceedings of SIGIR*, 39–48.

[12] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. **ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.** In *Proceedings of NAACL-HLT*, 3715–3734.

[13] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. **Dense Passage Retrieval for Open-Domain Question Answering.** In *Proceedings of EMNLP*, 6769–6781.

[14] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.** In *NeurIPS Datasets and Benchmarks Track*.

[15] Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. **MTEB: Massive Text Embedding Benchmark.** In *Proceedings of EACL*, 2014–2037.

[16] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. **Retrieval Augmented Language Model Pre-Training.** In *Proceedings of ICML*, 3929–3938.

[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** In *NeurIPS 33*.

[18] Jaime G. Carbonell and Jade Goldstein. 1998. **The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.** In *Proceedings of SIGIR*, 335–336.

[19] Yonatan Geifman and Ran El-Yaniv. 2017. **Selective Classification for Deep Neural Networks.** In *NeurIPS 30*.

[20] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020. **LayoutLM: Pre-training of Text and Layout for Document Image Understanding.** In *Proceedings of KDD*, 1192–1200.

[21] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. 2021. **LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding.** In *Proceedings of ACL-IJCNLP*, 2579–2591.

[22] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. 2021. **DocFormer: End-to-End Transformer for Document Understanding.** In *Proceedings of ICCV*, 993–1003.

[23] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. 2022. **OCR-Free Document Understanding Transformer.** In *Proceedings of ECCV*.

[24] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. 2021. **DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification.** In *NeurIPS 34*.

[25] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. 2022. **Not All Patches Are What You Need: Expediting Vision Transformers via Token Reorganizations.** In *ICLR*.

[26] Michael S. Ryoo, A. J. Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. 2021. **TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?** In *NeurIPS 34*.

[27] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2023. **Token Merging: Your ViT but Faster.** In *ICLR*.

[28] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. **RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.** In *Proceedings of NAACL-HLT*, 5835–5847.

[29] Ikuya Yamada, Akari Asai, and Hannaneh Hajishirzi. 2021. **Efficient Passage Retrieval with Hashing for Open-domain Question Answering.** In *Proceedings of ACL-IJCNLP*, 979–986.

[30] Ye Liu, Kazuma Hashimoto, Yingbo Zhou, Semih Yavuz, Caiming Xiong, and Philip S. Yu. 2021. **Dense Hierarchical Retrieval for Open-domain Question Answering.** In *Findings of EMNLP*, 188–200.
