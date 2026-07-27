# AdaColRAG

**Redundancy-Aware Visual Token Selection and Dense–Late Interaction Fusion for Efficient Document Retrieval**

AdaColRAG is a query-time visual token compression framework for multimodal document retrieval. The framework combines dense candidate retrieval with redundancy-aware visual token selection and dense–late interaction fusion to reduce visual computation while maintaining retrieval effectiveness.

## Method Overview

AdaColRAG consists of three major components:

1. **Dense candidate generation**
   - Uses a frozen dense retriever to identify a small set of relevant document pages.

2. **Redundancy-aware visual token selection**
   - Applies relevance pre-filtering and fast MMR-based selection to retain informative visual tokens while removing redundant representations.

3. **Dense–late interaction fusion**
   - Combines dense retrieval scores with ColPali-style late interaction scores for final ranking.

The released implementation corresponds to the final evaluated configuration:

- Top-50 candidate pages;
- fixed 112 visual tokens per candidate page;
- Top-448 relevance pre-filtering;
- redundancy-aware MMR selection;
- dense–late interaction fusion.

## Repository Structure

```text
AdaColRAG/
├── configs/          # Experiment configurations
├── environment/      # Conda environment definitions
├── src/              # Core implementation
├── scripts/          # Dataset preparation and reproduction scripts
├── tests/            # Unit tests
└── assets/           # Documentation figures
```

## Installation

```bash
conda env create -f environment/core.yml
conda activate adacolrag-core
pip install -e .
```

For model-specific components:

```bash
conda env create -f environment/colpali.yml
conda env create -f environment/visrag.yml
```

## Dataset

AdaColRAG supports ViDoRe-style visual document retrieval benchmarks.

Prepare datasets using:

```bash
python scripts/download_dataset.py --dataset <dataset_name>
```

The implementation expects:

```text
corpus/
queries/
qrels/
```

in the prepared dataset directory.

## Model Preparation

The framework uses:

- `vidore/colpali-v1.3-merged`
- `openbmb/VisRAG-Ret`

Download the checkpoints locally and configure their paths in the YAML files under `configs/`.

## Reproduction

Run embedding preparation:

```bash
bash scripts/export_embeddings.sh
```

Run retrieval evaluation:

```bash
bash scripts/run_experiment.sh
```

The generated outputs include retrieval metrics and efficiency statistics.

## Evaluation

The evaluation pipeline supports:

- nDCG@5 / nDCG@10;
- Recall and MRR metrics;
- visual token usage analysis;
- query-level retrieval analysis.

## Notes

- The repository contains code and reproduction scripts only.
- Datasets, pretrained checkpoints, and experiment outputs are not included.
- Experimental claims should be reproduced using the specified checkpoints, datasets, and configurations.
