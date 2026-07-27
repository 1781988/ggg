# Effective-Module Redesign

## Motivation from verified Finance EN results

The verified merged-checkpoint run showed:

- the overall AdaColRAG pipeline improved significantly over exhaustive ColPali;
- relevance--diversity selection improved over relevance-only selection at matched token use;
- dense fusion was a strong contributor;
- confidence recovery showed a positive but less certain contribution;
- the query-complexity budget controller did not significantly outperform Fixed-112;
- exact all-token MMR introduced substantial CPU selection overhead.

## Final method changes

- Primary mode changed from adaptive budgeting to fixed 112-token `fixed_mmr`.
- Added relevance prefiltering before greedy MMR; factor 4 is the frozen default.
- Added vectorized and cached dense/mean coarse scoring.
- Preserved the legacy adaptive controller only for archived reproducibility.
- Kept OCR as an optional extension.

## Experiment changes

- Finance EN is the development collection.
- Added a 13-point development matrix for budget, MMR prefilter, threshold, and fusion sensitivity.
- Replaced the old 17-system submission matrix with a 13-system controlled main matrix.
- Added separate redundancy-only and layout-only selector ablations.
- Added fusion-without-recovery and recovery-without-fusion controls.
- Industrial, Pharmaceuticals, and Finance FR are frozen held-out evaluations.

## Workflow changes

- Added deterministic pinned dataset revisions and shard inventories.
- Updated bootstrap comparisons to come from matrix YAML.
- Added quality--latency Pareto flags.
- Added cross-dataset delta and macro-average tables.
- Updated one-click workflow, manuscript injection, README, runbook, protocol, tests, and CI.
