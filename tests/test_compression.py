import numpy as np

from adacolrag.compression import BudgetController, QueryAwareSelector
from adacolrag.io import l2_normalize


def test_budget_is_bounded_and_aligned():
    controller = BudgetController(32, 128, 8, 0.35, 0.30, 0.35, 24)
    query = l2_normalize(np.eye(8, dtype=np.float32))
    budget, complexity = controller.estimate(query, np.asarray([0.5, 0.49, 0.1], dtype=np.float32))
    assert 32 <= budget <= 128
    assert budget % 8 == 0
    assert 0.0 <= complexity <= 1.0


def test_selector_returns_unique_indices():
    rng = np.random.default_rng(0)
    query = l2_normalize(rng.normal(size=(5, 16)))
    document = l2_normalize(rng.normal(size=(100, 16)))
    positions = rng.uniform(size=(100, 2)).astype(np.float32)
    selected = QueryAwareSelector(prefilter_factor=2.0).select(query, document, 32, positions)
    assert selected.shape == (32,)
    assert len(set(selected.tolist())) == 32
    assert selected.min() >= 0
    assert selected.max() < document.shape[0]


def test_prefiltered_selector_matches_exact_when_pool_covers_document():
    rng = np.random.default_rng(1)
    query = l2_normalize(rng.normal(size=(4, 12)))
    document = l2_normalize(rng.normal(size=(48, 12)))
    positions = rng.uniform(size=(48, 2)).astype(np.float32)
    exact = QueryAwareSelector(prefilter_factor=0.0).select(query, document, 16, positions)
    covering = QueryAwareSelector(prefilter_factor=3.0).select(query, document, 16, positions)
    assert exact.tolist() == covering.tolist()


def test_relevance_only_selector_uses_topk_fast_path():
    rng = np.random.default_rng(2)
    query = l2_normalize(rng.normal(size=(3, 8)))
    document = l2_normalize(rng.normal(size=(64, 8)))
    selected = QueryAwareSelector(
        redundancy_weight=0.0,
        layout_weight=0.0,
        prefilter_factor=2.0,
    ).select(query, document, 10)
    relevance = (query @ document.T).max(axis=0)
    expected = set(np.argsort(relevance)[-10:].tolist())
    assert set(selected.tolist()) == expected
