from adacolrag.config import load_config
from adacolrag.pipeline import AdaColRAGPipeline
from adacolrag.synthetic import build_synthetic_fixture


def test_synthetic_pipeline_retrieves_relevant_documents():
    config = load_config("configs/adacolrag.yaml")
    documents, queries, qrels = build_synthetic_fixture(seed=42)
    result = AdaColRAGPipeline(config).evaluate(documents, queries, qrels)
    assert result.metrics["nDCG@5"] >= 0.95
    assert result.efficiency["token_reduction"] > 0.30
    assert 0.0 <= result.reliability["fallback_rate"] <= 1.0
