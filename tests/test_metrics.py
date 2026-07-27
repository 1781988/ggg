from adacolrag.metrics import aggregate_retrieval_metrics


def test_metrics_perfect_ranking():
    rankings = {"q1": ["d1", "d2"]}
    qrels = {"q1": {"d1": 1}}
    metrics = aggregate_retrieval_metrics(rankings, qrels)
    assert metrics["nDCG@5"] == 1.0
    assert metrics["Recall@1"] == 1.0
    assert metrics["MRR@10"] == 1.0
