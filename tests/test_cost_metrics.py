"""Lock the cost/latency instrumentation contract — the reader-independent wedge metric.

Deterministic + offline (hashing embedder, synthetic dataset). Guards against regressions in the
runner's per-adapter cost reporting and the structural '0 LLM at ingest/query' claim for Midas.
"""
from eval.adapters.baseline_raw import BaselineRawAdapter
from eval.adapters.midas_adapter import MidasAdapter
from eval.datasets import synthetic
from eval.runner import run_adapter
from midas import HashingEmbedder


def test_midas_cost_metrics_present_and_no_internal_llm():
    adapter = MidasAdapter(embedder=HashingEmbedder(), rerank=False)
    m = run_adapter(adapter, synthetic(), token_budget=110)
    assert "ingest_ms/event" in m and m["ingest_ms/event"] >= 0.0
    assert "query_ms/q" in m and m["query_ms/q"] >= 0.0
    # The wedge: Midas makes no LLM calls at ingest/query.
    assert m["uses_internal_llm"] is False


def test_baseline_no_internal_llm():
    m = run_adapter(BaselineRawAdapter(), synthetic(), token_budget=110)
    assert m["uses_internal_llm"] is False
    assert m["ingest_ms/event"] >= 0.0
