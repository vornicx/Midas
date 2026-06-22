"""ColBERTReranker — late-interaction MaxSim scoring (Phase 3).

The MaxSim + logit-calibration logic is tested with a fake late-interaction model (token-level
vectors), so it is deterministic and needs no model download — the real model is exercised by the
BEAM eval. Scores are logits, so the assertions check ordering, not absolute values.
"""
from __future__ import annotations

import numpy as np

from midas.colbert import ColBERTReranker


class _FakeLI:
    """Maps text -> (n_tokens, dim) array, mimicking fastembed LateInteractionTextEmbedding."""

    def __init__(self, by_text):
        self._by_text = by_text

    def query_embed(self, q):
        yield self._by_text[q]

    def embed(self, docs):
        for d in docs:
            yield self._by_text[d]


def _reranker(by_text):
    rr = ColBERTReranker.__new__(ColBERTReranker)  # bypass __init__ → no model download
    rr._model = _FakeLI(by_text)
    rr.model_name = "fake"
    return rr


def test_colbert_ranks_better_token_match_higher():
    # query has two orthogonal tokens; d0 matches both, d1 matches only the first.
    rr = _reranker({
        "q": np.array([[1, 0], [0, 1]], dtype=np.float32),
        "d0": np.array([[1, 0], [0, 1]], dtype=np.float32),  # MaxSim/nq = (1+1)/2 = 1.0
        "d1": np.array([[1, 0], [1, 0]], dtype=np.float32),  # MaxSim/nq = (1+0)/2 = 0.5
    })
    s = rr.rerank("q", ["d0", "d1"])
    assert len(s) == 2
    assert s[0] > s[1]               # better token coverage ranks higher


def test_colbert_empty_documents():
    rr = _reranker({"q": np.array([[1.0, 0.0]], dtype=np.float32)})
    assert rr.rerank("q", []) == []


def test_colbert_handles_degenerate_doc():
    rr = _reranker({
        "q": np.array([[1.0, 0.0]], dtype=np.float32),
        "d0": np.zeros((0, 2), dtype=np.float32),  # empty doc → neutral, no crash
    })
    assert rr.rerank("q", ["d0"]) == [0.0]
