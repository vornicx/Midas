"""SparseLexicalIndex — learned-sparse lexical scoring as a BM25 drop-in (Phase 2).

Scoring is a sparse dot product over shared dimensions, walked via an inverted index. Tested with a
fake fastembed model so the logic is deterministic and needs no model download — the real model is
exercised by the BEAM eval.
"""
from __future__ import annotations

import numpy as np

from midas.sparse import SparseLexicalIndex
from midas.types import MemoryRecord


class _Sparse:
    def __init__(self, indices, values):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float32)


class _FakeModel:
    """Maps text -> sparse vector, mimicking fastembed SparseTextEmbedding."""

    def __init__(self, by_text):
        self._by_text = by_text

    def embed(self, texts):
        for t in texts:
            yield self._by_text[t]

    def query_embed(self, query):
        yield self._by_text[query]


def test_sparse_dot_product_scoring_and_ranking():
    # r0 ~ dims {1:2, 2:1}; r1 ~ dims {2:1, 3:5}; query ~ dims {2:1, 3:1}
    fake = _FakeModel({
        "doc0": _Sparse([1, 2], [2.0, 1.0]),
        "doc1": _Sparse([2, 3], [1.0, 5.0]),
        "q": _Sparse([2, 3], [1.0, 1.0]),
    })
    records = [MemoryRecord(id="r0", content="doc0"), MemoryRecord(id="r1", content="doc1")]
    idx = SparseLexicalIndex(records, model=fake)
    scores = idx.scores("q")
    assert scores["r0"] == 1.0          # overlap on dim 2 only: 1*1
    assert scores["r1"] == 6.0          # dim2 (1*1) + dim3 (1*5)
    assert max(scores, key=scores.get) == "r1"


def test_sparse_no_overlap_is_dropped():
    fake = _FakeModel({
        "doc0": _Sparse([10, 11], [1.0, 1.0]),
        "q": _Sparse([2, 3], [1.0, 1.0]),
    })
    idx = SparseLexicalIndex([MemoryRecord(id="r0", content="doc0")], model=fake)
    assert idx.scores("q") == {}        # no shared dimension → no score


def test_sparse_empty_corpus():
    fake = _FakeModel({"q": _Sparse([1], [1.0])})
    assert SparseLexicalIndex([], model=fake).scores("q") == {}
