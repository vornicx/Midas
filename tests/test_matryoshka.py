"""TruncatedEmbedder — Matryoshka dim truncation + renormalize (Phase 4).

Pure logic, tested with a fake inner embedder so no model download is needed; the MRL quality/speed
tradeoff itself is measured by the BEAM eval.
"""
from __future__ import annotations

import math

from midas.embeddings import TruncatedEmbedder


class _FakeInner:
    dim = 4
    model_name = "fake"
    max_text_chars = 100
    batch_size = 8

    def embed(self, text):
        return [3.0, 4.0, 1.0, 1.0]

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


def test_truncate_keeps_prefix_and_renormalizes():
    e = TruncatedEmbedder(_FakeInner(), 2)
    v = e.embed("x")
    assert len(v) == 2 and e.dim == 2
    assert abs(math.hypot(*v) - 1.0) < 1e-6                  # renormalized to unit length
    assert abs(v[0] - 0.6) < 1e-6 and abs(v[1] - 0.8) < 1e-6  # [3,4] -> [0.6, 0.8]


def test_truncate_has_distinct_cache_namespace():
    e = TruncatedEmbedder(_FakeInner(), 2)
    assert "mrl2" in e.model_name        # distinct model_name/dim → its own embedding-cache namespace


def test_truncate_embed_many():
    out = TruncatedEmbedder(_FakeInner(), 2).embed_many(["a", "b"])
    assert len(out) == 2 and all(len(v) == 2 for v in out)
