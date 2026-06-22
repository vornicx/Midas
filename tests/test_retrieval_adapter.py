"""Pins the query-adapter TRAINER (not the BEAM outcome, which is empirical): on constructed data with a
known query->positive linear relation, training must separate positives from negatives on a held-out
split better than the identity baseline; with no triplets it must stay the identity. numpy only."""
from __future__ import annotations

import numpy as np

from eval.retrieval_adapter import _normalize, train_adapter


def test_adapter_learns_a_known_query_to_positive_alignment() -> None:
    rng = np.random.default_rng(0)
    d = 16
    M = rng.standard_normal((d, d))  # the "true" query->positive map the adapter should approximate
    q = _normalize(rng.standard_normal((120, d)))
    p = _normalize(q @ M.T)                          # positive: aligned to q via M
    neg = _normalize(rng.standard_normal((120, d)))  # negative: unrelated

    triplets = [(q[i], p[i], neg[i]) for i in range(80)]  # train on the first 80
    W = train_adapter(triplets, d, epochs=100, lr=0.1, margin=0.2, l2=0.002)

    qt, pt, nt = q[80:], p[80:], neg[80:]  # held-out
    base = float(np.mean([qt[i] @ pt[i] - qt[i] @ nt[i] for i in range(len(qt))]))
    adpt = float(np.mean([(W @ qt[i]) @ pt[i] - (W @ qt[i]) @ nt[i] for i in range(len(qt))]))

    assert adpt > base       # the adapter helps on held-out data
    assert adpt > 0.0        # and genuinely separates positive from negative


def test_adapter_is_identity_without_triplets() -> None:
    W = train_adapter([], 8, epochs=10)
    assert np.allclose(W, np.eye(8))
