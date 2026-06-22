"""TurboVecIndex — compressed VectorIndex over TurboVec's IdMapIndex (Phase C).

Runs against the real `turbovec` wheel (skipped if not installed). Covers the string-id ↔ uint64
mapping, allowlist filtering, removal, and write/load — the contract a `TurboVecStore` builds on.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("turbovec")

from midas.turbovec_index import TurboVecIndex  # noqa: E402


def _unit(rows) -> "np.ndarray":
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _index(n=6, dim=8, seed=0):
    idx = TurboVecIndex(dim=dim, bit_width=4)
    vecs = _unit(np.random.RandomState(seed).randn(n, dim))
    idx.add_with_ids(vecs, [f"r{i}" for i in range(n)])
    return idx, vecs


def test_search_returns_string_ids_best_first():
    idx, vecs = _index()
    ids, scores = idx.search(vecs[0], k=3)
    assert ids[0] == "r0"                       # the self-vector is the top hit
    assert len(ids) == 3 and len(scores) == 3
    assert scores[0] >= scores[1] >= scores[2]  # descending similarity


def test_allowlist_is_a_hard_filter():
    idx, vecs = _index()
    ids, _ = idx.search(vecs[0], k=3, allowlist={"r3", "r4", "r5"})
    assert set(ids) <= {"r3", "r4", "r5"}       # nothing outside the allowlist
    assert "r0" not in ids                       # even though r0 is the closest


def test_remove_and_len():
    idx, _ = _index()
    assert len(idx) == 6
    assert idx.remove("r0") is True
    assert idx.remove("r0") is False             # idempotent
    assert len(idx) == 5


def test_write_load_roundtrip(tmp_path):
    idx, vecs = _index(n=5, seed=1)
    idx.search(vecs[0], k=1)                      # force prepare()
    p = str(tmp_path / "idx.tvim")
    idx.write(p)
    loaded = TurboVecIndex.load(p)
    ids, _ = loaded.search(vecs[0], k=1)
    assert ids[0] == "x0" or ids[0] == "r0"      # string-id map survived the round-trip
    assert len(loaded) == 5
