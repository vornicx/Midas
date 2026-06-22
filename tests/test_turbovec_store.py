"""TurboVecStore — compressed candidate-gen + exact rerank behind the MemoryStore contract (Phase C)."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("turbovec")

from midas import MemoryStore  # noqa: E402
from midas.turbovec_store import TurboVecStore  # noqa: E402
from midas.types import MemoryRecord  # noqa: E402


def _rec(i: int, vec) -> MemoryRecord:
    v = np.asarray(vec, dtype=np.float32)
    v = v / (float(np.linalg.norm(v)) or 1.0)
    return MemoryRecord(id=f"r{i}", content=f"c{i}", embedding=[float(x) for x in v])


def _store(dim=16, n=50, **kw):
    s = TurboVecStore(dim=dim, bit_width=4, rerank_pool=50, **kw)
    rng = np.random.RandomState(0)
    for i in range(n):
        s.put(_rec(i, rng.randn(dim)))
    return s


def test_exact_rerank_makes_self_top_hit():
    s = _store()
    r0 = s.get("r0")
    hits = s.search(r0.embedding, limit=5)
    assert hits[0][1].id == "r0"             # exact rerank recovers the self-match as #1
    assert hits[0][0] == pytest.approx(1.0, abs=1e-4)
    assert len(hits) == 5


def test_allowed_ids_pushed_into_search():
    s = _store()
    r0 = s.get("r0")
    hits = s.search(r0.embedding, limit=10, allowed_ids={"r10", "r11", "r12"})
    assert {h[1].id for h in hits} <= {"r10", "r11", "r12"}


def test_predicate_filter():
    s = _store()
    r0 = s.get("r0")
    hits = s.search(r0.embedding, limit=10, predicate=lambda r: r.id in {"r5", "r6"})
    assert {h[1].id for h in hits} == {"r5", "r6"}


def test_conforms_to_memorystore_and_crud():
    s = _store(n=10)
    assert isinstance(s, MemoryStore)
    assert s.delete("r0") is True
    assert s.get("r0") is None
    assert len(s.all()) == 9


def test_ram_saving_mode_strips_embeddings_to_disk(tmp_path):
    from midas.vector_source import SQLiteVectorSource

    vs = SQLiteVectorSource(tmp_path / "vecs.db")
    s = TurboVecStore(dim=16, bit_width=4, rerank_pool=50, vector_source=vs)
    rng = np.random.RandomState(0)
    recs = [_rec(i, rng.randn(16)) for i in range(50)]
    q = list(recs[0].embedding)  # capture the query vector before it is stripped
    for r in recs:
        s.put(r)

    hits = s.search(q, limit=5)  # first search flushes → indexes + strips embeddings to disk
    assert hits[0][1].id == "r0"                     # exact rerank from DISK still recovers self as #1
    assert s._records["r0"].embedding is None        # float32 no longer held in RAM
    assert s.get("r0").embedding is not None          # get() re-attaches it from disk
    assert len(vs) == 50                              # all full vectors persisted

    # MemoryStore conformance still holds, and delete cascades to the disk source
    assert isinstance(s, MemoryStore)
    assert s.delete("r0") is True
    assert len(vs) == 49


def test_ram_saving_returned_hits_carry_embeddings(tmp_path):
    from midas.vector_source import SQLiteVectorSource

    vs = SQLiteVectorSource(tmp_path / "v.db")
    s = TurboVecStore(dim=16, bit_width=4, rerank_pool=50, vector_source=vs)
    rng = np.random.RandomState(2)
    recs = [_rec(i, rng.randn(16)) for i in range(30)]
    q = list(recs[0].embedding)
    for r in recs:
        s.put(r)
    hits = s.search(q, limit=3)
    # Returned hits carry their full vector (so anchor_boost/novelty work), but the stored record
    # stays stripped (RAM-saving preserved).
    assert all(h[1].embedding is not None for h in hits)
    assert s._records["r0"].embedding is None

