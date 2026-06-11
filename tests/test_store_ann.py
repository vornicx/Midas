"""Opt-in IVF (ANN) search inside InMemoryStore/SQLiteStore at >= ann_threshold records."""
import pytest

np = pytest.importorskip("numpy")

from midas import HashingEmbedder, Memory, InMemoryStore
from midas.sqlite_store import SQLiteStore


def _fill(mem: Memory, n: int) -> None:
    mem.remember_many(
        [{"content": f"note {i} about topic-{i % 25} and item {i}", "kind": "note"} for i in range(n)]
    )


def test_ann_path_returns_relevant_results():
    store = InMemoryStore(ann_threshold=50, ann_nprobe=64)  # high nprobe ≈ exhaustive
    mem = Memory(store=store, embedder=HashingEmbedder(), min_relevance_ratio=0)
    _fill(mem, 300)
    mem.remember("the secret launch codename is Polaris", kind="fact", importance=5)

    hits = mem.recall("secret launch codename", limit=3)
    assert hits and "Polaris" in hits[0].record.content
    assert store._ann_cache is not None, "the IVF index should be active at this size"


def test_ann_index_invalidates_on_write():
    store = InMemoryStore(ann_threshold=50, ann_nprobe=64)
    mem = Memory(store=store, embedder=HashingEmbedder(), min_relevance_ratio=0)
    _fill(mem, 100)
    mem.recall("topic", limit=2)  # builds the index
    v1 = store._ann_cache[0]
    mem.remember("brand new fact about the deploy pipeline", kind="fact")
    hits = mem.recall("deploy pipeline", limit=2)
    assert any("deploy pipeline" in h.record.content for h in hits)
    assert store._ann_cache[0] != v1, "mutation must rebuild the index"


def test_ann_respects_predicate():
    store = InMemoryStore(ann_threshold=10, ann_nprobe=64)
    mem = Memory(store=store, embedder=HashingEmbedder(), min_relevance_ratio=0)
    _fill(mem, 60)
    mem.remember("constraint about topic-1 retention", kind="constraint", importance=5)
    hits = mem.recall("topic-1", kind="constraint", limit=5)
    assert hits and all(h.record.kind == "constraint" for h in hits)


def test_below_threshold_stays_exact():
    store = InMemoryStore(ann_threshold=10_000)
    mem = Memory(store=store, embedder=HashingEmbedder(), min_relevance_ratio=0)
    _fill(mem, 50)
    mem.recall("topic", limit=2)
    assert store._ann_cache is None, "small stores must keep the exact scan"


def test_sqlite_store_ann_passthrough(tmp_path):
    store = SQLiteStore(tmp_path / "ann.db", ann_threshold=50, ann_nprobe=64)
    mem = Memory(store=store, embedder=HashingEmbedder(), min_relevance_ratio=0)
    _fill(mem, 120)
    mem.remember("the database of record is PostgreSQL", kind="fact", importance=5)
    hits = mem.recall("database of record", limit=3)
    assert hits and "PostgreSQL" in hits[0].record.content
    store.close()
