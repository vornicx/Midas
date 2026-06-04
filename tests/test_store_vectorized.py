"""The vectorised store scan must return the same top-k as the pure-Python brute force."""
import pytest

from midas import HashingEmbedder, Memory
from midas.embeddings import cosine


def test_vectorized_search_matches_bruteforce():
    mem = Memory(embedder=HashingEmbedder())
    for i in range(400):  # > _VECTORIZE_THRESHOLD (256) -> numpy path
        mem.remember(f"item {i} about topic {i % 37} and detail {i * 7 % 53}", kind="note")
    store = mem.store
    q = mem.embedder.embed("topic 7 detail item")

    got = store.search(q, limit=10)
    ref = sorted(
        ((cosine(q, r.embedding), r) for r in store.all() if r.embedding is not None),
        key=lambda t: t[0],
        reverse=True,
    )[:10]

    got_scores = [s for s, _ in got]
    assert got_scores == pytest.approx([s for s, _ in ref], abs=1e-9)
    assert got_scores == sorted(got_scores, reverse=True)  # highest-first


def test_vectorized_search_respects_predicate():
    mem = Memory(embedder=HashingEmbedder())
    for i in range(300):  # 300 facts pass the predicate -> still hits the numpy path
        mem.remember(f"fact {i} topic {i % 10}", kind="fact", importance=5)
    for i in range(50):
        mem.remember(f"note {i} topic {i % 10}", kind="note")
    store = mem.store
    q = mem.embedder.embed("topic 3")

    hits = store.search(q, limit=10, predicate=lambda r: r.kind == "fact")
    assert hits and all(r.kind == "fact" for _, r in hits)


def test_cached_matrix_invalidates_on_new_records():
    mem = Memory(embedder=HashingEmbedder())
    for i in range(300):  # populate + build the cached matrix on first search
        mem.remember(f"alpha item {i}", kind="note")
    q = mem.embedder.embed("brand new unique zebra subject exactly")
    mem.store.search(q, limit=5)  # warms the cache

    mem.remember("brand new unique zebra subject exactly", kind="note")  # mutate -> cache dirty
    after = mem.store.search(q, limit=5)
    assert any("zebra subject exactly" in r.content for _, r in after), "new record must be retrievable"
