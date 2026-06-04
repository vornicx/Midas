"""Extractive consolidation (no LLM) — collapse near-duplicate restatements, keep the best copy.

Consolidation must remove REDUNDANCY (the same fact restated many times) without merging DISTINCT
memories (the supersession lesson — over-merging destroys recall) and without orphaning a supersession
chain. It is extractive: it drops whole duplicate records, never rewrites them, so provenance survives.
"""
from __future__ import annotations

from midas import Memory

DAY = 86_400.0
NOW = 1_000_000_000.0


class DupEmbedder:
    """All 'alpha …' restatements map to the same vector (near-duplicates); 'beta' is orthogonal."""

    dim = 2

    def embed(self, text: str) -> list[float]:
        table = {
            "alpha": [1.0, 0.0],
            "alpha one": [1.0, 0.0],
            "alpha two": [1.0, 0.0],
            "alpha three": [1.0, 0.0],
            "alpha four": [1.0, 0.0],
            "beta": [0.0, 1.0],
        }
        return table[text]


def test_consolidate_collapses_near_duplicates_keeps_highest_value() -> None:
    m = Memory(embedder=DupEmbedder(), now=lambda: NOW)
    v1 = m.remember("alpha one", importance=2, created_at=NOW - 10 * DAY)
    v2 = m.remember("alpha two", importance=4, created_at=NOW - 5 * DAY)  # highest memory_value
    v3 = m.remember("alpha three", importance=1, created_at=NOW - 20 * DAY)
    other = m.remember("beta", importance=1, created_at=NOW)

    dropped = m.consolidate(now=NOW, similarity_threshold=0.95)
    assert set(dropped) == {v1.id, v3.id}  # the redundant copies go
    assert m.store.get(v2.id) is not None   # the highest-value copy stays
    assert m.store.get(other.id) is not None  # the distinct memory is untouched
    assert m.store.get(v1.id) is None
    # The fact is still retrievable via the surviving representative.
    hits = m.recall("alpha", limit=5)
    assert any(h.record.id == v2.id for h in hits)


def test_consolidate_does_not_merge_distinct_memories() -> None:
    m = Memory(embedder=DupEmbedder(), now=lambda: NOW)
    a = m.remember("alpha one", created_at=NOW)
    b = m.remember("beta", created_at=NOW)
    assert m.consolidate(now=NOW, similarity_threshold=0.95) == []  # orthogonal -> nothing merged
    assert m.store.get(a.id) is not None and m.store.get(b.id) is not None


def test_consolidate_threshold_gates_merging() -> None:
    # A threshold above the pair's cosine (1.0 here) must merge nothing.
    m = Memory(embedder=DupEmbedder(), now=lambda: NOW)
    m.remember("alpha one", created_at=NOW)
    m.remember("alpha two", created_at=NOW)
    assert m.consolidate(now=NOW, similarity_threshold=1.01) == []
    assert len(m.store.all()) == 2


def test_consolidate_preserves_supersession_chain() -> None:
    m = Memory(embedder=DupEmbedder(), now=lambda: NOW)
    a = m.remember("alpha one", created_at=NOW - 3 * DAY)
    b = m.remember("alpha two", created_at=NOW - 2 * DAY)
    a.superseded_by = b.id  # simulate a chain a -> b without supersede side effects
    d = m.remember("alpha three", created_at=NOW - 1 * DAY)
    e = m.remember("alpha four", created_at=NOW)

    dropped = m.consolidate(now=NOW, similarity_threshold=0.95)
    # Chain members are never dropped (would orphan _resolve_head); the free duplicates collapse to one.
    assert a.id not in dropped and b.id not in dropped
    assert m.store.get(a.id) is not None and m.store.get(b.id) is not None
    assert len(dropped) == 1 and dropped[0] in {d.id, e.id}


def test_consolidate_empty_store_is_noop() -> None:
    m = Memory(embedder=DupEmbedder(), now=lambda: NOW)
    assert m.consolidate(now=NOW) == []
