"""Reinforcement importance (no LLM) — repetition signals salience (the inverse of novelty).

When a turn restates an existing memory, that memory matters MORE (importance climbs) and is current
again (recency refreshes) — instead of the restatement being demoted as "redundant". This is the
principled fix to novelty's failure: a fact you keep bringing up is usually a core fact.
"""
from __future__ import annotations

from midas import Memory

DAY = 86_400.0
NOW = 1_000_000_000.0


class FixedEmbedder:
    """'x …' all map to one vector (restatements); 'y' is orthogonal."""

    dim = 2

    def embed(self, text: str) -> list[float]:
        return {"x": [1.0, 0.0], "x again": [1.0, 0.0], "x thrice": [1.0, 0.0], "y": [0.0, 1.0]}[text]


def test_restatement_boosts_the_matched_memory() -> None:
    m = Memory(embedder=FixedEmbedder(), reinforce=True, now=lambda: NOW)
    a = m.remember("x", importance=2)
    b = m.remember("x again", importance=2)  # restates a -> reinforces it
    assert m.store.get(a.id).importance == 3  # a climbed
    assert b.importance == 2                   # the new copy keeps its own score


def test_orthogonal_turn_is_not_reinforced() -> None:
    m = Memory(embedder=FixedEmbedder(), reinforce=True, now=lambda: NOW)
    a = m.remember("x", importance=2)
    m.remember("y", importance=2)  # unrelated
    assert m.store.get(a.id).importance == 2  # unchanged


def test_repeated_mentions_accumulate() -> None:
    m = Memory(embedder=FixedEmbedder(), reinforce=True, now=lambda: NOW)
    a = m.remember("x", importance=1)
    m.remember("x again", importance=1)   # +1
    m.remember("x thrice", importance=1)  # +1
    assert m.store.get(a.id).importance == 3  # 1 -> 3 over two restatements


def test_reinforce_refreshes_recency() -> None:
    m = Memory(embedder=FixedEmbedder(), reinforce=True, now=lambda: NOW)
    a = m.remember("x", importance=2, created_at=NOW - 10 * DAY)
    assert a.updated_at == NOW - 10 * DAY
    m.remember("x again", importance=2, created_at=NOW)  # restated now
    assert m.store.get(a.id).updated_at == NOW  # recency refreshed — it was just reaffirmed


def test_bump_is_clamped_at_five() -> None:
    m = Memory(embedder=FixedEmbedder(), reinforce=True, reinforce_bump=10, now=lambda: NOW)
    a = m.remember("x", importance=4)
    m.remember("x again", importance=4)
    assert m.store.get(a.id).importance == 5  # clamped, not 14


def test_capture_reinforces_and_skips_a_restatement() -> None:
    m = Memory(reinforce=True, now=lambda: NOW)  # default hashing embedder + dedup 0.95
    fact = "My launch date is September 14 2026."
    f = m.remember(fact, importance=3)
    result = m.capture(fact, kind="fact")  # exact restatement
    assert not result.stored and "reinforced" in result.reason
    assert m.store.get(f.id).importance == 4  # the existing memory was boosted
    assert len(m.store.all()) == 1            # no duplicate stored


def test_off_by_default_leaves_importance_unchanged() -> None:
    m = Memory(embedder=FixedEmbedder(), now=lambda: NOW)  # reinforce defaults to False
    a = m.remember("x", importance=2)
    m.remember("x again", importance=2)
    assert m.store.get(a.id).importance == 2
