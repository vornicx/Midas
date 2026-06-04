"""Selective forgetting (decay-based, no LLM) — the maintenance half of long-horizon memory.

Forgetting must bound growth by evicting the aging, low-importance, non-durable tail WITHOUT ever
dropping the durable semantic tier (facts/preferences/constraints, high-importance) or orphaning a
supersession chain. These tests pin those guarantees; the deterministic recall@k retention is
measured separately in the eval harness.
"""
from __future__ import annotations

from midas import Memory

DAY = 86_400.0
NOW = 1_000_000_000.0


def _mem() -> Memory:
    # Default HashingEmbedder is fine: forgetting is value-based (importance x recency), not
    # relevance-based, so embedding content is irrelevant except in the supersession-chain test.
    return Memory(now=lambda: NOW)


def test_memory_value_in_unit_range_and_orders_by_importance_then_recency() -> None:
    m = _mem()
    fresh_critical = m.remember("a", kind="note", importance=5, created_at=NOW)
    old_trivial = m.remember("b", kind="note", importance=1, created_at=NOW - 60 * DAY)
    v_hi = m.memory_value(fresh_critical, now=NOW)
    v_lo = m.memory_value(old_trivial, now=NOW)
    assert 0.0 <= v_lo < v_hi <= 1.0


def test_tier_names_the_short_medium_long_horizons() -> None:
    m = _mem()
    short = m.remember("today", importance=1, created_at=NOW - 0.5 * DAY)
    medium = m.remember("this week", importance=1, created_at=NOW - 3 * DAY)
    long = m.remember("last month", importance=1, created_at=NOW - 30 * DAY)
    assert m.tier(short, now=NOW) == "short"
    assert m.tier(medium, now=NOW) == "medium"
    assert m.tier(long, now=NOW) == "long"


def test_forget_decayed_drops_old_low_value_keeps_recent() -> None:
    m = _mem()
    old = m.remember("old trivia", kind="chat", importance=1, created_at=NOW - 40 * DAY)
    recent = m.remember("recent trivia", kind="chat", importance=1, created_at=NOW - 0.1 * DAY)
    forgotten = m.forget_decayed(now=NOW, min_value=0.3)
    assert old.id in forgotten
    assert recent.id not in forgotten
    assert m.store.get(old.id) is None
    assert m.store.get(recent.id) is not None


def test_forget_decayed_protects_durable_kinds_even_at_low_value() -> None:
    m = _mem()
    # An OLD, low-importance fact has a low memory_value, but is durable -> must survive.
    fact = m.remember("user is allergic to penicillin", kind="fact", importance=1, created_at=NOW - 90 * DAY)
    pref = m.remember("prefers dark mode", kind="preference", importance=1, created_at=NOW - 90 * DAY)
    constraint = m.remember("never deploy on Fridays", kind="constraint", importance=1, created_at=NOW - 90 * DAY)
    assert m.memory_value(fact, now=NOW) < 0.9  # genuinely low value...
    forgotten = m.forget_decayed(now=NOW, min_value=1.0)  # ...threshold that would drop everything
    assert forgotten == []  # all three are durable -> nothing unprotected to drop
    # And the durable fact is still recallable afterwards.
    assert any(h.record.id == fact.id for h in m.recall("penicillin allergy", limit=5))


def test_forget_decayed_protects_high_importance_regardless_of_age() -> None:
    m = _mem()
    critical = m.remember("prod DB password rotated", kind="note", importance=5, created_at=NOW - 99 * DAY)
    trivial = m.remember("idle chatter", kind="note", importance=1, created_at=NOW - 99 * DAY)
    forgotten = m.forget_decayed(now=NOW, min_value=1.0)
    assert critical.id not in forgotten  # importance >= protect_importance (4) -> kept
    assert trivial.id in forgotten


def test_forget_decayed_max_records_caps_to_highest_value() -> None:
    m = _mem()
    # Five unprotected notes of increasing value (older+less important -> lower).
    recs = [
        m.remember(f"note {i}", kind="note", importance=1, created_at=NOW - (50 - 10 * i) * DAY)
        for i in range(5)
    ]
    forgotten = m.forget_decayed(now=NOW, max_records=2)
    assert len(m.store.all()) == 2
    # The two highest-value (most recent) survive; the three lowest are forgotten.
    survivors = {r.id for r in m.store.all()}
    assert survivors == {recs[3].id, recs[4].id}
    assert set(forgotten) == {recs[0].id, recs[1].id, recs[2].id}


def test_forget_decayed_audit_trail_is_lowest_value_first() -> None:
    m = _mem()
    mid = m.remember("mid", kind="note", importance=1, created_at=NOW - 20 * DAY)
    lowest = m.remember("lowest", kind="note", importance=1, created_at=NOW - 80 * DAY)
    forgotten = m.forget_decayed(now=NOW, min_value=0.5)
    # Both dropped, but the audit list is ordered lowest-value (most decayed) first.
    assert forgotten == [lowest.id, mid.id]


def test_forget_decayed_keeps_supersession_chain_intact() -> None:
    # Two near-identical facts so the second supersedes the first (chain old -> new).
    class FixedEmbedder:
        dim = 2

        def embed(self, text: str) -> list[float]:
            return {"old launch": [1.0, 0.0], "new launch": [1.0, 0.0], "unrelated topic": [0.0, 1.0]}[text]

    m = Memory(embedder=FixedEmbedder(), supersede=True, now=lambda: NOW)
    old = m.remember("old launch", kind="fact", importance=1, created_at=NOW - 90 * DAY)
    new = m.remember("new launch", kind="fact", importance=1, created_at=NOW - 89 * DAY)
    assert old.superseded_by == new.id  # the chain formed
    unrelated = m.remember("unrelated topic", kind="fact", importance=1, created_at=NOW - 90 * DAY)

    # protect_kinds=() disables the durable-kind shield so ONLY chain protection is under test.
    forgotten = m.forget_decayed(now=NOW, min_value=1.0, protect_kinds=())
    assert old.id not in forgotten and new.id not in forgotten  # whole chain kept
    assert unrelated.id in forgotten  # a non-chain low-value fact still goes
    # The chain still resolves: a query phrased like the OLD value reaches the CURRENT head.
    hits = m.recall("old launch", limit=5)
    assert hits and hits[0].record.id == new.id


def test_forget_decayed_noop_without_limits() -> None:
    m = _mem()
    m.remember("a", kind="note", importance=1, created_at=NOW - 90 * DAY)
    assert m.forget_decayed(now=NOW) == []  # no min_value and no max_records -> nothing to do
    assert len(m.store.all()) == 1
