"""MMR re-selection in recall — diversity for multi-turn / summary questions (no LLM)."""
from __future__ import annotations

from midas import Memory
from midas.types import MemoryRecord, RecallHit


def _hit(rid: str, vec: list[float], rel: float) -> RecallHit:
    return RecallHit(
        record=MemoryRecord(id=rid, content=rid, embedding=vec),
        score=rel, relevance=rel, importance_norm=0.0, recency=0.0,
    )


def test_mmr_prefers_diverse_over_near_duplicate():
    mem = Memory()
    a = _hit("a", [1.0, 0.0], 0.90)
    b = _hit("b", [0.99, 0.14], 0.88)   # near-duplicate of a (cosine ~0.99)
    c = _hit("c", [0.0, 1.0], 0.70)     # diverse, slightly lower relevance
    out = mem._mmr_select([a, b, c], limit=2, lam=0.5)
    ids = [h.record.id for h in out]
    assert ids[0] == "a"                 # highest relevance is picked first
    assert ids[1] == "c"                 # then the diverse hit, not the near-duplicate b


def test_mmr_lambda_1_is_plain_relevance_order():
    mem = Memory()
    a = _hit("a", [1.0, 0.0], 0.90)
    b = _hit("b", [0.99, 0.14], 0.88)
    c = _hit("c", [0.0, 1.0], 0.70)
    out = mem._mmr_select([a, b, c], limit=3, lam=1.0)  # λ=1 → redundancy ignored
    assert [h.record.id for h in out] == ["a", "b", "c"]  # pure relevance order


def test_recall_accepts_mmr_lambda():
    mem = Memory()
    mem.remember("alpha topic one detail")
    mem.remember("beta topic two detail")
    hits = mem.recall("topic", limit=2, mmr_lambda=0.5)
    assert len(hits) <= 2  # runs end-to-end through recall()
