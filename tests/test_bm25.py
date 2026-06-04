"""Unit tests for the BM25 lexical ranker (used by hybrid retrieval)."""
from midas import HashingEmbedder, Memory
from midas.bm25 import BM25


def _records(*contents):
    mem = Memory(embedder=HashingEmbedder())
    for c in contents:
        mem.remember(c, kind="note")
    return mem.store.all()


def test_bm25_ranks_lexical_match_highest():
    recs = _records(
        "the primary database is postgresql",
        "we discussed the weather for a while",
        "reminder to buy coffee for the office",
    )
    scores = BM25(recs).scores("which database did we choose")
    db_rec = next(r for r in recs if "database" in r.content)
    assert scores, "query sharing a term should produce scores"
    assert scores.get(db_rec.id, 0.0) == max(scores.values())


def test_bm25_no_overlap_returns_empty():
    recs = _records("hello world", "the quick brown fox")
    assert BM25(recs).scores("zzz nonexistent qqq") == {}
