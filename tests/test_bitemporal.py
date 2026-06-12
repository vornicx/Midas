"""Bitemporal belief history: validity windows on supersession + recall(as_of=...)."""
from midas import HashingEmbedder, Memory
from midas.sqlite_store import SQLiteStore


def _mem(**kwargs) -> Memory:
    return Memory(
        embedder=HashingEmbedder(),
        supersede=True,
        supersede_threshold=0.3,
        supersede_lower_threshold=0.1,
        **kwargs,
    )


def test_superseded_record_carries_validity_bound():
    mem = _mem()
    old = mem.remember("the launch date is September 14", kind="fact", created_at=1_000.0)
    mem.remember("actually the launch date moved to October 2", kind="fact", created_at=2_000.0)
    retired = mem.store.get(old.id)
    assert retired.superseded_by is not None
    assert retired.metadata["superseded_at"] == 2_000.0


def test_as_of_resolves_the_belief_current_at_that_time():
    mem = _mem()
    mem.remember("the launch date is September 14", kind="fact", created_at=1_000.0)
    mem.remember("actually the launch date moved to October 2", kind="fact", created_at=2_000.0)
    mem.remember("actually the launch date moved again to November 5", kind="fact", created_at=3_000.0)

    current = mem.recall("when is the launch date?", limit=1, now=3_500.0)
    assert "November 5" in current[0].record.content

    middle = mem.recall("when is the launch date?", limit=1, as_of=2_500.0)
    assert "October 2" in middle[0].record.content

    original = mem.recall("when is the launch date?", limit=1, as_of=1_500.0)
    assert "September 14" in original[0].record.content


def test_as_of_excludes_records_born_later():
    mem = _mem()
    mem.remember("the gateway is Kong", kind="fact", created_at=5_000.0)
    hits = mem.recall("which gateway?", limit=3, as_of=1_000.0)
    assert hits == [], "a record created later must not exist for an earlier as-of query"


def test_supersession_survives_sqlite_reopen(tmp_path):
    """Regression: revision used to mutate the mirror without store.put — SQLite lost it."""
    db = str(tmp_path / "bitemporal.db")
    mem = Memory(
        store=SQLiteStore(db), embedder=HashingEmbedder(),
        supersede=True, supersede_threshold=0.3, supersede_lower_threshold=0.1,
    )
    old = mem.remember("the launch date is September 14", kind="fact", created_at=1_000.0)
    mem.remember("actually the launch date moved to October 2", kind="fact", created_at=2_000.0)
    mem.store.close()

    reopened = Memory(store=SQLiteStore(db), embedder=HashingEmbedder(), supersede=True)
    retired = reopened.store.get(old.id)
    assert retired.superseded_by is not None, "belief revision must survive a restart"
    assert retired.metadata["superseded_at"] == 2_000.0
    hits = reopened.recall("when is the launch date?", limit=1, now=2_500.0)
    assert "October 2" in hits[0].record.content
    reopened.store.close()
