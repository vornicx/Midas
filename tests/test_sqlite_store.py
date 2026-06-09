"""SQLiteStore: records persist across reopen, and search works after reload."""
from midas import HashingEmbedder, Memory
from midas.sqlite_store import SQLiteStore


def test_persists_and_searches_after_reopen(tmp_path):
    db = str(tmp_path / "mem.db")
    mem = Memory(store=SQLiteStore(db), embedder=HashingEmbedder())
    mem.remember("the primary database is postgresql", kind="constraint", importance=5)
    mem.remember("we chatted about lunch", kind="chat")
    mem.store.close()

    mem2 = Memory(store=SQLiteStore(db), embedder=HashingEmbedder())
    assert len(mem2.store.all()) == 2, "records should survive a reopen"
    hits = mem2.recall("which database did we choose?", limit=1)
    assert hits and "postgresql" in hits[0].record.content, "search works after reload"
    mem2.store.close()


def test_delete_and_clear_persist(tmp_path):
    db = str(tmp_path / "mem2.db")
    store = SQLiteStore(db)
    mem = Memory(store=store, embedder=HashingEmbedder())
    rec = mem.remember("ephemeral note", kind="note")
    assert store.delete(rec.id) is True
    mem.remember("another note", kind="note")
    store.clear()
    store.close()

    reopened = SQLiteStore(db)
    assert reopened.all() == [], "clear() should persist"
    reopened.close()


def test_provenance_persists_after_reopen(tmp_path):
    db = str(tmp_path / "mem3.db")
    mem = Memory(store=SQLiteStore(db), embedder=HashingEmbedder())
    mem.remember(
        "User confirmed deploy target is staging.",
        kind="constraint",
        provenance="user_confirmation",
        actor="user",
        source="mcp:s1",
    )
    mem.store.close()

    reopened = Memory(store=SQLiteStore(db), embedder=HashingEmbedder())
    rec = reopened.store.all()[0]
    assert rec.provenance == "user_confirmation"
    assert rec.actor == "user"
    assert rec.source == "mcp:s1"
    reopened.store.close()
