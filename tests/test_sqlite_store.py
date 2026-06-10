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


def test_two_open_stores_share_writes_live(tmp_path):
    """Two stores on the same file (≈ two MCP client processes) see each other's writes without a
    reopen — the cross-session memory story, live."""
    db = str(tmp_path / "shared.db")
    a = Memory(store=SQLiteStore(db), embedder=HashingEmbedder())
    b = Memory(store=SQLiteStore(db), embedder=HashingEmbedder())

    a.remember("decision: queue runs on Redis Streams", kind="fact", importance=5)
    hits = b.recall("what does the queue run on?", limit=1)
    assert hits and "Redis Streams" in hits[0].record.content, "B should see A's write live"

    rec = b.remember("note from the other session", kind="note")
    assert a.store.get(rec.id) is not None, "A should see B's write live"

    assert b.store.delete(hits[0].record.id) is True
    assert a.store.get(hits[0].record.id) is None, "A should see B's delete live"
    a.store.close()
    b.store.close()


def test_concurrent_threaded_writes(tmp_path):
    """The shared connection is lock-guarded: writes from worker threads (how MCP frameworks run
    sync tools) must not corrupt or raise."""
    import threading

    db = str(tmp_path / "threads.db")
    store = SQLiteStore(db)
    mem = Memory(store=store, embedder=HashingEmbedder())
    errors: list[Exception] = []

    def write(n: int) -> None:
        try:
            for i in range(20):
                mem.remember(f"thread {n} note {i}", kind="note")
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"threaded writes raised: {errors!r}"
    assert len(store.all()) == 80
    store.close()


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
