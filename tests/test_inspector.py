"""The Inspector's API (the testable core under the local UI): browse/search, belief history, project
state, diff, governance audit, and forget-with-receipt. Deterministic — HashingEmbedder, no LLM."""
from __future__ import annotations

import time

from midas import HashingEmbedder, Memory
from midas.coding import remember_architecture_decision
from midas.inspector import (
    api_audit,
    api_diff,
    api_forget,
    api_overview,
    api_project_state,
    api_record,
    api_records,
    api_stats,
)


def _mem() -> Memory:
    return Memory(embedder=HashingEmbedder())


def test_api_records_browse_search_and_filter() -> None:
    mem = _mem()
    mem.remember("The primary database is PostgreSQL.", kind="constraint", created_at=200)
    mem.remember("user: lol ok", kind="chat", created_at=100)
    browse = api_records(mem)
    assert len(browse) == 2 and browse[0]["content"].startswith("The primary")  # newest first
    assert len(api_records(mem, kind="constraint")) == 1
    assert any("PostgreSQL" in r["content"] for r in api_records(mem, q="which database"))


def test_api_record_includes_belief_history() -> None:
    mem = _mem()
    v1 = mem.remember("Cache: Redis.", kind="constraint", created_at=100)
    v2 = mem.remember("Cache: in-process LRU.", kind="constraint", created_at=200)
    v1.superseded_by = v2.id
    out = api_record(mem, v1.id)
    assert out and [h["content"] for h in out["history"]] == ["Cache: Redis.", "Cache: in-process LRU."]
    assert api_record(mem, "nope") is None


def test_api_project_state_groups_by_code_kind() -> None:
    mem = _mem()
    remember_architecture_decision(mem, "Apollo uses PostgreSQL.", project="apollo")
    out = api_project_state(mem, "apollo")
    assert out["architecture_decision"][0]["content"].startswith("Apollo")


def test_api_diff_added_excludes_old() -> None:
    mem = _mem()
    mem.remember("a brand new fact", kind="fact", created_at=time.time())
    mem.remember("an ancient fact", kind="fact", created_at=100)
    contents = [r["content"] for r in api_diff(mem, hours=24)["added"]]
    assert "a brand new fact" in contents and "an ancient fact" not in contents


def test_api_audit_and_forget_with_receipt() -> None:
    mem = _mem()
    rec = mem.remember(
        "User confirmed: deploying to staging is approved.",
        kind="constraint", provenance="user_confirmation", actor="user", source="mcp:s1",
    )
    audit = api_audit(mem, "deploy to staging", "external_action")
    assert audit["allowed"] is True and audit["evidence"]

    out = api_forget(mem, rec.id)
    assert out["ok"] and out["receipt"]["forgotten_count"] == 1
    assert api_record(mem, rec.id) is None  # actually gone


def test_api_stats() -> None:
    mem = _mem()
    mem.remember("a", kind="fact")
    mem.remember("b", kind="note")
    s = api_stats(mem)
    assert s["total"] == 2 and s["kinds"]["fact"] == 1


def test_api_overview_health_metrics() -> None:
    mem = _mem()
    mem.remember("Apollo uses PostgreSQL.", kind="constraint", importance=5,
                 actor="user", source="mcp:s1", created_at=time.time())
    v1 = mem.remember("Cache: Redis.", kind="constraint", created_at=100)
    v2 = mem.remember("Cache: in-process LRU.", kind="constraint", created_at=200)
    v1.superseded_by = v2.id
    o = api_overview(mem)
    assert o["total"] == 3 and o["live"] == 2 and o["superseded"] == 1
    assert o["high_importance"] >= 1
    assert 0.0 <= o["attributable"] <= 1.0  # only the first record has source + actor
    assert o["by_kind"]["constraint"] == 3
    assert "by_provenance" in o and "projects" in o


def test_api_conflicts_and_loops() -> None:
    from midas.continuity import remember_commitment
    from midas.inspector import api_close_loop, api_conflicts, api_loops

    mem = _mem()
    mem.remember("The primary database is PostgreSQL.", kind="constraint", actor="claude-code")
    mem.remember("The primary database is MySQL.", kind="constraint", actor="cursor")
    conflicts = api_conflicts(mem)
    assert len(conflicts) == 1 and conflicts[0]["signal"] == "value"
    assert {"a", "b", "similarity"} <= set(conflicts[0])

    loop = remember_commitment(mem, "Migrate the sessions table to UUID keys.")
    assert [r["id"] for r in api_loops(mem)] == [loop.id]
    out = api_close_loop(mem, loop.id, "migrated in PR #42")
    assert out["ok"] and api_loops(mem) == []
    assert api_close_loop(mem, "nope", "x")["ok"] is False


def test_api_audit_chain(tmp_path) -> None:
    from midas.inspector import api_audit_chain
    from midas.sqlite_store import SQLiteStore

    assert api_audit_chain(_mem()) == {"supported": False}  # in-memory store has no chain

    mem = Memory(store=SQLiteStore(str(tmp_path / "m.sqlite3")), embedder=HashingEmbedder())
    mem.remember("The primary database is PostgreSQL.", kind="constraint")
    out = api_audit_chain(mem)
    assert out["supported"] and out["ok"] and out["entries"] == 1
    assert out["tail"][0]["op"] == "put"
    import json as _json
    assert "PostgreSQL" not in _json.dumps(out)  # hashes only, never content
