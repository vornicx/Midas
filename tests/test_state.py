"""Control-plane views: memory_state (current durable beliefs of a scope) and memory_diff (what changed
since a point in time). Deterministic — HashingEmbedder, no LLM."""
from __future__ import annotations

import pytest

from midas import HashingEmbedder, Memory
from midas.state import memory_diff, memory_state


def _mem() -> Memory:
    return Memory(embedder=HashingEmbedder())


def test_memory_state_returns_current_durable_beliefs_newest_first() -> None:
    mem = _mem()
    a = mem.remember("Decision: primary DB is PostgreSQL.", kind="constraint", created_at=100)
    b = mem.remember("The launch date is September 14.", kind="fact", created_at=200)
    mem.remember("user: lol ok", kind="chat", created_at=300)  # filler kind — excluded from state

    state = memory_state(mem)
    assert [r.id for r in state] == [b.id, a.id]  # newest-first; the chat turn is not project state


def test_memory_state_excludes_superseded_and_filters_scope() -> None:
    mem = _mem()
    old = mem.remember("Cache: Redis.", kind="constraint", created_at=100, metadata={"project": "apollo"})
    new = mem.remember("Cache: in-process LRU.", kind="constraint", created_at=200, metadata={"project": "apollo"})
    mem.remember("Deploy region: us-east-1.", kind="fact", created_at=150, metadata={"project": "zeus"})
    old.superseded_by = new.id  # belief revised

    apollo = memory_state(mem, scope={"project": "apollo"})
    assert [r.id for r in apollo] == [new.id]  # superseded Redis gone; the zeus fact is out of scope


def test_memory_diff_separates_added_from_revised_without_double_counting() -> None:
    mem = _mem()
    old = mem.remember("Cache: Redis.", kind="constraint", created_at=100)
    added = mem.remember("New constraint: rate limit is 10/min.", kind="constraint", created_at=300)
    new = mem.remember("Cache: in-process LRU.", kind="constraint", created_at=400)
    old.superseded_by = new.id  # a revision in the window

    diff = memory_diff(mem, since=250)
    assert [r.id for r in diff.added] == [added.id]  # genuinely new belief
    assert [(o.id, n.id) for o, n in diff.revised] == [(old.id, new.id)]  # the revision pair
    assert new.id not in {r.id for r in diff.added}  # the revision's new side isn't also counted as added


def test_memory_diff_empty_when_nothing_changed_since() -> None:
    mem = _mem()
    mem.remember("Old fact.", kind="fact", created_at=100)
    assert memory_diff(mem, since=1000).is_empty()


def test_mcp_state_and_diff_tools_are_wired() -> None:
    import os

    os.environ.setdefault("MIDAS_MCP_EMBEDDER", "hashing")
    pytest.importorskip("mcp")  # optional extra; skip when the MCP SDK isn't installed
    from midas.mcp_server import memory_diff as diff_tool
    from midas.mcp_server import memory_state as state_tool
    from midas.mcp_server import remember

    remember("Decision: Apollo uses PostgreSQL for the primary database.", kind="constraint", importance=5)
    state = state_tool(limit=50)
    assert state["count"] >= 1
    assert any("PostgreSQL" in r["content"] for r in state["state"])

    diff = diff_tool(hours=24.0)
    assert set(diff) >= {"added", "revised", "since_hours"}
    assert any("PostgreSQL" in r["content"] for r in diff["added"])
