"""Tests for the Midas MCP server tools (offline hashing embedder — fast, no model load)."""
import os

import pytest

pytest.importorskip("mcp")  # optional extra; skip (don't crash collection) when MCP SDK is absent

os.environ["MIDAS_MCP_EMBEDDER"] = "hashing"  # must be set before importing the server

from midas.mcp_server import (  # noqa: E402
    build_context,
    capture,
    forget,
    forget_all,
    maintain,
    recall,
    remember,
    server,
    stats,
)


def test_remember_recall_roundtrip():
    forget_all()
    out = remember("The primary database is PostgreSQL.", kind="constraint", importance=5)
    assert "remembered" in out
    remember("Launch date moved to September 14.", kind="fact", importance=5)

    hits = recall("which database did we pick?", limit=2)
    assert isinstance(hits, list) and hits, "recall should return hits"
    # source-traceable shape
    assert all({"id", "score", "kind", "content"} <= set(h) for h in hits)
    assert any("PostgreSQL" in h["content"] for h in hits), "relevant memory should surface"


def test_build_context_returns_budgeted_prompt():
    forget_all()
    remember("Launch date moved to September 14.", kind="fact", importance=5)
    ctx = build_context("when do we launch?", token_budget=100)
    assert isinstance(ctx, str) and "September 14" in ctx


def test_forget_all_clears():
    remember("ephemeral note", kind="note")
    forget_all()
    assert recall("ephemeral", limit=5) == []


def test_stats_and_forget_by_id():
    forget_all()
    out = remember("Decision: use PostgreSQL.", kind="constraint", importance=5)
    rid = out.split("(", 1)[1].rstrip(")")  # "remembered (<id>)"
    remember("a chat turn", kind="chat")

    s = stats()
    assert s["total"] == 2 and s["by_kind"].get("constraint") == 1
    assert forget(rid) == "forgotten"
    assert stats()["total"] == 1


def test_stats_reports_temporal_tiers():
    forget_all()
    remember("a freshly created note", kind="note")
    s = stats()
    assert set(s["by_tier"]) == {"short", "medium", "long"}
    assert s["by_tier"]["short"] == 1  # just created -> short tier


def test_remember_auto_derives_importance_from_content():
    forget_all()
    remember("haha yeah sounds good", kind="chat")  # filler -> importance 1
    remember("My account number is 4421 and renewal is in 2027.", kind="chat")  # specific -> higher
    # maintain(max_records=1) forgets the lowest-value record: the filler, keeping the specific fact.
    out = maintain(max_records=1)
    assert out["before"] == 2 and out["remaining"] == 1
    assert out["forgotten"] == 1 and len(out["removed_ids"]) == 1
    assert any("account number" in h["content"] for h in recall("account number", limit=3))


def test_maintain_consolidates_duplicates_with_audit():
    forget_all()
    remember("The launch is on September 14.", kind="fact", importance=3)
    remember("The launch is on September 14.", kind="fact", importance=3)  # exact duplicate
    out = maintain(consolidate_threshold=0.99)
    assert out["consolidated"] == 1 and out["remaining"] == 1
    assert len(out["removed_ids"]) == 1  # the deletion audit trail


def test_capture_tool_gates_by_policy():
    forget_all()
    fact = capture("My API rate limit is 5000 requests per hour.", kind="fact")
    assert fact["stored"] is True and fact["importance"] >= 2
    filler = capture("haha ok cool")
    assert filler["stored"] is False and "floor" in filler["reason"]
    dup = capture("My API rate limit is 5000 requests per hour.", kind="fact")
    assert dup["stored"] is False and "duplicate" in dup["reason"]
    assert stats()["total"] == 1  # only the fact was kept


def test_server_injects_memory_instructions():
    # The injected prompt is how installing Midas makes an agent start remembering on its own.
    assert server.instructions and "capture" in server.instructions.lower()
