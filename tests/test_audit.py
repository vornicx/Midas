"""Audit trail — the compliance artifact: belief-revision history, attributability score, and the full
"why did the agent (not) act" record. Deterministic — HashingEmbedder, no LLM."""
from __future__ import annotations

import pytest

from midas import HashingEmbedder, Memory
from midas.audit import audit_completeness, audit_use, belief_history


def _mem() -> Memory:
    return Memory(embedder=HashingEmbedder())


def test_belief_history_reconstructs_the_full_chain_from_any_member() -> None:
    mem = _mem()
    v1 = mem.remember("Cache: Redis.", kind="constraint", created_at=100)
    v2 = mem.remember("Cache: Memcached.", kind="constraint", created_at=200)
    v3 = mem.remember("Cache: in-process LRU.", kind="constraint", created_at=300)
    v1.superseded_by = v2.id
    v2.superseded_by = v3.id

    for member in (v1.id, v2.id, v3.id):  # from any version, the full timeline oldest -> newest
        assert [r.id for r in belief_history(mem, member)] == [v1.id, v2.id, v3.id]


def test_audit_completeness_scores_attributability() -> None:
    mem = _mem()
    full = mem.remember("decided X", kind="fact", source="mcp:s1", actor="user")
    partial = mem.remember("noticed Y", kind="fact")  # no source / actor
    assert audit_completeness([full]) == 1.0
    assert audit_completeness([full, partial]) == 0.5
    assert audit_completeness([]) == 1.0


def test_audit_use_produces_a_traceable_artifact() -> None:
    mem = _mem()
    mem.remember(
        "User confirmed: deploying to staging is approved.",
        kind="constraint", provenance="user_confirmation", actor="user", source="mcp:s1",
    )
    art = audit_use(mem, "deploy to staging", "external_action")
    assert art["allowed"] is True
    assert art["intended_use"] == "external_action"
    assert art["audit_completeness"] == 1.0
    ev = art["evidence"][0]
    assert ev["provenance"] == "user_confirmation" and ev["actor"] == "user"
    assert "belief_history" in ev and ev["belief_history"]


def test_mcp_audit_use_tool_is_wired() -> None:
    import os

    os.environ.setdefault("MIDAS_MCP_EMBEDDER", "hashing")
    pytest.importorskip("mcp")
    from midas.mcp_server import audit_use as audit_tool
    from midas.mcp_server import remember

    remember("User confirmed: publish the changelog.", kind="constraint", provenance="user_confirmation", actor="user")
    out = audit_tool(query="publish the changelog", intended_use="external_action")
    assert {"allowed", "evidence", "audit_completeness", "reason"} <= set(out)
