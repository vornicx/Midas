"""Coding-agent memory vocabulary (Fase B): code memories map to core kinds + a code_kind/project tag,
and project_state gives the live, grouped onboarding view. Deterministic — HashingEmbedder, no LLM."""
from __future__ import annotations

import pytest

from midas import HashingEmbedder, Memory
from midas.coding import (
    project_state,
    remember_architecture_decision,
    remember_bug_fixed,
    remember_code,
    remember_forbidden_action,
)


def _mem() -> Memory:
    return Memory(embedder=HashingEmbedder())


def test_remember_code_maps_to_core_kind_and_tags_project() -> None:
    r = remember_architecture_decision(_mem(), "Primary DB is PostgreSQL.", project="apollo")
    assert r.kind == "constraint"  # architecture_decision -> a constraint in the core model
    assert r.metadata["code_kind"] == "architecture_decision"
    assert r.metadata["project"] == "apollo"


def test_forbidden_action_is_a_user_confirmed_constraint() -> None:
    # The A×B tie: a forbidden action is a user-confirmed constraint, so the Guard treats it as real.
    r = remember_forbidden_action(_mem(), "Never run destructive migrations without confirmation.", project="apollo")
    assert r.kind == "constraint"
    assert r.provenance == "user_confirmation"
    assert r.metadata["code_kind"] == "forbidden_action"


def test_project_state_groups_by_code_kind_and_scopes_by_project() -> None:
    mem = _mem()
    remember_architecture_decision(mem, "Primary DB is PostgreSQL.", project="apollo")
    remember_bug_fixed(mem, "Fixed N+1 in transactions list via composite index.", project="apollo")
    remember_forbidden_action(mem, "Never force-push to main.", project="apollo")
    remember_architecture_decision(mem, "Zeus uses MySQL.", project="zeus")  # different project

    state = project_state(mem, "apollo")
    assert set(state) == {"architecture_decision", "bug_fixed", "forbidden_action"}
    assert all(r.metadata["project"] == "apollo" for recs in state.values() for r in recs)
    assert len(state["architecture_decision"]) == 1  # the zeus decision is out of scope


def test_project_state_shows_current_decision_after_revision() -> None:
    mem = _mem()
    old = remember_architecture_decision(mem, "Cache: Redis.", project="apollo", created_at=100)
    new = remember_architecture_decision(mem, "Cache: in-process LRU.", project="apollo", created_at=200)
    old.superseded_by = new.id  # decision revised

    assert [r.id for r in project_state(mem, "apollo")["architecture_decision"]] == [new.id]


def test_unknown_code_kind_fails_fast() -> None:
    with pytest.raises(ValueError, match="unknown code_kind"):
        remember_code(_mem(), "x", "not_a_kind", project="apollo")


def test_mcp_project_state_tool_is_wired() -> None:
    import os

    os.environ.setdefault("MIDAS_MCP_EMBEDDER", "hashing")
    pytest.importorskip("mcp")
    from midas.mcp_server import _mem
    from midas.mcp_server import project_state as project_state_tool

    remember_architecture_decision(_mem, "Apollo uses event sourcing for the ledger.", project="apollo-mcp")
    out = project_state_tool(project="apollo-mcp")
    assert out["project"] == "apollo-mcp"
    assert out["counts"].get("architecture_decision", 0) >= 1
    assert any("event sourcing" in r["content"] for r in out["state"]["architecture_decision"])
