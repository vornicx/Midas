"""Coding-agent memory vocabulary (Fase B): code memories map to core kinds + a code_kind/project tag,
and project_state gives the live, grouped onboarding view. Deterministic — HashingEmbedder, no LLM."""
from __future__ import annotations

import pytest

from midas import HashingEmbedder, Memory
from midas.coding import (
    is_forbidden,
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


def test_is_forbidden_catches_a_matching_action() -> None:
    mem = _mem()
    remember_forbidden_action(mem, "Never force-push to the main branch.", project="apollo")
    hits = is_forbidden(mem, "force-push to main now", "apollo")
    assert hits and hits[0].metadata["code_kind"] == "forbidden_action"


def test_is_forbidden_allows_unrelated_actions() -> None:
    mem = _mem()
    remember_forbidden_action(mem, "Never force-push to the main branch.", project="apollo")
    assert is_forbidden(mem, "run the unit test suite", "apollo") == []


def test_superseded_forbidden_rule_no_longer_blocks() -> None:
    mem = _mem()
    rule = remember_forbidden_action(mem, "Never deploy on Fridays.", project="apollo", created_at=100)
    repeal = remember_code(mem, "Friday deploys are now allowed.", "convention", project="apollo", created_at=200)
    rule.superseded_by = repeal.id  # the rule was retired
    assert is_forbidden(mem, "deploy on Friday afternoon", "apollo") == []


def test_is_forbidden_semantic_catches_paraphrase() -> None:
    # A controlled embedder: the paraphrase is close to the rule and the benign action is far, while
    # NEITHER shares content words with the rule — so only the semantic path can catch the paraphrase.
    class _Stub:
        _M = {
            "Never delete user data.": [1.0, 0.0],
            "remove the accounts table": [0.96, 0.28],  # cosine ~0.96 with the rule
            "read the deployment logs": [0.0, 1.0],      # cosine 0 with the rule
        }

        def embed(self, text: str) -> list[float]:
            return self._M.get(text, [0.0, 0.0])

    mem = Memory(embedder=_Stub())
    remember_forbidden_action(mem, "Never delete user data.", project="apollo")
    assert is_forbidden(mem, "remove the accounts table", "apollo")  # semantic catch (lexical misses)
    assert is_forbidden(mem, "remove the accounts table", "apollo", use_embeddings=False) == []  # lexical alone misses
    assert is_forbidden(mem, "read the deployment logs", "apollo") == []  # neither signal fires


def test_mcp_check_forbidden_action_tool() -> None:
    import os

    os.environ.setdefault("MIDAS_MCP_EMBEDDER", "hashing")
    pytest.importorskip("mcp")
    from midas.mcp_server import _mem, check_forbidden_action

    remember_forbidden_action(_mem, "Never drop the production database.", project="apollo-fb")
    assert check_forbidden_action(action="drop the production database", project="apollo-fb")["forbidden"] is True
    assert check_forbidden_action(action="read the production logs", project="apollo-fb")["forbidden"] is False


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


def test_capture_policy_is_code_aware() -> None:
    from midas import AGENT_MEMORY_INSTRUCTIONS

    for tool in ("remember_code", "project_state", "check_forbidden_action"):
        assert tool in AGENT_MEMORY_INSTRUCTIONS


def test_mcp_remember_code_tool() -> None:
    import os

    os.environ.setdefault("MIDAS_MCP_EMBEDDER", "hashing")
    pytest.importorskip("mcp")
    from midas.mcp_server import project_state as project_state_tool
    from midas.mcp_server import remember_code as remember_code_tool

    out = remember_code_tool(
        content="Apollo uses gRPC between services.", code_kind="architecture_decision", project="apollo-rc"
    )
    assert out["metadata"]["code_kind"] == "architecture_decision"
    assert out["metadata"]["project"] == "apollo-rc"
    state = project_state_tool(project="apollo-rc")
    assert any("gRPC" in r["content"] for r in state["state"]["architecture_decision"])


def test_forbidden_rule_never_authorizes_an_action() -> None:
    """A prohibition is a gate, not an authorization: a forbidden-action rule (stamped as a confirmed
    constraint) must never *justify acting* — even though its provenance would otherwise satisfy the
    external/destructive policy. Regression for the guard counting a 'never do X' rule as approval."""
    from midas.guard import decide_memory_use

    mem = _mem()
    rule = remember_forbidden_action(
        mem, "Never run destructive database migrations without confirmation.", project="apollo"
    )
    assert rule.provenance == "user_confirmation"  # stamped as a confirmed constraint…
    # …yet it must NOT authorize a destructive or external action:
    assert decide_memory_use([rule], intended_use="destructive_action").allowed is False
    assert decide_memory_use([rule], intended_use="external_action").allowed is False
    # a genuine, non-prohibition confirmation still authorizes:
    ok = mem.remember(
        "User confirmed: deploying to staging is approved.",
        kind="constraint", provenance="user_confirmation",
    )
    assert decide_memory_use([ok], intended_use="external_action").allowed is True
