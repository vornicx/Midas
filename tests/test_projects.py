"""Projects as a first-class, derived view: a project is read from a record's metadata (project tag /
namespace / capture origin), with per-project stats. Deterministic — HashingEmbedder, no LLM."""
from __future__ import annotations

from midas import HashingEmbedder, Memory
from midas.coding import remember_architecture_decision
from midas.projects import list_projects, project_of, project_overview


def _mem() -> Memory:
    return Memory(embedder=HashingEmbedder())


def test_project_of_derivation_precedence() -> None:
    mem = _mem()
    assert project_of(mem.remember("a", kind="fact", metadata={"namespace": "alpha"})) == "alpha"
    assert project_of(mem.remember("b", kind="fact", metadata={"origin": {"cwd": "beta"}})) == "beta"
    # the explicit code project tag wins over namespace/origin
    rec = remember_architecture_decision(mem, "c", project="apollo", metadata={"namespace": "x"})
    assert project_of(rec) == "apollo"


def test_list_projects_and_overview() -> None:
    mem = _mem()
    A = dict(actor="claude-code", source="mcp:claude-code:s1")
    remember_architecture_decision(mem, "Apollo uses PostgreSQL.", project="apollo", **A)
    remember_architecture_decision(mem, "Apollo cache is an LRU.", project="apollo", **A)
    mem.remember("a beta note", kind="note", metadata={"namespace": "beta"})

    projs = {p["name"]: p for p in list_projects(mem)}
    assert projs["apollo"]["total"] == 2 and projs["apollo"]["live"] == 2
    assert "beta" in projs

    ov = project_overview(mem, "apollo")
    assert ov["total"] == 2 and ov["attributable"] == 1.0  # apollo records carry source + actor


def test_inspector_project_api() -> None:
    from midas.inspector import api_project, api_projects

    mem = _mem()
    remember_architecture_decision(mem, "Apollo uses PostgreSQL.", project="apollo")
    assert any(p["name"] == "apollo" for p in api_projects(mem))
    d = api_project(mem, "apollo")
    assert d["overview"]["name"] == "apollo" and "architecture_decision" in d["state"]
    assert "governance" in d and "by_provenance" in d["governance"]


def test_project_governance() -> None:
    from midas.coding import remember_forbidden_action
    from midas.projects import project_governance

    mem = _mem()
    A = dict(actor="claude-code", source="mcp:claude-code:s1")
    remember_architecture_decision(mem, "Apollo uses PostgreSQL.", project="apollo", **A)
    remember_forbidden_action(mem, "Never force-push to main.", project="apollo", **A)
    g = project_governance(mem, "apollo")
    assert len(g["forbidden"]) == 1
    assert g["by_actor"].get("claude-code") == 2
    assert sum(g["by_provenance"].values()) == 2
    assert g["confirmations"] >= 1  # the forbidden rule is a user-confirmed constraint
