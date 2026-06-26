"""Projects as a first-class view over memory — derived, not stored.

A "project" is read off a record's metadata, no new schema: the explicit code `project` tag (the coding
vocabulary), else the `namespace` (per-project scoping / `MIDAS_MCP_NAMESPACE=auto`), else the capture
`origin.cwd` (Phase-1 traceability). So all three signals roll up into one project view, and a fresh agent
run organizes itself by project automatically. Deterministic, no LLM.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .audit import audit_completeness

if TYPE_CHECKING:
    from .memory import Memory
    from .types import MemoryRecord


def project_of(record: "MemoryRecord") -> str | None:
    """The project a record belongs to — explicit code tag, else namespace, else capture cwd."""
    md = record.metadata or {}
    origin = md.get("origin") or {}
    return md.get("project") or md.get("namespace") or origin.get("cwd") or None


def _by_project(mem: "Memory") -> dict[str, list["MemoryRecord"]]:
    groups: dict[str, list[MemoryRecord]] = {}
    for r in mem.store.all():
        name = project_of(r)
        if name:
            groups.setdefault(name, []).append(r)
    return groups


def list_projects(mem: "Memory") -> list[dict[str, Any]]:
    """Every project with at-a-glance stats, most-recently-active first."""
    out = []
    for name, recs in _by_project(mem).items():
        out.append({
            "name": name,
            "total": len(recs),
            "live": sum(1 for r in recs if r.superseded_by is None),
            "attributable": round(audit_completeness(recs), 2),
            "last_active": max((r.updated_at for r in recs), default=0.0),
        })
    out.sort(key=lambda p: -p["last_active"])
    return out


def project_overview(mem: "Memory", name: str) -> dict[str, Any]:
    """Per-project health: counts, attributability, recency, and the kind distribution."""
    recs = [r for r in mem.store.all() if project_of(r) == name]
    by_kind: dict[str, int] = {}
    for r in recs:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
    return {
        "name": name,
        "total": len(recs),
        "live": sum(1 for r in recs if r.superseded_by is None),
        "superseded": sum(1 for r in recs if r.superseded_by is not None),
        "attributable": round(audit_completeness(recs), 2),
        "last_active": max((r.updated_at for r in recs), default=0.0),
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
    }


def project_governance(mem: "Memory", name: str) -> dict[str, Any]:
    """The governance picture of a project: its active prohibitions, the trust mix (by provenance), who
    contributed (by actor), and how much has been revised. The same view the paid Team dashboard renders,
    multi-user. `forbidden` is returned as records — the caller serializes."""
    recs = [r for r in mem.store.all() if project_of(r) == name]
    by_prov: dict[str, int] = {}
    by_actor: dict[str, int] = {}
    for r in recs:
        by_prov[r.provenance] = by_prov.get(r.provenance, 0) + 1
        actor = r.actor or "—"
        by_actor[actor] = by_actor.get(actor, 0) + 1
    forbidden = [r for r in recs
                 if r.superseded_by is None and (r.metadata or {}).get("code_kind") == "forbidden_action"]
    return {
        "forbidden": forbidden,
        "by_provenance": dict(sorted(by_prov.items(), key=lambda kv: -kv[1])),
        "by_actor": dict(sorted(by_actor.items(), key=lambda kv: -kv[1])),
        "confirmations": sum(1 for r in recs if r.provenance == "user_confirmation"),
        "revisions": sum(1 for r in recs if r.superseded_by is not None),
    }
