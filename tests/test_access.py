"""RBAC scoped access — a caller sees only the scopes it's granted (plus global, unless strict).
Deterministic — HashingEmbedder, no LLM."""
from __future__ import annotations

from midas import HashingEmbedder, Memory
from midas.access import can_access, visible_records


def _mem() -> Memory:
    return Memory(embedder=HashingEmbedder())


def test_visible_records_scopes_by_namespace_with_global() -> None:
    mem = _mem()
    a = mem.remember("team A secret", kind="fact", metadata={"namespace": "team-a"})
    b = mem.remember("team B secret", kind="fact", metadata={"namespace": "team-b"})
    g = mem.remember("the org-wide coding standard", kind="constraint")  # unscoped = global

    ids = {r.id for r in visible_records(mem, {"team-a"})}
    assert a.id in ids and g.id in ids and b.id not in ids  # team-a + global, never team-b


def test_strict_mode_excludes_global() -> None:
    mem = _mem()
    mem.remember("global", kind="constraint")
    a = mem.remember("team A", kind="fact", metadata={"namespace": "team-a"})
    assert {r.id for r in visible_records(mem, {"team-a"}, strict=True)} == {a.id}


def test_none_scopes_is_unrestricted() -> None:
    mem = _mem()
    mem.remember("x", kind="fact", metadata={"namespace": "team-a"})
    assert len(visible_records(mem, None)) == 1


def test_can_access() -> None:
    mem = _mem()
    a = mem.remember("x", kind="fact", metadata={"namespace": "team-a"})
    assert can_access(a, {"team-a"}) and not can_access(a, {"team-b"}) and can_access(a, None)
