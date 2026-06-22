"""Memory control-plane views — the project-state layer over raw recall.

Top-k recall answers "find the turn about X". A long-horizon agent also needs two things a similarity
search is bad at, because the query ("what's the current state?", "what changed?") doesn't resemble any
single stored turn:

  memory_state(mem)  the CURRENT durable state of a scope/project — the live, non-superseded
                     decisions/constraints/facts/preferences, newest first. For onboarding into a
                     project ("what do we currently believe about Apollo?") and for planning.
  memory_diff(mem)   what CHANGED since a timestamp — beliefs newly added and beliefs revised
                     (old -> new). The "what's new since our last session" primitive.

Both are deterministic and call no LLM — they read the store and the supersession links Midas already
maintains. They complement `recall`/`build_context` (similarity) rather than replace them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .types import MemoryKind, MemoryRecord

if TYPE_CHECKING:
    from .memory import Memory

# The durable, decision-bearing kinds — a project's "state" is its constraints/facts/preferences/
# missions, not the raw chat turns or scratch notes (those stay recall-only).
DURABLE_KINDS: tuple[MemoryKind, ...] = ("constraint", "fact", "preference", "mission")


def _scope_match(record: MemoryRecord, scope: dict[str, Any] | None) -> bool:
    """Equality scope (namespace / project / user / agent): every key in `scope` must match the
    record's metadata. `None`/empty scope matches everything."""
    if not scope:
        return True
    return all(record.metadata.get(k) == v for k, v in scope.items())


def memory_state(
    mem: "Memory",
    *,
    scope: dict[str, Any] | None = None,
    kinds: tuple[MemoryKind, ...] | None = DURABLE_KINDS,
    limit: int = 50,
) -> list[MemoryRecord]:
    """The live durable state of a scope: non-superseded records of `kinds`, newest-first. Not a
    similarity search — it returns the project's current beliefs regardless of query phrasing, which is
    exactly where broad "what's the current state?" questions under-retrieve with top-k. No LLM."""
    records = [
        r
        for r in mem.store.all()
        if r.superseded_by is None
        and (kinds is None or r.kind in kinds)
        and _scope_match(r, scope)
    ]
    records.sort(key=lambda r: r.updated_at, reverse=True)
    return records[:limit] if limit else records


@dataclass(frozen=True)
class MemoryDiff:
    """What changed in a scope since a point in time. `added` = new current beliefs; `revised` =
    (old, new) pairs where a belief was superseded by a later one."""

    added: list[MemoryRecord] = field(default_factory=list)
    revised: list[tuple[MemoryRecord, MemoryRecord]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.added and not self.revised


def memory_diff(
    mem: "Memory",
    since: float,
    *,
    scope: dict[str, Any] | None = None,
) -> MemoryDiff:
    """Beliefs added or revised since `since` (epoch seconds). A revision is dated by when its
    replacement was created (Midas stores the supersession link, not a separate timestamp). The
    "what's new since our last session" view for long-horizon agents. No LLM."""
    records = list(mem.store.all())
    by_id = {r.id: r for r in records}
    # ids that REPLACE an older belief — their "new" record is a revision, not a fresh add.
    superseding_ids = {r.superseded_by for r in records if r.superseded_by is not None}

    added: list[MemoryRecord] = []
    revised: list[tuple[MemoryRecord, MemoryRecord]] = []
    for r in records:
        if not _scope_match(r, scope):
            continue
        if r.superseded_by is not None:
            new = by_id.get(r.superseded_by)
            if new is not None and new.created_at >= since:
                revised.append((r, new))
        elif r.created_at >= since and r.id not in superseding_ids:
            added.append(r)

    added.sort(key=lambda r: r.created_at, reverse=True)
    revised.sort(key=lambda pair: pair[1].created_at, reverse=True)
    return MemoryDiff(added=added, revised=revised)
