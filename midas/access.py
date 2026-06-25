"""Access control (RBAC) — scope a memory store to what a caller may see.

Multi-tenant / multi-team Midas: team A must not be able to recall team B's memories. Records carry a
scope tag in `metadata` (namespace / project / user); a caller is granted a set of allowed scopes, and
this layer filters the store to those. Deterministic, no LLM. Unscoped records are treated as **global**
(visible to everyone) unless `strict=True` (then a caller sees only its own scopes).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .types import MemoryRecord

if TYPE_CHECKING:
    from .memory import Memory


def can_access(
    record: MemoryRecord,
    allowed_scopes: Iterable[str] | None,
    *,
    scope_key: str = "namespace",
    strict: bool = False,
) -> bool:
    """Whether a caller granted `allowed_scopes` may read `record`. `None` ⇒ unrestricted. An unscoped
    record (no `metadata[scope_key]`) is global unless `strict`."""
    if allowed_scopes is None:
        return True
    scope = record.metadata.get(scope_key)
    if scope is None:
        return not strict
    return scope in set(allowed_scopes)


def visible_records(
    mem: "Memory",
    allowed_scopes: Iterable[str] | None,
    *,
    scope_key: str = "namespace",
    strict: bool = False,
) -> list[MemoryRecord]:
    """The records a caller granted `allowed_scopes` may see (`None` ⇒ all). The deterministic RBAC view
    over the store; pair it with `recall`'s `metadata_filter` to enforce scope at query time too."""
    if allowed_scopes is None:
        return list(mem.store.all())
    allowed = set(allowed_scopes)
    return [
        r for r in mem.store.all() if can_access(r, allowed, scope_key=scope_key, strict=strict)
    ]
