"""Midas memory types.

The record shape intentionally mirrors the TypeScript Atlas prototype
(`kind`, `importance` 1-5, `content`, `metadata`) so the model stays portable
across the eventual TS/Python split.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

MemoryKind = Literal["note", "chat", "mission", "fact", "preference", "constraint"]
MEMORY_KINDS: tuple[MemoryKind, ...] = (
    "note",
    "chat",
    "mission",
    "fact",
    "preference",
    "constraint",
)

MemoryProvenance = Literal["planning", "action", "observation", "user_confirmation"]
MEMORY_PROVENANCE: tuple[MemoryProvenance, ...] = (
    "planning",
    "action",
    "observation",
    "user_confirmation",
)


@dataclass
class MemoryRecord:
    id: str
    content: str
    kind: MemoryKind = "note"
    importance: int = 1  # 1 (incidental) .. 5 (critical)
    source: str | None = None
    provenance: MemoryProvenance = "observation"
    actor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)  # epoch seconds
    updated_at: float = field(default_factory=time.time)
    embedding: list[float] | None = None
    superseded_by: str | None = None  # id of the memory that updates/replaces this one (belief revision)


@dataclass
class RecallHit:
    """A recalled record with its blended score and the components behind it
    (the breakdown is kept visible on purpose — it is part of what we evaluate)."""

    record: MemoryRecord
    score: float
    relevance: float
    importance_norm: float
    recency: float
