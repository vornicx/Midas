"""Dataset schema shared by every loader and adapter.

Keeping one neutral shape (Event / Question / Sample / Dataset) is what lets the
synthetic set, LoCoMo and LongMemEval all flow through the same runner unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """One ingestible memory event: a conversation turn, a fact, a doc chunk."""

    id: str
    content: str
    kind: str = "chat"  # chat | fact | constraint | preference | note | mission
    importance: int = 1
    speaker: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    id: str
    text: str
    answer: str | None = None  # gold answer (for correctness); None = unanswerable (should abstain)
    gold_event_ids: list[str] = field(default_factory=list)  # supporting events (recall@k)
    category: str = "fact"  # fact | current | stable | unanswerable | confusable | temporal | …
    stale_answer: str | None = None  # for 'current'/'confusable': outdated or wrong-entity trap value
    metadata: dict[str, Any] = field(default_factory=dict)  # e.g. {"asked_at": epoch} for time-aware recall


@dataclass
class Sample:
    """A long-horizon scenario: a stream of events plus questions asked over it."""

    id: str
    events: list[Event]
    questions: list[Question]


@dataclass
class Dataset:
    name: str
    samples: list[Sample]
