from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..schema import Event


@dataclass
class RetrievalResult:
    """What an adapter returns for a query: the assembled context to feed an LLM,
    the source event ids it surfaced (for recall@k), and its token cost."""

    context: str
    retrieved_event_ids: list[str] = field(default_factory=list)
    tokens: int = 0
    # Top retrieved turns in RELEVANCE order (the ones an answer was likely drawn from). Used by the
    # NLI answer-verifier so it checks contradiction against the *source*, not every distractor.
    top_texts: list[str] = field(default_factory=list)


@runtime_checkable
class MemoryAdapter(Protocol):
    """Uniform surface every system-under-test implements. This is also, on purpose,
    the shape of the standalone SDK (option B): ingest events, then query for context."""

    name: str

    def reset(self) -> None: ...

    def ingest(self, events: list[Event]) -> None: ...

    def query(
        self, question: str, *, token_budget: int, now: float | None = None
    ) -> RetrievalResult: ...
