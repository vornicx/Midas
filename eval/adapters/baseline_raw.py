from __future__ import annotations

from midas import approx_tokens

from ..schema import Event
from .base import RetrievalResult


class BaselineRawAdapter:
    """No memory layer: stuff the raw history into the context window, most-recent
    first, truncated to the budget. The 'just use a big context window' baseline —
    it pays the full token cost and silently drops whatever doesn't fit."""

    name = "baseline-raw"
    uses_internal_llm = False  # pure truncation, no LLM

    def __init__(self) -> None:
        self._events: list[Event] = []

    def reset(self) -> None:
        self._events = []

    def ingest(self, events: list[Event]) -> None:
        self._events.extend(events)

    def query(self, question: str, *, token_budget: int, now: float | None = None) -> RetrievalResult:
        lines: list[str] = []
        ids: list[str] = []
        used = 0
        for event in reversed(self._events):  # most recent first
            line = f"- {event.content}"
            cost = approx_tokens(line)
            if used + cost > token_budget:
                break
            lines.append(line)
            ids.append(event.id)
            used += cost
        lines.reverse()
        return RetrievalResult(
            context="\n".join(lines), retrieved_event_ids=ids, tokens=used
        )
