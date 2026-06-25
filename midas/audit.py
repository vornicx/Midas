"""Audit trail — the compliance artifact for a governed memory layer.

A regulated deployment (the story Banco Santander AI Lab's mech-gov-framework validates) must be able to
PROVE, after the fact, why an agent did or did not act: which memories justified it, who produced them and
when, with what provenance, whether they were still current, and what prior beliefs they revised. The
guard already decides; this module packages the *evidence* into a reproducible, source-traceable record.
No LLM.

  belief_history(mem, id)         the full supersession chain a record belongs to (oldest → newest) —
                                  "what did we believe, and what did each version replace, and when".
  audit_completeness(records)     fraction of evidence fully attributable (has both a source and an actor)
                                  — a compliance score: how much of what justified an action is traceable.
  audit_use(mem, query, use)      the guard decision + each evidence record's full provenance + its belief
                                  history + the completeness score — the artifact you hand an auditor.
"""
from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

from .types import MemoryRecord

if TYPE_CHECKING:
    from .memory import Memory


def forgetting_receipt(
    removed: list[MemoryRecord], *, actor: str | None = None, reason: str | None = None
) -> dict[str, Any]:
    """A tamper-evident **erasure certificate** — prove WHAT was forgotten without retaining the content
    (the right-to-be-forgotten artifact). For each removed record it keeps the id + a sha256 of
    `id\\x00content`: enough to later prove a specific item was erased, not enough to reconstruct it. Pair
    with `Memory.forget_matching(...)`'s returned records. Deterministic, no LLM."""
    return {
        "forgotten_count": len(removed),
        "at": time.time(),
        "actor": actor,
        "reason": reason,
        "items": [
            {
                "id": r.id,
                "kind": r.kind,
                "content_sha256": hashlib.sha256(f"{r.id}\x00{r.content}".encode()).hexdigest(),
            }
            for r in removed
        ],
    }


def audit_record(r: MemoryRecord) -> dict[str, Any]:
    """The complete, source-traceable audit fields of one memory (no embedding)."""
    return {
        "id": r.id,
        "content": r.content,
        "kind": r.kind,
        "importance": r.importance,
        "provenance": r.provenance,
        "actor": r.actor,
        "source": r.source,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
        "superseded_by": r.superseded_by,
    }


def belief_history(mem: "Memory", record_id: str) -> list[MemoryRecord]:
    """The full belief-revision chain `record_id` belongs to, OLDEST → newest. Reconstructs the timeline
    from the `superseded_by` links: each version and what replaced it, so an auditor can see "we believed
    v1 from t1, revised to v2 at t2, …, current is vN". Empty if the id is unknown."""
    by_id = {r.id: r for r in mem.store.all()}
    if record_id not in by_id:
        return []
    # superseded_by points at the NEWER record; map newer_id -> the record it replaced (older).
    replaced_by_new = {r.superseded_by: r.id for r in by_id.values() if r.superseded_by}

    # Walk back to the oldest ancestor.
    oldest, seen = record_id, {record_id}
    while oldest in replaced_by_new and replaced_by_new[oldest] not in seen:
        oldest = replaced_by_new[oldest]
        seen.add(oldest)

    # Walk forward via superseded_by to the current head, collecting the chain.
    chain: list[MemoryRecord] = []
    cur: str | None = oldest
    walked: set[str] = set()
    while cur and cur in by_id and cur not in walked:
        walked.add(cur)
        rec = by_id[cur]
        chain.append(rec)
        cur = rec.superseded_by
    return chain


def audit_completeness(records: list[MemoryRecord]) -> float:
    """Fraction of `records` fully attributable — has BOTH a `source` and an `actor`. A compliance score:
    how much of the evidence behind an action can be traced to who produced it and where. 1.0 for empty."""
    if not records:
        return 1.0
    full = sum(1 for r in records if r.source and r.actor)
    return full / len(records)


def audit_use(
    mem: "Memory",
    query: str,
    intended_use: str,
    **recall_kwargs: Any,
) -> dict[str, Any]:
    """The audit artifact for a proposed memory-justified use: the guard decision + the full provenance
    and belief history of each supporting memory + the attributability score. What you hand an auditor to
    answer "why did the agent (not) do this?" — reproducible and source-traceable, no LLM."""
    decision = mem.guard_reliance(query, intended_use=intended_use, **recall_kwargs)
    by_id = {r.id: r for r in mem.store.all()}
    evidence_records = [by_id[e.id] for e in decision.evidence if e.id in by_id]
    return {
        "query": query,
        "intended_use": decision.intended_use,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "required_provenance": list(decision.required_provenance),
        "audit_completeness": audit_completeness(evidence_records),
        "evidence": [
            {**audit_record(r), "belief_history": [audit_record(h) for h in belief_history(mem, r.id)]}
            for r in evidence_records
        ],
        "blocked_ids": list(decision.blocked_ids),
    }
