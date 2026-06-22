"""Memory provenance control-plane and guard checks.

Midas stores verbatim source turns; this module decides how much an agent may rely on those
turns for a proposed use. The key boundary: low-trust memory can guide planning, but external
actions must be grounded in explicit user confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .types import MEMORY_PROVENANCE, MemoryProvenance, MemoryRecord

MemoryUse = Literal["planning", "answer", "external_action", "destructive_action"]
MEMORY_USES: tuple[MemoryUse, ...] = (
    "planning",
    "answer",
    "external_action",
    "destructive_action",
)

_ALLOWED_BY_USE: dict[MemoryUse, frozenset[MemoryProvenance]] = {
    "planning": frozenset(MEMORY_PROVENANCE),
    "answer": frozenset(("action", "observation", "user_confirmation")),
    "external_action": frozenset(("user_confirmation",)),
    "destructive_action": frozenset(("user_confirmation",)),
}


@dataclass(frozen=True)
class ProvenanceStamp:
    provenance: MemoryProvenance
    actor: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Armorer:
    """Small control-plane helper for stamping memories with provenance.

    `actor` should identify the agent/process that produced the memory. Another agent may still use
    the memory for planning, but Guard will require user confirmation before acting on it.
    """

    def __init__(self, *, actor: str | None = None, session: str = "default") -> None:
        self.actor = actor
        self.session = session

    def stamp(
        self,
        provenance: MemoryProvenance,
        *,
        source: str | None = None,
        **metadata: Any,
    ) -> ProvenanceStamp:
        _validate_provenance(provenance)
        data = {"session": self.session, **metadata}
        return ProvenanceStamp(
            provenance=provenance,
            actor=self.actor,
            source=source or f"session:{self.session}",
            metadata=data,
        )

    def remember_kwargs(
        self,
        provenance: MemoryProvenance,
        *,
        source: str | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        stamp = self.stamp(provenance, source=source, **metadata)
        return {
            "provenance": stamp.provenance,
            "actor": stamp.actor,
            "source": stamp.source,
            "metadata": stamp.metadata,
        }


@dataclass(frozen=True)
class EvidenceRef:
    id: str
    kind: str
    provenance: MemoryProvenance
    actor: str | None
    source: str | None
    content: str


@dataclass(frozen=True)
class MemoryUseDecision:
    allowed: bool
    intended_use: MemoryUse
    reason: str
    required_provenance: tuple[MemoryProvenance, ...]
    evidence: tuple[EvidenceRef, ...] = ()
    blocked_ids: tuple[str, ...] = ()


class Guard:
    """Policy evaluator for whether memory may justify a proposed use."""

    def __init__(self, *, acting_agent: str | None = None) -> None:
        self.acting_agent = acting_agent

    def decide(
        self,
        records: list[MemoryRecord] | tuple[MemoryRecord, ...],
        *,
        intended_use: MemoryUse,
    ) -> MemoryUseDecision:
        return decide_memory_use(records, intended_use=intended_use, acting_agent=self.acting_agent)


def decide_memory_use(
    records: list[MemoryRecord] | tuple[MemoryRecord, ...],
    *,
    intended_use: MemoryUse,
    acting_agent: str | None = None,
) -> MemoryUseDecision:
    """Return whether recalled memory is sufficient evidence for the intended use.

    Rules are deliberately conservative:
    - planning can use any provenance, because it does not change the outside world;
    - answers can use observations, completed actions, and user confirmations, but not internal plans;
    - external/destructive actions require `user_confirmation` memory. If an action was observed by
      agent A, agent B still needs fresh user approval before acting on it;
    - a superseded belief is no longer current truth, so it cannot justify an answer or an action on
      its own — even one that was user-confirmed when current (planning may still weigh the history).
      The guard verifies currency itself rather than trusting recall to have filtered the stale record.
    """
    if intended_use not in MEMORY_USES:
        allowed = ", ".join(MEMORY_USES)
        raise ValueError(f"invalid memory use '{intended_use}' (expected one of: {allowed})")
    allowed_prov = _ALLOWED_BY_USE[intended_use]
    evidence = tuple(_evidence_ref(r) for r in records)
    required = tuple(sorted(allowed_prov))
    if not records:
        return MemoryUseDecision(
            allowed=False,
            intended_use=intended_use,
            reason="no recalled memory evidence; ask or proceed without relying on memory",
            required_provenance=required,
            evidence=evidence,
        )

    supporting = []
    blocked = []
    for record in records:
        _validate_provenance(record.provenance)
        if record.provenance not in allowed_prov:
            blocked.append(record.id)
            continue
        # Currency (defense-in-depth): a superseded belief is stale, so on its own it cannot justify an
        # answer or an action — even if it was user-confirmed when current. Planning may still weigh
        # superseded history. The guard checks this itself instead of trusting recall to have filtered it.
        if intended_use != "planning" and record.superseded_by is not None:
            blocked.append(record.id)
            continue
        if (
            intended_use in ("external_action", "destructive_action")
            and record.provenance != "user_confirmation"
        ):
            blocked.append(record.id)
            continue
        if (
            intended_use in ("external_action", "destructive_action")
            and acting_agent
            and record.actor
            and record.actor != acting_agent
            and record.provenance != "user_confirmation"
        ):
            blocked.append(record.id)
            continue
        supporting.append(record)

    if not supporting:
        return MemoryUseDecision(
            allowed=False,
            intended_use=intended_use,
            reason=(
                f"{intended_use} requires memory provenance in {list(required)}; "
                "ask the user to confirm before relying on blocked memory"
            ),
            required_provenance=required,
            evidence=evidence,
            blocked_ids=tuple(blocked),
        )
    reason = f"{len(supporting)} supporting memories satisfy the {intended_use} provenance policy"
    if blocked:
        reason += f"; {len(blocked)} recalled memories were excluded as insufficient evidence"
    return MemoryUseDecision(
        allowed=True,
        intended_use=intended_use,
        reason=reason,
        required_provenance=required,
        evidence=tuple(_evidence_ref(r) for r in supporting),
        blocked_ids=tuple(blocked),
    )


def _evidence_ref(record: MemoryRecord) -> EvidenceRef:
    return EvidenceRef(
        id=record.id,
        kind=record.kind,
        provenance=record.provenance,
        actor=record.actor,
        source=record.source,
        content=record.content,
    )


def _validate_provenance(value: str) -> None:
    if value not in MEMORY_PROVENANCE:
        allowed = ", ".join(MEMORY_PROVENANCE)
        raise ValueError(f"invalid memory provenance '{value}' (expected one of: {allowed})")
