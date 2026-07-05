"""Continuity control-plane — resume a long-horizon agent in one call, and keep multi-agent memory
coherent.

Three primitives that similarity search cannot express, all deterministic and LLM-free:

  memory_conflicts(mem)   live beliefs that CONTRADICT each other and neither superseded the other —
                          the failure mode of one memory shared by many agents (Claude Code writes
                          "the DB is PostgreSQL", Cursor writes "the DB is MySQL", both stay live).
                          Supersession only revises what the SAME write path saw; this surfaces what
                          slipped through, so a human (or the agent) can resolve it.
  open_loops(mem)         unresolved commitments — work an agent said it would do and never closed.
                          Continuity is not only facts: it is unfinished intentions.
  resume(mem)             the session-onboarding pack in ONE call: pinned directives, forbidden rules,
                          current state, what changed since last time, open loops, and unresolved
                          conflicts — token-budgeted and prompt-ready.

Conflict detection composes the same gates belief revision uses (embedding similarity + topical anchor
+ entity overlap) with a contradiction signal: the local NLI model when available, else a same-slot/
different-value heuristic (numbers disagree, or one side negates the other). Approximate by design —
it returns *candidates to resolve*, ranked, never auto-deletes anything.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .memory import _content_words, _fmt_date, _proper_entities, approx_tokens, format_record
from .state import DURABLE_KINDS, MemoryDiff, _scope_match, memory_diff, memory_state
from .types import MemoryRecord

if TYPE_CHECKING:
    from .memory import Memory

_NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")
_NEG_RE = re.compile(
    r"\b(?:not|no longer|never|don'?t|doesn'?t|isn'?t|aren'?t|won'?t|can'?t|cannot|"
    r"shouldn'?t|stop(?:ped)?|forbidden|banned|disallowed|deprecated)\b",
    re.IGNORECASE,
)
# Sentence-initial capitalised function/cue words that _proper_entities picks up as noise ("The
# launch…", "Update: …"). Harmless for supersession (a negative guard), fatal here — they make every
# pair look like it names different things — so the conflict detector strips them.
_ENTITY_NOISE = frozenset(
    "the a an this that these those we i you it they he she update updated now note reminder new "
    "also actually however meanwhile today yesterday tomorrow from after before once finally".split()
)


def _entities(text: str) -> set[str]:
    return _proper_entities(text) - _ENTITY_NOISE


def _numbers(text: str) -> set[str]:
    return set(_NUM_RE.findall(text))


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


@dataclass(frozen=True)
class Conflict:
    """Two live records that appear to disagree. `signal` says why: "nli" (the local NLI model scored
    them contradictory), "numeric" (same topic, different numbers), or "negation" (one side negates).
    `strength` ranks conflicts across signals (NLI contradiction score, else similarity)."""

    a: MemoryRecord
    b: MemoryRecord
    similarity: float
    signal: str
    strength: float


def _contradiction(mem: "Memory", a: MemoryRecord, b: MemoryRecord, similarity: float,
                   nli: Any, nli_threshold: float) -> tuple[str, float] | None:
    """The contradiction signal for a same-topic candidate pair, or None if they merely relate."""
    if _norm(a.content) == _norm(b.content):
        return None  # restatements agree — that's consolidation work, not a conflict
    # Topical anchor (the supersession gate): the pair must share salient words at all.
    shared = _content_words(a.content) & _content_words(b.content)
    if len(shared) < 2:
        return None
    if nli is not None:
        # Contradiction is not symmetric in NLI models; take the stronger direction.
        score = max(nli.contradiction(a.content, b.content), nli.contradiction(b.content, a.content))
        return ("nli", score) if score >= nli_threshold else None
    # Heuristic fallback — "same slot, different value", three ways to disagree. `da`/`db` are the
    # named things each side mentions that the other does not (shared entities cancel out).
    na, nb = _numbers(a.content), _numbers(b.content)
    ea, eb = _entities(a.content), _entities(b.content)
    da, db = ea - eb, eb - ea
    # 1) the frame (content words minus the named value) matches but ONE named entity differs per
    #    side: "the primary database is PostgreSQL" vs "... is MySQL". Restricted to agreeing numbers
    #    — several differing names AND differing numbers means two facts about two subjects
    #    (Apollo launches May 3 / Artemis launches June 9), not one slot with two values.
    frame_a, frame_b = _content_words(a.content) - ea - na, _content_words(b.content) - eb - nb
    if len(da) == 1 and len(db) == 1 and na == nb and len(frame_a & frame_b) >= 2:
        return ("value", similarity)
    # For the remaining signals, each side naming its own thing means different facts, not a conflict.
    if da and db:
        return None
    # 2) same topic, different numbers: "launch is Sept 14" vs "launch is Sept 20".
    if na and nb and na != nb:
        return ("numeric", similarity)
    # 3) exactly one side negates: "we support X" vs "we no longer support X".
    if bool(_NEG_RE.search(a.content)) != bool(_NEG_RE.search(b.content)):
        return ("negation", similarity)
    return None


def memory_conflicts(
    mem: "Memory",
    *,
    scope: dict[str, Any] | None = None,
    kinds: tuple[str, ...] | None = DURABLE_KINDS,
    min_similarity: float = 0.35,
    nli: Any | None = None,
    nli_threshold: float = 0.5,
    pool: int = 500,
    neighbors: int = 10,
    limit: int = 20,
) -> list[Conflict]:
    """Live beliefs that contradict each other with neither superseding the other — ranked candidates
    for resolution (confirm one, forget the other, or capture the corrected value; Midas never resolves
    them silently). Uses the memory's own NLI model when it has one (pass `nli=` to override); without
    NLI it falls back to a same-slot heuristic: numbers disagree, or exactly one side negates.
    Bounded work: the newest `pool` live records × `neighbors` nearest each. No LLM."""
    nli = nli if nli is not None else getattr(mem, "_nli", None)
    live = [
        r for r in mem.store.all()
        if r.superseded_by is None and r.embedding is not None
        and (kinds is None or r.kind in kinds) and _scope_match(r, scope)
    ]
    live.sort(key=lambda r: r.updated_at, reverse=True)
    live = live[:pool]
    live_ids = {r.id for r in live}

    conflicts: list[Conflict] = []
    seen: set[tuple[str, str]] = set()
    for record in live:
        near = mem.store.search(
            record.embedding, limit=neighbors,
            predicate=lambda r, rid=record.id: r.id != rid and r.id in live_ids,
        )
        for score, other in near:
            if score < min_similarity:
                break  # search is sorted by similarity desc
            pair = (record.id, other.id) if record.id < other.id else (other.id, record.id)
            if pair in seen:
                continue
            seen.add(pair)
            hit = _contradiction(mem, record, other, score, nli, nli_threshold)
            if hit:
                signal, strength = hit
                conflicts.append(Conflict(a=record, b=other, similarity=score,
                                          signal=signal, strength=strength))
    conflicts.sort(key=lambda c: c.strength, reverse=True)
    return conflicts[:limit]


# ---- open loops (commitments) -------------------------------------------------------------------

def remember_commitment(
    mem: "Memory",
    content: str,
    *,
    project: str | None = None,
    due: str | None = None,
    actor: str | None = None,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MemoryRecord:
    """Record a commitment — work the agent (or user) said WILL be done. Importance 4 by default so
    open promises don't decay out of memory; close it with `close_loop` when the work lands."""
    meta = {**(metadata or {})}
    if project:
        meta["project"] = project
    if due:
        meta["due"] = due
    return mem.remember(content, kind="commitment", importance=4, provenance="planning",
                        actor=actor, source=source, metadata=meta)


def open_loops(
    mem: "Memory",
    *,
    scope: dict[str, Any] | None = None,
    limit: int = 50,
) -> list[MemoryRecord]:
    """Live (unclosed) commitments, OLDEST first — the longest-open promise is the most overdue.
    Closed loops are superseded by their resolution, so they drop out here but keep their history."""
    records = [
        r for r in mem.store.all()
        if r.superseded_by is None and r.kind == "commitment"
        and not r.metadata.get("closes")  # a resolution record is the close, not a new loop
        and _scope_match(r, scope)
    ]
    records.sort(key=lambda r: r.created_at)
    return records[:limit] if limit else records


def close_loop(mem: "Memory", loop_id: str, resolution: str, *, actor: str | None = None) -> MemoryRecord:
    """Close a commitment: store the resolution (provenance=action) and supersede the open loop with
    it, so `open_loops` no longer returns it while the full promise→resolution chain stays auditable."""
    target = mem.store.get(loop_id)
    if target is None or target.kind != "commitment":
        raise ValueError(f"no open commitment with id '{loop_id}'")
    closing = mem.remember(
        f"Done: {resolution}", kind="commitment", importance=2, provenance="action", actor=actor,
        metadata={k: v for k, v in target.metadata.items() if k in ("project", "namespace")}
        | {"closes": loop_id},
    )
    target.superseded_by = closing.id
    target.metadata = {**target.metadata, "superseded_at": closing.created_at}
    mem.store.put(target)
    return closing


# ---- resume: the one-call session-onboarding pack -------------------------------------------------

def resume(
    mem: "Memory",
    *,
    scope: dict[str, Any] | None = None,
    project: str | None = None,
    since: float | None = None,
    token_budget: int = 900,
    now: float | None = None,
) -> dict[str, Any]:
    """Everything an agent needs to pick up where it left off, in one deterministic call: pinned
    standing directives, live forbidden rules, the current durable state, what changed since `since`
    (default: the last 7 days), open commitments, and unresolved conflicts. `context` is the same
    content as a token-budgeted, prompt-ready block (sections dropped tail-first when over budget).
    Complements `build_context` (query-driven recall); this is the query-less session start. No LLM."""
    now = now if now is not None else time.time()
    since = since if since is not None else now - 7 * 86_400.0
    if project:
        scope = {**(scope or {}), "project": project}

    live = [r for r in mem.store.all() if r.superseded_by is None and _scope_match(r, scope)]
    pinned = sorted((r for r in live if r.metadata.get("standing") or r.kind == "mission"),
                    key=lambda r: r.updated_at, reverse=True)
    forbidden = sorted((r for r in live if r.metadata.get("code_kind") == "forbidden_action"),
                       key=lambda r: r.updated_at, reverse=True)
    skip = {r.id for r in pinned} | {r.id for r in forbidden}
    state = [r for r in memory_state(mem, scope=scope, limit=0) if r.id not in skip]
    diff: MemoryDiff = memory_diff(mem, since, scope=scope)
    loops = open_loops(mem, scope=scope)
    conflicts = memory_conflicts(mem, scope=scope, limit=5)

    lines: list[str] = [f"RESUME (today is {_fmt_date(now)}; changes since {_fmt_date(since)})"]
    used = approx_tokens(lines[0])
    truncated = False

    def emit(header: str, rows: list[str]) -> None:
        nonlocal used, truncated
        if not rows:
            return
        block = [header] + rows
        for line in block:
            cost = approx_tokens(line)
            if used + cost > token_budget:
                truncated = True
                return
            lines.append(line)
            used += cost

    fmt = lambda r: format_record(r, now=now)  # noqa: E731
    emit("PINNED:", [fmt(r) for r in pinned])
    emit("FORBIDDEN:", [fmt(r) for r in forbidden])
    emit("CHANGED (added):", [fmt(r) for r in diff.added])
    emit("CHANGED (revised):",
         [f"{fmt(o)}\n    -> now: {' '.join(n.content.split())[:300]}" for o, n in diff.revised])
    emit("CURRENT STATE:", [fmt(r) for r in state])
    emit("OPEN LOOPS:", [fmt(r) for r in loops])
    emit("CONFLICTS (unresolved — verify before relying on either):",
         [f"- [{c.signal}] \"{' '.join(c.a.content.split())[:200]}\"  vs  "
          f"\"{' '.join(c.b.content.split())[:200]}\"" for c in conflicts])

    return {
        "generated_at": now,
        "since": since,
        "pinned": pinned,
        "forbidden": forbidden,
        "state": state,
        "added": diff.added,
        "revised": diff.revised,
        "open_loops": loops,
        "conflicts": conflicts,
        "context": "\n".join(lines),
        "truncated": truncated,
    }
