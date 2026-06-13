"""Auto-memory policy — the parameters Midas imposes so an agent remembers the *relevant* things.

Two halves work together:
  - `MemoryPolicy`: the machine-enforced parameters (importance floor, accepted kinds, dedup). When the
    agent calls `capture`, Midas applies these and *rejects* trivia/duplicates — so relevance is enforced
    by Midas, not left to the agent's judgement.
  - `AGENT_MEMORY_INSTRUCTIONS`: the prompt the MCP server injects into the agent (via the MCP
    `instructions` field, surfaced automatically on connection) telling it the recall -> work -> capture
    loop. The agent captures liberally; Midas's policy decides what is actually kept.

This is what makes "install Midas and it starts remembering" work: the instructions drive the agent to
capture, and the policy (below) imposes what's worth keeping — all with no LLM at ingest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Kinds that carry durable, reusable signal (excludes nothing by default — the importance floor does the
# filtering; this is here so a deployment can, say, only auto-keep typed facts/preferences/constraints).
_DEFAULT_KINDS: tuple[str, ...] = ("note", "chat", "fact", "preference", "constraint", "mission")


@dataclass(frozen=True)
class MemoryPolicy:
    """The relevance parameters Midas imposes when auto-capturing a turn.

    min_importance: content scoring below this (via `ContentImportance`, 1-5) is too trivial to store —
        the no-LLM gate that keeps chit-chat out. Default 2 keeps anything with a real fact/name/number.
    accept_kinds: only these kinds are auto-captured.
    dedup_threshold: skip a turn whose cosine similarity to an existing memory is >= this (0 disables) —
        so restating a known fact doesn't pile up duplicates.
    """

    min_importance: int = 2
    accept_kinds: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_KINDS)
    dedup_threshold: float = 0.95


DEFAULT_POLICY = MemoryPolicy()


# The prompt the MCP server injects into the agent (MCP `instructions`). Written to be model-agnostic and
# short — it is surfaced on every connection, so it states the loop and trusts Midas to enforce the rest.
AGENT_MEMORY_INSTRUCTIONS = (
    "Use Midas memory on every task. It is local, source-traceable, and uses no LLM at ingest/query.\n\n"
    "1) RECALL FIRST. Call `build_context` with the user's goal; use the returned facts silently. Use "
    "`recall`/`inspect_memory` only when you need audit details.\n\n"
    "2) CAPTURE DURABLE SIGNAL — DISTILLED. Call `capture` for reusable facts, decisions, preferences, "
    "constraints, corrections, and completed actions. Prefer ONE compact, self-contained statement "
    "(the entities, the value, and when) over raw turns — a memory that answers on its own retrieves "
    "far better than a conversational fragment. Skip pure small talk. Midas scores, dedups, and rejects "
    "trivia, so capture can be brief and needs no LLM. Set kind/provenance accurately; use "
    "provenance=\"user_confirmation\" only for explicit user confirmation.\n\n"
    "3) GUARD ACTIONS. Memory may guide planning, but before external/destructive actions based on memory, "
    "call `check_memory_use`. If it is not allowed, ask the user to confirm in this turn.\n\n"
    "4) FORGET ON REQUEST. Use `forget_matching` as a dry-run first, show matches, then repeat with "
    "dry_run=false after confirmation.\n\n"
    "Midas stores verbatim source records and bounds memory automatically; compact context is for cheap "
    "reader prompts, audit tools are for traceability."
)


def policy_summary(policy: MemoryPolicy) -> str:
    """One-line, human/agent-readable summary of the enforced parameters (for prompts / `stats`)."""
    return (
        f"keep items scoring importance >= {policy.min_importance}/5, "
        f"kinds {list(policy.accept_kinds)}, "
        f"skip near-duplicates (cosine >= {policy.dedup_threshold}); "
        "guard external/destructive actions to user_confirmation provenance"
    )
