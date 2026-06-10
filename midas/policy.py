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
    "You have a persistent Midas memory: local, source-traceable, with no LLM at ingest. Use it on "
    "every task.\n\n"
    "1) RECALL FIRST. Before answering or acting, call `build_context` (or `recall`) with the user's "
    "goal to load what you already know about them: prior decisions, stated preferences, established "
    "facts, hard constraints, and past corrections. Use it silently to stay consistent.\n\n"
    "2) CAPTURE AS YOU GO. Call `capture` to save anything durable and reusable across sessions. You do "
    "NOT need to judge relevance perfectly — Midas scores each item and skips trivia and duplicates for "
    "you, telling you what it kept. When in doubt, capture. Especially capture:\n"
    "   - facts the user states about themselves, their project, or their environment  (kind=\"fact\")\n"
    "   - decisions and their rationale  (kind=\"note\")\n"
    "   - the user's stated preferences  (kind=\"preference\")\n"
    "   - hard requirements / constraints  (kind=\"constraint\")\n"
    "   - corrections — when the user overrides or changes something said earlier\n"
    "   Tag provenance on every `remember`/`capture` call:\n"
    "   - provenance=\"planning\" for internal plans, hypotheses, or proposed next steps.\n"
    "   - provenance=\"action\" for completed agent/tool actions and their observed result.\n"
    "   - provenance=\"observation\" for passive observations from files, tools, logs, or retrieved data.\n"
    "   - provenance=\"user_confirmation\" only when the user explicitly confirms the content.\n"
    "   Skip pure small talk and acknowledgements; Midas enforces the relevance floor regardless.\n\n"
    "3) GUARD MEMORY BEFORE ACTION. Memory can guide planning, but it cannot by itself authorize "
    "external or destructive actions. Before relying on recalled memory to act outside the chat, call "
    "`check_memory_use` with intended_use=\"external_action\" or \"destructive_action\". If the decision "
    "is not allowed, ask the user to confirm in the current turn. External/destructive actions may rely "
    "only on user_confirmation provenance.\n\n"
    "Everything is stored verbatim with its source, so recall is auditable, and memory is bounded "
    "automatically (low-value, stale items are forgotten) — so capturing freely is safe and cheap."
)


def policy_summary(policy: MemoryPolicy) -> str:
    """One-line, human/agent-readable summary of the enforced parameters (for prompts / `stats`)."""
    return (
        f"keep items scoring importance >= {policy.min_importance}/5, "
        f"kinds {list(policy.accept_kinds)}, "
        f"skip near-duplicates (cosine >= {policy.dedup_threshold}); "
        "guard external/destructive actions to user_confirmation provenance"
    )
