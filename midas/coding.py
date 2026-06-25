"""Coding-agent memory vocabulary — the Fase-B vertical layer.

Generic memory stores "facts". A long-horizon *coding* agent needs a sharper vocabulary: which
architecture decisions are live, which bugs were already fixed, what conventions apply, and which
actions are forbidden. This module is a thin, NON-INVASIVE layer over the core: each code memory maps
to an existing `MemoryKind` plus a `metadata["code_kind"]` tag and a `project` scope — so core recall,
supersession, forgetting, and the Guard all keep working unchanged, while `project_state` gives a code
agent a structured onboarding view ("what do we currently believe about this project, by category").

No new core types, no LLM. The capture helpers stamp the right kind/importance/provenance so the
governance layer behaves correctly (e.g. a `forbidden_action` is a user-confirmed constraint).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .types import MemoryKind, MemoryProvenance, MemoryRecord

if TYPE_CHECKING:
    from .memory import Memory

# code_kind -> (core MemoryKind, default importance, default provenance).
# The mapping is the point: it encodes how each code memory should behave in retrieval and governance.
_CODE_KIND_MAP: dict[str, tuple[MemoryKind, int, MemoryProvenance]] = {
    "architecture_decision": ("constraint", 5, "observation"),
    "dependency_choice":     ("fact", 4, "observation"),
    "convention":            ("constraint", 4, "observation"),
    "bug_fixed":             ("note", 3, "action"),
    "recurring_failure":     ("note", 4, "action"),
    # forbidden actions are user-confirmed constraints — they are meant to gate what the agent may do.
    "forbidden_action":      ("constraint", 5, "user_confirmation"),
    "command_worked":        ("note", 2, "action"),
    "command_failed":        ("note", 3, "action"),
}
CODE_KINDS: tuple[str, ...] = tuple(_CODE_KIND_MAP)


def remember_code(
    mem: "Memory",
    content: str,
    code_kind: str,
    *,
    project: str,
    provenance: MemoryProvenance | None = None,
    importance: int | None = None,
    actor: str | None = None,
    metadata: dict[str, Any] | None = None,
    created_at: float | None = None,
) -> MemoryRecord:
    """Store a code memory: it maps to a core kind + a `code_kind` tag + a `project` scope, so the rest
    of Midas treats it like any other record while `project_state` can present it by category. Pass an
    explicit `provenance`/`importance` to override the per-kind default (e.g. an architecture decision
    the user actually confirmed → `provenance="user_confirmation"`)."""
    try:
        core_kind, def_importance, def_provenance = _CODE_KIND_MAP[code_kind]
    except KeyError:
        raise ValueError(f"unknown code_kind '{code_kind}' (expected one of: {', '.join(CODE_KINDS)})")
    meta = {**(metadata or {}), "code_kind": code_kind, "project": project}
    return mem.remember(
        content,
        kind=core_kind,
        importance=def_importance if importance is None else importance,
        provenance=def_provenance if provenance is None else provenance,
        actor=actor,
        metadata=meta,
        created_at=created_at,
    )


# Thin, first-class wrappers for the most common code memories (DX for the vertical).
def remember_architecture_decision(mem: "Memory", content: str, *, project: str, **kw: Any) -> MemoryRecord:
    return remember_code(mem, content, "architecture_decision", project=project, **kw)


def remember_bug_fixed(mem: "Memory", content: str, *, project: str, **kw: Any) -> MemoryRecord:
    return remember_code(mem, content, "bug_fixed", project=project, **kw)


def remember_convention(mem: "Memory", content: str, *, project: str, **kw: Any) -> MemoryRecord:
    return remember_code(mem, content, "convention", project=project, **kw)


def remember_forbidden_action(mem: "Memory", content: str, *, project: str, **kw: Any) -> MemoryRecord:
    return remember_code(mem, content, "forbidden_action", project=project, **kw)


def project_state(mem: "Memory", project: str, *, limit: int = 200) -> dict[str, list[MemoryRecord]]:
    """The current code-state of a project, grouped by `code_kind` — the onboarding view for a coding
    agent. Returns only LIVE (non-superseded) code memories in the project, newest-first within each
    group, so a revised architecture decision shows its current value and a retired forbidden rule
    drops out. Deterministic, no LLM."""
    records = [
        r
        for r in mem.store.all()
        if r.superseded_by is None
        and r.metadata.get("project") == project
        and r.metadata.get("code_kind")
    ]
    records.sort(key=lambda r: r.updated_at, reverse=True)
    grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
    for r in records[:limit]:
        grouped[r.metadata["code_kind"]].append(r)
    return dict(grouped)


_WORD_RE = re.compile(r"[a-z0-9]+")
# Glue words that shouldn't count toward an action↔rule match (the rule's *negation* — "never", "don't"
# — is intentionally dropped: a forbidden rule and its action describe the SAME act, just framed opposite).
_STOP = frozenset(
    "a an the to of in on at for and or but no not never dont do does did is are was were be been "
    "this that these those it its with without your you we i our us me my".split()
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 1 and w not in _STOP}


def _cosine(a: Any, b: Any) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na and nb else 0.0


def is_forbidden(
    mem: "Memory",
    action: str,
    project: str,
    *,
    min_overlap: float = 0.5,
    min_shared: int = 2,
    min_similarity: float = 0.58,
    use_embeddings: bool = True,
) -> list[MemoryRecord]:
    """Return the live `forbidden_action` rules a proposed `action` falls under (empty list ⇒ allowed).

    A coding agent calls this BEFORE acting: if it returns rules, refuse and cite them. A rule matches on
    EITHER of two signals (no LLM, deterministic for a fixed embedder; superseded/retired rules ignored):

    - **lexical** — shares ≥ `min_shared` content words AND covers ≥ `min_overlap` of the action's words
      (the two-word floor keeps a single common verb from false-matching). High precision, exact phrasings.
    - **semantic** — cosine(action, rule) ≥ `min_similarity` over the embedder's vectors, so a paraphrase
      ("remove the accounts table" under "never delete user data") is caught even with zero shared words.
      Only meaningful with a *semantic* embedder (`LocalEmbedder`); on the hashing embedder it degenerates
      to lexical. **Measured on a labelled set (bge-base, n=24, `eval.forbidden_eval`): the distributions
      OVERLAP** — paraphrase cosines 0.56–0.72, benign 0.49–0.69 — so there is no clean threshold. Best
      semantic F1 0.79 (recall 0.92, precision 0.69 at ~0.58) ⇒ **~31% false positives.** So semantic is an
      **ADVISORY recall signal, NOT a reliable hard block**: callers should treat a semantic-only match as
      "verify with the user" and use the LEXICAL signal for high-confidence refusal (the MCP
      `check_forbidden_action` splits the two). The agent also sees forbidden rules via
      `project_state`/context (defence in depth). `min_similarity` is absolute and embedder-dependent.

    Set `use_embeddings=False` for the lexical-only (high-precision) behaviour."""
    action_words = _words(action)
    action_emb = None
    if use_embeddings:
        try:
            action_emb = mem.embedder.embed(action)
        except Exception:
            action_emb = None

    hits: list[MemoryRecord] = []
    for r in mem.store.all():
        if not (
            r.superseded_by is None
            and r.metadata.get("project") == project
            and r.metadata.get("code_kind") == "forbidden_action"
        ):
            continue
        shared = action_words & _words(r.content)
        lexical = (
            len(action_words) >= min_shared
            and len(shared) >= min_shared
            and len(shared) / len(action_words) >= min_overlap
        )
        semantic = (
            action_emb is not None
            and r.embedding is not None
            and _cosine(action_emb, r.embedding) >= min_similarity
        )
        if lexical or semantic:
            hits.append(r)
    return hits
