"""Midas as an MCP server — agent memory for any MCP client (Claude Desktop, IDEs, agents).

Exposes the Midas SDK over the Model Context Protocol. Same design as the library: **no LLM and no
network at ingest/query** (local embeddings only), and recall returns **source-traceable** memories.

Setup:
    uv pip install -e ".[mcp,local]"
    python -m midas.mcp_server          # or the `midas-mcp` console script

Register with an MCP client (e.g. Claude Desktop) — command: `python`, args: `["-m", "midas.mcp_server"]`.

Config (env):
    MIDAS_MCP_EMBEDDER     = local | hashing   (default: local if fastembed is present, else hashing)
    MIDAS_MCP_DB           = path to a SQLite file (persist memory across restarts; default: in-memory)
    MIDAS_MCP_MAX_RECORDS  = cap the store; over it, `remember` auto-forgets the lowest-value tail
                             (no LLM) so memory stays bounded out of the box (default: unbounded)
    MIDAS_MCP_MIN_IMPORTANCE = relevance floor for `capture` (1-5); turns scoring below it are skipped
                             (default: 2 — keeps anything with a real fact/name/number, drops chit-chat)
    MIDAS_MCP_CLIENT       = the client/agent id (claude-code | codex | cursor…) — stamped as actor and
                             into source so every memory is traceable (default: agent; `midas init` sets it)
    MIDAS_MCP_ACTOR        = override the actor id stamped on MCP memories (default: the client)
    MIDAS_MCP_NAMESPACE    = default namespace stamped on writes and applied to reads — lets many
                             projects/agents share one DB file with scoped memory (default: none)
    MIDAS_MCP_SUPERSEDE    = 1/0 typed belief revision for stale facts (default: 1)
    MIDAS_MCP_SUPERSEDE_CONVO = 1 enables strict-cue chat revision (default: 0)
    MIDAS_MCP_NLI          = 1 gates revision with the local NLI model (default: 0)
    MIDAS_MCP_AUTO_MAINTAIN = minutes between background maintenance passes (consolidate
                             near-duplicates + bound the store) — "sleep-time" upkeep at $0,
                             no LLM, while the agent is idle (default: 0 = off)
    MIDAS_MCP_PINNED       = standing-directive slots pinned into every build_context — durable
                             user rules ("from now on…", "always…") detected at ingest (no LLM)
                             and kept in context regardless of query relevance. Measured on BEAM
                             instruction-following recall@k: 0.26 off -> 0.44 at 2 (default) ->
                             0.51 at 4, overall unchanged-to-better (0 = off)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict

from midas import (
    AGENT_MEMORY_INSTRUCTIONS, MEMORY_PROVENANCE, MEMORY_USES, HashingEmbedder, Memory,
    MemoryPolicy, StructuralImportance, __version__, policy_summary,
)
from midas.state import memory_diff as _diff_view, memory_state as _state_view
from midas.audit import audit_use as _audit_use
from midas.coding import (
    is_forbidden as _is_forbidden,
    project_state as _project_state_view,
    remember_code as _remember_code_impl,
)

# Auto-retention cap: when set, the store is kept at or below this many records by no-LLM selective
# forgetting after each write — bounded memory for long-running/enterprise deployments.
_MAX_RECORDS = int(os.getenv("MIDAS_MCP_MAX_RECORDS", "0")) or None
# The relevance parameters Midas imposes on `capture` (the no-LLM "what's worth keeping" gate).
_POLICY = MemoryPolicy(min_importance=int(os.getenv("MIDAS_MCP_MIN_IMPORTANCE", "2")))
# The MCP client/agent this server serves (set per-client by `midas init`: claude-code / codex / cursor…)
# — stamped as the actor and into the source so every memory is traceable to *who* produced it.
_CLIENT = os.getenv("MIDAS_MCP_CLIENT", "agent")
_ACTOR = os.getenv("MIDAS_MCP_ACTOR") or _CLIENT
def _resolve_namespace() -> str:
    """Default scope for this server instance. A per-call `namespace` arg overrides it; empty = unscoped
    (reads see everything, writes carry no namespace) — fully backward compatible. MIDAS_MCP_NAMESPACE=
    "auto" derives the scope from the PROJECT the server was launched in (the git repo name, else the cwd
    basename), so each project gets its own partition in the one shared store — automatically."""
    ns = os.getenv("MIDAS_MCP_NAMESPACE", "")
    if ns != "auto":
        return ns
    try:
        import subprocess

        root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, timeout=2)
        if root.returncode == 0 and root.stdout.strip():
            return os.path.basename(root.stdout.strip())
    except Exception:
        pass
    return os.path.basename(os.getcwd()) or ""


# Default scope for this server instance (see _resolve_namespace; "auto" = per-project).
_NAMESPACE = _resolve_namespace()


def _ns(namespace: str) -> str:
    return namespace or _NAMESPACE


def _source(session: str) -> str:
    """A structured, traceable source for every write: who (client) + which thread (session)."""
    return f"mcp:{_CLIENT}:{session}"


def _resolve_origin() -> dict:
    """Where the server runs — the git commit/branch + cwd — stamped on every memory so a belief traces
    back to the code state it came from. Resolved once at startup (cheap); empty outside a repo."""
    cwd = os.path.basename(os.getcwd())
    try:
        import subprocess

        def _git(*a: str) -> str:
            r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=2)
            return r.stdout.strip() if r.returncode == 0 else ""

        commit = _git("rev-parse", "--short", "HEAD")
        if commit:
            return {"git": commit, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "cwd": cwd}
    except Exception:
        pass
    return {"cwd": cwd} if cwd else {}


_ORIGIN = _resolve_origin()


def _ns_metadata(namespace: str, session: str) -> dict:
    ns = _ns(namespace)
    md: dict = {"session": session}
    if ns:
        md["namespace"] = ns
    if _ORIGIN:
        md["origin"] = _ORIGIN  # where (git/cwd) it was captured — links the memory to the code state
    return md


def _ns_filter(namespace: str) -> dict | None:
    ns = _ns(namespace)
    return {"namespace": ns} if ns else None
_SUPERSEDE = os.getenv("MIDAS_MCP_SUPERSEDE", "1") != "0"
_SUPERSEDE_CONVO = os.getenv("MIDAS_MCP_SUPERSEDE_CONVO", "0") == "1"
_USE_NLI = os.getenv("MIDAS_MCP_NLI", "0") == "1"
_PINNED = int(os.getenv("MIDAS_MCP_PINNED", "2") or "0")


_MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_memory() -> Memory:
    """Local semantic memory by default (graceful fallback to the offline hashing embedder), with
    optional SQLite persistence via MIDAS_MCP_DB. Importance is derived from content (no LLM) when a
    caller doesn't supply it, so forgetting/tiering have a salience to work with."""
    embedder = HashingEmbedder()
    reranker = None
    choice = os.getenv("MIDAS_MCP_EMBEDDER", "local").lower()
    if choice != "hashing":
        try:
            from midas import LocalEmbedder, LocalReranker

            if choice == "multilingual" or "/" in choice:
                # Non-English memory: bge-base is English-only (measured: Spanish answer_dumb
                # 0.68 vs 1.00 with the multilingual model). The cross-encoder reranker is
                # English-trained (ms-marco), so it stays off for non-English models.
                model = _MULTILINGUAL_MODEL if choice == "multilingual" else os.getenv("MIDAS_MCP_EMBEDDER")
                embedder = LocalEmbedder(model_name=model)
            else:
                embedder, reranker = LocalEmbedder(), LocalReranker()
        except Exception as exc:  # fastembed missing or model load failed
            print(f"[midas-mcp] local embedder unavailable ({exc}); using hashing", file=sys.stderr)

    store = None
    # One shared, persistent store BY DEFAULT: ~/.midas/memory.sqlite3 — so every MCP client (Claude
    # Code, Codex, Cursor…) that launches `midas-mcp` reads and writes the SAME memory, with zero config.
    # Override with MIDAS_MCP_DB=<path>, or MIDAS_MCP_DB=":memory:" for an ephemeral store.
    db = os.getenv("MIDAS_MCP_DB")
    if db is None:
        db = os.path.expanduser("~/.midas/memory.sqlite3")
    elif db == ":memory:":
        db = ""
    # Opt-in sub-linear (IVF) search for very large stores — approximate (recall ~0.95 at the
    # default nprobe; BENCHMARKS §4), so exact scan stays the default. Kicks in at >= 10k records.
    ann = os.getenv("MIDAS_MCP_ANN", "0") == "1"
    if db:
        from midas.sqlite_store import SQLiteStore

        os.makedirs(os.path.dirname(db), exist_ok=True)
        store = SQLiteStore(db, ann_threshold=10_000 if ann else None)
        print(f"[midas-mcp] memory: {db}", file=sys.stderr)
    elif ann:
        from midas import InMemoryStore

        store = InMemoryStore(ann_threshold=10_000)
    if _NAMESPACE:
        print(f"[midas-mcp] namespace (project scope): {_NAMESPACE}", file=sys.stderr)
    nli = None
    if _USE_NLI:
        from midas.nli import LocalNLI

        nli = LocalNLI()
    return Memory(
        embedder=embedder, reranker=reranker, store=store,
        importance_scorer=StructuralImportance(), policy=_POLICY,  # measured-better fact-vs-question
        supersede=_SUPERSEDE or _SUPERSEDE_CONVO,
        supersede_conversational=_SUPERSEDE_CONVO,
        nli=nli,
        include_provenance=False,
        detect_standing=_PINNED > 0,  # tag durable user directives at ingest (no LLM)
    )


try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
except ImportError as exc:  # pragma: no cover
    raise SystemExit('Midas MCP server needs the MCP SDK: uv pip install -e ".[mcp]"') from exc

_mem = build_memory()
# `instructions` are surfaced to the model by the MCP client on connect — this is how installing Midas
# makes an agent start remembering on its own: it injects the recall->work->capture loop + the policy.
server = FastMCP("midas-memory", instructions=AGENT_MEMORY_INSTRUCTIONS)
# Report Midas's own version in the MCP handshake (FastMCP otherwise reports the MCP SDK version).
server._mcp_server.version = __version__


@server.tool(
    title="Remember",
    annotations=ToolAnnotations(title="Remember", readOnlyHint=False, destructiveHint=False),
)
def remember(
    content: str,
    kind: str = "note",
    importance: int = 0,
    session: str = "default",
    provenance: str = "observation",
    actor: str = "",
    namespace: str = "",
) -> str:
    """Store a memory for later recall.

    content: text to remember (a fact, decision, preference, or conversation turn).
    kind: note | chat | fact | preference | constraint | mission.
    importance: 1-5 (higher is weighted up in recall and protected from forgetting). 0 = auto-derive
        from content (no LLM) — concrete facts score higher than chit-chat.
    session: conversation/thread id used to group related memories.
    provenance: planning | action | observation | user_confirmation. Use user_confirmation only when
        the user explicitly confirmed the content; external actions may rely only on that provenance.
    actor: agent/process that produced the memory (default: MIDAS_MCP_ACTOR).
    namespace: scope tag (e.g. a project or user id); defaults to MIDAS_MCP_NAMESPACE.
    """
    imp = int(importance) or None  # 0 -> derive importance from content
    rec = _mem.remember(
        content,
        kind=kind,
        importance=imp,
        source=_source(session),
        provenance=provenance,
        actor=actor or _ACTOR,
        metadata=_ns_metadata(namespace, session),
    )
    if _MAX_RECORDS and len(_mem.store.all()) > _MAX_RECORDS:
        _mem.forget_decayed(max_records=_MAX_RECORDS)  # keep memory bounded (no LLM)
    return f"remembered ({rec.id})"


@server.tool(
    title="Capture turn",
    annotations=ToolAnnotations(title="Capture turn", readOnlyHint=False, destructiveHint=False),
)
def capture(
    content: str,
    kind: str = "chat",
    session: str = "default",
    provenance: str = "observation",
    actor: str = "",
    namespace: str = "",
) -> dict:
    """Offer a turn to memory; Midas decides whether to keep it (no LLM).

    Forward anything that might be durable — a fact, decision, preference, constraint, or correction.
    Midas scores its importance and keeps it only if it clears the relevance policy and isn't a
    duplicate, so you can capture freely without polluting memory. Returns whether it was stored and
    why (so you learn the bar). This is the workhorse for hands-off, automatic remembering.
    kind: note | chat | fact | preference | constraint | mission.
    provenance: planning | action | observation | user_confirmation.
    """
    result = _mem.capture(
        content,
        kind=kind,
        source=_source(session),
        provenance=provenance,
        actor=actor or _ACTOR,
        metadata=_ns_metadata(namespace, session),
    )
    if result.stored and _MAX_RECORDS and len(_mem.store.all()) > _MAX_RECORDS:
        _mem.forget_decayed(max_records=_MAX_RECORDS)
    return {
        "stored": result.stored,
        "reason": result.reason,
        "importance": result.importance,
        "id": result.record.id if result.record else None,
        "provenance": result.record.provenance if result.record else None,
        "actor": result.record.actor if result.record else None,
    }


@server.tool(
    title="Remember code memory",
    annotations=ToolAnnotations(title="Remember code memory", readOnlyHint=False, destructiveHint=False),
)
def remember_code(
    content: str,
    code_kind: str,
    project: str,
    provenance: str = "",
    importance: int = 0,
    session: str = "default",
    namespace: str = "",
) -> dict:
    """Capture a CODE memory tagged by category + project, for the coding-agent views. code_kind: one of
    architecture_decision | dependency_choice | convention | bug_fixed | recurring_failure |
    forbidden_action | command_worked | command_failed. Use forbidden_action for rules the agent must not
    violate (they gate `check_forbidden_action`); afterwards `project_state` shows the live state by
    category. No LLM. Set provenance="user_confirmation" only when the user explicitly confirmed it.
    """
    rec = _remember_code_impl(
        _mem, content, code_kind, project=project,
        provenance=(provenance or None), importance=(int(importance) or None),
        actor=_ACTOR, source=_source(session),
        metadata=_ns_metadata(namespace, session),
    )
    if _MAX_RECORDS and len(_mem.store.all()) > _MAX_RECORDS:
        _mem.forget_decayed(max_records=_MAX_RECORDS)  # keep memory bounded (no LLM)
    return _serialize_record(rec)


def _round_score(value: float) -> float:
    return round(float(value), 3)


def _serialize_record(record) -> dict:
    """Human-auditable memory evidence. Deliberately omits the embedding."""
    return {
        "id": record.id,
        "kind": record.kind,
        "importance": record.importance,
        "provenance": record.provenance,
        "actor": record.actor,
        "source": record.source,
        "metadata": record.metadata,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "superseded_by": record.superseded_by,
        "content": record.content,
    }


def _serialize_recall_hit(hit, *, explain: bool = True) -> dict:
    """Structured, source-traceable recall evidence for MCP clients.

    Keep this separate from `build_context`: agents need compact prompt context by default, while audit
    workflows need the deterministic retrieval signals that explain why a record surfaced.
    """
    record = hit.record
    item = {
        **_serialize_record(record),
        "score": _round_score(hit.score),
    }
    if explain:
        item["explanation"] = {
            "score": _round_score(hit.score),
            "relevance": _round_score(hit.relevance),
            "importance_norm": _round_score(hit.importance_norm),
            "recency": _round_score(hit.recency),
            "formula": "score = relevance + 0.3*importance_norm + 0.2*recency",
        }
        # Keep the raw components at the top level too so simple clients can filter/sort without digging.
        item.update(
            {
                "relevance": _round_score(hit.relevance),
                "importance_norm": _round_score(hit.importance_norm),
                "recency": _round_score(hit.recency),
            }
        )
    return item


@server.tool(
    title="Recall",
    annotations=ToolAnnotations(title="Recall", readOnlyHint=True, openWorldHint=False),
)
def recall(
    query: str,
    limit: int = 5,
    kind: str | None = None,
    min_importance: int = 0,
    min_relevance: float = 0.0,
    hybrid: bool = False,
    fusion: str = "rrf",
    pool: int = 0,
    explain: bool = True,
    namespace: str = "",
    as_of: str = "",
) -> list[dict]:
    """Retrieve relevant memories with deterministic, source-traceable evidence.

    Returns exact stored text plus provenance/source/timestamps and, by default, score components
    (relevance, importance_norm, recency). No LLM rewrites or rationales are generated.
    Set hybrid=true to fuse BM25 lexical matching with semantic recall — useful when the query is
    an exact identifier (an error code, a function name) rather than a paraphrase.
    as_of: ISO date (YYYY-MM-DD) for a HISTORICAL query — "what did memory say on that date":
    later records are excluded and revised beliefs resolve to the version valid then.
    """
    as_of_ts: float | None = None
    if as_of:
        import datetime as _dt

        as_of_ts = _dt.datetime.fromisoformat(as_of).replace(tzinfo=_dt.timezone.utc).timestamp()
    return [
        _serialize_recall_hit(h, explain=bool(explain))
        for h in _mem.recall(
            query,
            limit=int(limit),
            kind=kind or None,
            min_importance=int(min_importance) or None,
            min_relevance=float(min_relevance) or None,
            hybrid=bool(hybrid),
            fusion=fusion,
            pool=int(pool) or None,
            metadata_filter=_ns_filter(namespace),
            as_of=as_of_ts,
        )
    ]


@server.tool(
    title="Inspect memory",
    annotations=ToolAnnotations(title="Inspect memory", readOnlyHint=True, openWorldHint=False),
)
def inspect_memory(memory_id: str) -> dict:
    """Inspect one stored memory by id without search, mutation, or embedding exposure."""
    record = _mem.store.get(memory_id)
    if record is None:
        return {"found": False, "id": memory_id}
    return {"found": True, **_serialize_record(record)}


@server.tool(
    title="Build context",
    annotations=ToolAnnotations(title="Build context", readOnlyHint=True, openWorldHint=False),
)
def build_context(
    query: str,
    token_budget: int = 512,
    limit: int = 6,
    hybrid: bool = False,
    namespace: str = "",
) -> str:
    """Assemble a budgeted, prompt-ready context block for a query.

    Highest-value memories first, with same-session neighbours pulled in, trimmed to token_budget.
    Drop the returned string straight into an LLM prompt. It uses lean memory lines by default;
    call `recall` or `inspect_memory` when you need full provenance/source evidence. Each line is dated
    and the header anchors today's date so the reader can resolve relative time. Standing user
    directives ("from now on…", "always…") are pinned in regardless of query relevance.
    """
    return _mem.assemble(
        query,
        token_budget=int(token_budget),
        limit=int(limit),
        window=1,
        thread_key="session",
        hybrid=bool(hybrid),
        metadata_filter=_ns_filter(namespace),
        pinned_limit=_PINNED,
        # The "Today is ..." anchor + per-memory relative ages — the LLM-free temporal grounding
        # that lifted temporal recall@k 0.86 -> 0.95 in the eval; identical recency scoring.
        now=time.time(),
    )


@server.tool(
    title="Memory state",
    annotations=ToolAnnotations(title="Memory state", readOnlyHint=True, openWorldHint=False),
)
def memory_state(namespace: str = "", kinds: str = "", limit: int = 50) -> dict:
    """The CURRENT durable state of a project/scope: the live (non-superseded) decisions, constraints,
    facts, and preferences, newest first. Use this to ONBOARD into a project or before planning — when
    a broad 'what's the current state?' has no single matching turn, so `recall` under-retrieves. Not a
    similarity search and no LLM. kinds: optional comma list (constraint,fact,preference,mission) to
    narrow; empty = all durable kinds.
    """
    kind_tuple = tuple(k.strip() for k in kinds.split(",") if k.strip())
    extra = {"kinds": kind_tuple} if kind_tuple else {}
    records = _state_view(_mem, scope=_ns_filter(namespace), limit=int(limit), **extra)
    return {"count": len(records), "state": [_serialize_record(r) for r in records]}


@server.tool(
    title="Memory diff",
    annotations=ToolAnnotations(title="Memory diff", readOnlyHint=True, openWorldHint=False),
)
def memory_diff(hours: float = 24.0, namespace: str = "") -> dict:
    """What CHANGED in memory in the last `hours`: beliefs newly added, and beliefs revised (old -> new).
    The 'what's new since our last session' view for resuming long-horizon work without re-reading
    everything. Deterministic, no LLM.
    """
    since = time.time() - float(hours) * 3600.0
    diff = _diff_view(_mem, since, scope=_ns_filter(namespace))
    return {
        "since_hours": float(hours),
        "added": [_serialize_record(r) for r in diff.added],
        "revised": [
            {"old": _serialize_record(o), "new": _serialize_record(n)} for o, n in diff.revised
        ],
    }


@server.tool(
    title="Project state",
    annotations=ToolAnnotations(title="Project state", readOnlyHint=True, openWorldHint=False),
)
def project_state(project: str, limit: int = 200) -> dict:
    """The current code-state of a project for a coding agent, grouped by code_kind
    (architecture_decision, bug_fixed, convention, forbidden_action, dependency_choice, ...). Live
    (non-superseded) memories only — call this to ONBOARD into a project, or to see what is decided and
    what is forbidden before acting. Deterministic, no LLM. Capture code memories with the SDK's
    `midas.coding.remember_*` helpers (they tag `code_kind` + `project`).
    """
    grouped = _project_state_view(_mem, project, limit=int(limit))
    return {
        "project": project,
        "counts": {k: len(recs) for k, recs in grouped.items()},
        "state": {k: [_serialize_record(r) for r in recs] for k, recs in grouped.items()},
    }


@server.tool(
    title="Check forbidden action",
    annotations=ToolAnnotations(title="Check forbidden action", readOnlyHint=True, openWorldHint=False),
)
def check_forbidden_action(action: str, project: str) -> dict:
    """Before a code action, check it against the project's live `forbidden_action` rules. Two tiers, by
    confidence (deterministic, no LLM; superseded/retired rules ignored):
      - `forbidden` (HIGH confidence, lexical match) → do NOT do it; refuse and cite `rules`.
      - `possibly_forbidden` (ADVISORY, semantic/paraphrase match) → ASK the user before proceeding; the
        semantic signal is noisy (~31% false positives, `eval.forbidden_eval`), so it warns, not blocks.
    Separate from the provenance/currency gate (`check_memory_use`).
    """
    lexical = _is_forbidden(_mem, action, project, use_embeddings=False)
    matched = _is_forbidden(_mem, action, project)  # lexical + semantic
    lex_ids = {r.id for r in lexical}
    semantic_only = [r for r in matched if r.id not in lex_ids]
    return {
        "action": action,
        "project": project,
        "forbidden": bool(lexical),
        "rules": [_serialize_record(r) for r in lexical],
        "possibly_forbidden": bool(semantic_only),
        "candidate_rules": [_serialize_record(r) for r in semantic_only],
    }


@server.tool(
    title="Audit memory use",
    annotations=ToolAnnotations(title="Audit memory use", readOnlyHint=True, openWorldHint=False),
)
def audit_use(query: str, intended_use: str = "external_action", namespace: str = "") -> dict:
    """The compliance audit artifact for a memory-justified use: the guard decision + the full provenance
    and belief-revision history of every supporting memory + an attributability score (fraction of evidence
    with both a source and an actor). Hand this to an auditor to prove WHY an action is or isn't justified.
    Source-traceable, deterministic, no LLM.
    """
    return _audit_use(_mem, query, intended_use, metadata_filter=_ns_filter(namespace))


@server.tool(
    title="Memory policy",
    annotations=ToolAnnotations(title="Memory policy", readOnlyHint=True, openWorldHint=False),
)
def memory_policy() -> dict:
    """Return the exact MCP-injected memory policy text and guard parameters."""
    return {
        "instructions": AGENT_MEMORY_INSTRUCTIONS,
        "relevance_policy": policy_summary(_POLICY),
        "provenance": list(MEMORY_PROVENANCE),
        "guard_uses": list(MEMORY_USES),
    }


@server.tool(
    title="Check memory use",
    annotations=ToolAnnotations(title="Check memory use", readOnlyHint=True, openWorldHint=False),
)
def check_memory_use(
    query: str,
    intended_use: str = "planning",
    limit: int = 5,
    acting_agent: str = "",
    namespace: str = "",
) -> dict:
    """Decide whether recalled memory may justify the intended use.

    intended_use: planning | answer | external_action | destructive_action.
    External/destructive actions require user_confirmation provenance; otherwise ask the user first.
    """
    decision = _mem.guard_reliance(
        query,
        intended_use=intended_use,
        acting_agent=acting_agent or _ACTOR,
        limit=int(limit),
        metadata_filter=_ns_filter(namespace),
    )
    return asdict(decision)


@server.tool(
    title="Forget memory",
    annotations=ToolAnnotations(title="Forget memory", readOnlyHint=False, destructiveHint=True),
)
def forget(memory_id: str) -> str:
    """Delete a single memory by id (ids come from `recall`). Supersession chains through the
    deleted record are relinked, so belief-revision history stays walkable."""
    return "forgotten" if _mem.forget(memory_id) else f"no memory with id {memory_id}"


@server.tool(
    title="Forget matching",
    annotations=ToolAnnotations(title="Forget matching", readOnlyHint=False, destructiveHint=True),
)
def forget_matching(
    query: str,
    min_relevance: float = 0.5,
    limit: int = 20,
    dry_run: bool = True,
    namespace: str = "",
) -> dict:
    """Topic-level erasure ("forget what you know about X") with a reviewable audit.

    Matches memories at relevance >= min_relevance. By default this is a DRY RUN: it returns what
    *would* be deleted so you (or the user) can review; call again with dry_run=false to delete.
    Deletion bypasses durability protections — an explicit erasure request outranks retention —
    and returns the full list of removed memories as the audit trail.
    """
    matched = _mem.forget_matching(
        query,
        min_relevance=float(min_relevance),
        limit=int(limit),
        metadata_filter=_ns_filter(namespace),
        dry_run=bool(dry_run),
    )
    return {
        "dry_run": bool(dry_run),
        "matched": len(matched),
        "records": [_serialize_record(r) for r in matched],  # the erasure audit trail
        "note": "dry run — nothing deleted; repeat with dry_run=false to erase"
        if dry_run
        else "deleted",
    }


@server.tool(
    title="Forget all",
    annotations=ToolAnnotations(title="Forget all", readOnlyHint=False, destructiveHint=True),
)
def forget_all() -> str:
    """Clear all stored memories (fresh start)."""
    n = len(_mem.store.all())
    _mem.store.clear()
    return f"cleared {n} memories"


@server.tool(
    title="Maintain memory",
    annotations=ToolAnnotations(title="Maintain memory", readOnlyHint=False, destructiveHint=True),
)
def maintain(consolidate_threshold: float = 0.0, max_records: int = 0, min_value: float = 0.0) -> dict:
    """Run a no-LLM memory-maintenance pass and return the deletion audit.

    Bounds storage and keeps recall clean without sending anything to an LLM — the enterprise
    retention / "right to be forgotten" lever, with a full audit of exactly what was removed:
      - consolidate_threshold: if > 0, dedup near-duplicate restatements at this cosine (e.g. 0.95).
      - max_records: if > 0, forget the lowest-value tail until at most this many remain.
      - min_value: if > 0, forget every (non-durable, unprotected) memory scoring below this value.
    Durable memories (facts/preferences/constraints, high importance) and supersession chains are
    never dropped. Returns counts and the ids removed (auditable).
    """
    before = len(_mem.store.all())
    consolidated = (
        _mem.consolidate(similarity_threshold=float(consolidate_threshold))
        if consolidate_threshold and consolidate_threshold > 0
        else []
    )
    forgotten = (
        _mem.forget_decayed(
            max_records=int(max_records) or None,
            min_value=float(min_value) or None,
        )
        if (max_records or min_value)
        else []
    )
    return {
        "before": before,
        "remaining": len(_mem.store.all()),
        "consolidated": len(consolidated),
        "forgotten": len(forgotten),
        "removed_ids": consolidated + forgotten,  # the deletion audit trail
    }


@server.tool(
    title="Memory stats",
    annotations=ToolAnnotations(title="Memory stats", readOnlyHint=True, openWorldHint=False),
)
def stats(namespace: str = "") -> dict:
    """Memory stats: total count, breakdown by kind, namespace, and the temporal-tier distribution.

    tiers: short (<= 1 day) / medium (<= 1 week) / long (older) — the short/medium/multi-day horizons.
    Pass namespace to scope the counts to one project/agent scope.
    """
    ns = _ns(namespace)
    by_kind: dict[str, int] = {}
    by_provenance: dict[str, int] = {p: 0 for p in MEMORY_PROVENANCE}
    by_tier: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    by_namespace: dict[str, int] = {}
    for record in _mem.store.all():
        rec_ns = record.metadata.get("namespace") or "(none)"
        by_namespace[rec_ns] = by_namespace.get(rec_ns, 0) + 1
        if ns and record.metadata.get("namespace") != ns:
            continue
        by_kind[record.kind] = by_kind.get(record.kind, 0) + 1
        by_provenance[record.provenance] = by_provenance.get(record.provenance, 0) + 1
        by_tier[_mem.tier(record)] += 1
    return {
        "total": sum(by_kind.values()),
        "namespace": ns or None,
        "by_kind": by_kind,
        "by_provenance": by_provenance,
        "by_tier": by_tier,
        "by_namespace": by_namespace,
        "guard_uses": list(MEMORY_USES),
    }


@server.prompt()
def memory_session(goal: str = "") -> str:
    """A drop-in prompt that switches an agent into Midas auto-memory mode for a session.

    Clients that don't surface server `instructions` automatically can pin this as a system prompt.
    """
    g = f' for this goal: "{goal.strip()}"' if goal.strip() else ""
    return (
        AGENT_MEMORY_INSTRUCTIONS
        + f"\n\nStart now{g}: call `build_context` with the goal to load relevant memory, then proceed — "
        "calling `capture` as durable facts, decisions, preferences, constraints, or corrections come up.\n"
        f"Relevance policy in effect: {policy_summary(_POLICY)}."
    )


@server.prompt()
def distill(session: str = "default") -> str:
    """Drive a no-Midas-LLM distillation pass: YOUR model turns the recent conversation into compact,
    self-contained memories.

    This is the architecturally-honest version of the frontier's LLM-at-ingest memory (Mem0/Letta/
    LIGHT): the gains in those systems come from distilling raw turns into high-signal facts — but
    they pay a separate LLM per session. Here the model you're ALREADY running does it, and Midas
    stays no-LLM and $0. A distilled fact ("X's deploy target is staging, as of 2026-06") retrieves
    far better than the conversational fragment it came from.
    """
    return (
        "Distill the durable knowledge from this session into memory, using your own reasoning "
        "(no extra tool/model needed):\n"
        "1) Identify the facts, decisions, preferences, constraints, and corrections worth keeping.\n"
        "2) For each, write ONE compact, self-contained sentence — name the entities, the value, and "
        "when it holds — so it answers on its own without the surrounding chat.\n"
        f"3) Store each with `capture` (session=\"{session}\", kind=fact|preference|constraint|note, "
        "provenance=user_confirmation only if the user explicitly confirmed it). Midas dedups and "
        "scores; restating an existing memory just reinforces it, so err on the side of capturing.\n"
        "4) Do NOT dump raw turns — distillation is the point."
    )


def _run_maintenance_pass() -> dict:
    """One no-LLM upkeep pass: collapse near-duplicate restatements and re-bound the store.

    This is the sleep-time idea (reorganise memory while the agent is idle) at Midas prices:
    extractive consolidation + value-ranked forgetting, $0, nothing leaves the box — versus
    LLM-agent rewrites of memory in systems like Letta's sleep-time agents."""
    consolidated = _mem.consolidate(similarity_threshold=0.95)
    forgotten = _mem.forget_decayed(max_records=_MAX_RECORDS) if _MAX_RECORDS else []
    return {"consolidated": len(consolidated), "forgotten": len(forgotten)}


def _auto_maintain_loop(interval_seconds: float) -> None:
    import time as _time

    while True:
        _time.sleep(interval_seconds)
        try:
            result = _run_maintenance_pass()
            if result["consolidated"] or result["forgotten"]:
                print(f"[midas-mcp] auto-maintain: {result}", file=sys.stderr)
        except Exception as exc:  # never let upkeep kill the server
            print(f"[midas-mcp] auto-maintain error: {exc}", file=sys.stderr)


def run_server(transport: str = "stdio", host: str = "127.0.0.1", port: int = 7077) -> None:
    """Run the MCP server. transport='stdio' (default — each client launches it) or 'http' (one shared
    server at an MCP URL: http://host:port/mcp). The `midas` CLI calls this; `midas-mcp` uses stdio."""
    interval_min = int(os.getenv("MIDAS_MCP_AUTO_MAINTAIN", "0") or "0")
    if interval_min > 0:
        import threading

        threading.Thread(
            target=_auto_maintain_loop, args=(interval_min * 60.0,), daemon=True
        ).start()
        print(f"[midas-mcp] auto-maintain every {interval_min} min (no LLM)", file=sys.stderr)
    if transport in ("http", "streamable-http"):
        server.settings.host = host
        server.settings.port = port
        print(f"[midas-mcp] MCP URL -> http://{host}:{port}/mcp  (Ctrl-C to stop)", file=sys.stderr)
        server.run(transport="streamable-http")
    else:
        server.run()


def main() -> None:
    run_server("stdio")


if __name__ == "__main__":
    main()
