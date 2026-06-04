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
"""
from __future__ import annotations

import os
import sys

from midas import ContentImportance, HashingEmbedder, Memory

# Auto-retention cap: when set, the store is kept at or below this many records by no-LLM selective
# forgetting after each write — bounded memory for long-running/enterprise deployments.
_MAX_RECORDS = int(os.getenv("MIDAS_MCP_MAX_RECORDS", "0")) or None


def build_memory() -> Memory:
    """Local semantic memory by default (graceful fallback to the offline hashing embedder), with
    optional SQLite persistence via MIDAS_MCP_DB. Importance is derived from content (no LLM) when a
    caller doesn't supply it, so forgetting/tiering have a salience to work with."""
    embedder = HashingEmbedder()
    reranker = None
    if os.getenv("MIDAS_MCP_EMBEDDER", "local").lower() != "hashing":
        try:
            from midas import LocalEmbedder, LocalReranker

            embedder, reranker = LocalEmbedder(), LocalReranker()
        except Exception as exc:  # fastembed missing or model load failed
            print(f"[midas-mcp] local embedder unavailable ({exc}); using hashing", file=sys.stderr)

    store = None
    db = os.getenv("MIDAS_MCP_DB")
    if db:
        from midas.sqlite_store import SQLiteStore

        store = SQLiteStore(db)
        print(f"[midas-mcp] persisting memory to {db}", file=sys.stderr)
    return Memory(
        embedder=embedder, reranker=reranker, store=store, importance_scorer=ContentImportance()
    )


try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit('Midas MCP server needs the MCP SDK: uv pip install -e ".[mcp]"') from exc

_mem = build_memory()
server = FastMCP("midas-memory")


@server.tool()
def remember(content: str, kind: str = "note", importance: int = 0, session: str = "default") -> str:
    """Store a memory for later recall.

    content: text to remember (a fact, decision, preference, or conversation turn).
    kind: note | chat | fact | preference | constraint | mission.
    importance: 1-5 (higher is weighted up in recall and protected from forgetting). 0 = auto-derive
        from content (no LLM) — concrete facts score higher than chit-chat.
    session: conversation/thread id used to group related memories.
    """
    imp = int(importance) or None  # 0 -> derive importance from content
    rec = _mem.remember(content, kind=kind, importance=imp, metadata={"session": session})
    if _MAX_RECORDS and len(_mem.store.all()) > _MAX_RECORDS:
        _mem.forget_decayed(max_records=_MAX_RECORDS)  # keep memory bounded (no LLM)
    return f"remembered ({rec.id})"


@server.tool()
def recall(query: str, limit: int = 5) -> list[dict]:
    """Retrieve the most relevant memories for a query.

    Returns source-traceable hits (id, score, kind, original content) — no LLM rewriting, so each
    result is auditable back to the exact stored text.
    """
    return [
        {
            "id": h.record.id,
            "score": round(float(h.score), 3),
            "kind": h.record.kind,
            "content": h.record.content,
        }
        for h in _mem.recall(query, limit=int(limit))
    ]


@server.tool()
def build_context(query: str, token_budget: int = 512) -> str:
    """Assemble a budgeted, prompt-ready context block for a query.

    Highest-value memories first, with same-session neighbours pulled in, trimmed to token_budget.
    Drop the returned string straight into an LLM prompt.
    """
    return _mem.assemble(query, token_budget=int(token_budget), window=1, thread_key="session")


@server.tool()
def forget(memory_id: str) -> str:
    """Delete a single memory by id (ids come from `recall`)."""
    return "forgotten" if _mem.store.delete(memory_id) else f"no memory with id {memory_id}"


@server.tool()
def forget_all() -> str:
    """Clear all stored memories (fresh start)."""
    n = len(_mem.store.all())
    _mem.store.clear()
    return f"cleared {n} memories"


@server.tool()
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


@server.tool()
def stats() -> dict:
    """Memory stats: total count, breakdown by kind, and the temporal-tier distribution.

    tiers: short (<= 1 day) / medium (<= 1 week) / long (older) — the short/medium/multi-day horizons.
    """
    by_kind: dict[str, int] = {}
    by_tier: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    for record in _mem.store.all():
        by_kind[record.kind] = by_kind.get(record.kind, 0) + 1
        by_tier[_mem.tier(record)] += 1
    return {"total": sum(by_kind.values()), "by_kind": by_kind, "by_tier": by_tier}


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
