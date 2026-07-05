"""Tests for the Midas MCP server tools (offline hashing embedder — fast, no model load)."""
import os

import pytest

pytest.importorskip("mcp")  # optional extra; skip (don't crash collection) when MCP SDK is absent

os.environ["MIDAS_MCP_EMBEDDER"] = "hashing"  # must be set before importing the server

from midas.mcp_server import (  # noqa: E402
    build_context,
    capture,
    check_memory_use,
    forget,
    forget_all,
    forget_matching,
    inspect_memory,
    maintain,
    memory_policy,
    recall,
    remember,
    server,
    stats,
)


def test_remember_recall_roundtrip():
    forget_all()
    out = remember("The primary database is PostgreSQL.", kind="constraint", importance=5)
    assert "remembered" in out
    remember("Launch date moved to September 14.", kind="fact", importance=5)

    hits = recall("which database did we pick?", limit=2)
    assert isinstance(hits, list) and hits, "recall should return hits"
    # source-traceable, auditable shape
    required = {
        "id",
        "score",
        "relevance",
        "importance_norm",
        "recency",
        "explanation",
        "kind",
        "importance",
        "provenance",
        "actor",
        "source",
        "metadata",
        "created_at",
        "updated_at",
        "superseded_by",
        "content",
    }
    assert all(required <= set(h) for h in hits)
    assert all("embedding" not in h for h in hits)
    assert all(
        {"score", "relevance", "importance_norm", "recency", "formula"} <= set(h["explanation"])
        for h in hits
    )
    assert any("PostgreSQL" in h["content"] for h in hits), "relevant memory should surface"


def test_recall_supports_audit_filters_and_compact_mode():
    forget_all()
    remember("The primary database is PostgreSQL.", kind="constraint", importance=5)
    remember("The team lunch is at noon.", kind="note", importance=1)

    hits = recall("database", kind="constraint", min_importance=5, min_relevance=0.0, limit=5)
    assert hits and all(h["kind"] == "constraint" for h in hits)
    assert all(h["importance"] >= 5 for h in hits)

    compact = recall("database", explain=False, limit=1)
    assert compact and "explanation" not in compact[0]
    assert "relevance" not in compact[0]
    assert "score" in compact[0]


def test_recall_accepts_hybrid_parameters():
    forget_all()
    remember("Redis backs the queue worker.", kind="fact", importance=4)
    hits = recall("queue worker", hybrid=True, fusion="rrf", pool=5, limit=1)
    assert hits and "Redis" in hits[0]["content"]
    assert "explanation" in hits[0]


def test_inspect_memory_by_id_is_read_only_and_auditable():
    forget_all()
    out = remember(
        "Decision: deploy from the release branch.",
        kind="constraint",
        importance=5,
        session="release",
        provenance="user_confirmation",
        actor="user",
    )
    rid = out.split("(", 1)[1].rstrip(")")

    before = stats()
    inspected = inspect_memory(rid)

    assert inspected["found"] is True
    assert inspected["id"] == rid
    assert inspected["content"] == "Decision: deploy from the release branch."
    assert inspected["kind"] == "constraint"
    assert inspected["importance"] == 5
    assert inspected["provenance"] == "user_confirmation"
    assert inspected["actor"] == "user"
    assert inspected["source"] == "mcp:agent:release"  # mcp:<client>:<session> (client defaults to "agent")
    assert inspected["metadata"]["session"] == "release"
    assert "origin" in inspected["metadata"]  # Phase-1 traceability: where (git/cwd) it was captured
    assert inspected["superseded_by"] is None
    assert "created_at" in inspected and "updated_at" in inspected
    assert "embedding" not in inspected
    assert stats() == before


def test_inspect_memory_not_found():
    forget_all()
    assert inspect_memory("missing") == {"found": False, "id": "missing"}


def test_build_context_returns_budgeted_prompt():
    forget_all()
    remember("Launch date moved to September 14.", kind="fact", importance=5)
    ctx = build_context("when do we launch?", token_budget=100)
    assert isinstance(ctx, str) and "September 14" in ctx


def test_build_context_is_lean_while_recall_keeps_audit_details():
    forget_all()
    remember(
        "Decision: use PostgreSQL for the primary database.",
        kind="constraint",
        importance=5,
        provenance="user_confirmation",
        actor="user",
        session="db",
    )

    ctx = build_context("primary database", token_budget=100)
    hits = recall("primary database", limit=1)

    assert "PostgreSQL" in ctx
    assert "id:" not in ctx
    assert "actor:" not in ctx
    assert "source:" not in ctx
    assert hits and hits[0]["provenance"] == "user_confirmation"
    assert hits[0]["source"] == "mcp:agent:db"  # mcp:<client>:<session>
    assert "score" in hits[0]


def test_forget_all_clears():
    remember("ephemeral note", kind="note")
    forget_all()
    assert recall("ephemeral", limit=5) == []


def test_stats_and_forget_by_id():
    forget_all()
    out = remember(
        "Decision: use PostgreSQL.",
        kind="constraint",
        importance=5,
        provenance="user_confirmation",
        actor="user",
    )
    rid = out.split("(", 1)[1].rstrip(")")  # "remembered (<id>)"
    remember("a chat turn", kind="chat")

    s = stats()
    assert s["total"] == 2 and s["by_kind"].get("constraint") == 1
    assert s["by_provenance"]["user_confirmation"] == 1
    assert forget(rid) == "forgotten"
    assert stats()["total"] == 1


def test_stats_reports_temporal_tiers():
    forget_all()
    remember("a freshly created note", kind="note")
    s = stats()
    assert set(s["by_tier"]) == {"short", "medium", "long"}
    assert s["by_tier"]["short"] == 1  # just created -> short tier


def test_remember_auto_derives_importance_from_content():
    forget_all()
    remember("haha yeah sounds good", kind="chat")  # filler -> importance 1
    remember("My account number is 4421 and renewal is in 2027.", kind="chat")  # specific -> higher
    # maintain(max_records=1) forgets the lowest-value record: the filler, keeping the specific fact.
    out = maintain(max_records=1)
    assert out["before"] == 2 and out["remaining"] == 1
    assert out["forgotten"] == 1 and len(out["removed_ids"]) == 1
    assert any("account number" in h["content"] for h in recall("account number", limit=3))


def test_maintain_consolidates_duplicates_with_audit():
    forget_all()
    remember("The launch is on September 14.", kind="fact", importance=3)
    remember("The launch is on September 14.", kind="fact", importance=3)  # exact duplicate
    out = maintain(consolidate_threshold=0.99)
    assert out["consolidated"] == 1 and out["remaining"] == 1
    assert len(out["removed_ids"]) == 1  # the deletion audit trail


def test_capture_tool_gates_by_policy():
    forget_all()
    fact = capture("My API rate limit is 5000 requests per hour.", kind="fact")
    assert fact["stored"] is True and fact["importance"] >= 2
    filler = capture("haha ok cool")
    assert filler["stored"] is False and "floor" in filler["reason"]
    dup = capture("My API rate limit is 5000 requests per hour.", kind="fact")
    assert dup["stored"] is False and "duplicate" in dup["reason"]
    assert stats()["total"] == 1  # only the fact was kept


def test_guard_blocks_action_until_user_confirmation():
    forget_all()
    remember("Deploy target is staging.", kind="constraint", importance=5, provenance="observation")
    blocked = check_memory_use("deploy target", intended_use="external_action")
    assert blocked["allowed"] is False

    forget_all()
    remember(
        "User confirmed deploy target is staging.",
        kind="constraint",
        importance=5,
        provenance="user_confirmation",
        actor="user",
    )
    allowed = check_memory_use("deploy target", intended_use="external_action")
    assert allowed["allowed"] is True


def test_memory_policy_exposes_injected_text_and_guard_options():
    out = memory_policy()
    assert "RECALL FIRST" in out["instructions"]
    assert "user_confirmation" in out["provenance"]
    assert "external_action" in out["guard_uses"]


def test_namespace_scopes_writes_and_reads():
    forget_all()
    remember("the gateway is Kong", kind="fact", importance=4, namespace="proj-a")
    remember("the gateway is Envoy", kind="fact", importance=4, namespace="proj-b")

    a = recall("gateway", namespace="proj-a", limit=5)
    assert a and all(h["metadata"]["namespace"] == "proj-a" for h in a)
    assert not any("Envoy" in h["content"] for h in a)

    everything = recall("gateway", limit=5)  # unscoped reads still see all namespaces
    assert len(everything) == 2

    ctx = build_context("which gateway?", namespace="proj-b")
    assert "Envoy" in ctx and "Kong" not in ctx

    s = stats(namespace="proj-a")
    assert s["total"] == 1 and s["by_namespace"] == {"proj-a": 1, "proj-b": 1}


def test_capture_stamps_namespace():
    forget_all()
    out = capture("Deploy key expires on 2027-03-01.", kind="fact", namespace="proj-a")
    assert out["stored"] is True
    hit = recall("deploy key", namespace="proj-a", limit=1)
    assert hit and hit[0]["metadata"]["namespace"] == "proj-a"


def test_forget_matching_dry_run_then_delete():
    forget_all()
    remember("my badge PIN is 0042", kind="fact", importance=5)
    remember("the standup is at 9:30", kind="note", importance=2)

    preview = forget_matching("badge PIN", min_relevance=0.2)
    assert preview["dry_run"] is True and preview["matched"] == 1
    assert stats()["total"] == 2, "dry run must not delete"
    assert any("badge PIN" in r["content"] for r in preview["records"])

    erased = forget_matching("badge PIN", min_relevance=0.2, dry_run=False)
    assert erased["matched"] == 1 and stats()["total"] == 1
    assert recall("badge PIN", limit=3) == [] or all(
        "0042" not in h["content"] for h in recall("badge PIN", limit=3)
    )


def test_build_context_anchors_today():
    forget_all()
    remember("Launch date moved to September 14.", kind="fact", importance=5)
    ctx = build_context("when do we launch?", token_budget=120)
    assert "# Today is" in ctx, "the temporal anchor should reach real agents"


def test_maintenance_pass_consolidates_and_bounds():
    from midas.mcp_server import _run_maintenance_pass

    forget_all()
    remember("The launch is on September 14.", kind="fact", importance=3)
    remember("The launch is on September 14.", kind="fact", importance=3)  # near-duplicate
    out = _run_maintenance_pass()
    assert out["consolidated"] == 1
    assert stats()["total"] == 1


def test_recall_as_of_returns_the_historical_belief():
    forget_all()
    from midas.mcp_server import _mem

    # 1.7e9 ≈ 2023-11-14, 1.75e9 ≈ 2025-06-15; the as-of date falls between the two.
    _mem.remember("the launch date is September 14", kind="fact", created_at=1.7e9)
    _mem.remember("actually the launch date moved to October 2", kind="fact", created_at=1.75e9)

    current = recall("when is the launch?", limit=2)
    assert any("October 2" in h["content"] for h in current)

    historical = recall("when is the launch?", limit=2, as_of="2024-06-01")
    assert historical and all("October 2" not in h["content"] for h in historical)
    assert any("September 14" in h["content"] for h in historical)
    # The injected prompt is how installing Midas makes an agent start remembering on its own.
    assert server.instructions and "capture" in server.instructions.lower()
    assert "check_memory_use" in server.instructions


def test_build_context_pins_standing_directives():
    forget_all()
    capture("From now on, always reply in Spanish.", kind="chat")
    for i in range(20):
        remember(f"database migration note {i} about postgres tables", kind="note")
    ctx = build_context("how do I migrate the postgres schema?", token_budget=300)
    assert "reply in Spanish" in ctx, "the standing directive must be pinned into every context"


def test_policy_nudges_distillation():
    out = memory_policy()
    assert "DISTILLED" in out["instructions"]
    assert "compact, self-contained" in out["instructions"]


def test_distill_prompt_drives_no_llm_distillation():
    from midas.mcp_server import distill

    text = distill(session="proj-a")
    assert "compact, self-contained" in text
    assert "capture" in text and "proj-a" in text
    assert "no extra tool/model needed" in text or "no Midas-LLM" in text or "no extra" in text


def test_bearer_auth_middleware_gates_http() -> None:
    """The HTTP transport must reject requests without the exact bearer token (401 + WWW-Authenticate),
    pass authorized ones through, and leave non-http scopes (lifespan) untouched."""
    import asyncio

    from midas.mcp_server import _BearerAuth

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    mw = _BearerAuth(app, "s3cret")

    def call(scope_type: str, auth: str | None):
        headers = [(b"authorization", auth.encode())] if auth else []
        events: list = []

        async def send(ev):
            events.append(ev)

        asyncio.run(mw({"type": scope_type, "headers": headers}, None, send))
        return events

    assert call("http", "Bearer s3cret")[0]["status"] == 200
    assert call("http", "Bearer wrong")[0]["status"] == 401
    assert call("http", None)[0]["status"] == 401
    assert call("lifespan", None)[0]["status"] == 200  # non-http scopes pass straight through
