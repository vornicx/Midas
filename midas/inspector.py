"""Midas Inspector — a LOCAL, glass-box UI over your memory.

`midas inspect` starts a tiny local web server (stdlib only) that reads your SQLite store and shows what
your agent actually remembers: the verbatim records, their provenance and source, the belief-revision
history (with time-travel), the current project state, what changed since a date, and the governance/
audit view — and lets you correct, pin, or forget. It runs on localhost over your own file: **zero
egress, no LLM, no account.** This is the differentiator competitors can't offer — their memory is
LLM-rewritten facts with nothing to show; Midas's is source-traceable and governed.

The HTTP layer is a thin router; the real work is the `api_*` functions below (pure, SDK-only, unit-
tested). Run:  midas inspect --db ~/.midas/memory.sqlite3   (or `python -m midas.inspector`)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from .audit import audit_completeness, audit_record, audit_use, belief_history, forgetting_receipt
from .coding import project_state
from .projects import list_projects, project_governance, project_of, project_overview
from .state import memory_diff

if TYPE_CHECKING:
    from .memory import Memory


# --- API (pure, SDK-only — the testable core) --------------------------------------------------

def api_records(mem: "Memory", *, q: str = "", kind: str = "", limit: int = 200) -> list[dict[str, Any]]:
    """Browse memory: a relevance-ranked search when `q` is given, else newest-first; optional kind
    filter. Returns verbatim, source-traceable records (no embeddings)."""
    if q:
        recs = [h.record for h in mem.recall(q, limit=limit, min_relevance_ratio=0.0)]
    else:
        recs = sorted(mem.store.all(), key=lambda r: r.updated_at, reverse=True)
    if kind:
        recs = [r for r in recs if r.kind == kind]
    return [audit_record(r) for r in recs[:limit]]


def api_record(mem: "Memory", record_id: str) -> dict[str, Any] | None:
    """One record + its full belief-revision history (oldest → newest) — the time-travel view."""
    rec = next((r for r in mem.store.all() if r.id == record_id), None)
    if rec is None:
        return None
    return {"record": audit_record(rec), "history": [audit_record(h) for h in belief_history(mem, record_id)]}


def api_project_state(mem: "Memory", project: str) -> dict[str, list[dict[str, Any]]]:
    """The live code-state of a project, grouped by code_kind (decisions / bugs / forbidden / …)."""
    return {k: [audit_record(r) for r in recs] for k, recs in project_state(mem, project).items()}


def api_projects(mem: "Memory") -> list[dict[str, Any]]:
    """Every project with at-a-glance stats — the Projects home."""
    return list_projects(mem)


def api_project(mem: "Memory", name: str) -> dict[str, Any]:
    """Per-project detail: overview stats + governance (prohibitions, trust mix, actors) + the code-state
    grouped by kind + recent records."""
    recs = sorted((r for r in mem.store.all() if project_of(r) == name),
                  key=lambda r: r.updated_at, reverse=True)
    gov = project_governance(mem, name)
    return {
        "overview": project_overview(mem, name),
        "governance": {
            "forbidden": [audit_record(r) for r in gov["forbidden"]],
            "by_provenance": gov["by_provenance"],
            "by_actor": gov["by_actor"],
            "confirmations": gov["confirmations"],
            "revisions": gov["revisions"],
        },
        "state": {k: [audit_record(r) for r in v] for k, v in project_state(mem, name).items()},
        "recent": [audit_record(r) for r in recs[:12]],
    }


def api_diff(mem: "Memory", *, hours: float = 24.0) -> dict[str, Any]:
    """What was added or revised in the last `hours` — the 'what changed since last session' view."""
    diff = memory_diff(mem, time.time() - hours * 3600.0)
    return {
        "added": [audit_record(r) for r in diff.added],
        "revised": [{"old": audit_record(o), "new": audit_record(n)} for o, n in diff.revised],
    }


def api_audit(mem: "Memory", query: str, use: str = "external_action") -> dict[str, Any]:
    """The governance/audit artifact: would memory authorize this use, and why — with full provenance."""
    return audit_use(mem, query, use)


def api_forget(mem: "Memory", record_id: str) -> dict[str, Any]:
    """Forget a record and return a content-hashed erasure receipt (proves it was erased)."""
    rec = next((r for r in mem.store.all() if r.id == record_id), None)
    if rec is None:
        return {"ok": False, "error": "not found"}
    receipt = forgetting_receipt([rec], actor="inspector", reason="forgotten via inspector")
    mem.forget(record_id)
    return {"ok": True, "receipt": receipt}


def api_stats(mem: "Memory") -> dict[str, Any]:
    recs = list(mem.store.all())
    kinds: dict[str, int] = {}
    for r in recs:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    return {"total": len(recs), "live": sum(1 for r in recs if r.superseded_by is None), "kinds": kinds}


def api_conflicts(mem: "Memory", *, limit: int = 20) -> list[dict[str, Any]]:
    """Live beliefs that contradict each other with neither superseding the other — the multi-agent
    view. Ranked candidates for a HUMAN to resolve (forget one, or capture the corrected value)."""
    from .continuity import memory_conflicts

    return [
        {"signal": c.signal, "similarity": round(c.similarity, 3),
         "a": audit_record(c.a), "b": audit_record(c.b)}
        for c in memory_conflicts(mem, limit=limit)
    ]


def api_loops(mem: "Memory") -> list[dict[str, Any]]:
    """Open commitments (promised work never closed), oldest — most overdue — first."""
    from .continuity import open_loops

    return [audit_record(r) for r in open_loops(mem)]


def api_close_loop(mem: "Memory", loop_id: str, resolution: str) -> dict[str, Any]:
    """Close a commitment from the UI: records the resolution and supersedes the open loop."""
    from .continuity import close_loop

    try:
        rec = close_loop(mem, loop_id, resolution or "closed via inspector", actor="inspector")
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "closed_by": audit_record(rec)}


def api_audit_chain(mem: "Memory", *, limit: int = 50) -> dict[str, Any]:
    """The tamper-evident mutation log: chain verification + the most recent entries (hashes only)."""
    store = mem.store
    if not hasattr(store, "verify_audit_log"):
        return {"supported": False}
    result = store.verify_audit_log()
    return {"supported": True, **result, "tail": store.audit_log(limit=limit)}


def api_overview(mem: "Memory") -> dict[str, Any]:
    """The memory-health dashboard a team/enterprise needs at a glance: counts, attributability (the
    compliance metric: fraction with both a source and an actor), revision activity, recency, and the
    distribution by kind / provenance / project / recency tier. All computed — no fabricated counters."""
    recs = list(mem.store.all())
    total = len(recs)
    now = time.time()
    by_kind: dict[str, int] = {}
    by_prov: dict[str, int] = {}
    by_tier: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    projects: dict[str, int] = {}
    imp_sum = added_24h = added_7d = 0
    for r in recs:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
        by_prov[r.provenance] = by_prov.get(r.provenance, 0) + 1
        by_tier[mem.tier(r, now=now)] += 1
        imp_sum += r.importance
        proj = (r.metadata or {}).get("project")
        if proj:
            projects[proj] = projects.get(proj, 0) + 1
        if now - r.created_at < 86400:
            added_24h += 1
        if now - r.created_at < 7 * 86400:
            added_7d += 1
    rank = lambda d: dict(sorted(d.items(), key=lambda kv: -kv[1]))  # noqa: E731
    return {
        "total": total,
        "live": sum(1 for r in recs if r.superseded_by is None),
        "superseded": sum(1 for r in recs if r.superseded_by is not None),
        "high_importance": sum(1 for r in recs if r.importance >= 4),
        "avg_importance": round(imp_sum / total, 2) if total else 0,
        "attributable": round(audit_completeness(recs), 2),
        "added_24h": added_24h,
        "added_7d": added_7d,
        "by_kind": rank(by_kind),
        "by_provenance": rank(by_prov),
        "by_tier": by_tier,
        "projects": rank(projects),
    }


def api_timeseries(mem: "Memory", *, days: int = 30) -> dict[str, Any]:
    """Daily capture volume for the last `days` days (UTC-bucketed) — the activity trend a static
    count can't show. Zero-filled so gaps in captures show as true zeros, not missing days."""
    import datetime as _dt

    days = max(1, int(days))
    today = _dt.datetime.now(tz=_dt.timezone.utc).date()
    buckets: dict[str, int] = {
        (today - _dt.timedelta(days=i)).isoformat(): 0 for i in range(days - 1, -1, -1)
    }
    cutoff = time.time() - days * 86_400.0
    for r in mem.store.all():
        if r.created_at < cutoff:
            continue
        day = _dt.datetime.fromtimestamp(r.created_at, tz=_dt.timezone.utc).date().isoformat()
        if day in buckets:
            buckets[day] += 1
    return {"days": days, "buckets": [{"date": d, "count": c} for d, c in buckets.items()]}


def api_meta(mem: "Memory", *, db: str, embedder_name: str) -> dict[str, Any]:
    """Which store/version/embedder this Inspector instance is actually looking at — so a user
    juggling several stores (or a bug report) doesn't have to guess. No memory content."""
    from midas import __version__

    store = mem.store
    return {
        "version": __version__,
        "db": db,
        "embedder": embedder_name,
        "records": len(store.all()),
        "schema_version": store.schema_version() if hasattr(store, "schema_version") else None,
        "audit_chain": hasattr(store, "verify_audit_log"),
    }


# --- The embedded UI ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html><html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Midas Inspector</title>
<link rel="icon" href="data:,">
<script>(function(){try{var t=localStorage.getItem('midas-theme');
if(!t){t=matchMedia('(prefers-color-scheme: light)').matches?'light':'dark';}
document.documentElement.dataset.theme=t;}catch(e){}})();</script>
<style>
:root{
 --bg:#0d0d15;--plane:#0a0a10;--surf:rgba(255,255,255,.034);--surf2:rgba(255,255,255,.065);--surf3:rgba(255,255,255,.09);
 --text:#eceef5;--steel:#8b93ad;--muted:#6f7795;
 --line:rgba(255,255,255,.10);--line2:rgba(255,255,255,.15);
 --gold:#FFD700;--gold2:#e9bd28;--gsoft:rgba(255,215,0,.10);--gline:rgba(255,215,0,.22);--gold-ink:#3a2e00;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 34px rgba(0,0,0,.28);
 --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
 --good-soft:rgba(12,163,12,.14);--warning-soft:rgba(250,178,25,.16);--critical-soft:rgba(208,59,59,.14);
 --good-line:rgba(12,163,12,.34);--warning-line:rgba(250,178,25,.4);--critical-line:rgba(208,59,59,.34);
 --hue-aqua:#199e70;--hue-amber:#c98500;--hue-blue:#3987e5;--hue-magenta:#d55181;--hue-violet:#9085e9;--hue-orange:#d95926;
 --focus:0 0 0 3px rgba(255,215,0,.32);
}
:root[data-theme="light"]{
 --bg:#f7f6f2;--plane:#f2f1ec;--surf:#ffffff;--surf2:#fdfcf8;--surf3:#f4f2ec;
 --text:#17160f;--steel:#5c5a4e;--muted:#83806f;
 --line:rgba(20,18,8,.11);--line2:rgba(20,18,8,.18);
 --gold:#a86f00;--gold2:#8f5e00;--gsoft:rgba(168,111,0,.10);--gline:rgba(168,111,0,.30);--gold-ink:#3a2e00;
 --shadow:0 1px 2px rgba(20,18,8,.06),0 10px 28px rgba(20,18,8,.10);
 --good:#0a7d0a;--warning:#a56600;--serious:#b3502f;--critical:#b4302f;
 --good-soft:rgba(10,125,10,.10);--warning-soft:rgba(165,102,0,.12);--critical-soft:rgba(180,48,47,.10);
 --good-line:rgba(10,125,10,.30);--warning-line:rgba(165,102,0,.34);--critical-line:rgba(180,48,47,.30);
 --hue-aqua:#1baf7a;--hue-amber:#b07a00;--hue-blue:#2a78d6;--hue-magenta:#c1477a;--hue-violet:#4a3aa7;--hue-orange:#c14f1e;
 --focus:0 0 0 3px rgba(168,111,0,.28);
}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;color:var(--text);-webkit-font-smoothing:antialiased;letter-spacing:-.006em;
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;background:var(--bg);transition:background .2s,color .2s}
:root[data-theme="dark"] body{background:
 radial-gradient(900px 520px at 10% -8%,rgba(92,92,156,.18),transparent 60%),
 radial-gradient(760px 460px at 110% 2%,rgba(255,215,0,.07),transparent 55%),var(--bg)}
::selection{background:var(--gline)}a{color:var(--gold);text-decoration:none}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:var(--line2);border-radius:8px;border:2px solid transparent;background-clip:content-box}
:focus-visible{outline:none;box-shadow:var(--focus);border-radius:6px}
button,input,select{font-family:inherit}
@keyframes rise{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:scale(.96) translateY(6px)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){*{animation-duration:.001s!important;transition-duration:.001s!important}}
.app{display:grid;grid-template-columns:250px 1fr;min-height:100vh}
.side{border-right:1px solid var(--line);padding:20px 14px;display:flex;flex-direction:column;gap:2px;
position:sticky;top:0;height:100vh;overflow-y:auto}
:root[data-theme="dark"] .side{background:linear-gradient(180deg,rgba(255,255,255,.018),transparent)}
.brand{display:flex;align-items:center;gap:11px;padding:6px 8px 18px}
.brand .m{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;color:#15151f;font-weight:900;font-size:17px;
background:linear-gradient(145deg,#ffe874,#e0ad00);box-shadow:0 4px 16px rgba(255,200,0,.32),inset 0 1px 0 rgba(255,255,255,.5);flex-shrink:0}
.brand .bt{display:flex;flex-direction:column;line-height:1.18;flex:1;min-width:0}
.brand .bt b{font-size:15px;font-weight:700;letter-spacing:-.01em}
.brand .bt span{font-size:10px;color:var(--steel);text-transform:uppercase;letter-spacing:.15em}
.themebtn{flex-shrink:0;width:30px;height:30px;display:grid;place-items:center;border-radius:9px;border:1px solid var(--line2);
background:var(--surf);color:var(--steel);cursor:pointer;transition:border-color .15s,color .15s}
.themebtn:hover{color:var(--gold);border-color:var(--gline)}.themebtn svg{width:15px;height:15px}
.tag.imp-hi{border-color:var(--gline);color:var(--gold)}
.navlbl{font-size:10px;text-transform:uppercase;letter-spacing:.16em;color:var(--steel);opacity:.65;padding:14px 11px 5px}
.navlbl:first-of-type{padding-top:8px}
.nav a{position:relative;display:flex;align-items:center;gap:11px;padding:8px 11px;border-radius:10px;color:var(--steel);
cursor:pointer;font-weight:500;font-size:13.5px;transition:background .15s,color .15s}
.nav a:hover{background:var(--surf2);color:var(--text)}.nav a.on{background:var(--gsoft);color:var(--gold)}
.nav a.on::before{content:"";position:absolute;left:-14px;top:8px;bottom:8px;width:3px;border-radius:0 3px 3px 0;background:var(--gold)}
:root[data-theme="dark"] .nav a.on::before{box-shadow:0 0 12px var(--gold)}
.nav svg{width:16px;height:16px;opacity:.9;flex-shrink:0}
.navcount{margin-left:auto;font-size:10.5px;padding:1px 6px;border-radius:999px;background:var(--surf3);color:var(--steel);font-variant-numeric:tabular-nums}
.navcount.warn{background:var(--warning-soft);color:var(--warning)}
.foot{margin-top:auto;padding-top:14px;border-top:1px solid var(--line);display:flex;flex-direction:column;gap:8px}
.footrow{display:flex;flex-wrap:wrap;gap:6px}
.pill{display:inline-flex;gap:6px;align-items:center;background:var(--surf);border:1px solid var(--line);
border-radius:8px;padding:5px 9px;font-size:11px;color:var(--steel)}.pill b{color:var(--text);font-variant-numeric:tabular-nums}
.meta{font-size:10.5px;color:var(--muted);line-height:1.5;overflow-wrap:anywhere}
.kbdhint{font-size:10.5px;color:var(--muted);display:flex;align-items:center;gap:5px;cursor:pointer}
.kbdhint:hover{color:var(--steel)}
kbd{font:10px/1 ui-monospace,monospace;padding:2px 5px;border-radius:5px;border:1px solid var(--line2);background:var(--surf2);color:var(--steel)}
.main{padding:34px 40px 60px;max-width:1180px}
.vhead{animation:rise .38s cubic-bezier(.16,1,.3,1) both;margin-bottom:22px;display:flex;align-items:flex-start;justify-content:space-between;gap:20px}
.vhead h1{font-size:24px;margin:0;font-weight:700;letter-spacing:-.02em}
.vhead p{margin:6px 0 0;color:var(--steel);font-size:13.5px;max-width:64ch}
.vhead .side-stat{flex-shrink:0;text-align:right;padding-top:2px}
.vhead .side-stat b{font-size:22px;font-variant-numeric:tabular-nums;display:block;line-height:1}
.vhead .side-stat span{font-size:11px;color:var(--steel)}
.controls{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-bottom:20px}
input,select,textarea{background:var(--surf);border:1px solid var(--line2);color:var(--text);border-radius:11px;
padding:11px 13px;font:inherit;font-size:13.5px;outline:none;transition:border-color .15s,box-shadow .15s}
input::placeholder,textarea::placeholder{color:var(--muted)}
input:focus,select:focus,textarea:focus{border-color:var(--gline);box-shadow:var(--focus)}
select{appearance:none;-webkit-appearance:none;cursor:pointer;padding-right:36px;background-repeat:no-repeat;background-position:right 13px center;
background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238b93ad' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E")}
.search{flex:1;min-width:220px}
.btn{background:linear-gradient(145deg,#ffe874,#e7b400);color:#15151f;border:none;border-radius:11px;
padding:10px 17px;font-weight:600;font-size:13.5px;cursor:pointer;box-shadow:0 2px 10px rgba(255,190,0,.18);
transition:transform .12s,filter .15s,box-shadow .15s}
.btn:hover{filter:brightness(1.05);transform:translateY(-1px);box-shadow:0 5px 18px rgba(255,190,0,.28)}.btn:active{transform:none}
.btn.ghost{background:transparent;border:1px solid var(--line2);color:var(--text);box-shadow:none}
.btn.ghost:hover{background:var(--surf2);filter:none;transform:none}
.btn.danger{background:transparent;border:1px solid var(--critical-line);color:var(--critical);box-shadow:none}
.btn.danger:hover{background:var(--critical-soft);transform:none;filter:none}
.btn.sm{padding:6px 12px;font-size:12px;border-radius:9px}
.btn:disabled{opacity:.5;cursor:default;transform:none!important}
.card{border:1px solid var(--line);border-radius:14px;padding:15px 17px;margin-bottom:10px;background:var(--surf);
animation:rise .34s cubic-bezier(.16,1,.3,1) both;transition:border-color .16s,background .16s,box-shadow .16s}
.card:hover{border-color:var(--line2);box-shadow:var(--shadow)}.card.dim{opacity:.56}
.meta-row{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-bottom:9px}
.tag{font-size:11px;padding:2.5px 9px 2.5px 8px;border-radius:7px;background:var(--surf3);color:var(--steel);font-weight:500;
border:1px solid transparent;display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.tag .dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.tag.imp{background:transparent;border-color:var(--line2)}
.tag.proj{border-color:var(--line2);color:var(--steel)}
.tag.status-good{background:var(--good-soft);color:var(--good);border-color:var(--good-line)}
.tag.status-critical{background:var(--critical-soft);color:var(--critical);border-color:var(--critical-line)}
.tag.status-warning{background:var(--warning-soft);color:var(--warning);border-color:var(--warning-line)}
.tag.gold{background:var(--gsoft);color:var(--gold);border-color:var(--gline)}
.when{margin-left:auto;font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums;cursor:default}
.content{font-size:14px;line-height:1.55;white-space:pre-wrap}
.src{margin-top:10px;font-family:ui-monospace,"SF Mono",monospace;font-size:11px;color:var(--muted)}
.acts{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}
.muted{color:var(--steel)}.mut2{color:var(--muted)}.gold-text{color:var(--gold)}
.empty{text-align:center;color:var(--steel);padding:56px 24px;border:1px dashed var(--line2);border-radius:16px;background:var(--surf)}
.empty svg{width:34px;height:34px;opacity:.5;margin-bottom:12px}
.empty .el{font-size:13.5px}
.tl{margin-top:12px;border-left:1.5px solid var(--gline);padding-left:18px;display:flex;flex-direction:column;gap:13px}
.tl .ev{position:relative}.tl .ev::before{content:"";position:absolute;left:-25px;top:5px;width:9px;height:9px;border-radius:50%;background:var(--steel);border:2px solid var(--bg)}
.tl .ev.cur::before{background:var(--gold)}
:root[data-theme="dark"] .tl .ev.cur::before{box-shadow:0 0 0 4px rgba(255,215,0,.18),0 0 12px var(--gold)}
.tl .t{font-size:11px;color:var(--steel);font-variant-numeric:tabular-nums;display:flex;gap:8px;align-items:center}
.tl .meta-row{margin-bottom:5px}
.verdict{animation:rise .34s cubic-bezier(.16,1,.3,1) both;border-radius:18px;padding:22px 24px;border:1px solid var(--line2);
margin-bottom:20px;display:flex;align-items:center;gap:18px;background:var(--surf)}
.verdict.ok{background:linear-gradient(120deg,var(--good-soft),transparent 70%);border-color:var(--good-line)}
.verdict.no{background:linear-gradient(120deg,var(--critical-soft),transparent 70%);border-color:var(--critical-line)}
.verdict .big{display:grid;place-items:center;width:52px;height:52px;border-radius:50%;flex-shrink:0}
.verdict .big svg{width:24px;height:24px}
.verdict.ok .big{background:var(--good-soft);color:var(--good);box-shadow:0 0 0 1px var(--good-line)}
.verdict.no .big{background:var(--critical-soft);color:var(--critical);box-shadow:0 0 0 1px var(--critical-line)}
.verdict h2{margin:0;font-size:18px;font-weight:700;letter-spacing:.02em}.verdict .r{color:var(--steel);font-size:13px;margin-top:4px}
.verdict .r b{color:var(--text);font-variant-numeric:tabular-nums}
.meter{height:6px;background:var(--surf3);border-radius:999px;overflow:hidden;margin-top:2px;width:170px}
.meter i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--gold2),var(--gold))}
h3.grp{margin:24px 0 12px;font-size:11px;text-transform:uppercase;letter-spacing:.13em;color:var(--steel);font-weight:600;display:flex;align-items:center;gap:8px}
h3.grp:first-child{margin-top:0}
h3.grp .n{font-variant-numeric:tabular-nums;color:var(--muted);font-weight:500;text-transform:none;letter-spacing:0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px;animation:rise .34s cubic-bezier(.16,1,.3,1) both}
.stat{position:relative;border:1px solid var(--line);border-radius:16px;padding:17px;background:var(--surf);overflow:hidden;transition:border-color .16s,transform .16s,box-shadow .16s}
:root[data-theme="dark"] .stat{box-shadow:inset 0 1px 0 rgba(255,255,255,.05)}
.stat:hover{border-color:var(--line2);transform:translateY(-2px);box-shadow:var(--shadow)}
.stat .sv{font-size:29px;font-weight:700;color:var(--text);line-height:1;letter-spacing:-.025em;font-variant-numeric:tabular-nums}
.stat .sl{font-size:12.5px;color:var(--text);margin-top:9px;font-weight:500}.stat .ss{font-size:10.5px;color:var(--steel);margin-top:3px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px;animation:rise .34s cubic-bezier(.16,1,.3,1) both}
.panel{border:1px solid var(--line);border-radius:16px;padding:16px 18px 18px;background:var(--surf)}
:root[data-theme="dark"] .panel{box-shadow:inset 0 1px 0 rgba(255,255,255,.05)}
.panel h3.grp{margin:0 0 13px}
.ogrid{display:grid;grid-template-columns:1.5fr 1fr;gap:14px;align-items:start;margin-bottom:14px;animation:rise .34s cubic-bezier(.16,1,.3,1) both}
.ocol{display:flex;flex-direction:column;gap:14px}
.rrow{display:flex;align-items:center;gap:11px;padding:10px 0;border-bottom:1px solid var(--line)}
.rrow:last-child{border-bottom:0;padding-bottom:2px}
.rrow .rc{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--steel);flex-shrink:0}
.plist{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;animation:rise .34s cubic-bezier(.16,1,.3,1) both}
.pcard{border:1px solid var(--line);border-radius:14px;padding:16px;background:var(--surf);cursor:pointer;transition:border-color .16s,transform .16s,box-shadow .16s}
:root[data-theme="dark"] .pcard{box-shadow:inset 0 1px 0 rgba(255,255,255,.05)}
.pcard:hover{border-color:var(--line2);transform:translateY(-2px);box-shadow:var(--shadow)}
.pcard:focus-visible{border-color:var(--gline)}
.pcard .pn{font-size:15px;font-weight:600;color:var(--gold);margin-bottom:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pcard .pstat{font-size:12px;color:var(--steel);margin-top:3px}.pcard .pstat b{color:var(--text);font-variant-numeric:tabular-nums}
.pcard .pkinds{display:flex;gap:4px;margin-top:10px}
.back{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:var(--steel);cursor:pointer;margin-bottom:10px}
.back:hover{color:var(--gold)}.back svg{width:13px;height:13px}
@media(max-width:900px){.ogrid{grid-template-columns:1fr}}
.bar{display:flex;align-items:center;gap:11px;margin:9px 0;position:relative}
.bar .bk{width:132px;flex-shrink:0;font-size:12.5px;color:var(--steel);line-height:1.25;display:flex;align-items:center;gap:7px}
.bar .bt{flex:1;height:9px;background:var(--surf3);border-radius:999px;overflow:hidden;cursor:default}
.bar .bt i{display:block;height:100%;border-radius:999px;transition:width .5s cubic-bezier(.16,1,.3,1)}
.bar .bv{width:40px;text-align:right;font-size:12.5px;color:var(--text);font-variant-numeric:tabular-nums;font-weight:500}
.stackbar{display:flex;height:12px;border-radius:999px;overflow:hidden;background:var(--surf3);margin:6px 0 10px}
.stackbar i{height:100%}
.stacklegend{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--steel)}
.stacklegend span{display:inline-flex;align-items:center;gap:6px}
.cpair{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}
.conflict-head{display:flex;align-items:center;gap:9px;margin-bottom:4px;flex-wrap:wrap}
@media(max-width:760px){.cpair{grid-template-columns:1fr}.app{grid-template-columns:1fr}
.side{position:static;height:auto;flex-direction:column}.side>.nav,.side>.navlbl{display:none}
.side{padding:14px}.brand{padding-bottom:10px}
.mobnav{display:flex!important}
.foot{display:none}.main{padding:22px 18px}.vhead{flex-direction:column;gap:8px}.vhead .side-stat{text-align:left}}
.mobnav{display:none;gap:6px;overflow-x:auto;padding-bottom:4px;-webkit-overflow-scrolling:touch}
.mobnav a{flex-shrink:0;display:flex;align-items:center;gap:6px;padding:7px 12px;border-radius:999px;border:1px solid var(--line2);
color:var(--steel);font-size:12.5px;white-space:nowrap;cursor:pointer}
.mobnav a.on{color:var(--gold);border-color:var(--gline);background:var(--gsoft)}
.mobnav svg{width:14px;height:14px}
/* chart: sparkline */
.chartwrap{position:relative}
.spark{width:100%;height:120px;display:block;overflow:visible}
.spark .fill{fill:url(#sparkgrad)}
.spark .line{fill:none;stroke:var(--gold);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.spark .hit{fill:transparent}
.spark .cursor{stroke:var(--line2);stroke-width:1}
.spark .dot{fill:var(--gold);stroke:var(--bg);stroke-width:2}
.charttip{position:absolute;pointer-events:none;background:var(--surf2);border:1px solid var(--line2);border-radius:9px;
padding:6px 10px;font-size:11.5px;box-shadow:var(--shadow);white-space:nowrap;opacity:0;transition:opacity .1s;z-index:5;transform:translate(-50%,-100%)}
.charttip.on{opacity:1}
.charttip b{font-variant-numeric:tabular-nums}
/* toasts */
#toasts{position:fixed;right:18px;bottom:18px;display:flex;flex-direction:column;gap:8px;z-index:200;max-width:340px}
.toast{background:var(--surf2);border:1px solid var(--line2);border-radius:12px;padding:11px 14px;box-shadow:var(--shadow);
font-size:13px;display:flex;align-items:flex-start;gap:10px;animation:pop .22s cubic-bezier(.16,1,.3,1) both}
.toast svg{width:16px;height:16px;flex-shrink:0;margin-top:1px}
.toast.ok svg{color:var(--good)}.toast.err svg{color:var(--critical)}
.toast .tx{flex:1;line-height:1.4}.toast .tx small{display:block;color:var(--steel);font-size:11px;margin-top:2px;font-family:ui-monospace,monospace}
/* modal */
#modalwrap{position:fixed;inset:0;background:rgba(5,5,10,.55);backdrop-filter:blur(2px);display:none;place-items:center;z-index:150;padding:20px}
:root[data-theme="light"] #modalwrap{background:rgba(30,28,20,.35)}
#modalwrap.on{display:grid}
.modal{background:var(--plane);border:1px solid var(--line2);border-radius:18px;padding:22px 24px;width:100%;max-width:440px;
box-shadow:0 20px 60px rgba(0,0,0,.4);animation:pop .2s cubic-bezier(.16,1,.3,1) both}
.modal h3{margin:0 0 8px;font-size:16.5px}
.modal p{margin:0 0 16px;color:var(--steel);font-size:13.5px;line-height:1.5}
.modal textarea{width:100%;min-height:80px;resize:vertical;margin-bottom:14px}
.modal .mbtns{display:flex;justify-content:flex-end;gap:9px}
/* command palette */
#palwrap{position:fixed;inset:0;background:rgba(5,5,10,.5);display:none;place-items:flex-start;justify-content:center;z-index:160;padding-top:14vh}
:root[data-theme="light"] #palwrap{background:rgba(30,28,20,.32)}
#palwrap.on{display:grid}
.pal{width:100%;max-width:520px;background:var(--plane);border:1px solid var(--line2);border-radius:16px;box-shadow:0 24px 70px rgba(0,0,0,.45);
overflow:hidden;animation:pop .16s cubic-bezier(.16,1,.3,1) both}
.pal input{width:100%;border:none;border-radius:0;border-bottom:1px solid var(--line);padding:16px 18px;font-size:15px;background:transparent}
.pal input:focus{box-shadow:none}
.pal ul{list-style:none;margin:0;padding:8px;max-height:340px;overflow-y:auto}
.pal li{display:flex;align-items:center;gap:11px;padding:10px 12px;border-radius:10px;cursor:pointer;font-size:13.5px;color:var(--steel)}
.pal li.sel{background:var(--gsoft);color:var(--gold)}
.pal li svg{width:15px;height:15px;flex-shrink:0}
.pal li .k{margin-left:auto;font-size:10.5px;color:var(--muted)}
.skiplink{position:absolute;left:-9999px;top:0;background:var(--gold);color:#15151f;padding:10px 16px;border-radius:0 0 10px 0;font-weight:600;z-index:300}
.skiplink:focus{left:0}
</style></head>
<body>
<a class="skiplink" href="#main">Skip to content</a>
<div class="app">
<aside class="side">
<div class="brand"><span class="m">M</span><span class="bt"><b>Midas</b><span>Inspector</span></span>
<button class="themebtn" id="themebtn" type="button" aria-label="Toggle theme" title="Toggle theme (dark/light)"></button>
</div>
<div class="mobnav" id="mobnav"></div>
<div class="navlbl">Memory</div>
<nav class="nav" id="nav" role="tablist">
<a data-t="overview" role="tab"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg><span class="lbl">Overview</span></a>
<a data-t="browse" role="tab"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h11"/></svg><span class="lbl">Browse</span></a>
<a data-t="changed" role="tab"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 2"/></svg><span class="lbl">Changed</span></a>
</nav>
<div class="navlbl">Coding</div>
<nav class="nav">
<a data-t="project" role="tab"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg><span class="lbl">Projects</span></a>
</nav>
<div class="navlbl">Governance</div>
<nav class="nav">
<a data-t="gov" role="tab"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l8 3v6c0 5-3.5 7.6-8 9-4.5-1.4-8-4-8-9V6z"/></svg><span class="lbl">Governance</span></a>
<a data-t="conflicts" role="tab"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v4m0 4h.01M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg><span class="lbl">Conflicts</span><span class="navcount" id="cnt-conflicts" hidden></span></a>
<a data-t="loops" role="tab"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg><span class="lbl">Open loops</span><span class="navcount" id="cnt-loops" hidden></span></a>
<a data-t="auditlog" role="tab"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h10M4 18h6"/><circle cx="18" cy="16" r="3"/><path d="M20.5 18.5L23 21"/></svg><span class="lbl">Audit log</span></a>
</nav>
<div class="foot">
<div class="footrow" id="foot"></div>
<div class="meta" id="metafoot"></div>
<div class="kbdhint" id="palhint"><kbd>&#8984;K</kbd> quick jump</div>
</div>
</aside>
<main class="main" id="main" tabindex="-1"></main>
</div>
<div id="toasts" aria-live="polite"></div>
<div id="modalwrap"><div class="modal" id="modal" role="dialog" aria-modal="true"></div></div>
<div id="palwrap"><div class="pal"><input id="palinput" placeholder="Jump to a view, or type to search memory…" autocomplete="off">
<ul id="pallist"></ul></div></div>
<script>
const main=document.getElementById('main');
let tab='overview';
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const ago=t=>{if(!t)return'';const s=(Date.now()/1000)-t;const d=s/86400;
 return d>=1?Math.round(d)+'d ago':s>=3600?Math.round(s/3600)+'h ago':Math.max(1,Math.round(s/60))+'m ago'};
const fmtDate=t=>t?new Date(t*1000).toISOString().replace('T',' ').slice(0,16)+' UTC':'';
const get=async u=>(await fetch(u)).json();
const post=async(u,b)=>(await fetch(u,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)})).json();

// ---- icons ----
const ICON_CHECK='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>';
const ICON_CROSS='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 6 6 18M6 6l12 12"/></svg>';
const ICON_BACK='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 18l-6-6 6-6"/></svg>';
const ICON_SEARCH_SM='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>';
const EMPTY_ICON='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/></svg>';
function emptyState(text){return `<div class="empty">${EMPTY_ICON}<div class="el">${text}</div></div>`;}

// ---- categorical color system (fixed assignment — never reassigned by frequency/order) ----
const KIND_HUE={note:null,chat:'aqua',mission:'amber',fact:'blue',preference:'magenta',constraint:'violet',commitment:'orange'};
const PROV_HUE={observation:null,planning:'violet',action:'blue'};
const hueVar=h=>h?`var(--hue-${h})`:'var(--steel)';
const kindColor=k=>hueVar(KIND_HUE[k]);
const provColor=p=>p==='user_confirmation'?'var(--gold)':hueVar(PROV_HUE[p]);
function kindTag(k){return `<span class="tag"><i class="dot" style="background:${kindColor(k)}"></i>${esc(k)}</span>`;}
function provTag(p){
 if(p==='user_confirmation')return `<span class="tag gold"><i class="dot" style="background:var(--gold)"></i>${esc(p)}</span>`;
 return `<span class="tag"><i class="dot" style="background:${provColor(p)}"></i>${esc(p)}</span>`;
}

// ---- shared card renderer ----
function card(r,acts=''){
 const sup=r.superseded_by?'<span class="tag status-critical">superseded</span>':'';
 const proj=(r.metadata&&r.metadata.project)?`<span class="tag proj">${esc(r.metadata.project)}</span>`:'';
 const impCls=r.importance>=4?'tag imp imp-hi':'tag imp';
 return `<div class="card${r.superseded_by?' dim':''}"><div class="meta-row">
  ${kindTag(r.kind)}<span class="${impCls}" title="importance ${r.importance} of 5">imp ${r.importance}</span>
  ${provTag(r.provenance)}${proj}${sup}
  <span class="when" title="${fmtDate(r.created_at)}">${ago(r.created_at)}</span></div>
  <div class="content">${esc(r.content)}</div>
  ${(r.actor||r.source)?`<div class="src">${esc(r.actor||'')}${r.source?' &middot; '+esc(r.source):''}</div>`:''}
  ${acts?`<div class="acts">${acts}</div><div id="d-${r.id}"></div>`:''}</div>`;
}
function head(t,d){return `<div class="vhead"><div><h1>${t}</h1><p>${d}</p></div></div>`;}

// ---- bar chart (shared) ----
function barList(entries,colorFn,opts={}){
 if(!entries.length)return `<div class="muted" style="padding:4px 0">&mdash;</div>`;
 const mx=Math.max(...entries.map(e=>e[1]));
 const total=entries.reduce((a,e)=>a+e[1],0);
 return entries.map(([k,v])=>{
  const pct=mx?Math.round(v/mx*100):0, sh=total?Math.round(v/total*100):0;
  const label=opts.labelFmt?opts.labelFmt(k):k;
  const color=colorFn?colorFn(k):null;
  const dotHtml=colorFn?`<i class="dot" style="background:${color}"></i>`:'';
  const fill=colorFn?color:'linear-gradient(90deg,var(--gold2),var(--gold))';
  return `<div class="bar" title="${esc(label)}: ${v} (${sh}% of ${total})">
   <div class="bk">${dotHtml}${esc(label)}</div>
   <div class="bt"><i style="width:${pct}%;background:${fill}"></i></div>
   <div class="bv">${v}</div></div>`;
 }).join('');
}
function recencyBar(t){
 const total=(t.short+t.medium+t.long)||1;
 const seg=(v,color)=>v?`<i style="width:${v/total*100}%;background:${color}"></i>`:'';
 return `<div class="stackbar">${seg(t.short,'var(--gold)')}${seg(t.medium,'var(--hue-blue)')}${seg(t.long,'var(--steel)')}</div>
  <div class="stacklegend">
   <span><i class="dot" style="background:var(--gold)"></i>Short &middot; &le;1d &middot; ${t.short}</span>
   <span><i class="dot" style="background:var(--hue-blue)"></i>Medium &middot; &le;7d &middot; ${t.medium}</span>
   <span><i class="dot" style="background:var(--steel)"></i>Long &middot; ${t.long}</span></div>`;
}

// ---- sparkline (real chart: hover crosshair + tooltip) ----
function sparkline(buckets,elId){
 const w=640,h=112,pad=6,n=buckets.length||1;
 const mx=Math.max(1,...buckets.map(b=>b.count));
 const x=i=>pad+i*(w-2*pad)/(n-1||1), y=v=>h-pad-(v/mx)*(h-2*pad);
 const pts=buckets.map((b,i)=>[x(i),y(b.count)]);
 const line=pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
 const area=line+` L${x(n-1).toFixed(1)},${h-pad} L${x(0).toFixed(1)},${h-pad} Z`;
 return `<div class="chartwrap"><svg class="spark" viewBox="0 0 ${w} ${h}" id="${elId}" preserveAspectRatio="none">
  <defs><linearGradient id="sparkgrad" x1="0" y1="0" x2="0" y2="1">
   <stop offset="0%" stop-color="var(--gold)" stop-opacity=".28"/><stop offset="100%" stop-color="var(--gold)" stop-opacity="0"/>
  </linearGradient></defs>
  <path class="fill" d="${area}"/><path class="line" d="${line}"/>
  <line class="cursor" x1="0" y1="0" x2="0" y2="${h}" style="display:none" id="${elId}-cur"/>
  <circle class="dot" r="3.5" style="display:none" id="${elId}-dot"/>
  <rect class="hit" x="0" y="0" width="${w}" height="${h}"/></svg>
  <div class="charttip" id="${elId}-tip"></div></div>`;
}
function wireSparkline(elId,buckets){
 const svg=document.getElementById(elId); if(!svg||!buckets.length)return;
 const w=640,h=112,pad=6,n=buckets.length;
 const mx=Math.max(1,...buckets.map(b=>b.count));
 const x=i=>pad+i*(w-2*pad)/(n-1||1), y=v=>h-pad-(v/mx)*(h-2*pad);
 const cur=document.getElementById(elId+'-cur'),dot=document.getElementById(elId+'-dot'),tip=document.getElementById(elId+'-tip');
 const move=e=>{
  const rect=svg.getBoundingClientRect();
  const px=(e.clientX-rect.left)/rect.width*w;
  let i=Math.round((px-pad)/((w-2*pad)/(n-1||1)));
  i=Math.max(0,Math.min(n-1,i));
  const b=buckets[i];
  cur.setAttribute('x1',x(i));cur.setAttribute('x2',x(i));cur.style.display='block';
  dot.setAttribute('cx',x(i));dot.setAttribute('cy',y(b.count));dot.style.display='block';
  tip.style.left=((x(i)/w)*rect.width)+'px'; tip.style.top=((y(b.count)/h)*rect.height)+'px';
  tip.classList.add('on'); tip.innerHTML=`<b>${b.count}</b> capture${b.count===1?'':'s'} &middot; ${b.date}`;
 };
 svg.addEventListener('mousemove',move);
 svg.addEventListener('mouseleave',()=>{cur.style.display='none';dot.style.display='none';tip.classList.remove('on');});
}

// ---- toasts ----
function toast(msg,kind='ok',sub=''){
 const el=document.createElement('div');
 el.className=`toast ${kind==='ok'?'ok':'err'}`;
 el.innerHTML=`${kind==='ok'?ICON_CHECK:ICON_CROSS}<div class="tx">${esc(msg)}${sub?`<small>${esc(sub)}</small>`:''}</div>`;
 document.getElementById('toasts').appendChild(el);
 setTimeout(()=>{el.style.opacity='0';el.style.transition='opacity .25s';setTimeout(()=>el.remove(),260);},3600);
}

// ---- modal (replaces confirm()/prompt()) ----
function modal({title,body,confirmLabel='Confirm',cancelLabel='Cancel',danger=false,withTextarea=false,placeholder=''}){
 return new Promise(resolve=>{
  const wrap=document.getElementById('modalwrap'),m=document.getElementById('modal');
  m.innerHTML=`<h3>${esc(title)}</h3><p>${body}</p>`
   +(withTextarea?`<textarea id="modaltext" placeholder="${esc(placeholder)}"></textarea>`:'')
   +`<div class="mbtns"><button class="btn ghost" id="mcancel">${esc(cancelLabel)}</button>
      <button class="btn ${danger?'danger':''}" id="mok">${esc(confirmLabel)}</button></div>`;
  wrap.classList.add('on');
  const ta=document.getElementById('modaltext');
  (ta||document.getElementById('mok')).focus();
  function close(v){wrap.classList.remove('on');document.removeEventListener('keydown',onKey);wrap.onclick=null;resolve(v);}
  function onKey(e){if(e.key==='Escape')close(null);else if(e.key==='Enter'&&!withTextarea)close(true);}
  document.addEventListener('keydown',onKey);
  document.getElementById('mcancel').onclick=()=>close(null);
  document.getElementById('mok').onclick=()=>close(withTextarea?(ta.value||''):true);
  wrap.onclick=e=>{if(e.target===wrap)close(null);};
 });
}
async function forgetPrompt(id){
 const ok=await modal({title:'Forget this memory?',
  body:'A tamper-evident erasure receipt is recorded (a hash proving something was removed) &mdash; the content itself is gone for good.',
  confirmLabel:'Forget',danger:true});
 if(!ok)return;
 const res=await post('/api/forget',{id});
 if(res.ok)toast('Memory forgotten','ok',res.receipt&&res.receipt.items[0]?'sha '+res.receipt.items[0].content_sha256.slice(0,16)+'&hellip;':'');
 else toast(res.error||'Could not forget that memory','err');
 loadFoot(); if(V[tab])V[tab]();
}
async function closeLoopPrompt(id){
 const res=await modal({title:'Close this loop',
  body:'How was it resolved? Recorded as the resolution, and the open commitment is superseded by it.',
  withTextarea:true,placeholder:'e.g. merged in PR #42',confirmLabel:'Close loop'});
 if(res===null)return;
 const out=await post('/api/close_loop',{id,resolution:res});
 if(out.ok)toast('Loop closed'); else toast(out.error||'Could not close that loop','err');
 loadFoot(); if(V[tab])V[tab]();
}

// ---- footer (store stats + meta + nav badges) ----
async function loadFoot(){
 const s=await get('/api/stats');
 document.getElementById('foot').innerHTML=`<span class="pill"><b>${s.total}</b> records</span><span class="pill"><b>${s.live}</b> live</span>`;
 get('/api/meta').then(m=>{
  const base=(m.db||'').split('/').pop()||m.db;
  const mf=document.getElementById('metafoot');
  mf.innerHTML=`${esc(base)} &middot; v${esc(m.version)}<br>${esc(m.embedder)} embedder${m.audit_chain?' &middot; audit chain':''}`;
  mf.title=m.db;
 }).catch(()=>{});
 Promise.all([get('/api/conflicts'),get('/api/loops')]).then(([cf,lp])=>{
  const c1=document.getElementById('cnt-conflicts'),c2=document.getElementById('cnt-loops');
  if(cf.length){c1.textContent=cf.length;c1.hidden=false;c1.classList.add('warn');}else c1.hidden=true;
  if(lp.length){c2.textContent=lp.length;c2.hidden=false;}else c2.hidden=true;
 }).catch(()=>{});
}

// ---- belief history (record detail expand) ----
async function hist(id){
 const el=document.getElementById('d-'+id); if(!el)return;
 if(el.innerHTML){el.innerHTML='';return;}
 const d=await get('/api/record/'+id); const last=d.history.length-1;
 el.innerHTML=`<div class="tl">`+d.history.map((h,i)=>`<div class="ev${i===last?' cur':''}">
  <div class="t">${fmtDate(h.created_at)}${i===last?' &middot; current':''} ${provTag(h.provenance)}</div>
  <div>${esc(h.content)}</div></div>`).join('')+`</div>`;
}

// ---- views ----
const V={
 async overview(){
  const [o,recent,ts]=await Promise.all([get('/api/overview'),get('/api/records?limit=8'),get('/api/timeseries?days=30')]);
  const stat=(l,v,s='')=>`<div class="stat"><div class="sv">${v}</div><div class="sl">${l}</div>${s?`<div class="ss">${s}</div>`:''}</div>`;
  const row=r=>`<div class="rrow">${kindTag(r.kind)}<span class="rc">${esc(r.content)}</span>${r.superseded_by?'<span class="tag status-critical">superseded</span>':''}<span class="when" title="${fmtDate(r.created_at)}">${ago(r.created_at)}</span></div>`;
  main.innerHTML=head('Overview','The health of your memory at a glance &mdash; counts, attributability, activity, and recent captures.')
   +`<div class="stats">${stat('Total memories',o.total)}${stat('Live',o.live,'current beliefs')}${stat('Superseded',o.superseded,'revised')}${stat('Attributable',Math.round(o.attributable*100)+'%','has source + actor')}${stat('High importance',o.high_importance,'imp &ge; 4')}${stat('Added &middot; 7d',o.added_7d,o.added_24h+' in 24h')}</div>
   <div class="panel" style="margin-bottom:14px"><h3 class="grp">Activity <span class="n">last 30 days</span></h3>${sparkline(ts.buckets,'ov-spark')}</div>
   <div class="ogrid"><div class="panel"><h3 class="grp">Recent activity</h3>${recent.length?recent.map(row).join(''):'<div class="muted">No memories yet.</div>'}</div>
   <div class="ocol">
    <div class="panel"><h3 class="grp">By kind</h3>${barList(Object.entries(o.by_kind),kindColor)}</div>
    <div class="panel"><h3 class="grp">By provenance</h3>${barList(Object.entries(o.by_provenance),provColor)}</div>
    <div class="panel"><h3 class="grp">Recency</h3>${recencyBar(o.by_tier)}</div>
    <div class="panel"><h3 class="grp">Projects</h3>${barList(Object.entries(o.projects),null)}</div>
   </div></div>`;
  wireSparkline('ov-spark',ts.buckets);
 },
 async browse(){
  main.innerHTML=head('Browse','Every memory, verbatim and source-traceable &mdash; search or scan.')
   +`<div class="controls"><input id="q" class="search" placeholder="Search your memory&hellip; (press / to focus)">
     <select id="kind"><option value="">all kinds</option>${['note','chat','mission','fact','preference','constraint','commitment'].map(k=>`<option>${k}</option>`).join('')}</select>
     <select id="prov"><option value="">all provenance</option>${['observation','planning','action','user_confirmation'].map(p=>`<option>${p}</option>`).join('')}</select>
     <select id="sort"><option value="new">newest</option><option value="old">oldest</option><option value="imp">importance</option></select>
     <button class="btn" id="go">Search</button></div><div id="list"></div>`;
  const run=async()=>{
   let recs=await get('/api/records?q='+encodeURIComponent(q.value)+'&kind='+encodeURIComponent(kind.value)+'&limit=200');
   if(prov.value)recs=recs.filter(r=>r.provenance===prov.value);
   if(sort.value==='old')recs=recs.slice().reverse();
   else if(sort.value==='imp')recs=recs.slice().sort((a,b)=>b.importance-a.importance);
   document.getElementById('list').innerHTML=recs.length?recs.map(r=>card(r,
    `<button class="btn ghost sm" onclick="hist('${r.id}')">history</button>
     <button class="btn ghost sm" onclick="forgetPrompt('${r.id}')">forget</button>`)).join(''):emptyState('No memories match.');
  };
  document.getElementById('go').onclick=run;
  document.getElementById('q').onkeydown=e=>{if(e.key==='Enter')run()};
  kind.onchange=run; prov.onchange=run; sort.onchange=run;
  run();
  setTimeout(()=>document.getElementById('q').focus(),10);
 },
 async changed(){
  main.innerHTML=head('What changed','Beliefs added or revised since a point in time.')
   +`<div class="controls">last <input id="h" type="number" value="24" min="1" style="width:84px"> hours
     <button class="btn" id="go">Show</button></div><div id="out"></div>`;
  const run=async()=>{
   const d=await get('/api/diff?hours='+h.value);
   document.getElementById('out').innerHTML=`<h3 class="grp">Added <span class="n">${d.added.length}</span></h3>`
    +(d.added.length?d.added.map(r=>card(r)).join(''):emptyState('Nothing added in this window.'))
    +`<h3 class="grp">Revised <span class="n">${d.revised.length}</span></h3>`
    +(d.revised.length?d.revised.map(p=>`<div class="card"><div class="meta-row">
       <span class="tag status-warning">revised</span><span class="when" title="${fmtDate(p.new.created_at)}">${ago(p.new.created_at)}</span></div>
       <div class="content"><span class="mut2" style="text-decoration:line-through">${esc(p.old.content)}</span>
       <span class="gold-text"> &rarr; ${esc(p.new.content)}</span></div></div>`).join(''):emptyState('Nothing revised in this window.'));
  };
  document.getElementById('go').onclick=run;
  document.getElementById('h').onkeydown=e=>{if(e.key==='Enter')run()};
  run();
 },
 async project(){
  const ps=await get('/api/projects');
  main.innerHTML=head('Projects','Every project Midas has memory for &mdash; derived from its tag, scope, and capture origin.')
   +(ps.length?`<div class="plist">`+ps.map(p=>`<div class="pcard" tabindex="0" data-p="${esc(p.name)}">
     <div class="pn">${esc(p.name)}</div>
     <div class="pstat"><b>${p.live}</b> live &middot; ${p.total} total</div>
     <div class="pstat">${Math.round(p.attributable*100)}% attributable &middot; ${ago(p.last_active)}</div></div>`).join('')+`</div>`
    :emptyState('No projects yet &mdash; captures get grouped by project / namespace / cwd.'));
  document.querySelectorAll('.pcard').forEach(c=>{
   const open=()=>{location.hash='project='+encodeURIComponent(c.dataset.p);};
   c.onclick=open; c.onkeydown=e=>{if(e.key==='Enter')open();};
  });
 },
 async gov(){
  main.innerHTML=head('Governance &amp; audit','Would memory authorize this action &mdash; and why? The audit trail.')
   +`<div class="controls"><input id="q" class="search" placeholder="proposed action / query">
     <select id="use"><option>external_action</option><option>destructive_action</option><option>answer</option><option>planning</option></select>
     <button class="btn" id="go">Check</button></div><div id="out"></div>`;
  const run=async()=>{
   const a=await get('/api/audit?query='+encodeURIComponent(q.value)+'&use='+use.value);
   const pct=Math.round((a.audit_completeness||0)*100);
   document.getElementById('out').innerHTML=`<div class="verdict ${a.allowed?'ok':'no'}"><div class="big" style="color:${a.allowed?'var(--good)':'var(--critical)'}">${a.allowed?ICON_CHECK:ICON_CROSS}</div>
    <div><h2 style="color:${a.allowed?'var(--good)':'var(--critical)'}">${a.allowed?'ALLOWED':'BLOCKED'}</h2>
    <div class="r">${esc(a.reason)}</div>
    <div class="r">attributable evidence: <b>${pct}%</b><div class="meter"><i style="width:${pct}%"></i></div></div></div></div>
    <h3 class="grp">Evidence <span class="n">${a.evidence.length}</span></h3>`
    +(a.evidence.length?a.evidence.map(e=>card(e,`<button class="btn ghost sm" onclick="hist('${e.id}')">history</button>`)).join(''):emptyState('No supporting memory.'));
  };
  document.getElementById('go').onclick=run;
  document.getElementById('q').onkeydown=e=>{if(e.key==='Enter')run()};
 },
 async conflicts(){
  main.innerHTML=head('Conflicts','Live beliefs that contradict each other &mdash; neither superseded the other. Resolve by forgetting the wrong one, or capture the corrected value.');
  const cs=await get('/api/conflicts');
  main.innerHTML+=cs.length?cs.map(c=>`<div class="card"><div class="conflict-head">
    <span class="tag status-warning">${esc(c.signal)}</span><span class="tag" title="cosine similarity">${Math.round(c.similarity*100)}% similar</span></div>
    <div class="cpair">${[c.a,c.b].map(r=>card(r,`<button class="btn ghost sm" onclick="forgetPrompt('${r.id}')">forget this one</button>`)).join('')}</div></div>`).join('')
   :emptyState('No unresolved conflicts &mdash; every live belief agrees. &#10003;');
 },
 async loops(){
  main.innerHTML=head('Open loops','Promised work that was never closed &mdash; oldest (most overdue) first.');
  const ls=await get('/api/loops');
  main.innerHTML+=ls.length?ls.map(r=>card(r,`<button class="btn ghost sm" onclick="closeLoopPrompt('${r.id}')">close loop</button>`)).join('')
   :emptyState('No open loops &mdash; every commitment is closed. &#10003;');
 },
 async auditlog(){
  main.innerHTML=head('Audit log','The tamper-evident mutation chain &mdash; every write, revision, and deletion, hash-linked. No memory content, only hashes.');
  const a=await get('/api/audit_chain');
  if(!a.supported){main.innerHTML+=emptyState('This store has no audit chain (in-memory store, or audit=off).');return;}
  main.innerHTML+=`<div class="verdict ${a.ok?'ok':'no'}"><div class="big" style="color:${a.ok?'var(--good)':'var(--critical)'}">${a.ok?ICON_CHECK:ICON_CROSS}</div>
   <div><h2 style="color:${a.ok?'var(--good)':'var(--critical)'}">${a.ok?'CHAIN INTACT':'CHAIN BROKEN'}</h2>
   <div class="r"><b>${a.entries}</b> entries${a.ok?'':' &middot; first invalid seq: '+a.first_invalid_seq+' &mdash; history after this point is untrusted'}</div></div></div>
   <h3 class="grp">Most recent <span class="n">${a.tail.length}</span></h3>
   <div class="panel">`+a.tail.slice().reverse().map(e=>{
    const opTag=e.op==='put'?`<span class="tag">put</span>`:`<span class="tag status-critical">${esc(e.op)}</span>`;
    return `<div class="rrow"><span class="mut2" style="width:40px;font-variant-numeric:tabular-nums">#${e.seq}</span>${opTag}
     <span class="rc" style="font-family:ui-monospace,monospace;font-size:12px">${esc(e.record_id.slice(0,8))} &middot; sha ${esc(e.content_sha.slice(0,14))}&hellip;</span>
     <span class="when" title="${fmtDate(e.at)}">${ago(e.at)}</span></div>`;
   }).join('')+`</div>`;
 },
};

async function openProj(name){
 const d=await get('/api/project?name='+encodeURIComponent(name));
 const o=d.overview,g=d.governance;
 const stat=(l,v,s='')=>`<div class="stat"><div class="sv">${v}</div><div class="sl">${l}</div>${s?`<div class="ss">${s}</div>`:''}</div>`;
 const actorLabel=k=>k==='&mdash;'||k==='—'?'(unattributed)':k;
 main.innerHTML=`<div class="vhead"><div><div class="back" id="backp">${ICON_BACK}Projects</div><h1>${esc(name)}</h1><p>Per-project state, governance, and attributability.</p></div></div>
  <div class="stats">${stat('Total',o.total)}${stat('Live',o.live,'current')}${stat('Superseded',o.superseded,'revised')}${stat('Attributable',Math.round(o.attributable*100)+'%','source + actor')}${stat('Confirmations',g.confirmations,'user-confirmed')}${stat('Forbidden',g.forbidden.length,'active rules')}</div>
  <h3 class="grp">Governance</h3>
  <div class="panels"><div class="panel"><h3 class="grp">Trust &middot; by provenance</h3>${barList(Object.entries(g.by_provenance),provColor)}</div>
   <div class="panel"><h3 class="grp">Contributors &middot; by actor</h3>${barList(Object.entries(g.by_actor),null,{labelFmt:actorLabel})}</div></div>`
  +Object.entries(d.state).map(([k,rs])=>`<h3 class="grp">${esc(k.replace(/_/g,' '))} <span class="n">${rs.length}</span></h3>`+rs.map(r=>card(r)).join('')).join('')
  +`<h3 class="grp">Recent <span class="n">${d.recent.length}</span></h3>`+(d.recent.length?d.recent.map(r=>card(r)).join(''):emptyState('&mdash;'));
 document.getElementById('backp').onclick=()=>{location.hash='project';};
}

// ---- routing (hash-driven; hashchange IS wired, unlike the old build) ----
function setActive(t){document.querySelectorAll('.side nav a,#mobnav a').forEach(a=>a.classList.toggle('on',a.dataset.t===t));}
function route(){
 const h=location.hash.slice(1);
 if(h.startsWith('project=')){tab='project';setActive('project');openProj(decodeURIComponent(h.slice(8)));return;}
 const s=V[h]?h:'overview';
 tab=s;setActive(s);V[s]();
}
function navigateTo(t){if(location.hash.slice(1)===t){route();}else{location.hash=t;}}
window.addEventListener('hashchange',route);
document.querySelectorAll('.side nav a[data-t]').forEach(a=>a.onclick=()=>navigateTo(a.dataset.t));

function buildMobileNav(){
 const items=[...document.querySelectorAll('.side nav a[data-t]')];
 document.getElementById('mobnav').innerHTML=items.map(a=>
  `<a data-t="${a.dataset.t}">${a.querySelector('svg').outerHTML}${a.querySelector('.lbl').outerHTML}</a>`).join('');
 document.querySelectorAll('#mobnav a').forEach(a=>a.onclick=()=>navigateTo(a.dataset.t));
}

// ---- command palette (Cmd/Ctrl+K) ----
let palSel=0,palMatches=[];
function palItems(){return [...document.querySelectorAll('.side nav a[data-t]')].map(a=>({
 type:'nav',t:a.dataset.t,label:a.querySelector('.lbl').textContent,icon:a.querySelector('svg').outerHTML}));}
function openPalette(){
 const wrap=document.getElementById('palwrap'),input=document.getElementById('palinput');
 wrap.classList.add('on');input.value='';input.focus();renderPalette('');
}
function closePalette(){document.getElementById('palwrap').classList.remove('on');}
function renderPalette(q){
 const items=palItems(),ql=q.trim().toLowerCase();
 palMatches=ql?items.filter(i=>i.label.toLowerCase().includes(ql)):items;
 if(ql)palMatches=palMatches.concat([{type:'search',label:`Search memory for "${q}"`,q}]);
 palSel=0;
 document.getElementById('pallist').innerHTML=palMatches.length?palMatches.map((m,i)=>
  `<li class="${i===palSel?'sel':''}" data-i="${i}">${m.icon||ICON_SEARCH_SM}<span>${esc(m.label)}</span>${m.type==='nav'?'<span class="k">&crarr;</span>':''}</li>`
 ).join(''):`<li class="muted" style="cursor:default">No matches</li>`;
 document.querySelectorAll('#pallist li[data-i]').forEach(li=>li.onclick=()=>choosePalette(+li.dataset.i));
}
function highlightPal(){document.querySelectorAll('#pallist li[data-i]').forEach(li=>li.classList.toggle('sel',+li.dataset.i===palSel));}
function choosePalette(i){
 const m=palMatches[i]; if(!m)return; closePalette();
 if(m.type==='nav')navigateTo(m.t);
 else{navigateTo('browse');setTimeout(()=>{const qi=document.getElementById('q');if(qi){qi.value=m.q;document.getElementById('go').click();}},70);}
}
document.getElementById('palhint').onclick=openPalette;
document.getElementById('palinput').oninput=e=>renderPalette(e.target.value);
document.getElementById('palwrap').onclick=e=>{if(e.target.id==='palwrap')closePalette();};
document.addEventListener('keydown',e=>{
 const mod=e.metaKey||e.ctrlKey;
 if(mod&&e.key.toLowerCase()==='k'){e.preventDefault();openPalette();return;}
 const wrap=document.getElementById('palwrap');
 if(wrap.classList.contains('on')){
  if(e.key==='Escape')closePalette();
  else if(e.key==='ArrowDown'){e.preventDefault();palSel=Math.min(palMatches.length-1,palSel+1);highlightPal();}
  else if(e.key==='ArrowUp'){e.preventDefault();palSel=Math.max(0,palSel-1);highlightPal();}
  else if(e.key==='Enter'){e.preventDefault();choosePalette(palSel);}
  return;
 }
 if(e.key==='/'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName)){
  const qi=document.getElementById('q'); if(qi){e.preventDefault();qi.focus();}
 }
});

// ---- theme toggle ----
function applyThemeIcon(){
 const t=document.documentElement.dataset.theme;
 document.getElementById('themebtn').innerHTML=t==='light'
  ?'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
  :'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/></svg>';
}
document.getElementById('themebtn').onclick=()=>{
 const next=document.documentElement.dataset.theme==='light'?'dark':'light';
 document.documentElement.dataset.theme=next;
 try{localStorage.setItem('midas-theme',next);}catch(e){}
 applyThemeIcon();
};
applyThemeIcon();

// ---- boot ----
buildMobileNav();
loadFoot();
route();
</script></body></html>
"""


# --- HTTP layer (thin router) ------------------------------------------------------------------

def _make_handler(mem: "Memory", *, db: str = "", embedder_name: str = ""):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", code)

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                return self._send(INDEX_HTML.encode(), "text/html; charset=utf-8")
            if u.path == "/api/stats":
                return self._json(api_stats(mem))
            if u.path == "/api/overview":
                return self._json(api_overview(mem))
            if u.path == "/api/records":
                return self._json(api_records(mem, q=qs.get("q", ""), kind=qs.get("kind", ""),
                                              limit=int(qs.get("limit", 200))))
            if u.path.startswith("/api/record/"):
                rec = api_record(mem, u.path.rsplit("/", 1)[1])
                return self._json(rec) if rec else self._json({"error": "not found"}, 404)
            if u.path == "/api/project_state":
                return self._json(api_project_state(mem, qs.get("project", "")))
            if u.path == "/api/projects":
                return self._json(api_projects(mem))
            if u.path == "/api/project":
                return self._json(api_project(mem, qs.get("name", "")))
            if u.path == "/api/diff":
                return self._json(api_diff(mem, hours=float(qs.get("hours", 24))))
            if u.path == "/api/audit":
                return self._json(api_audit(mem, qs.get("query", ""), qs.get("use", "external_action")))
            if u.path == "/api/conflicts":
                return self._json(api_conflicts(mem, limit=int(qs.get("limit", 20))))
            if u.path == "/api/loops":
                return self._json(api_loops(mem))
            if u.path == "/api/audit_chain":
                return self._json(api_audit_chain(mem, limit=int(qs.get("limit", 50))))
            if u.path == "/api/timeseries":
                return self._json(api_timeseries(mem, days=int(qs.get("days", 30))))
            if u.path == "/api/meta":
                return self._json(api_meta(mem, db=db, embedder_name=embedder_name))
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/api/forget":
                return self._json(api_forget(mem, body.get("id", "")))
            if u.path == "/api/close_loop":
                return self._json(api_close_loop(mem, body.get("id", ""), body.get("resolution", "")))
            self._json({"error": "not found"}, 404)

        def log_message(self, *args: Any) -> None:  # keep the console quiet
            pass

    return Handler


def serve(db: str, *, host: str = "127.0.0.1", port: int = 7777, embedder: str = "local",
          open_browser: bool = True) -> None:
    from .memory import Memory
    from .sqlite_store import SQLiteStore

    embedder_name = "hashing"
    if embedder == "local":
        try:
            from .embeddings import LocalEmbedder

            emb = LocalEmbedder()
            embedder_name = "local"
        except Exception:
            from .embeddings import HashingEmbedder

            emb = HashingEmbedder()
    else:
        from .embeddings import HashingEmbedder

        emb = HashingEmbedder()

    mem = Memory(store=SQLiteStore(db), embedder=emb)
    httpd = ThreadingHTTPServer((host, port), _make_handler(mem, db=db, embedder_name=embedder_name))
    url = f"http://{host}:{port}"
    print(f"Midas Inspector — {db}\n  → {url}  (local only, zero egress; Ctrl-C to stop)")
    if open_browser:
        import webbrowser

        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="midas inspect", description="Local glass-box UI over your memory.")
    ap.add_argument("--db", default=os.environ.get("MIDAS_MCP_DB") or os.path.expanduser("~/.midas/memory.sqlite3"),
                    help="path to the SQLite store (default: $MIDAS_MCP_DB or ~/.midas/memory.sqlite3)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--embedder", choices=("local", "hashing"), default="local")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    serve(args.db, host=args.host, port=args.port, embedder=args.embedder, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
