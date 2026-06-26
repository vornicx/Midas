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


def api_overview(mem: "Memory") -> dict[str, Any]:
    """The memory-health dashboard a team/enterprise needs at a glance: counts, attributability (the
    compliance metric: fraction with both a source and an actor), revision activity, recency, and the
    distribution by kind / provenance / project. All computed — no fabricated governance counters."""
    recs = list(mem.store.all())
    total = len(recs)
    now = time.time()
    by_kind: dict[str, int] = {}
    by_prov: dict[str, int] = {}
    projects: dict[str, int] = {}
    imp_sum = added_24h = added_7d = 0
    for r in recs:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
        by_prov[r.provenance] = by_prov.get(r.provenance, 0) + 1
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
        "projects": rank(projects),
    }


# --- The embedded UI ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Midas Inspector</title><style>
:root{--ink:#13131f;--gold:#FFD700;--gsoft:#ffd7001f;--steel:#9aa4c0;--text:#eef0f7;--line:#ffffff14;
--green:#34d399;--red:#f87171;--blue:#7aa2ff}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;color:var(--text);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
background:radial-gradient(1100px 560px at 8% -12%,#2a2a4d40,transparent),
radial-gradient(820px 480px at 112% 6%,#ffd70012,transparent),var(--ink)}
a{color:var(--gold);text-decoration:none}
.app{display:grid;grid-template-columns:230px 1fr;min-height:100vh}
.side{border-right:1px solid var(--line);padding:18px 14px;display:flex;flex-direction:column;gap:4px;
position:sticky;top:0;height:100vh}
.brand{display:flex;align-items:center;gap:10px;font-weight:800;font-size:16px;padding:4px 8px 16px}
.brand .m{display:grid;place-items:center;width:30px;height:30px;border-radius:9px;color:#1a1a2e;font-weight:900;
background:linear-gradient(145deg,#ffe96b,#e6b800);box-shadow:0 3px 14px #ffd70044}
.nav a{display:flex;align-items:center;gap:11px;padding:9px 11px;border-radius:10px;color:var(--steel);
cursor:pointer;font-weight:500}
.nav a:hover{background:#ffffff0a;color:var(--text)}.nav a.on{background:var(--gsoft);color:var(--gold)}
.nav svg{width:17px;height:17px}
.foot{margin-top:auto;padding-top:12px;border-top:1px solid var(--line)}
.pill{display:inline-flex;gap:5px;align-items:center;background:#ffffff0a;border:1px solid var(--line);
border-radius:999px;padding:3px 10px;font-size:11px;color:var(--steel);margin:4px 5px 0 0}
.pill b{color:var(--text)}
.main{padding:28px 32px;max-width:1000px}
.vhead h1{font-size:21px;margin:0}.vhead p{margin:3px 0 18px;color:var(--steel);font-size:13px}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:18px}
input,select{background:#0000003a;border:1px solid var(--line);color:var(--text);border-radius:10px;
padding:10px 12px;font:inherit;outline:none}input:focus,select:focus{border-color:#ffd70066}
.search{flex:1;min-width:220px}
.btn{background:linear-gradient(145deg,#ffe96b,#e6b800);color:#1a1a2e;border:none;border-radius:10px;
padding:9px 16px;font-weight:600;cursor:pointer}.btn:hover{filter:brightness(1.06)}
.btn.ghost{background:transparent;border:1px solid var(--line);color:var(--text)}
.btn.sm{padding:5px 11px;font-size:12px;border-radius:8px}
.card{border:1px solid var(--line);border-radius:13px;padding:14px 16px;margin-bottom:11px;
background:linear-gradient(180deg,#ffffff08,#ffffff03);transition:border-color .15s}
.card:hover{border-color:#ffffff24}.card.dim{opacity:.5}
.meta{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-bottom:9px}
.tag{font-size:11px;padding:2px 9px;border-radius:999px;background:#ffffff12;color:var(--steel);font-weight:500}
.tag.p-user_confirmation{background:#ffd7001f;color:var(--gold)}.tag.p-action{background:#7aa2ff26;color:var(--blue)}
.tag.sup{background:#f871711f;color:var(--red)}.tag.imp{background:#ffffff0a}
.when{margin-left:auto;font-size:11px;color:var(--steel)}
.content{font-size:14px;white-space:pre-wrap}
.src{margin-top:9px;font-family:ui-monospace,monospace;font-size:11px;color:var(--steel)}
.acts{margin-top:11px;display:flex;gap:8px}
.muted{color:var(--steel)}.gold{color:var(--gold)}.green{color:var(--green)}.red{color:var(--red)}
.empty{text-align:center;color:var(--steel);padding:46px;border:1px dashed var(--line);border-radius:14px}
.tl{margin-top:11px;border-left:2px solid #ffd70038;padding-left:16px;display:flex;flex-direction:column;gap:9px}
.tl .ev{position:relative}.tl .ev::before{content:"";position:absolute;left:-23px;top:6px;width:9px;height:9px;
border-radius:50%;background:#8a8aa6}.tl .ev.cur::before{background:var(--gold);box-shadow:0 0 0 4px #ffd70033}
.tl .t{font-size:11px;color:var(--steel)}
.verdict{border-radius:15px;padding:18px 20px;border:1px solid var(--line);margin-bottom:18px;
display:flex;align-items:center;gap:16px}
.verdict.ok{background:#34d3990f;border-color:#34d39944}.verdict.no{background:#f871710f;border-color:#f8717144}
.verdict .big{font-size:26px}.verdict h2{margin:0;font-size:18px}.verdict .r{color:var(--steel);font-size:13px;margin-top:3px}
.meter{height:6px;background:#ffffff14;border-radius:999px;overflow:hidden;margin-top:7px;width:150px}
.meter i{display:block;height:100%;background:var(--gold)}
h3.grp{margin:22px 0 11px;font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--gold)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:11px;margin-bottom:22px}
.stat{border:1px solid var(--line);border-radius:13px;padding:15px 16px;background:linear-gradient(180deg,#ffffff08,#ffffff03)}
.stat .sv{font-size:27px;font-weight:700;color:var(--gold);line-height:1;font-variant-numeric:tabular-nums}
.stat .sl{font-size:12px;color:var(--text);margin-top:7px}.stat .ss{font-size:10.5px;color:var(--steel);margin-top:3px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}
.panel{border:1px solid var(--line);border-radius:13px;padding:6px 16px 14px;background:#ffffff05}
.bar{display:flex;align-items:center;gap:10px;margin:8px 0}
.bar .bk{width:118px;font-size:12px;color:var(--steel);text-transform:capitalize;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar .bt{flex:1;height:7px;background:#ffffff12;border-radius:999px;overflow:hidden}
.bar .bt i{display:block;height:100%;background:linear-gradient(90deg,#e6b800,#ffe96b)}
.bar .bv{width:36px;text-align:right;font-size:12px;color:var(--text);font-variant-numeric:tabular-nums}
@media(max-width:720px){.app{grid-template-columns:1fr}.side{position:static;height:auto;flex-direction:row;
flex-wrap:wrap}.nav{display:flex;gap:2px}.foot{display:none}}
</style></head><body>
<div class="app">
<aside class="side">
<div class="brand"><span class="m">M</span> Midas <span class="muted" style="font-weight:500;font-size:12px">Inspector</span></div>
<nav class="nav" id="nav">
<a data-t="overview" class="on"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>Overview</a>
<a data-t="browse"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h11"/></svg>Browse</a>
<a data-t="project"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>Project</a>
<a data-t="changed"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 2"/></svg>Changed</a>
<a data-t="gov"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l8 3v6c0 5-3.5 7.6-8 9-4.5-1.4-8-4-8-9V6z"/></svg>Governance</a>
</nav>
<div class="foot" id="foot"></div>
</aside>
<main class="main" id="main"></main>
</div>
<script>
const main=document.getElementById('main'),foot=document.getElementById('foot');let tab='browse';
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const ago=t=>{if(!t)return'';const s=(Date.now()/1000)-t;const d=s/86400;
 return d>=1?Math.round(d)+'d ago':s>=3600?Math.round(s/3600)+'h ago':Math.max(1,Math.round(s/60))+'m ago'};
const get=async u=>(await fetch(u)).json();
const post=async(u,b)=>(await fetch(u,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)})).json();
function card(r,acts=''){const sup=r.superseded_by?'<span class="tag sup">superseded</span>':'';
 return `<div class="card${r.superseded_by?' dim':''}"><div class="meta">
 <span class="tag">${esc(r.kind)}</span><span class="tag imp">imp ${r.importance}</span>
 <span class="tag p-${esc(r.provenance)}">${esc(r.provenance)}</span>${sup}
 <span class="when">${ago(r.created_at)}</span></div>
 <div class="content">${esc(r.content)}</div>
 ${(r.actor||r.source)?`<div class="src">${esc(r.actor||'')}${r.source?' · '+esc(r.source):''}</div>`:''}
 ${acts?`<div class="acts">${acts}</div><div id="d-${r.id}"></div>`:''}</div>`}
function head(t,d){return `<div class="vhead"><h1>${t}</h1><p>${d}</p></div>`}
async function loadFoot(){const s=await get('/api/stats');
 foot.innerHTML=`<span class="pill"><b>${s.total}</b> records</span><span class="pill"><b>${s.live}</b> live</span>`
 +Object.entries(s.kinds).map(([k,n])=>`<span class="pill">${esc(k)} <b>${n}</b></span>`).join('');}
const V={
 async overview(){const o=await get('/api/overview');
  const card=(l,v,s='')=>`<div class="stat"><div class="sv">${v}</div><div class="sl">${l}</div>${s?`<div class="ss">${s}</div>`:''}</div>`;
  const bars=ob=>{const e=Object.entries(ob);if(!e.length)return '<div class="muted">—</div>';const mx=Math.max(...e.map(x=>x[1]));
   return e.map(([k,v])=>`<div class="bar"><div class="bk">${esc(k)}</div><div class="bt"><i style="width:${Math.round(v/mx*100)}%"></i></div><div class="bv">${v}</div></div>`).join('');};
  main.innerHTML=head('Overview','The health of your memory at a glance — counts, attributability, and activity.')
   +`<div class="stats">${card('Total memories',o.total)}${card('Live',o.live,'current beliefs')}${card('Superseded',o.superseded,'revised')}${card('Attributable',Math.round(o.attributable*100)+'%','has source + actor')}${card('High importance',o.high_importance,'imp ≥ 4')}${card('Added · 7d',o.added_7d,o.added_24h+' in 24h')}</div>
    <div class="panels"><div class="panel"><h3 class="grp">By kind</h3>${bars(o.by_kind)}</div>
    <div class="panel"><h3 class="grp">By provenance</h3>${bars(o.by_provenance)}</div>
    <div class="panel"><h3 class="grp">Projects</h3>${bars(o.projects)}</div></div>`;},
 async browse(){main.innerHTML=head('Browse','Every memory, verbatim and source-traceable — search or scan.')
  +`<div class="controls"><input id="q" class="search" placeholder="Search your memory…">
   <select id="kind"><option value="">all kinds</option><option>fact</option><option>constraint</option>
   <option>preference</option><option>note</option><option>chat</option><option>mission</option></select>
   <button class="btn" id="go">Search</button></div><div id="list"></div>`;
  const run=async()=>{const recs=await get('/api/records?q='+encodeURIComponent(q.value)+'&kind='+encodeURIComponent(kind.value));
   list.innerHTML=recs.length?recs.map(r=>card(r,
    `<button class="btn ghost sm" onclick="hist('${r.id}')">history</button>
     <button class="btn ghost sm" onclick="forget('${r.id}')">forget</button>`)).join(''):'<div class="empty">No memories.</div>';};
  go.onclick=run;q.onkeydown=e=>{if(e.key==='Enter')run()};run();},
 async project(){main.innerHTML=head('Project state','The live, governed state of a project — by category.')
  +`<div class="controls"><input id="p" class="search" placeholder="project (e.g. apollo)">
   <button class="btn" id="go">Load</button></div><div id="out"></div>`;
  go.onclick=async()=>{const g=await get('/api/project_state?project='+encodeURIComponent(p.value));
   out.innerHTML=Object.keys(g).length?Object.entries(g).map(([k,rs])=>
    `<h3 class="grp">${esc(k.replace(/_/g,' '))} · ${rs.length}</h3>`+rs.map(r=>card(r)).join('')).join('')
    :'<div class="empty">Nothing tagged for that project.</div>';};
  p.onkeydown=e=>{if(e.key==='Enter')go.click()};},
 async changed(){main.innerHTML=head('What changed','Beliefs added or revised since a point in time.')
  +`<div class="controls">last <input id="h" type="number" value="24" style="width:84px"> hours
   <button class="btn" id="go">Show</button></div><div id="out"></div>`;
  go.onclick=async()=>{const d=await get('/api/diff?hours='+h.value);
   out.innerHTML=`<h3 class="grp">Added · ${d.added.length}</h3>`+(d.added.map(r=>card(r)).join('')||'<div class="muted">—</div>')
   +`<h3 class="grp">Revised · ${d.revised.length}</h3>`+(d.revised.map(p=>`<div class="card"><div class="meta">
     <span class="tag sup">revised</span><span class="when">${ago(p.new.created_at)}</span></div>
     <div class="content"><span class="red" style="text-decoration:line-through">${esc(p.old.content)}</span>
     <span class="gold"> → ${esc(p.new.content)}</span></div></div>`).join('')||'<div class="muted">—</div>');};
  go.click();},
 async gov(){main.innerHTML=head('Governance &amp; audit','Would memory authorize this action — and why? The audit trail.')
  +`<div class="controls"><input id="q" class="search" placeholder="proposed action / query">
   <select id="use"><option>external_action</option><option>destructive_action</option><option>answer</option><option>planning</option></select>
   <button class="btn" id="go">Check</button></div><div id="out"></div>`;
  go.onclick=async()=>{const a=await get('/api/audit?query='+encodeURIComponent(q.value)+'&use='+use.value);
   const pct=Math.round((a.audit_completeness||0)*100);
   out.innerHTML=`<div class="verdict ${a.allowed?'ok':'no'}"><div class="big">${a.allowed?'✓':'✕'}</div>
    <div><h2 class="${a.allowed?'green':'red'}">${a.allowed?'ALLOWED':'BLOCKED'}</h2>
    <div class="r">${esc(a.reason)}</div><div class="r">attributable evidence
    <div class="meter"><i style="width:${pct}%"></i></div></div></div></div>
    <h3 class="grp">Evidence · ${a.evidence.length}</h3>`+(a.evidence.map(e=>card(e)).join('')||'<div class="empty">No supporting memory.</div>');};
  q.onkeydown=e=>{if(e.key==='Enter')go.click()};},
};
async function hist(id){const el=document.getElementById('d-'+id);if(el.innerHTML){el.innerHTML='';return}
 const d=await get('/api/record/'+id);const last=d.history.length-1;
 el.innerHTML=`<div class="tl">`+d.history.map((h,i)=>`<div class="ev${i===last?' cur':''}">
  <div class="t">${ago(h.created_at)}${i===last?' · current':''}</div><div>${esc(h.content)}</div></div>`).join('')+`</div>`;}
async function forget(id){if(!confirm('Forget this memory? A tamper-evident erasure receipt is recorded.'))return;
 await post('/api/forget',{id});loadFoot();V[tab]();}
document.querySelectorAll('#nav a').forEach(a=>a.onclick=()=>{tab=a.dataset.t;
 document.querySelectorAll('#nav a').forEach(x=>x.classList.toggle('on',x===a));V[tab]();});
loadFoot();V.overview();
</script></body></html>"""


# --- HTTP layer (thin router) ------------------------------------------------------------------

def _make_handler(mem: "Memory"):
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
            if u.path == "/api/diff":
                return self._json(api_diff(mem, hours=float(qs.get("hours", 24))))
            if u.path == "/api/audit":
                return self._json(api_audit(mem, qs.get("query", ""), qs.get("use", "external_action")))
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/api/forget":
                return self._json(api_forget(mem, body.get("id", "")))
            self._json({"error": "not found"}, 404)

        def log_message(self, *args: Any) -> None:  # keep the console quiet
            pass

    return Handler


def serve(db: str, *, host: str = "127.0.0.1", port: int = 7777, embedder: str = "local",
          open_browser: bool = True) -> None:
    from .memory import Memory
    from .sqlite_store import SQLiteStore

    if embedder == "local":
        try:
            from .embeddings import LocalEmbedder

            emb = LocalEmbedder()
        except Exception:
            from .embeddings import HashingEmbedder

            emb = HashingEmbedder()
    else:
        from .embeddings import HashingEmbedder

        emb = HashingEmbedder()

    mem = Memory(store=SQLiteStore(db), embedder=emb)
    httpd = ThreadingHTTPServer((host, port), _make_handler(mem))
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
    ap.add_argument("--db", default=os.environ.get("MIDAS_MCP_DB") or "memory.sqlite3",
                    help="path to the SQLite store (default: $MIDAS_MCP_DB or ./memory.sqlite3)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--embedder", choices=("local", "hashing"), default="local")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    serve(args.db, host=args.host, port=args.port, embedder=args.embedder, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
