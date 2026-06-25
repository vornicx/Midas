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

from .audit import audit_record, audit_use, belief_history, forgetting_receipt
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


# --- The embedded UI ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Midas Inspector</title><style>
:root{--ink:#1A1A2E;--gold:#FFD700;--steel:#B0C4DE}
*{box-sizing:border-box}body{margin:0;background:var(--ink);color:#fff;font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
a{color:var(--gold)}header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid #ffffff1a}
.logo{display:grid;place-items:center;width:26px;height:26px;border-radius:6px;background:var(--gold);color:var(--ink);font-weight:800}
.tabs{display:flex;gap:6px;margin-left:auto}.tab{padding:6px 12px;border-radius:8px;cursor:pointer;color:var(--steel)}
.tab.on{background:#ffffff14;color:#fff}main{padding:20px;max-width:1100px;margin:0 auto}
input,select{background:#00000040;border:1px solid #ffffff26;color:#fff;border-radius:8px;padding:8px 10px;font:inherit}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
.card{border:1px solid #ffffff1a;background:#ffffff08;border-radius:12px;padding:12px 14px;margin-bottom:10px}
.badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px;background:#ffffff14;color:var(--steel);margin-right:6px}
.gold{color:var(--gold)}.muted{color:var(--steel)}.mono{font-family:ui-monospace,monospace;font-size:12px}
.btn{background:var(--gold);color:var(--ink);border:none;border-radius:8px;padding:6px 12px;font-weight:600;cursor:pointer}
.btn.ghost{background:transparent;border:1px solid #ffffff33;color:#fff}
.detail{border-left:2px solid var(--gold);padding-left:12px;margin-top:8px}
h3{margin:18px 0 8px}.ok{color:#4ade80}.no{color:#f87171}
</style></head><body>
<header><span class="logo">M</span><b>Midas Inspector</b><span class="muted" id="stat"></span>
<nav class="tabs">
<span class="tab on" data-t="browse">Browse</span><span class="tab" data-t="project">Project</span>
<span class="tab" data-t="changed">Changed</span><span class="tab" data-t="gov">Governance</span></nav></header>
<main id="app"></main>
<script>
const app=document.getElementById('app'), stat=document.getElementById('stat');
let tab='browse';
const esc=s=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmt=t=>t?new Date(t*1000).toISOString().slice(0,16).replace('T',' '):'';
async function get(u){return (await fetch(u)).json()}
async function post(u,b){return (await fetch(u,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)})).json()}
function recCard(r,extra=''){return `<div class="card"><div><span class="badge">${esc(r.kind)}</span>
<span class="badge">imp ${r.importance}</span><span class="badge">${esc(r.provenance)}</span>
${r.superseded_by?'<span class="badge no">superseded</span>':''}<span class="muted">${fmt(r.created_at)}</span></div>
<div style="margin-top:6px">${esc(r.content)}</div>
<div class="muted mono" style="margin-top:4px">${esc(r.actor||'')} ${esc(r.source||'')}</div>
<div style="margin-top:8px">${extra}</div></div>`}
async function loadStat(){const s=await get('/api/stats');stat.textContent=`· ${s.total} records (${s.live} live)`}
const tabs={
 async browse(){app.innerHTML=`<div class="row"><input id="q" placeholder="Search memory…" style="flex:1">
   <select id="kind"><option value="">all kinds</option><option>fact</option><option>constraint</option>
   <option>preference</option><option>note</option><option>chat</option><option>mission</option></select>
   <button class="btn" id="go">Search</button></div><div id="list"></div>`;
  const run=async()=>{const q=document.getElementById('q').value,k=document.getElementById('kind').value;
   const recs=await get('/api/records?q='+encodeURIComponent(q)+'&kind='+encodeURIComponent(k));
   document.getElementById('list').innerHTML=recs.map(r=>recCard(r,
     `<button class="btn ghost" onclick="openRec('${r.id}')">history</button>
      <button class="btn ghost" onclick="forget('${r.id}')">forget</button>
      <div id="d-${r.id}"></div>`)).join('')||'<div class="muted">no records</div>';};
  document.getElementById('go').onclick=run;document.getElementById('q').onkeydown=e=>{if(e.key==='Enter')run()};run();},
 async project(){app.innerHTML=`<div class="row"><input id="p" placeholder="project (e.g. apollo)">
   <button class="btn" id="go">Load state</button></div><div id="out"></div>`;
  document.getElementById('go').onclick=async()=>{const g=await get('/api/project_state?project='+encodeURIComponent(document.getElementById('p').value));
   document.getElementById('out').innerHTML=Object.keys(g).length?Object.entries(g).map(([k,rs])=>
     `<h3 class="gold">${esc(k)}</h3>`+rs.map(r=>recCard(r)).join('')).join(''):'<div class="muted">nothing for that project</div>';};},
 async changed(){app.innerHTML=`<div class="row"><input id="h" type="number" value="24" style="width:90px"> hours
   <button class="btn" id="go">Show diff</button></div><div id="out"></div>`;
  document.getElementById('go').onclick=async()=>{const d=await get('/api/diff?hours='+document.getElementById('h').value);
   document.getElementById('out').innerHTML=`<h3 class="gold">Added (${d.added.length})</h3>`+(d.added.map(r=>recCard(r)).join('')||'<div class="muted">—</div>')
     +`<h3 class="gold">Revised (${d.revised.length})</h3>`+(d.revised.map(p=>`<div class="card"><div class="muted no">${esc(p.old.content)}</div>
       <div class="gold">→ ${esc(p.new.content)}</div></div>`).join('')||'<div class="muted">—</div>');};},
 async gov(){app.innerHTML=`<div class="row"><input id="q" placeholder="proposed action / query" style="flex:1">
   <select id="use"><option>external_action</option><option>destructive_action</option><option>answer</option><option>planning</option></select>
   <button class="btn" id="go">Check</button></div><div id="out"></div>`;
  document.getElementById('go').onclick=async()=>{const a=await get('/api/audit?query='+encodeURIComponent(document.getElementById('q').value)+'&use='+document.getElementById('use').value);
   document.getElementById('out').innerHTML=`<div class="card"><b class="${a.allowed?'ok':'no'}">${a.allowed?'ALLOWED':'BLOCKED'}</b>
     <span class="badge">${esc(a.intended_use)}</span> <span class="badge">attributable ${(a.audit_completeness*100|0)}%</span>
     <div class="muted" style="margin-top:6px">${esc(a.reason)}</div></div>
     <h3 class="gold">Evidence (${a.evidence.length})</h3>`+(a.evidence.map(e=>recCard(e)).join('')||'<div class="muted">none</div>');};},
};
async function openRec(id){const el=document.getElementById('d-'+id);if(el.innerHTML){el.innerHTML='';return}
 const d=await get('/api/record/'+id);el.innerHTML='<div class="detail"><b class="gold">Belief history</b>'
  +d.history.map(h=>`<div class="muted">${fmt(h.created_at)} — ${esc(h.content)}</div>`).join('')+'</div>';}
async function forget(id){if(!confirm('Forget this memory? An erasure receipt is recorded.'))return;
 await post('/api/forget',{id});loadStat();tabs[tab]();}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{tab=t.dataset.t;
 document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));tabs[tab]();});
loadStat();tabs.browse();
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
