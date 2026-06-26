"""The `midas` command — one entry point for setup, serving, updating, and inspecting.

    midas init       create the shared memory + wire up your MCP clients (Claude Code, Codex, Cursor…)
    midas serve      run the MCP server (stdio; or --http for an MCP URL all clients can share)
    midas status     show the store, version, and which clients are configured
    midas update     upgrade Midas to the latest version
    midas inspect    open the local Memory Inspector

**One memory, shared.** By default every client points at `~/.midas/memory.sqlite3`, so Claude Code,
Codex and Cursor all read and write the SAME memory — autonomously, with no per-client paths to keep in
sync. `midas init` does the wiring for you.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DB = str(Path.home() / ".midas" / "memory.sqlite3")

# What `midas init` writes into each client: just launch `midas-mcp`. No DB path needed — the server
# defaults to the shared store, so every client lands on the same memory automatically.
_STDIO_BLOCK = {"command": "midas-mcp", "env": {"MIDAS_MCP_EMBEDDER": "local"}}


def _store_path(db: str | None) -> str:
    return str(Path(db or os.getenv("MIDAS_MCP_DB") or DEFAULT_DB).expanduser())


def _ensure_store(db: str) -> None:
    p = Path(db)
    p.parent.mkdir(parents=True, exist_ok=True)
    from midas.sqlite_store import SQLiteStore

    SQLiteStore(str(p))  # creates a valid, empty SQLite store if missing


def _safe_read(p: Path) -> str:
    try:
        return p.read_text()
    except Exception:
        return ""


def _claude_desktop_path() -> Path | None:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    if sys.platform.startswith("win"):
        ad = os.getenv("APPDATA")
        return Path(ad) / "Claude" / "claude_desktop_config.json" if ad else None
    return Path.home() / ".config/Claude/claude_desktop_config.json"


# ---- init -------------------------------------------------------------------------------------

def _merge_mcp_json(path: Path, block: dict, *, dry: bool) -> str:
    """Add/refresh a `midas` entry in a client's mcpServers JSON, non-destructively (merge + backup)."""
    cfg: dict = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text() or "{}")
        except Exception:
            return f"⚠  {path} is not valid JSON — add `midas` by hand"
    servers = cfg.setdefault("mcpServers", {})
    verb = "update" if "midas" in servers else "add"
    if dry:
        return f"would {verb} → {path}"
    if path.exists():
        shutil.copy(path, str(path) + ".midas-bak")
    servers["midas"] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return f"{verb}ed → {path}"


def _cli_add(cmd: list[str], *, dry: bool, force: bool = False, remove: list[str] | None = None) -> str | None:
    """Configure a client via its own CLI (claude/codex). None if that CLI isn't installed. With force,
    remove any existing `midas` entry first so re-running init can't collide."""
    if not shutil.which(cmd[0]):
        return None
    if dry:
        return ("would re-add via " if force else "would run: ") + " ".join(cmd)
    if force and remove:
        subprocess.run(remove, capture_output=True, text=True)  # best-effort; ignore if not present
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"{'re-' if force else ''}configured via `{cmd[0]} mcp add`"
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or e.stdout or "").strip().splitlines()
        hint = "" if force else " (already added — re-run `midas init --force`)"
        return f"`{cmd[0]}` failed: {tail[-1][:78] if tail else 'error'}{hint}"


def _cli_remove(exe: str, remove: list[str]) -> str | None:
    """Remove the midas entry via a client's CLI (for uninstall). None if the CLI isn't installed."""
    if not shutil.which(exe):
        return None
    r = subprocess.run(remove, capture_output=True, text=True)
    return "removed" if r.returncode == 0 else "not configured"


def cmd_init(args: argparse.Namespace) -> int:
    db = _store_path(args.db)
    if not args.dry_run:
        _ensure_store(db)
    print(f"✓ shared memory: {db}")
    scoped = getattr(args, "project_scoped", False)
    print("  mode: per-project — memory auto-separates by project (git repo / cwd).\n" if scoped
          else "  all clients below point here → they share one memory, autonomously.\n")

    def env_for(client_id: str) -> dict:
        e = {"MIDAS_MCP_EMBEDDER": "local", "MIDAS_MCP_CLIENT": client_id}
        if scoped:
            e["MIDAS_MCP_NAMESPACE"] = "auto"   # each project gets its own partition
        return e

    def eflags(e: dict) -> list[str]:
        out: list[str] = []
        for k, v in e.items():
            out += ["-e", f"{k}={v}"]
        return out

    force = getattr(args, "force", False)
    results: list[tuple[str, str]] = []
    # Clients with their own CLI (cleanest, no file editing) — each stamped with its client id:
    r = _cli_add(["claude", "mcp", "add", "midas", "-s", "user", *eflags(env_for("claude-code")),
                  "--", "midas-mcp"], dry=args.dry_run, force=force,
                 remove=["claude", "mcp", "remove", "midas", "-s", "user"])
    if r:
        results.append(("Claude Code", r))
    codex_note = "  — set MIDAS_MCP_CLIENT=codex" + (" + MIDAS_MCP_NAMESPACE=auto" if scoped else "") + \
                 " in ~/.codex/config.toml"
    r = _cli_add(["codex", "mcp", "add", "midas", "--", "midas-mcp"], dry=args.dry_run, force=force,
                 remove=["codex", "mcp", "remove", "midas"])
    if r:
        results.append(("Codex", r + codex_note))

    # Clients configured by a JSON file (only the ones already present, unless --all):
    targets = [("Cursor", "cursor", Path.home() / ".cursor/mcp.json"),
               ("Windsurf", "windsurf", Path.home() / ".codeium/windsurf/mcp_config.json")]
    cd = _claude_desktop_path()
    if cd:
        targets.append(("Claude Desktop", "claude-desktop", cd))
    for name, client_id, path in targets:
        if path.exists() or args.all:
            block = {"command": "midas-mcp", "env": env_for(client_id)}
            results.append((name, _merge_mcp_json(path, block, dry=args.dry_run)))

    for name, msg in results:
        print(f"  • {name}: {msg}")
    if not results:
        print("  (no known MCP clients detected — see the manual line below)")
    print("\nAny other client → point it at command `midas-mcp` (no env required).")
    if not args.dry_run:
        print("Restart your clients to apply. Verify with `midas status`.")
    return 0


# ---- serve ------------------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    if args.db:  # set BEFORE importing mcp_server (it builds its store at import)
        os.environ["MIDAS_MCP_DB"] = _store_path(args.db)
    from midas import mcp_server

    mcp_server.run_server("http" if args.http else "stdio", host=args.host, port=args.port)
    return 0


# ---- status -----------------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    from midas import __version__

    db = _store_path(args.db)
    print(f"Midas {__version__}")
    print(f"memory: {db}")
    if Path(db).exists():
        from midas.sqlite_store import SQLiteStore

        recs = list(SQLiteStore(db).all())
        live = sum(1 for r in recs if r.superseded_by is None)
        print(f"  {len(recs)} records ({live} live) · {Path(db).stat().st_size // 1024} KB")
    else:
        print("  (not created yet — run `midas init`)")

    print("clients:")
    for name, p in _client_paths():
        wired = p.exists() and "midas" in _safe_read(p)
        flag = "✓" if wired else "·"
        note = "" if p.exists() else "  (not found)"
        print(f"  {flag} {name}{note}")
    return 0


# ---- update -----------------------------------------------------------------------------------

def cmd_update(args: argparse.Namespace) -> int:
    if args.pip or not shutil.which("uv"):
        cmd = [sys.executable, "-m", "pip", "install", "-U", "midas-memory[mcp,local]"]
    else:
        cmd = ["uv", "tool", "upgrade", "midas-memory"]
    print("Updating Midas:  " + " ".join(cmd))
    if args.dry_run:
        return 0
    rc = subprocess.run(cmd).returncode
    if rc != 0 and not args.pip and shutil.which("uv"):
        print("\n(if you installed Midas with pip, run `midas update --pip`)")
    return rc


# ---- inspect ----------------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> int:
    from midas import inspector

    inspector.serve(_store_path(args.db), host=args.host, port=args.port,
                    embedder=args.embedder, open_browser=not args.no_browser)
    return 0


# ---- shared client detection ------------------------------------------------------------------

def _client_paths() -> list[tuple[str, Path]]:
    paths = [("Claude Code", Path.home() / ".claude.json"),
             ("Cursor", Path.home() / ".cursor/mcp.json"),
             ("Codex", Path.home() / ".codex/config.toml"),
             ("Windsurf", Path.home() / ".codeium/windsurf/mcp_config.json")]
    cd = _claude_desktop_path()
    if cd:
        paths.append(("Claude Desktop", cd))
    return paths


def _wired_clients() -> list[str]:
    return [name for name, p in _client_paths() if p.exists() and "midas" in _safe_read(p)]


# ---- doctor -----------------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    from midas import __version__

    ok = True

    def check(good: bool, label: str, hint: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"  {'✓' if good else '⚠'}  {label}" + (f"  — {hint}" if hint and not good else ""))

    print(f"Midas {__version__}  ·  Python {sys.version.split()[0]}\n")
    check(shutil.which("midas-mcp") is not None, "midas-mcp on PATH",
          "reinstall, or use the absolute path from `which midas-mcp` in your client config")

    db = _store_path(args.db)
    if Path(db).exists():
        from midas.sqlite_store import SQLiteStore
        try:
            store = SQLiteStore(db)
            check(True, f"store: {db}  ({len(store.all())} records, schema v{store.schema_version()})")
        except Exception as exc:
            check(False, f"store: {db}", f"cannot open: {exc}")
    else:
        check(False, f"store: {db}", "not created — run `midas init`")

    try:
        import fastembed  # noqa: F401
        check(True, "local embedder available (fastembed)")
    except Exception:
        check(False, "local embedder (fastembed)",
              'install `uv tool install "midas-memory[mcp,local]"` — Midas falls back to hashing meanwhile')

    wired = _wired_clients()
    check(bool(wired), f"clients wired: {', '.join(wired) if wired else 'none'}", "run `midas init`")
    print("\n" + ("Everything looks good." if ok else "Some checks need attention (see ⚠ above)."))
    return 0 if ok else 1


# ---- export / import --------------------------------------------------------------------------

def _rec_to_dict(r) -> dict:
    emb = r.embedding
    if emb is not None and not isinstance(emb, list):
        emb = [float(x) for x in emb]  # numpy -> JSON-able list
    return {"id": r.id, "content": r.content, "kind": r.kind, "importance": r.importance,
            "source": r.source, "provenance": r.provenance, "actor": r.actor,
            "metadata": r.metadata or {}, "created_at": r.created_at, "updated_at": r.updated_at,
            "superseded_by": r.superseded_by, "embedding": emb}


def cmd_export(args: argparse.Namespace) -> int:
    from midas.sqlite_store import SQLiteStore

    db = _store_path(args.db)
    if not Path(db).exists():
        print(f"no store at {db}", file=sys.stderr)
        return 1
    recs = list(SQLiteStore(db).all())
    text = json.dumps({"midas_export": 1, "count": len(recs),
                       "records": [_rec_to_dict(r) for r in recs]}, indent=2)
    if args.out and args.out != "-":
        Path(args.out).expanduser().write_text(text + "\n")
        print(f"exported {len(recs)} records → {args.out}")
    else:
        sys.stdout.write(text + "\n")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from midas.sqlite_store import SQLiteStore
    from midas.types import MemoryRecord

    data = json.loads(Path(args.file).expanduser().read_text())
    records = data["records"] if isinstance(data, dict) else data
    db = _store_path(args.db)
    _ensure_store(db)
    store = SQLiteStore(db)
    existing = {r.id for r in store.all()}
    added = skipped = 0
    for d in records:
        if not args.overwrite and d.get("id") in existing:
            skipped += 1
            continue
        store.put(MemoryRecord(
            id=d["id"], content=d["content"], kind=d["kind"], importance=d["importance"],
            source=d.get("source"), provenance=d.get("provenance", "observation"), actor=d.get("actor"),
            metadata=d.get("metadata") or {}, created_at=d["created_at"], updated_at=d["updated_at"],
            superseded_by=d.get("superseded_by"), embedding=d.get("embedding")))
        added += 1
    note = f"  ({skipped} already present, skipped — use --overwrite to replace)" if skipped else ""
    print(f"imported {added} records into {db}{note}")
    return 0


# ---- uninstall --------------------------------------------------------------------------------

def cmd_uninstall(args: argparse.Namespace) -> int:
    results: list[tuple[str, str]] = []
    r = _cli_remove("claude", ["claude", "mcp", "remove", "midas", "-s", "user"])
    if r:
        results.append(("Claude Code", r))
    r = _cli_remove("codex", ["codex", "mcp", "remove", "midas"])
    if r:
        results.append(("Codex", r))
    json_targets = [("Cursor", Path.home() / ".cursor/mcp.json"),
                    ("Windsurf", Path.home() / ".codeium/windsurf/mcp_config.json")]
    cd = _claude_desktop_path()
    if cd:
        json_targets.append(("Claude Desktop", cd))
    for name, path in json_targets:
        if not path.exists():
            continue
        try:
            cfg = json.loads(path.read_text() or "{}")
        except Exception:
            results.append((name, "left as-is (unparseable)"))
            continue
        if "midas" in cfg.get("mcpServers", {}):
            shutil.copy(path, str(path) + ".midas-bak")
            del cfg["mcpServers"]["midas"]
            path.write_text(json.dumps(cfg, indent=2) + "\n")
            results.append((name, "removed"))
    for name, msg in results:
        print(f"  • {name}: {msg}")
    if not results:
        print("  nothing to remove from known clients.")

    db = _store_path(args.db)
    if args.purge:
        for suffix in ("", "-wal", "-shm"):
            f = Path(db + suffix)
            if f.exists():
                f.unlink()
        print(f"\npurged the memory store: {db}")
    else:
        print(f"\nyour memory is kept at {db}  (use `--purge` to delete it too).")
    return 0


# ---- projects ---------------------------------------------------------------------------------

def cmd_projects(args: argparse.Namespace) -> int:
    from datetime import datetime
    from types import SimpleNamespace

    from midas.projects import list_projects
    from midas.sqlite_store import SQLiteStore

    db = _store_path(args.db)
    if not Path(db).exists():
        print(f"no store at {db} — run `midas init`", file=sys.stderr)
        return 1
    projs = list_projects(SimpleNamespace(store=SQLiteStore(db)))
    if not projs:
        print("no projects yet — memory gets grouped by project / namespace / cwd.")
        return 0
    print(f"{'PROJECT':<26}{'LIVE':>6}{'TOTAL':>7}{'ATTRIB':>8}  LAST ACTIVE")
    for p in projs:
        la = datetime.fromtimestamp(p["last_active"]).strftime("%Y-%m-%d") if p["last_active"] else "—"
        print(f"{p['name'][:26]:<26}{p['live']:>6}{p['total']:>7}{round(p['attributable'] * 100):>7}%  {la}")
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    from types import SimpleNamespace

    from midas.projects import project_governance, project_overview
    from midas.sqlite_store import SQLiteStore

    db = _store_path(args.db)
    if not Path(db).exists():
        print(f"no store at {db} — run `midas init`", file=sys.stderr)
        return 1
    mem = SimpleNamespace(store=SQLiteStore(db))
    ov = project_overview(mem, args.name)
    if ov["total"] == 0:
        print(f"no memory for project '{args.name}'.")
        return 1
    gov = project_governance(mem, args.name)
    join = lambda d: ", ".join(f"{k} {v}" for k, v in d.items()) or "—"  # noqa: E731
    print(args.name)
    print(f"  {ov['live']} live / {ov['total']} total · {round(ov['attributable'] * 100)}% attributable "
          f"· {ov['superseded']} revised")
    print(f"  governance: {gov['confirmations']} user-confirmed · {len(gov['forbidden'])} forbidden rules")
    print(f"  by kind:       {join(ov['by_kind'])}")
    print(f"  by provenance: {join(gov['by_provenance'])}")
    print(f"  by actor:      {join(gov['by_actor'])}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="midas", description="Local-first, governed memory for AI agents.")
    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("init", help="create the shared memory + wire up your MCP clients")
    pi.add_argument("--db", help=f"store path (default: {DEFAULT_DB})")
    pi.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    pi.add_argument("--all", action="store_true", help="also configure clients that aren't installed yet")
    pi.add_argument("--force", action="store_true", help="re-add even if a midas entry already exists")
    pi.add_argument("--project-scoped", action="store_true",
                    help="memory auto-separates per project (git repo / cwd) instead of one shared pool")
    pi.set_defaults(func=cmd_init)

    ps = sub.add_parser("serve", help="run the MCP server (stdio, or --http for an MCP URL)")
    ps.add_argument("--http", action="store_true", help="serve over HTTP at an MCP URL all clients share")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=7077)
    ps.add_argument("--db")
    ps.set_defaults(func=cmd_serve)

    pst = sub.add_parser("status", help="show the store, version, and configured clients")
    pst.add_argument("--db")
    pst.set_defaults(func=cmd_status)

    pu = sub.add_parser("update", help="upgrade Midas to the latest version")
    pu.add_argument("--pip", action="store_true", help="force a pip upgrade instead of uv tool")
    pu.add_argument("--dry-run", action="store_true")
    pu.set_defaults(func=cmd_update)

    pn = sub.add_parser("inspect", help="open the local Memory Inspector")
    pn.add_argument("--db")
    pn.add_argument("--host", default="127.0.0.1")
    pn.add_argument("--port", type=int, default=7777)
    pn.add_argument("--embedder", choices=("local", "hashing"), default="local")
    pn.add_argument("--no-browser", action="store_true")
    pn.set_defaults(func=cmd_inspect)

    pd = sub.add_parser("doctor", help="diagnose your install, store, and client config")
    pd.add_argument("--db")
    pd.set_defaults(func=cmd_doctor)

    pe = sub.add_parser("export", help="export all memory to JSON (backup / move machines)")
    pe.add_argument("--db")
    pe.add_argument("-o", "--out", help="output file (default: stdout)")
    pe.set_defaults(func=cmd_export)

    pm = sub.add_parser("import", help="import memory from a JSON file")
    pm.add_argument("file", help="the .json produced by `midas export`")
    pm.add_argument("--db")
    pm.add_argument("--overwrite", action="store_true", help="overwrite records with the same id")
    pm.set_defaults(func=cmd_import)

    pun = sub.add_parser("uninstall", help="remove midas from your MCP clients (memory kept unless --purge)")
    pun.add_argument("--db")
    pun.add_argument("--purge", action="store_true", help="also delete the memory store")
    pun.set_defaults(func=cmd_uninstall)

    ppr = sub.add_parser("projects", help="list the projects Midas has memory for")
    ppr.add_argument("--db")
    ppr.set_defaults(func=cmd_projects)

    ppd = sub.add_parser("project", help="show a project's state, governance, and attribution")
    ppd.add_argument("name")
    ppd.add_argument("--db")
    ppd.set_defaults(func=cmd_project)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        return
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
