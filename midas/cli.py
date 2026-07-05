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


def _vscode_user_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Code/User"
    if sys.platform.startswith("win"):
        ad = os.getenv("APPDATA")
        return (Path(ad) if ad else Path.home() / "AppData/Roaming") / "Code" / "User"
    return Path.home() / ".config/Code/User"


def _zed_settings_path() -> Path:
    if sys.platform.startswith("win"):
        ad = os.getenv("APPDATA")
        return (Path(ad) if ad else Path.home() / "AppData/Roaming") / "Zed" / "settings.json"
    return Path.home() / ".config/zed/settings.json"


# Every JSON-configured client Midas can wire: (display name, client id, config path, JSON key holding
# the server map, block shape). Shapes differ per client — see _client_block.
def _json_clients() -> list[tuple[str, str, Path, str, str]]:
    out = [("Cursor", "cursor", Path.home() / ".cursor/mcp.json", "mcpServers", "std"),
           ("Windsurf", "windsurf", Path.home() / ".codeium/windsurf/mcp_config.json",
            "mcpServers", "std"),
           ("VS Code", "vscode", _vscode_user_dir() / "mcp.json", "servers", "vscode"),
           ("Gemini CLI", "gemini-cli", Path.home() / ".gemini/settings.json", "mcpServers", "std"),
           ("Cline", "cline", _vscode_user_dir() / "globalStorage/saoudrizwan.claude-dev/settings"
            / "cline_mcp_settings.json", "mcpServers", "std"),
           ("Zed", "zed", _zed_settings_path(), "context_servers", "zed")]
    cd = _claude_desktop_path()
    if cd:
        out.append(("Claude Desktop", "claude-desktop", cd, "mcpServers", "std"))
    return out


def _client_block(shape: str, env: dict) -> dict:
    """The `midas` server entry in the schema each client expects."""
    if shape == "vscode":   # VS Code mcp.json declares the transport type explicitly
        return {"type": "stdio", "command": "midas-mcp", "env": env}
    if shape == "zed":      # Zed's context_servers entries are flat command + args
        return {"source": "custom", "command": "midas-mcp", "args": [], "env": env}
    return {"command": "midas-mcp", "env": env}


# ---- init -------------------------------------------------------------------------------------

def _merge_mcp_json(path: Path, block: dict, *, dry: bool, key: str = "mcpServers") -> str:
    """Add/refresh a `midas` entry in a client's server-map JSON, non-destructively (merge + backup).
    `key` is the JSON key holding the server map (mcpServers / servers / context_servers)."""
    cfg: dict = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text() or "{}")
        except Exception:
            return f"⚠  {path} is not valid JSON — add `midas` by hand"
    servers = cfg.setdefault(key, {})
    verb = "update" if "midas" in servers else "add"
    if dry:
        return f"would {verb} → {path}"
    if path.exists():
        shutil.copy(path, str(path) + ".midas-bak")
    servers["midas"] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return f"{verb}ed → {path}"


def _cli_add(cmd: list[str], *, dry: bool, force: bool = False,
             remove: list[str] | None = None) -> tuple[str, bool] | None:
    """Configure a client via its own CLI (claude/codex). None if that CLI isn't installed; otherwise
    (message, ok). With force, remove any existing `midas` entry first so re-running init can't collide."""
    if not shutil.which(cmd[0]):
        return None
    if dry:
        return ("would re-add via " if force else "would run: ") + " ".join(cmd), True
    if force and remove:
        subprocess.run(remove, capture_output=True, text=True)  # best-effort; ignore if not present
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"{'re-' if force else ''}configured via `{cmd[0]} mcp add`", True
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or e.stdout or "").strip().splitlines()
        hint = "" if force else " (already added — re-run `midas init --force`)"
        return f"`{cmd[0]}` failed: {tail[-1][:78] if tail else 'error'}{hint}", False


def _cli_remove(exe: str, remove: list[str]) -> str | None:
    """Remove the midas entry via a client's CLI (for uninstall). None if the CLI isn't installed."""
    if not shutil.which(exe):
        return None
    r = subprocess.run(remove, capture_output=True, text=True)
    return "removed" if r.returncode == 0 else "not configured"


def _receipt(db: str, *, scope_mode: str, namespace: str | None, clients: list[dict],
             warnings: list[str] | None = None) -> dict:
    """The machine-readable client-wiring receipt: a compact, pasteable proof of what is wired where —
    config paths, server command, scope, and policy — with NO memory contents (a stable audit boundary
    that agents/support can consume without scraping terminal prose)."""
    from datetime import datetime, timezone

    from midas import __version__
    from midas.policy import MemoryPolicy

    pol = MemoryPolicy(min_importance=int(os.getenv("MIDAS_MCP_MIN_IMPORTANCE", "2")))
    return {
        "midas_version": __version__,
        "receipt_kind": "client_wiring",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "memory_db": db,
        "scope_mode": scope_mode,
        "namespace": namespace,
        "server_command": "midas-mcp",
        "embedder": os.getenv("MIDAS_MCP_EMBEDDER", "local"),
        "policy": {
            "recall_first": True,
            "capture_enabled": True,
            "min_importance": pol.min_importance,
            "dedup_threshold": pol.dedup_threshold,
            "external_or_destructive_actions_require_check_memory_use": True,
        },
        "clients": clients,
        "warnings": warnings or [],
    }


def cmd_init(args: argparse.Namespace) -> int:
    db = _store_path(args.db)
    as_json = getattr(args, "json", False)
    dry = args.dry_run
    if not dry:
        _ensure_store(db)
    scoped = getattr(args, "project_scoped", False)
    force = getattr(args, "force", False)

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

    results: list[tuple[str, str]] = []
    receipt_clients: list[dict] = []

    def record(name: str, path: Path, *, detected: bool, wired: bool, changed: bool,
               backup: str | None = None, reason: str | None = None) -> None:
        receipt_clients.append({"client": name, "config_path": str(path), "detected": detected,
                                "wired": wired, "changed": changed, "backup_path": backup,
                                "reason": reason})

    def already_wired(path: Path) -> bool:
        return path.exists() and "midas" in _safe_read(path)

    # Clients with their own CLI (cleanest, no file editing) — each stamped with its client id:
    codex_note = "  — set MIDAS_MCP_CLIENT=codex" + (" + MIDAS_MCP_NAMESPACE=auto" if scoped else "") + \
                 " in ~/.codex/config.toml"
    for name, cfg_path, note, add_cmd, rm_cmd in (
        ("Claude Code", Path.home() / ".claude.json", "",
         ["claude", "mcp", "add", "midas", "-s", "user", *eflags(env_for("claude-code")),
          "--", "midas-mcp"],
         ["claude", "mcp", "remove", "midas", "-s", "user"]),
        ("Codex", Path.home() / ".codex/config.toml", codex_note,
         ["codex", "mcp", "add", "midas", "--", "midas-mcp"],
         ["codex", "mcp", "remove", "midas"]),
    ):
        res = _cli_add(add_cmd, dry=dry, force=force, remove=rm_cmd)
        if res is None:
            record(name, cfg_path, detected=False, wired=already_wired(cfg_path), changed=False,
                   reason=f"`{add_cmd[0]}` CLI not found")
            continue
        msg, ok = res
        results.append((name, msg + note))
        record(name, cfg_path, detected=True, wired=already_wired(cfg_path) if dry else ok,
               changed=ok and not dry, reason=msg if (dry or not ok) else None)

    # Clients configured by a JSON file (only the ones already present, unless --all):
    for name, client_id, path, key, shape in _json_clients():
        if not (path.exists() or args.all):
            record(name, path, detected=False, wired=False, changed=False,
                   reason="config not found (skipped — use --all to configure anyway)")
            continue
        existed, was = path.exists(), already_wired(path)
        block = _client_block(shape, env_for(client_id))
        msg = _merge_mcp_json(path, block, dry=dry, key=key)
        results.append((name, msg))
        ok = not msg.startswith("⚠")
        record(name, path, detected=existed, wired=was if dry else ok, changed=ok and not dry,
               backup=str(path) + ".midas-bak" if (ok and existed and not dry) else None,
               reason=msg if (dry or not ok) else None)

    if as_json:  # machine-readable receipt only — pipeable, and the human UX stays clean without it
        receipt = _receipt(db, scope_mode="project-scoped" if scoped else "shared",
                           namespace="auto" if scoped else None, clients=receipt_clients)
        receipt["dry_run"] = dry
        print(json.dumps(receipt, indent=2))
        return 0

    print(f"✓ shared memory: {db}")
    print("  mode: per-project — memory auto-separates by project (git repo / cwd).\n" if scoped
          else "  all clients below point here → they share one memory, autonomously.\n")
    for name, msg in results:
        print(f"  • {name}: {msg}")
    if not results:
        print("  (no known MCP clients detected — see the manual line below)")
    print("\nAny other client → point it at command `midas-mcp` (no env required).")
    if not dry:
        print("Restart your clients to apply. Verify with `midas status`.")
    return 0


# ---- serve ------------------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    if args.db:  # set BEFORE importing mcp_server (it builds its store at import)
        os.environ["MIDAS_MCP_DB"] = _store_path(args.db)
    if getattr(args, "token", None):
        os.environ["MIDAS_MCP_TOKEN"] = args.token
    from midas import mcp_server

    mcp_server.run_server("http" if args.http else "stdio", host=args.host, port=args.port)
    return 0


# ---- status -----------------------------------------------------------------------------------

def _midas_server_entry(path: Path, key: str = "mcpServers") -> dict | None:
    """Parse the `midas` server entry out of a client config (JSON server map / codex TOML), so the
    receipt can state which command + MIDAS_* env — i.e. which memory and scope — that client actually
    got. Only MIDAS_* env keys are surfaced (the audit boundary excludes anything else a user added)."""
    text = _safe_read(path)
    if not text:
        return None
    try:
        if path.suffix == ".toml":
            import tomllib

            servers = tomllib.loads(text).get(key) or {}
        else:
            servers = json.loads(text).get(key) or {}
    except Exception:
        return None
    entry = servers.get("midas")
    if not isinstance(entry, dict):
        return None
    env = {k: v for k, v in (entry.get("env") or {}).items() if str(k).startswith("MIDAS_")}
    return {"command": entry.get("command"), "env": env}


def _derive_scope(ns_values: set) -> tuple[str, str | None, list[str]]:
    """Map the namespaces found across wired clients onto the receipt's scope_mode."""
    if not ns_values or ns_values == {None}:
        return "shared", None, []
    if ns_values == {"auto"}:
        return "project-scoped", "auto", []
    if len(ns_values) == 1:
        return "manual-namespace", next(iter(ns_values)), []
    return "mixed", None, ["clients are wired with different namespaces: "
                           + ", ".join(sorted(str(v) for v in ns_values))]


def cmd_status(args: argparse.Namespace) -> int:
    from midas import __version__

    db = _store_path(args.db)
    store_info: dict = {"exists": Path(db).exists()}
    if store_info["exists"]:
        from midas.sqlite_store import SQLiteStore

        recs = list(SQLiteStore(db).all())
        store_info.update(records=len(recs),
                          live=sum(1 for r in recs if r.superseded_by is None),
                          size_kb=Path(db).stat().st_size // 1024)

    clients: list[dict] = []
    ns_values: set = set()
    for name, p, key in _client_paths():
        wired = p.exists() and "midas" in _safe_read(p)
        entry = _midas_server_entry(p, key) if wired else None
        env = entry["env"] if entry else {}
        if entry:
            ns_values.add(env.get("MIDAS_MCP_NAMESPACE"))
        clients.append({"client": name, "config_path": str(p), "detected": p.exists(),
                        "wired": wired, "server_command": entry["command"] if entry else None,
                        "client_id": env.get("MIDAS_MCP_CLIENT"),
                        "namespace": env.get("MIDAS_MCP_NAMESPACE"),
                        "reason": None if p.exists() else "config not found"})

    if getattr(args, "json", False):  # machine-readable receipt only (see issue #15)
        scope_mode, namespace, warnings = _derive_scope(ns_values)
        if not store_info["exists"]:
            warnings.append("memory store not created yet — run `midas init`")
        receipt = _receipt(db, scope_mode=scope_mode, namespace=namespace,
                           clients=clients, warnings=warnings)
        receipt["store"] = store_info
        print(json.dumps(receipt, indent=2))
        return 0

    print(f"Midas {__version__}")
    print(f"memory: {db}")
    if store_info["exists"]:
        print(f"  {store_info['records']} records ({store_info['live']} live)"
              f" · {store_info['size_kb']} KB")
    else:
        print("  (not created yet — run `midas init`)")

    print("clients:")
    for c in clients:
        flag = "✓" if c["wired"] else "·"
        note = "" if c["detected"] else "  (not found)"
        print(f"  {flag} {c['client']}{note}")
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

def _client_paths() -> list[tuple[str, Path, str]]:
    """Every client Midas knows how to check: (name, config path, JSON/TOML key of the server map)."""
    paths = [("Claude Code", Path.home() / ".claude.json", "mcpServers"),
             ("Codex", Path.home() / ".codex/config.toml", "mcp_servers")]
    paths += [(name, p, key) for name, _cid, p, key, _shape in _json_clients()]
    return paths


def _wired_clients() -> list[str]:
    return [name for name, p, _k in _client_paths() if p.exists() and "midas" in _safe_read(p)]


# ---- doctor -----------------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    from midas import __version__

    checks: list[dict] = []

    def check(good: bool, label: str, hint: str = "") -> None:
        checks.append({"ok": good, "check": label, "hint": hint if (hint and not good) else None})

    check(shutil.which("midas-mcp") is not None, "midas-mcp on PATH",
          "reinstall, or use the absolute path from `which midas-mcp` in your client config")

    db = _store_path(args.db)
    if Path(db).exists():
        from midas.sqlite_store import SQLiteStore
        try:
            store = SQLiteStore(db)
            check(True, f"store: {db}  ({len(store.all())} records, schema v{store.schema_version()})")
            chain = store.verify_audit_log()
            check(chain["ok"], f"audit chain intact ({chain['entries']} entries)",
                  f"broken at seq {chain['first_invalid_seq']} — see `midas audit`")
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

    try:
        import mcp  # noqa: F401
        check(True, "MCP server available (the `mcp` extra)")
    except Exception:
        check(False, "MCP server (the `mcp` extra)",
              'SDK is installed but the MCP server is not — `pip install "midas-memory[mcp]"` to expose it')

    wired = _wired_clients()
    check(bool(wired), f"clients wired: {', '.join(wired) if wired else 'none'}", "run `midas init`")
    ok = all(c["ok"] for c in checks)

    if getattr(args, "json", False):  # same audit boundary as the wiring receipt: no memory contents
        from datetime import datetime, timezone

        print(json.dumps({
            "midas_version": __version__,
            "receipt_kind": "doctor",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "python": sys.version.split()[0],
            "memory_db": db,
            "ok": ok,
            "checks": checks,
            "clients_wired": wired,
        }, indent=2))
        return 0 if ok else 1

    print(f"Midas {__version__}  ·  Python {sys.version.split()[0]}\n")
    for c in checks:
        print(f"  {'✓' if c['ok'] else '⚠'}  {c['check']}" + (f"  — {c['hint']}" if c["hint"] else ""))
    print("\n" + ("Everything looks good." if ok else "Some checks need attention (see ⚠ above)."))
    return 0 if ok else 1


# ---- audit ------------------------------------------------------------------------------------

def cmd_audit(args: argparse.Namespace) -> int:
    """Show / verify the tamper-evident mutation log: every put/delete/clear, hash-chained. Editing,
    removing, or reordering any past entry breaks every hash after it — `--verify` (the default check)
    walks the whole chain. Entries carry hashes only, never memory content."""
    from midas.sqlite_store import SQLiteStore

    db = _store_path(args.db)
    if not Path(db).exists():
        print(f"no store at {db} — run `midas init`", file=sys.stderr)
        return 1
    store = SQLiteStore(db)
    result = store.verify_audit_log()
    entries = store.audit_log(limit=int(args.limit) if args.limit else 10)

    if getattr(args, "json", False):
        from datetime import datetime, timezone

        from midas import __version__

        print(json.dumps({
            "midas_version": __version__,
            "receipt_kind": "audit_chain",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "memory_db": db,
            "ok": result["ok"],
            "entries": result["entries"],
            "first_invalid_seq": result["first_invalid_seq"],
            "tail": entries,
        }, indent=2))
        return 0 if result["ok"] else 1

    state = "intact" if result["ok"] else f"BROKEN at seq {result['first_invalid_seq']}"
    print(f"audit chain: {state} · {result['entries']} entries · {db}")
    if entries:
        from datetime import datetime

        print(f"\nlast {len(entries)}:")
        for e in entries:
            when = datetime.fromtimestamp(e["at"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  #{e['seq']:<6} {when}  {e['op']:<7} {e['record_id'][:8]:<9} "
                  f"sha:{e['content_sha'][:12]}…")
    if not result["ok"]:
        print("\n⚠  the log was modified after the fact — treat this store's history as untrusted.")
    return 0 if result["ok"] else 1


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
    for name, _cid, path, key, _shape in _json_clients():
        if not path.exists():
            continue
        try:
            cfg = json.loads(path.read_text() or "{}")
        except Exception:
            results.append((name, "left as-is (unparseable)"))
            continue
        if "midas" in cfg.get(key, {}):
            shutil.copy(path, str(path) + ".midas-bak")
            del cfg[key]["midas"]
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


# ---- bench ------------------------------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> int:
    """Reproduce the governance benchmark suite — deterministic, $0, no datasets, no API key."""
    try:
        from eval import benches  # ships with the repo (dev tooling), not the installed wheel
    except Exception:
        print("The benchmark suite lives in the repo (it's kept out of the wheel). Clone and run it:\n"
              "  git clone https://github.com/vornicx/Midas && cd Midas\n"
              "  uv run python -m eval.benches", file=sys.stderr)
        return 1
    result = benches.run(verbose=True)
    green = result["memory_safety"]["ASR"] == 0.0 and result["memory_safety"]["benign_pass"] == 1.0
    return 0 if green else 1


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
    pi.add_argument("--json", action="store_true",
                    help="print a machine-readable client wiring receipt instead of text")
    pi.set_defaults(func=cmd_init)

    ps = sub.add_parser("serve", help="run the MCP server (stdio, or --http for an MCP URL)")
    ps.add_argument("--http", action="store_true", help="serve over HTTP at an MCP URL all clients share")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=7077)
    ps.add_argument("--token", help="require this bearer token on every HTTP request "
                    "(same as MIDAS_MCP_TOKEN)")
    ps.add_argument("--db")
    ps.set_defaults(func=cmd_serve)

    pst = sub.add_parser("status", help="show the store, version, and configured clients")
    pst.add_argument("--db")
    pst.add_argument("--json", action="store_true",
                     help="print a machine-readable client wiring receipt instead of text")
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
    pd.add_argument("--json", action="store_true",
                    help="print a machine-readable diagnosis instead of text")
    pd.set_defaults(func=cmd_doctor)

    pa = sub.add_parser("audit", help="show/verify the tamper-evident mutation log (hash chain)")
    pa.add_argument("--db")
    pa.add_argument("--limit", type=int, default=10, help="how many recent entries to show")
    pa.add_argument("--json", action="store_true",
                    help="print a machine-readable audit receipt instead of text")
    pa.set_defaults(func=cmd_audit)

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

    pb = sub.add_parser("bench", help="reproduce the governance benchmark suite ($0, deterministic)")
    pb.set_defaults(func=cmd_bench)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        return
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
