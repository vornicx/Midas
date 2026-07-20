"""The `midas` CLI: client-config merging (non-destructive, idempotent, with backups), dry-run safety,
and status. We never touch real client configs here — only temp files and read-only checks."""
from __future__ import annotations

import argparse
import json

from midas import cli


def test_merge_mcp_json_adds_then_updates(tmp_path) -> None:
    p = tmp_path / "mcp.json"
    msg1 = cli._merge_mcp_json(p, cli._STDIO_BLOCK, dry=False)
    assert "add" in msg1 and p.exists()
    assert json.loads(p.read_text())["mcpServers"]["midas"]["command"] == "midas-mcp"

    msg2 = cli._merge_mcp_json(p, cli._STDIO_BLOCK, dry=False)
    assert "update" in msg2
    assert (tmp_path / "mcp.json.midas-bak").exists()  # backed up before the second (existing) write
    assert list(json.loads(p.read_text())["mcpServers"]) == ["midas"]  # no duplicate entry


def test_merge_preserves_other_servers(tmp_path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    cli._merge_mcp_json(p, cli._STDIO_BLOCK, dry=False)
    assert set(json.loads(p.read_text())["mcpServers"]) == {"other", "midas"}


def test_merge_dry_run_writes_nothing(tmp_path) -> None:
    p = tmp_path / "mcp.json"
    msg = cli._merge_mcp_json(p, cli._STDIO_BLOCK, dry=True)
    assert "would add" in msg and not p.exists()


def test_init_dry_run_creates_no_store(tmp_path) -> None:
    db = tmp_path / "memory.sqlite3"
    rc = cli.cmd_init(argparse.Namespace(db=str(db), dry_run=True, all=False))
    assert rc == 0 and not db.exists()


def test_init_creates_the_store(tmp_path, monkeypatch) -> None:
    # Never call real client CLIs in tests.
    monkeypatch.setattr(cli.shutil, "which", lambda exe: None)
    _sandbox_client_paths(monkeypatch, tmp_path)
    db = tmp_path / "sub" / "memory.sqlite3"
    rc = cli.cmd_init(argparse.Namespace(db=str(db), dry_run=False, all=False))
    assert rc == 0 and db.exists()  # the shared store is created (parent dir too)


def _sandbox_client_paths(monkeypatch, tmp_path) -> None:
    """Point every client-config lookup into tmp_path. VS Code, Zed, and Claude Desktop resolve
    OS-specific roots (macOS Library/, Windows %APPDATA%) that ignore a patched `Path.home()` —
    without this, the suite behaves differently per OS and could touch a CI runner's real configs."""
    monkeypatch.setattr(cli.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cli, "_vscode_user_dir", lambda: tmp_path / "Code/User")
    monkeypatch.setattr(cli, "_zed_settings_path", lambda: tmp_path / "zed/settings.json")
    monkeypatch.setenv("APPDATA", str(tmp_path))


def test_init_json_receipt_dry_run(tmp_path, capsys) -> None:
    db = tmp_path / "memory.sqlite3"
    rc = cli.cmd_init(argparse.Namespace(db=str(db), dry_run=True, all=False, json=True))
    assert rc == 0 and not db.exists()  # dry-run still writes nothing
    receipt = json.loads(capsys.readouterr().out)  # --json prints ONLY the receipt (pipeable)
    assert receipt["receipt_kind"] == "client_wiring" and receipt["dry_run"] is True
    assert receipt["memory_db"] == str(db) and receipt["server_command"] == "midas-mcp"
    assert receipt["scope_mode"] == "shared" and receipt["namespace"] is None
    assert receipt["policy"]["external_or_destructive_actions_require_check_memory_use"] is True
    names = {c["client"] for c in receipt["clients"]}
    assert {"Claude Code", "Codex", "Grok Build", "Cursor", "Windsurf"} <= names
    for c in receipt["clients"]:
        assert {"client", "config_path", "detected", "wired", "changed", "backup_path",
                "reason"} <= set(c)
        assert c["changed"] is False  # a dry run never changes a config


def test_init_json_receipt_real_run(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda exe: None)  # no client CLIs → no subprocesses
    _sandbox_client_paths(monkeypatch, tmp_path)
    cursor = tmp_path / ".cursor/mcp.json"
    cursor.parent.mkdir(parents=True)
    cursor.write_text("{}")
    db = tmp_path / "m.sqlite3"
    rc = cli.cmd_init(argparse.Namespace(db=str(db), dry_run=False, all=False, json=True,
                                         project_scoped=True))
    assert rc == 0 and db.exists()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["scope_mode"] == "project-scoped" and receipt["namespace"] == "auto"
    by = {c["client"]: c for c in receipt["clients"]}
    assert by["Cursor"]["wired"] and by["Cursor"]["changed"]
    assert by["Cursor"]["backup_path"].endswith(".midas-bak")
    assert by["Claude Code"]["detected"] is False and "not found" in by["Claude Code"]["reason"]
    assert by["Windsurf"]["wired"] is False and "skipped" in by["Windsurf"]["reason"]
    assert json.loads(cursor.read_text())["mcpServers"]["midas"]["env"]["MIDAS_MCP_NAMESPACE"] == "auto"

    # the status receipt then proves the same wiring back (the issue's one-command invariant)
    assert cli.cmd_status(argparse.Namespace(db=str(db), json=True)) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["store"]["exists"] and status["store"]["records"] == 0
    assert status["scope_mode"] == "project-scoped"
    sby = {c["client"]: c for c in status["clients"]}
    assert sby["Cursor"]["wired"] and sby["Cursor"]["client_id"] == "cursor"
    assert sby["Cursor"]["namespace"] == "auto" and sby["Cursor"]["server_command"] == "midas-mcp"


def test_status_json_receipt_nothing_wired(tmp_path, capsys, monkeypatch) -> None:
    _sandbox_client_paths(monkeypatch, tmp_path)
    rc = cli.cmd_status(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), json=True))
    assert rc == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["store"]["exists"] is False
    assert any("midas init" in w for w in receipt["warnings"])  # missing store is called out
    assert all(c["wired"] is False and c["reason"] == "config not found" for c in receipt["clients"])
    assert receipt["scope_mode"] == "shared"


def test_status_json_parses_codex_toml(tmp_path, capsys, monkeypatch) -> None:
    _sandbox_client_paths(monkeypatch, tmp_path)
    codex = tmp_path / ".codex/config.toml"
    codex.parent.mkdir(parents=True)
    codex.write_text('[mcp_servers.midas]\ncommand = "midas-mcp"\n')
    cli.cmd_status(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), json=True))
    receipt = json.loads(capsys.readouterr().out)
    by = {c["client"]: c for c in receipt["clients"]}
    assert by["Codex"]["wired"] and by["Codex"]["server_command"] == "midas-mcp"
    assert receipt["scope_mode"] == "shared"  # no namespace anywhere → one shared pool


def test_status_json_parses_grok_build_toml(tmp_path, capsys, monkeypatch) -> None:
    _sandbox_client_paths(monkeypatch, tmp_path)
    grok = tmp_path / ".grok/config.toml"
    grok.parent.mkdir(parents=True)
    grok.write_text(
        '[mcp_servers.midas]\ncommand = "midas-mcp"\n'
        '[mcp_servers.midas.env]\nMIDAS_MCP_EMBEDDER = "local"\n'
        'MIDAS_MCP_CLIENT = "grok-build"\nMIDAS_MCP_NAMESPACE = "auto"\n'
    )
    cli.cmd_status(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), json=True))
    receipt = json.loads(capsys.readouterr().out)
    by = {c["client"]: c for c in receipt["clients"]}
    assert by["Grok Build"]["wired"] and by["Grok Build"]["server_command"] == "midas-mcp"
    assert by["Grok Build"]["client_id"] == "grok-build"
    assert by["Grok Build"]["namespace"] == "auto"
    assert receipt["scope_mode"] == "project-scoped"


def test_client_block_shapes() -> None:
    env = {"MIDAS_MCP_EMBEDDER": "local"}
    assert cli._client_block("std", env) == {"command": "midas-mcp", "env": env}
    assert cli._client_block("vscode", env)["type"] == "stdio"       # VS Code declares the transport
    zed = cli._client_block("zed", env)
    assert zed["source"] == "custom" and zed["args"] == []           # Zed's context_servers schema


def test_merge_mcp_json_custom_key(tmp_path) -> None:
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"servers": {"other": {"command": "x"}}}))
    cli._merge_mcp_json(p, cli._client_block("vscode", {}), dry=False, key="servers")
    cfg = json.loads(p.read_text())
    assert set(cfg["servers"]) == {"other", "midas"}                 # merged under VS Code's key
    assert "mcpServers" not in cfg                                   # never invents the wrong key


def test_init_wires_new_clients(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda exe: None)
    _sandbox_client_paths(monkeypatch, tmp_path)
    vscode = tmp_path / "Code/User/mcp.json"
    vscode.parent.mkdir(parents=True)
    vscode.write_text("{}")
    gemini = tmp_path / ".gemini/settings.json"
    gemini.parent.mkdir(parents=True)
    gemini.write_text(json.dumps({"theme": "dark"}))
    rc = cli.cmd_init(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), dry_run=False,
                                         all=False, json=True))
    assert rc == 0
    by = {c["client"]: c for c in json.loads(capsys.readouterr().out)["clients"]}
    assert by["VS Code"]["wired"] and by["Gemini CLI"]["wired"]
    assert by["Zed"]["wired"] is False and by["Cline"]["wired"] is False   # not present → skipped
    assert json.loads(vscode.read_text())["servers"]["midas"]["type"] == "stdio"
    cfg = json.loads(gemini.read_text())
    assert cfg["mcpServers"]["midas"]["command"] == "midas-mcp"
    assert cfg["theme"] == "dark"                                    # other settings survive the merge

    # status sees them through their own keys
    cli.cmd_status(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), json=True))
    sby = {c["client"]: c for c in json.loads(capsys.readouterr().out)["clients"]}
    assert sby["VS Code"]["wired"] and sby["VS Code"]["server_command"] == "midas-mcp"
    assert sby["Gemini CLI"]["wired"] and sby["Gemini CLI"]["client_id"] == "gemini-cli"


def test_init_wires_grok_build_via_native_cli(tmp_path, capsys, monkeypatch) -> None:
    _sandbox_client_paths(monkeypatch, tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.shutil, "which", lambda exe: "/fake/grok" if exe == "grok" else None)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return argparse.Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    rc = cli.cmd_init(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), dry_run=False,
                                         all=False, json=True, project_scoped=True))
    assert rc == 0
    receipt = json.loads(capsys.readouterr().out)
    grok = {c["client"]: c for c in receipt["clients"]}["Grok Build"]
    assert grok["detected"] and grok["wired"] and grok["changed"]
    assert grok["config_path"] == str(tmp_path / ".grok/config.toml")
    assert calls == [[
        "grok", "mcp", "add", "--scope", "user", "midas",
        "-e", "MIDAS_MCP_EMBEDDER=local", "-e", "MIDAS_MCP_CLIENT=grok-build",
        "-e", "MIDAS_MCP_NAMESPACE=auto", "--", "midas-mcp",
    ]]


def test_uninstall_removes_grok_build_via_native_cli(tmp_path, capsys, monkeypatch) -> None:
    _sandbox_client_paths(monkeypatch, tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.shutil, "which", lambda exe: "/fake/grok" if exe == "grok" else None)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return argparse.Namespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.cmd_uninstall(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), purge=False)) == 0
    assert calls == [["grok", "mcp", "remove", "--scope", "user", "midas"]]
    assert "Grok Build: removed" in capsys.readouterr().out


def test_uninstall_removes_from_custom_key(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda exe: None)
    _sandbox_client_paths(monkeypatch, tmp_path)
    vscode = tmp_path / "Code/User/mcp.json"
    vscode.parent.mkdir(parents=True)
    vscode.write_text(json.dumps({"servers": {"midas": {"command": "midas-mcp"},
                                              "other": {"command": "x"}}}))
    rc = cli.cmd_uninstall(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), purge=False))
    assert rc == 0
    assert list(json.loads(vscode.read_text())["servers"]) == ["other"]  # midas gone, other kept


def test_doctor_json(tmp_path, capsys, monkeypatch) -> None:
    _sandbox_client_paths(monkeypatch, tmp_path)
    rc = cli.cmd_doctor(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["receipt_kind"] == "doctor" and out["ok"] is (rc == 0)
    assert any("store" in c["check"] and not c["ok"] for c in out["checks"])  # missing store flagged
    assert all({"ok", "check", "hint"} <= set(c) for c in out["checks"])


def test_derive_scope_modes() -> None:
    assert cli._derive_scope(set()) == ("shared", None, [])
    assert cli._derive_scope({None}) == ("shared", None, [])
    assert cli._derive_scope({"auto"}) == ("project-scoped", "auto", [])
    assert cli._derive_scope({"team-a"}) == ("manual-namespace", "team-a", [])
    mode, ns, warns = cli._derive_scope({"auto", None})
    assert mode == "mixed" and ns is None and warns  # disagreeing clients are surfaced, not hidden


def test_status_runs(capsys) -> None:
    rc = cli.cmd_status(argparse.Namespace(db=":memory:"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Midas" in out and "clients:" in out


def test_doctor_runs(capsys) -> None:
    rc = cli.cmd_doctor(argparse.Namespace(db=":memory:"))
    assert rc in (0, 1)  # may flag the missing store under :memory:
    assert "Midas" in capsys.readouterr().out


def test_export_import_roundtrip(tmp_path) -> None:
    from midas import HashingEmbedder, Memory
    from midas.sqlite_store import SQLiteStore

    src = str(tmp_path / "src.sqlite3")
    mem = Memory(store=SQLiteStore(src), embedder=HashingEmbedder())
    mem.remember("Apollo uses PostgreSQL.", kind="constraint", importance=5, actor="user", source="s1")
    mem.remember("The launch date is Sept 14.", kind="fact")

    out = tmp_path / "dump.json"
    assert cli.cmd_export(argparse.Namespace(db=src, out=str(out))) == 0
    assert json.loads(out.read_text())["count"] == 2

    dst = str(tmp_path / "dst.sqlite3")
    assert cli.cmd_import(argparse.Namespace(file=str(out), db=dst, overwrite=False)) == 0
    got = {r.content for r in SQLiteStore(dst).all()}
    assert "Apollo uses PostgreSQL." in got and "The launch date is Sept 14." in got

    cli.cmd_import(argparse.Namespace(file=str(out), db=dst, overwrite=False))  # idempotent
    assert len(SQLiteStore(dst).all()) == 2


def test_schema_version_and_newer_guard(tmp_path) -> None:
    import sqlite3

    import pytest

    from midas.sqlite_store import SCHEMA_VERSION, SQLiteStore

    db = str(tmp_path / "m.sqlite3")
    s = SQLiteStore(db)
    assert s.schema_version() == SCHEMA_VERSION
    s.close()

    con = sqlite3.connect(db)  # simulate a store written by a newer Midas
    con.execute("PRAGMA user_version = 999")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError):
        SQLiteStore(db)


def test_resolve_namespace_modes(monkeypatch) -> None:
    from midas import mcp_server

    monkeypatch.delenv("MIDAS_MCP_NAMESPACE", raising=False)
    assert mcp_server._resolve_namespace() == ""           # unscoped by default
    monkeypatch.setenv("MIDAS_MCP_NAMESPACE", "team-a")
    assert mcp_server._resolve_namespace() == "team-a"      # explicit scope
    monkeypatch.setenv("MIDAS_MCP_NAMESPACE", "auto")
    assert mcp_server._resolve_namespace()                 # auto derives a non-empty project scope


def test_source_and_origin_attribution() -> None:
    from midas import mcp_server

    s = mcp_server._source("s1")
    assert s.startswith("mcp:") and s.endswith(":s1") and s.count(":") == 2  # mcp:<client>:<session>
    o = mcp_server._resolve_origin()
    assert isinstance(o, dict) and ("git" in o or "cwd" in o)  # where it was captured (git/cwd)
