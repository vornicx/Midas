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


def test_init_creates_the_store(tmp_path) -> None:
    db = tmp_path / "sub" / "memory.sqlite3"
    rc = cli.cmd_init(argparse.Namespace(db=str(db), dry_run=False, all=False))
    assert rc == 0 and db.exists()  # the shared store is created (parent dir too)


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
