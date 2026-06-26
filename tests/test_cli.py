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
