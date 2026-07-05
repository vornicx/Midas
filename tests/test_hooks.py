"""The Claude Code SessionEnd auto-capture hook: transcript parsing, policy-gated capture,
fail-open behaviour, and idempotent install/uninstall into settings.json."""
from __future__ import annotations

import argparse
import io
import json

from midas.hooks import (
    HOOK_COMMAND,
    capture_session,
    install_claude_hook,
    uninstall_claude_hook,
    user_turns,
)


def _transcript(tmp_path, rows) -> str:
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_user_turns_parses_both_content_shapes_and_skips_noise(tmp_path) -> None:
    path = _transcript(tmp_path, [
        {"type": "user", "message": {"role": "user", "content": "plain string turn"}},
        {"type": "user", "message": {"role": "user",
                                     "content": [{"type": "text", "text": "block turn"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": "assistant prose"}},
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "meta noise"}},
        {"type": "user", "message": {"role": "user",
                                     "content": [{"type": "tool_result", "content": "x"}]}},
        {"type": "user", "message": {"role": "user", "content": "<system-reminder>injected</system-reminder>"}},
    ])
    assert user_turns(path) == ["plain string turn", "block turn"]
    assert user_turns(tmp_path / "missing.jsonl") == []  # fail-open: no transcript, no crash


def test_capture_session_stores_signal_and_skips_trivia(tmp_path) -> None:
    from midas.sqlite_store import SQLiteStore

    db = str(tmp_path / "m.sqlite3")
    path = _transcript(tmp_path, [
        {"type": "user", "message": {"role": "user",
         "content": "The primary database for Apollo is PostgreSQL 16."}},
        {"type": "user", "message": {"role": "user", "content": "lol ok cool"}},
    ])
    result = capture_session(
        {"session_id": "s1", "transcript_path": path, "cwd": "/repo"}, db=db)
    assert result["offered"] == 2 and result["stored"] == 1 and result["skipped"] == 1
    recs = SQLiteStore(db).all()
    assert len(recs) == 1 and "PostgreSQL" in recs[0].content
    assert recs[0].actor == "claude-code" and recs[0].source == "hook:claude-code:s1"


def test_capture_session_never_raises() -> None:
    out = capture_session({"transcript_path": "/nonexistent/x.jsonl"}, db=":memory:!!bad//path")
    assert out["stored"] == 0  # fail-open, whatever went wrong


def test_install_and_uninstall_hook_idempotent(tmp_path) -> None:
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"model": "opus", "hooks": {"PreToolUse": []}}))

    assert "would add" in install_claude_hook(settings, dry=True)
    assert json.loads(settings.read_text()).get("hooks", {}).get("SessionEnd") is None  # dry: untouched

    assert "added" in install_claude_hook(settings)
    cfg = json.loads(settings.read_text())
    assert cfg["model"] == "opus" and "PreToolUse" in cfg["hooks"]  # merge, never clobber
    assert cfg["hooks"]["SessionEnd"][0]["hooks"][0]["command"] == HOOK_COMMAND

    assert install_claude_hook(settings) == "already installed"     # idempotent
    assert len(json.loads(settings.read_text())["hooks"]["SessionEnd"]) == 1

    assert uninstall_claude_hook(settings) == "removed"
    assert json.loads(settings.read_text())["hooks"]["SessionEnd"] == []
    assert uninstall_claude_hook(settings) == "not installed"


def test_cli_hook_command_reads_stdin_and_exits_zero(tmp_path, capsys, monkeypatch) -> None:
    from midas import cli

    db = str(tmp_path / "m.sqlite3")
    path = _transcript(tmp_path, [
        {"type": "user", "message": {"role": "user",
         "content": "My API rate limit is 5000 requests per hour."}},
    ])
    monkeypatch.setattr("sys.stdin",
                        io.StringIO(json.dumps({"session_id": "s2", "transcript_path": path})))
    rc = cli.cmd_hook(argparse.Namespace(action="capture-session", db=db))
    assert rc == 0
    assert json.loads(capsys.readouterr().err)["stored"] == 1

    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert cli.cmd_hook(argparse.Namespace(action="capture-session", db=db)) == 0  # fail-open


def test_init_claude_hook_flag(tmp_path, capsys, monkeypatch) -> None:
    from midas import cli
    from tests.test_cli import _sandbox_client_paths

    monkeypatch.setattr(cli.shutil, "which", lambda exe: None)
    _sandbox_client_paths(monkeypatch, tmp_path)
    rc = cli.cmd_init(argparse.Namespace(db=str(tmp_path / "m.sqlite3"), dry_run=False, all=False,
                                         claude_hook=True))
    assert rc == 0 and "Claude Code hook" in capsys.readouterr().out
    cfg = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert cfg["hooks"]["SessionEnd"][0]["hooks"][0]["command"] == HOOK_COMMAND
