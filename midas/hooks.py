"""Claude Code session hook — "install Midas and it starts remembering", even when the agent never
calls `capture`.

`midas init --claude-hook` registers a SessionEnd hook in `~/.claude/settings.json` that runs
`midas hook capture-session`. When a Claude Code session ends, the hook receives the session payload
on stdin (session id, transcript path, cwd), reads the transcript JSONL, and OFFERS each user turn to
`Memory.capture` — Midas's no-LLM policy then decides what is actually kept (importance floor, dedup),
exactly as if the agent had called `capture` itself. User turns only in v1: they carry the durable
preferences/facts/decisions; assistant prose is mostly restatement.

Fail-open by design: a hook must never break the user's session, so every error path returns a result
instead of raising, and the CLI always exits 0.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _text_of(content: Any) -> str:
    """A message's text: transcripts store content as a plain string or a list of typed blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b.get("text", "").strip() for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


def user_turns(transcript_path: str | Path) -> list[str]:
    """The user-authored texts in a Claude Code transcript (JSONL), in order. Tool results, meta
    lines, and unparseable rows are skipped — a hook must tolerate format drift."""
    turns: list[str] = []
    try:
        lines = Path(transcript_path).read_text().splitlines()
    except Exception:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict) or row.get("type") != "user" or row.get("isMeta"):
            continue
        message = row.get("message") or {}
        if message.get("role") != "user":
            continue
        text = _text_of(message.get("content"))
        if text and not text.startswith("<"):  # skip tool_result / system-reminder style payloads
            turns.append(text)
    return turns


def capture_session(payload: dict, *, db: str | None = None) -> dict:
    """Offer a finished session's user turns to memory. `payload` is the hook's stdin JSON
    (session_id, transcript_path, cwd). Returns {offered, stored, skipped} — never raises."""
    try:
        from midas import Memory
        from midas.sqlite_store import SQLiteStore

        transcript = payload.get("transcript_path") or ""
        turns = user_turns(transcript)
        if not turns:
            return {"offered": 0, "stored": 0, "skipped": 0}

        import os
        path = db or os.getenv("MIDAS_MCP_DB") or str(Path.home() / ".midas" / "memory.sqlite3")
        mem = Memory(store=SQLiteStore(path))
        session = payload.get("session_id") or "session"
        cwd = payload.get("cwd") or ""
        stored = 0
        for text in turns:
            result = mem.capture(
                text, kind="chat", actor="claude-code",
                source=f"hook:claude-code:{session}",
                metadata={"session": session} | ({"origin": {"cwd": cwd}} if cwd else {}),
            )
            stored += bool(result.stored)
        return {"offered": len(turns), "stored": stored, "skipped": len(turns) - stored}
    except Exception as exc:  # fail-open: a memory hook must never break the session
        return {"offered": 0, "stored": 0, "skipped": 0, "error": str(exc)[:200]}


# The hook block `midas init --claude-hook` merges into ~/.claude/settings.json.
HOOK_COMMAND = "midas hook capture-session"


def hook_settings_block() -> dict:
    return {"type": "command", "command": HOOK_COMMAND}


def install_claude_hook(settings_path: Path, *, dry: bool = False) -> str:
    """Register the SessionEnd hook in Claude Code's settings.json, non-destructively (merge +
    backup; idempotent — re-running never duplicates the entry)."""
    import shutil

    cfg: dict = {}
    if settings_path.exists():
        try:
            cfg = json.loads(settings_path.read_text() or "{}")
        except Exception:
            return f"⚠  {settings_path} is not valid JSON — add the hook by hand"
    matchers = cfg.setdefault("hooks", {}).setdefault("SessionEnd", [])
    for m in matchers:
        if any(h.get("command") == HOOK_COMMAND for h in m.get("hooks", [])):
            return "already installed"
    if dry:
        return f"would add SessionEnd hook → {settings_path}"
    if settings_path.exists():
        shutil.copy(settings_path, str(settings_path) + ".midas-bak")
    matchers.append({"hooks": [hook_settings_block()]})
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return f"SessionEnd auto-capture hook added → {settings_path}"


def uninstall_claude_hook(settings_path: Path) -> str:
    """Remove the SessionEnd hook (for `midas uninstall`)."""
    if not settings_path.exists():
        return "not installed"
    try:
        cfg = json.loads(settings_path.read_text() or "{}")
    except Exception:
        return "left as-is (unparseable)"
    matchers = cfg.get("hooks", {}).get("SessionEnd", [])
    kept = [m for m in matchers
            if not any(h.get("command") == HOOK_COMMAND for h in m.get("hooks", []))]
    if len(kept) == len(matchers):
        return "not installed"
    import shutil

    shutil.copy(settings_path, str(settings_path) + ".midas-bak")
    cfg["hooks"]["SessionEnd"] = kept
    settings_path.write_text(json.dumps(cfg, indent=2) + "\n")
    return "removed"
