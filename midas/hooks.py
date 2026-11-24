"""Live prompt hook setup and compatibility with the original Claude SessionEnd hook."""
from __future__ import annotations

import base64
import json
import shutil
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


HOOK_COMMAND = "midas hook capture-session"  # legacy SessionEnd command


def _prompt_command(client: str) -> str:
    return f"midas-mcp --capture-prompt {client}"


def _update_prompt_hook(path: Path, client: str, *, install: bool, dry: bool = False,
                        db: str | None = None, scoped: bool = False) -> str:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8") or "{}") if path.exists() else {}
        if not isinstance(cfg, dict) or not isinstance(cfg.get("hooks", {}), dict):
            raise ValueError("invalid hook configuration")
    except (OSError, ValueError):
        return f"⚠  {path} is not a readable hook configuration"

    original = cfg.get("hooks", {})
    hooks = dict(original)
    command = _prompt_command(client)
    events = {"UserPromptSubmit": command}
    if client == "claude-code":
        events["SessionEnd"] = HOOK_COMMAND
    for event, owned in events.items():
        if event not in hooks:
            continue
        groups = []
        for group in hooks[event]:
            kept = [hook for hook in group.get("hooks", [])
                    if hook.get("command") != owned
                    and not str(hook.get("command", "")).startswith(owned + " ")]
            if kept or not group.get("hooks"):
                groups.append({**group, "hooks": kept} if group.get("hooks") else group)
        hooks[event] = groups

    if install:
        if db:
            command += " --db-encoded " + base64.urlsafe_b64encode(db.encode()).decode()
        if scoped:
            command += " --project-scoped"
        handler = {"type": "command", "command": command, "timeout": 5}
        if client == "claude-code":
            handler["async"] = True
        hooks.setdefault("UserPromptSubmit", []).append({"hooks": [handler]})
    if hooks == original:
        return "already installed" if install else "not installed"
    if dry:
        return f"would add UserPromptSubmit hook → {path}"
    if path.exists():
        shutil.copy(path, str(path) + ".midas-bak")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**cfg, "hooks": hooks}, indent=2) + "\n", encoding="utf-8")
    return f"UserPromptSubmit hook added → {path}" if install else "removed"


def install_claude_hook(settings_path: Path, *, dry: bool = False, db: str | None = None,
                        scoped: bool = False) -> str:
    return _update_prompt_hook(settings_path, "claude-code", install=True, dry=dry, db=db,
                               scoped=scoped)


def uninstall_claude_hook(settings_path: Path) -> str:
    return _update_prompt_hook(settings_path, "claude-code", install=False)


def codex_hook_path() -> Path:
    import os

    return Path(os.getenv("CODEX_HOME") or Path.home() / ".codex").expanduser() / "hooks.json"


def install_codex_hook(*, dry: bool = False, db: str | None = None, scoped: bool = False) -> str:
    return _update_prompt_hook(codex_hook_path(), "codex", install=True, dry=dry, db=db, scoped=scoped)


def uninstall_codex_hook() -> str:
    return _update_prompt_hook(codex_hook_path(), "codex", install=False)
