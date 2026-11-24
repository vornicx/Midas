"""Explicit, previewable imports of native Claude Code and Codex Markdown memories."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .importers import _FENCE_RE, parse_markdown_rules
from .types import MemoryKind


NATIVE_SOURCES = ("claude-memory", "codex-memory")
_CLAUDE_KINDS: dict[str, MemoryKind] = {
    "user": "preference", "feedback": "preference", "project": "fact", "reference": "note",
}
_CODEX_SECTIONS: dict[str, MemoryKind] = {
    "user preferences": "preference", "reusable knowledge": "fact",
    "failures and how to do differently": "note",
}
_INDEX_LINK = re.compile(r"^(?:[-*+]\s+)?!?\[[^\]]+\]\([^)]+\.md(?:#[^)]*)?\)")
_CODEX_ROUTING = {"rollout_summary_files", "keywords"}


@dataclass(frozen=True)
class NativeMemory:
    selection: str
    file: str
    line: int
    content: str
    kind: MemoryKind
    section: str
    group: str
    origin_scope: str
    modified: float


def _files(path: Path, source: str) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".md":
            raise ValueError("Native imports accept Markdown memory files.")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Memory file or directory does not exist: {path}")
    if source == "claude-memory":
        topics = sorted(p for p in path.glob("*.md")
                        if p.name.casefold() != "memory.md" and not p.is_symlink() and p.is_file())
        if topics:
            return topics
    # Codex v1 and v2 expose the same consolidated Markdown names. Importing a directory does not
    # also traverse raw memories, rollouts, generated skills, or linked paths.
    for name in ("MEMORY.md", "memory_summary.md") if source == "codex-memory" else ("MEMORY.md",):
        file = path / name
        if file.is_file() and not file.is_symlink():
            return [file]
    raise ValueError(f"No native Markdown memory found in {path}")


def _frontmatter(lines: list[str]) -> dict[str, str]:
    if not lines or lines[0].strip() != "---":
        return {}
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        return {}
    fields = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep and key in ("name", "type", "modified"):
            fields[key] = value.strip().strip("'\"")
    lines[:end + 1] = [""] * (end + 1)  # preserve source line numbers
    return fields


def _codex_groups(lines: list[str]) -> dict[int, tuple[str, str]]:
    group = origin_scope = ""
    in_fence = False
    groups = {}
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        if in_fence:
            continue
        if line.startswith("# Task Group:"):
            group, origin_scope = line.partition(":")[2].strip(), ""
            lines[i] = ""
        elif line.startswith(("scope:", "applies_to:")):
            applies_to = line.partition("applies_to:")[2].strip()
            if applies_to:
                origin_scope = applies_to
            lines[i] = ""
        groups[i + 1] = group, origin_scope
    return groups


def _modified(value: str, fallback: float) -> float:
    try:
        date = datetime.fromisoformat(value)
        return (date if date.tzinfo else date.replace(tzinfo=timezone.utc)).timestamp()
    except (ValueError, OSError, OverflowError):
        return fallback


def load_native(path: Path, source: str) -> list[NativeMemory]:
    out = []
    for file in _files(path, source):
        lines = file.read_text(encoding="utf-8-sig").splitlines()
        frontmatter = _frontmatter(lines) if source == "claude-memory" else {}
        groups = _codex_groups(lines) if source == "codex-memory" else {}
        if source == "claude-memory" and file.name.casefold() == "memory.md":
            lines = ["" if _INDEX_LINK.match(line.strip()) else line for line in lines]
        modified = _modified(frontmatter.get("modified", ""), file.stat().st_mtime)
        for item in parse_markdown_rules("\n".join(lines)):
            section = item.section.casefold().strip()
            if section in _CODEX_ROUTING and source == "codex-memory":
                continue
            group, origin_scope = groups.get(item.line, (frontmatter.get("name", ""), ""))
            kind = (_CLAUDE_KINDS.get(frontmatter.get("type", ""), "note")
                    if source == "claude-memory" else _CODEX_SECTIONS.get(section, "note"))
            fingerprint = f"{file.resolve()}\0{item.line}\0{item.content}\0{kind}\0{group}\0{origin_scope}"
            selection = hashlib.sha256(fingerprint.encode("utf-8", "surrogatepass")).hexdigest()[:16]
            out.append(NativeMemory(
                selection, str(file.resolve()), item.line, item.content, kind,
                item.section, group, origin_scope, modified,
            ))
    return out


def content_key(text: str) -> str:
    return " ".join(text.split()).casefold()
