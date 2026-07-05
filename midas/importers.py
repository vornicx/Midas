"""Import existing agent memory INTO Midas — the migration on-ramp.

Most teams already carry agent memory in files: a `CLAUDE.md`, a `.cursorrules`, exported JSONL from
another memory product. Instead of retyping it, `midas import --from …` ingests those as first-class
Midas records (with `imported_from` provenance metadata), so the constraints an agent already obeys
become recallable, supersedable, and governable.

Parsing is deliberately dumb and lossless-enough: bullets and standalone paragraphs become one record
each (a memory that answers on its own retrieves better than a whole file), fenced code blocks are
skipped, and headings become a "Section:" prefix so a rule keeps its context. No LLM.

Governance note: imported rules default to provenance="observation". They inform planning and recall,
but they cannot authorize external/destructive actions through the guard — pass `--confirmed` only
when you vouch that the file states user-confirmed policy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_MD_INLINE_RE = re.compile(r"\*\*([^*]+)\*\*|\*([^*]+)\*|__([^_]+)__|`([^`]+)`")


@dataclass(frozen=True)
class ImportedItem:
    content: str
    section: str  # the heading path the item sat under ("" at top level)


def _strip_inline_md(text: str) -> str:
    return _MD_INLINE_RE.sub(lambda m: next(g for g in m.groups() if g is not None), text).strip()


def parse_markdown_rules(text: str) -> list[ImportedItem]:
    """Extract one item per bullet / standalone paragraph from a rules-style markdown file
    (CLAUDE.md, .cursorrules, AGENTS.md…). Fenced code blocks are skipped; headings become context."""
    items: list[ImportedItem] = []
    section = ""
    in_fence = False
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        joined = _strip_inline_md(" ".join(paragraph))
        if len(joined) >= 20:  # a one-word line is noise, not a rule
            items.append(ImportedItem(content=joined, section=section))
        paragraph = []

    for raw in text.splitlines():
        if _FENCE_RE.match(raw):
            flush_paragraph()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = _HEADING_RE.match(raw)
        if heading:
            flush_paragraph()
            section = heading.group(2).strip()
            continue
        bullet = _BULLET_RE.match(raw)
        if bullet:
            flush_paragraph()
            content = _strip_inline_md(bullet.group(1))
            if len(content) >= 8:
                items.append(ImportedItem(content=content, section=section))
            continue
        if raw.strip():
            paragraph.append(raw.strip())
        else:
            flush_paragraph()
    flush_paragraph()
    return items


def to_record_dicts(
    items: list[ImportedItem],
    *,
    source_file: str,
    project: str | None = None,
    confirmed: bool = False,
) -> list[dict]:
    """Turn parsed items into `Memory.remember(**kwargs)` dicts. Section context is prefixed into the
    content (self-contained memories retrieve better), never invented."""
    out: list[dict] = []
    for item in items:
        content = f"{item.section}: {item.content}" if item.section else item.content
        meta: dict = {"imported_from": source_file}
        if project:
            meta["project"] = project
        out.append({
            "content": content,
            "kind": "constraint",
            "importance": 4,
            "provenance": "user_confirmation" if confirmed else "observation",
            "source": f"import:{source_file}",
            "metadata": meta,
        })
    return out
