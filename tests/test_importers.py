"""`midas import --from claude-md|cursorrules|jsonl` — file-based agent memory becomes first-class,
governable Midas records. No LLM, idempotent, governance-conscious provenance defaults."""
from __future__ import annotations

import argparse
import json

from midas import cli
from midas.importers import parse_markdown_rules, to_record_dicts

CLAUDE_MD = """# Project rules

Always run the full test suite before committing.

## Database

- The primary database is **PostgreSQL** — never MySQL.
- Migrations must be reviewed by a human.

```bash
# this fenced command block must be skipped
rm -rf build && make
```

## Style

1. Use `ruff` with line-length 100.
ok
"""


def test_parse_markdown_rules() -> None:
    items = parse_markdown_rules(CLAUDE_MD)
    contents = [i.content for i in items]
    assert "Always run the full test suite before committing." in contents
    assert "The primary database is PostgreSQL — never MySQL." in contents  # inline md stripped
    assert "Use ruff with line-length 100." in contents                     # numbered bullets too
    assert not any("rm -rf" in c for c in contents)                         # fenced blocks skipped
    assert not any(c == "ok" for c in contents)                             # one-word noise dropped
    by_content = {i.content: i.section for i in items}
    assert by_content["Migrations must be reviewed by a human."] == "Database"


def test_to_record_dicts_defaults_are_governance_safe() -> None:
    items = parse_markdown_rules(CLAUDE_MD)
    dicts = to_record_dicts(items, source_file="CLAUDE.md", project="apollo")
    assert all(d["provenance"] == "observation" for d in dicts)  # cannot authorize guarded actions
    assert all(d["kind"] == "constraint" and d["importance"] == 4 for d in dicts)
    assert all(d["metadata"] == {"imported_from": "CLAUDE.md", "project": "apollo"} for d in dicts)
    sectioned = [d for d in dicts if d["content"].startswith("Database: ")]
    assert sectioned  # heading context is kept in the content

    confirmed = to_record_dicts(items, source_file="CLAUDE.md", confirmed=True)
    assert all(d["provenance"] == "user_confirmation" for d in confirmed)


def test_cli_import_claude_md_roundtrip(tmp_path, capsys) -> None:
    from midas.sqlite_store import SQLiteStore

    md = tmp_path / "CLAUDE.md"
    md.write_text(CLAUDE_MD)
    db = str(tmp_path / "m.sqlite3")
    ns = argparse.Namespace(file=str(md), source="claude-md", db=db, project="apollo",
                            confirmed=False, overwrite=False)
    assert cli.cmd_import(ns) == 0
    out = capsys.readouterr().out
    assert "imported" in out and "provenance=observation" in out

    recs = SQLiteStore(db).all()
    assert any("PostgreSQL" in r.content for r in recs)
    assert all(r.metadata.get("imported_from") == "CLAUDE.md" for r in recs)
    n = len(recs)

    assert cli.cmd_import(ns) == 0                      # idempotent: same file, nothing duplicated
    assert len(SQLiteStore(db).all()) == n
    assert "already present" in capsys.readouterr().out


def test_cli_import_jsonl(tmp_path, capsys) -> None:
    from midas.sqlite_store import SQLiteStore

    f = tmp_path / "mem.jsonl"
    f.write_text(json.dumps({"content": "User prefers dark mode.", "kind": "preference",
                             "importance": 5}) + "\n\n" +
                 json.dumps({"content": "The launch date is September 14.", "kind": "fact"}) + "\n")
    db = str(tmp_path / "m.sqlite3")
    ns = argparse.Namespace(file=str(f), source="jsonl", db=db, project=None,
                            confirmed=False, overwrite=False)
    assert cli.cmd_import(ns) == 0
    recs = {r.content: r for r in SQLiteStore(db).all()}
    assert recs["User prefers dark mode."].kind == "preference"
    assert recs["User prefers dark mode."].importance == 5
    assert recs["The launch date is September 14."].source == "import:mem.jsonl"


def test_cli_import_empty_file_fails_cleanly(tmp_path, capsys) -> None:
    f = tmp_path / "empty.md"
    f.write_text("\n\n")
    ns = argparse.Namespace(file=str(f), source="claude-md", db=str(tmp_path / "m.sqlite3"),
                            project=None, confirmed=False, overwrite=False)
    assert cli.cmd_import(ns) == 1
    assert "nothing importable" in capsys.readouterr().err
