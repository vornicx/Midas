# midas-memory-mcp

The **MCP server** for [Midas](https://github.com/vornicx/Midas) — local-first, source-traceable
agent memory with **no LLM at ingest**. Local embeddings + ranking only: ingest is $0, nothing leaves
your machine, and every recalled memory points back to its source turn (no LLM-rewritten facts).

This package is a thin launcher. It installs the Midas SDK (`midas-memory`) with the MCP and
local-embedding extras and puts a `midas-memory-mcp` command on your PATH that any MCP client
(Claude Code, Cursor, Codex, Windsurf, …) can run.

## Install & run

```bash
uvx midas-memory-mcp          # run once, nothing to install
# or put the command on your PATH:
pipx install midas-memory-mcp
pip install midas-memory-mcp
```

## Wire it into a client (example: Claude Code)

```bash
claude mcp add midas -s user -- midas-memory-mcp
```

On connect, Midas injects a short memory policy into the agent and starts remembering on its own:
recall first, capture durable memory with provenance (`planning`, `action`, `observation`,
`user_confirmation`), and call `check_memory_use` before relying on memory for external/destructive
actions. Those actions require `user_confirmation`; otherwise the agent must ask the user to confirm.
Midas scores importance locally (no LLM), keeps what matters, revises typed stale beliefs, and drops
trivia and duplicates.

Optional environment variables: `MIDAS_MCP_DB` (SQLite path to persist across restarts),
`MIDAS_MCP_MAX_RECORDS`, `MIDAS_MCP_MIN_IMPORTANCE`, `MIDAS_MCP_EMBEDDER`,
`MIDAS_MCP_SUPERSEDE`, `MIDAS_MCP_SUPERSEDE_CONVO`, `MIDAS_MCP_NLI`, `MIDAS_MCP_ACTOR`.

For Cursor / Codex / Windsurf / Claude Desktop config, the Python SDK, reproducible benchmarks, and the
full eval methodology (anti-cheating checklist, failure cases, verbatim policy text), see the main repo:
**https://github.com/vornicx/Midas** — especially [`docs/methodology.md`](https://github.com/vornicx/Midas/blob/main/docs/methodology.md).

> Tip: if you installed the SDK directly (`uv tool install "midas-memory[mcp,local]"`), the same
> server is also available as the shorter `midas-mcp` command.

## License

Apache-2.0

<!-- mcp-name: io.github.vornicx/midas -->
