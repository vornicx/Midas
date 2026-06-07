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

On connect, Midas injects a short memory policy into the agent and starts remembering on its own —
it scores importance locally (no LLM), keeps what matters, and drops trivia and duplicates.

Optional environment variables: `MIDAS_MCP_DB` (SQLite path to persist across restarts),
`MIDAS_MCP_MAX_RECORDS`, `MIDAS_MCP_MIN_IMPORTANCE`, `MIDAS_MCP_EMBEDDER`.

For Cursor / Codex / Windsurf / Claude Desktop config, the Python SDK, and reproducible benchmarks,
see the main repo: **https://github.com/vornicx/Midas**

> Tip: if you installed the SDK directly (`uv tool install "midas-memory[mcp,local]"`), the same
> server is also available as the shorter `midas-mcp` command.

## License

MIT

<!-- mcp-name: io.github.vornicx/midas -->
