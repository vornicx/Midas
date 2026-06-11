# midas-memory-mcp (TypeScript) — experimental

A TypeScript port of [Midas](https://github.com/vornicx/Midas): local-first, source-traceable
memory for AI agents over MCP — **no LLM and no network at ingest or query**.

> **Status: experimental.** The Python server (`pip install "midas-memory[mcp,local]"`) is the
> reference implementation with semantic ONNX embeddings, NLI-gated belief revision, and the full
> eval harness behind it. This port covers the core for Node-first setups.

## Why it exists

Most MCP clients live in the Node ecosystem. This package gives them a zero-Python install:

```bash
npx midas-memory-mcp        # or: npm i -g midas-memory-mcp && midas-mcp
```

```json
{
  "mcpServers": {
    "midas": {
      "command": "npx",
      "args": ["-y", "midas-memory-mcp"],
      "env": { "MIDAS_MCP_DB": "/home/you/.midas/memory.sqlite3" }
    }
  }
}
```

## Parity with the Python server

- **Same SQLite schema and float32 blob encoding** — a TS server and a Python server can point at
  the **same DB file** and share one live memory (both probe SQLite's `data_version` and refresh
  on other connections' writes). Verified bidirectionally in tests.
- **Bit-comparable hashing embedder** (md5 token hashing, identical sign/index math) — vectors
  written by one runtime are recalled semantically by the other. Parity is pinned by a fixture
  generated from the Python implementation.
- **Same tool surface** — `remember`, `capture` (policy-gated, no LLM), `recall`
  (source-traceable), `build_context` (lean dated lines + "Today is" anchor), `inspect_memory`,
  `check_memory_use` (provenance guard), `forget`, `forget_matching` (dry-run + audit),
  `forget_all`, `maintain`, `stats`, `memory_policy` — plus the same env knobs
  (`MIDAS_MCP_DB`, `MIDAS_MCP_MAX_RECORDS`, `MIDAS_MCP_MIN_IMPORTANCE`, `MIDAS_MCP_NAMESPACE`,
  `MIDAS_MCP_ACTOR`, `MIDAS_MCP_SUPERSEDE`) and the same injected agent policy text.
- **Same shipping behaviour** — relevance × importance × recency ranking, the measured-safe
  scale-free parsimony floor (`minRelevanceRatio` 0.3), BM25+RRF hybrid (cached index), typed
  belief revision with supersession chains, no-LLM importance scoring, selective forgetting with
  durable-tier protection.

## Not ported (yet)

- **Semantic ONNX embeddings** (bge / multilingual) — the default embedder here is the offline
  hashing one (lexical-ish). For real semantic recall today, run the Python server; both can share
  the same DB.
- Local NLI contradiction gating, the cross-encoder reranker, and the eval harness.

## Requirements

Node **>= 22.5** (uses the built-in `node:sqlite`; you may see its experimental warning on stderr).

```bash
npm test   # builds + runs the suite, including the Python-parity fixture
```

MIT — same license, same repo, same [PRIVACY.md](../../PRIVACY.md) posture: everything stays on
your machine.
