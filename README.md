<h1 align="center">Midas</h1>

<p align="center"><b>Durable memory for AI agents — no LLM at ingest, $0, fully local, source-traceable.</b></p>

<p align="center">
  <a href="https://github.com/vornicx/Midas/actions/workflows/ci.yml"><img src="https://github.com/vornicx/Midas/actions/workflows/ci.yml/badge.svg" alt="tests"></a>
  <a href="https://pypi.org/project/midas-memory/"><img src="https://img.shields.io/pypi/v/midas-memory" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/midas-memory-mcp"><img src="https://img.shields.io/npm/v/midas-memory-mcp?label=npm" alt="npm"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

<p align="center">
  <img src="docs/demo.gif" alt="Midas stores facts with no LLM, then recalls them by meaning in a later session — local, $0, source-traceable" width="820">
</p>

Your AI assistant forgets everything between sessions. **Midas is a memory that lives next to it, on
your machine** — for coding agents, research agents, assistants. It remembers the durable stuff
across long, multi-session work *without* sending every turn through an LLM to "extract" facts. It
costs nothing per message, nothing leaves your computer, and every recalled memory traces back to
the exact moment it came from.

```bash
uv tool install "midas-memory[mcp,local]"     # the midas-mcp command, for any MCP client
# or, no Python:  npx -y midas-memory-mcp       # TypeScript port
# or, as a library:  pip install "midas-memory[local]"
```

---

## Why it's different

Other memory tools call an LLM to summarize every session — you pay in tokens forever, in latency,
and by sending every turn to a provider, and recall returns *rewritten* facts you can't audit. Midas
makes the opposite bet:

- **No LLM at ingest or query** → **$0 API spend, zero data egress**, fast local ops (embed-bound, ~tens of ms — no per-turn network round-trip).
- **Source-traceable** → recall returns the **verbatim source turns**, not LLM-rewritten facts. No extraction step that can silently hallucinate.
- **Stays current & bounded, all no-LLM** → typed belief revision (the old value is superseded, not duplicated), selective forgetting with an audit trail, dedup, time tiers.
- **One file, many clients** → point several MCP apps at one SQLite file and they share live memory (more below).
- **Eval-first** → every claim has a reproducible benchmark, *including the experiments that failed*.

## How it does on the benchmarks

Deterministic, reader-independent **retrieval** (`recall@k` — fraction of the gold supporting turns
pulled into context) on the **full public sets**, vs a recency-window baseline:

| Benchmark (full set) | baseline | **Midas** |
|---|---:|---:|
| **LongMemEval-`s`** — 500 questions, 246,750 turns | 0.01 | **0.92** |
| **LoCoMo** — 10 conversations, n=1,540 | 0.05 | **0.73** |
| **BEAM** — frontier benchmark, 100K → **10M tokens** | 0.00 | **0.56 → 0.32** |

And the cross-system metric, **judged answer-rate** (same gpt-4o judge the leaderboards use):

| Judged answer | baseline | **Midas** |
|---|---:|---:|
| LongMemEval-`s` (gpt-4o reader, ties LLM-ingest SOTA at **$0 ingest**) | — | **0.84** |
| BEAM-100K (gpt-4o judge, raw-turn floor, $0 ingest) | 0.05 | **0.40** |

All of it at **0 LLM calls, $0, and 0 data egress** at ingest. Full numbers, per-category
breakdowns, reproduce commands, and the head-to-head framing vs Mem0/Zep/Mastra are in
**[BENCHMARKS.md](BENCHMARKS.md)**.

> **Eval-first means we publish the misses too.** Hybrid retrieval, reranking, thread-diversification,
> dual-granularity indexing, and *naive distillation* were all measured to **not** help (or to hurt)
> and are documented as such. That honesty is the point — see BENCHMARKS.md and
> [`docs/frontier-2026.md`](docs/frontier-2026.md).

---

## Connect it to your coding agent

Midas is a standard **MCP server**: every client launches the same `midas-mcp` command with a few env
vars — only *where* you put the config differs. The universal block:

```json
{
  "mcpServers": {
    "midas": {
      "command": "midas-mcp",
      "env": {
        "MIDAS_MCP_EMBEDDER": "local",
        "MIDAS_MCP_DB": "/home/you/.midas/memory.sqlite3",
        "MIDAS_MCP_MAX_RECORDS": "50000",
        "MIDAS_MCP_MIN_IMPORTANCE": "2"
      }
    }
  }
}
```

**Claude Code** (CLI, no file editing):

```bash
claude mcp add midas -s user \
  -e MIDAS_MCP_EMBEDDER=local -e MIDAS_MCP_DB="$HOME/.midas/memory.sqlite3" \
  -e MIDAS_MCP_MAX_RECORDS=50000 -e MIDAS_MCP_MIN_IMPORTANCE=2 -- midas-mcp
```

<details>
<summary><b>Cursor · Claude Desktop · Codex CLI · Windsurf · VS Code / Cline / Zed · npx</b> (click to expand)</summary>

| Client | Where the config goes |
|---|---|
| **Cursor** | `~/.cursor/mcp.json` (all projects) or `.cursor/mcp.json` — paste the JSON block |
| **Claude Desktop** | Settings → Developer → Edit Config (`claude_desktop_config.json`) — paste the block, restart |
| **Codex CLI** | `codex mcp add midas -- midas-mcp`, or a `[mcp_servers.midas]` block in `~/.codex/config.toml` (TOML) |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` — paste the block, refresh |
| **Anything else** | point it at command `midas-mcp` with those env vars |
| **No Python** | `npx -y midas-memory-mcp` — the [TypeScript port](packages/midas-ts), same tools/schema (experimental: no semantic embeddings yet) |

> ⚠️ **#1 gotcha:** GUI apps don't share your shell `PATH`. If a client says *"command not found"*, use
> the absolute path from `which midas-mcp` (macOS/Linux) or `where midas-mcp` (Windows). On Windows use
> forward slashes in JSON paths.

</details>

**Once connected**, Midas injects a short policy into the agent (*recall first, then capture durable
facts/decisions/preferences/constraints/corrections*). The agent captures freely; **Midas decides
what's kept** — it scores importance (no LLM), drops trivia, skips duplicates, revises stale beliefs,
and forgets the low-value tail to stay bounded. A **provenance guard** (`check_memory_use`) blocks
memory-justified external/destructive actions unless they came from explicit user confirmation.

### One memory, many clients

Point Claude Code, Claude Desktop, Cursor… at the **same** `MIDAS_MCP_DB` file and they share one
*live* memory — each detects the others' writes (SQLite `data_version`) and refreshes, so a fact
captured in your IDE is recallable from your chat app seconds later, no restarts. Scope it per
project/agent/user with `MIDAS_MCP_NAMESPACE`.

<p align="center">
  <img src="docs/demo-multi-client.gif" alt="Two live processes share one Midas SQLite file: a recall that finds nothing, a capture from a different process, then the same never-restarted session recalls it" width="820">
</p>

<p align="center"><sub>Real run, reconstructed chrome — the capture/recall lines are verbatim output of two separate processes sharing one file.</sub></p>

<details>
<summary><b>All tools & env knobs</b></summary>

**Tools:** `remember`, `capture` (policy-gated auto-store), `recall` (source-traceable),
`build_context` (compact, dated, today-anchored prompt block), `check_memory_use` (guard),
`memory_policy`, `maintain` (dedup + forgetting, returns a deletion audit), `stats`,
`forget` (chain-safe), `forget_matching` (topic-level erasure, dry-run by default), `forget_all`.
Prompts: `memory_session`, `distill`.

**Env:** `MIDAS_MCP_DB` · `MIDAS_MCP_EMBEDDER` (`local` / `hashing` / `multilingual` / any fastembed id)
· `MIDAS_MCP_MAX_RECORDS` · `MIDAS_MCP_MIN_IMPORTANCE` · `MIDAS_MCP_NAMESPACE` · `MIDAS_MCP_ANN=1`
(sub-linear IVF for huge stores) · `MIDAS_MCP_SUPERSEDE` · `MIDAS_MCP_NLI=1` (NLI-gated revision) ·
`MIDAS_MCP_AUTO_MAINTAIN=<min>` (idle-time upkeep) · `MIDAS_MCP_PINNED` (pin standing directives).

</details>

---

## Use it from Python (the SDK)

```python
from midas import Memory, LocalEmbedder

mem = Memory(embedder=LocalEmbedder())   # fully local. (Or Memory() for a zero-setup offline embedder.)

mem.remember("Decision: the primary database is PostgreSQL.", kind="constraint", importance=5)
mem.remember("The launch date moved to September 14.", kind="fact", importance=5)
mem.capture("lol ok cool")               # filler — auto-scored below the floor, skipped (no LLM)

mem.assemble("when do we launch?", token_budget=128)          # prompt-ready, dated, source-traceable
for hit in mem.recall("which database did we pick?", limit=3):
    print(f"{hit.score:.2f}  {hit.record.content}")           # each hit traces to its source
```

<details>
<summary><b>Belief revision · forgetting · namespaces · bitemporal · LangGraph · persistence</b></summary>

```python
from midas import Memory, LocalEmbedder
from midas.nli import LocalNLI
from midas.sqlite_store import SQLiteStore

# Durable, shareable, no native extension. Safe across threads & processes (live data_version refresh).
mem = Memory(store=SQLiteStore("memory.db"), embedder=LocalEmbedder(),
             supersede=True, nli=LocalNLI())   # a turn that CONTRADICTS an old belief supersedes it

mem.forget_decayed(max_records=50_000)         # evict lowest value (importance × recency); protects facts
mem.consolidate(similarity_threshold=0.95)     # collapse near-duplicate restatements (keeps provenance)
mem.recall("when is the launch?", as_of=1_700_000_000)   # bitemporal: "what did we believe on date X"

# Right-to-be-forgotten — preview, then erase, with an audit trail:
mem.forget_matching("the user's home address", dry_run=True)
mem.forget_matching("the user's home address")

# Scoped memory: one store, many projects/users, no cross-talk:
mem.recall("api gateway", metadata_filter={"namespace": "proj-a"})

# Back LangGraph's long-term memory with Midas:
from midas.integrations.langgraph_store import MidasStore
store = MidasStore(); store.put(("user", "123"), "pref", {"text": "prefers dark mode"})
```

</details>

---

## Honest status

Midas is **early** (the API may change) but built narrow and measured-first. Where it stands, plainly:

- **Retrieval is its strength and is essentially maxed** for a no-LLM design — confirmed by our own
  A/Bs *and* by the frontier papers (the retriever is not the bottleneck). The numbers above are the
  result.
- **Distillation (turning raw turns into compact facts) is the frontier's extra lever — and a *naive*
  pass does not help here.** We built it, judged it on BEAM, and measured that replacing raw turns
  with summarized facts is *catastrophic* (it drops the temporal/changed-value detail), while
  augmenting is roughly neutral. So the optional distillation dial ships **off by default**, and we
  don't claim it as a win. The real lift needs sophisticated, structure-preserving extraction —
  open work. (Details: [`docs/frontier-2026.md`](docs/frontier-2026.md) §2b.)
- **Next:** a no-LLM *self-learning* recall policy aimed at precision — the one untested lever left.

## The eval harness

`eval/` (dev-only) runs Midas and competitors through synthetic / LoCoMo / LongMemEval / multiday /
conflicts-v1 / **BEAM** with deterministic `recall@k` + `precision@k`, cost/latency instrumentation, a
**dumb-reader ablation** (proves the numbers aren't reader-inflated), and an optional local-or-hosted
LLM judge. The anti-cheating checklist (no query rewriting, no LLM at ingest, no gold leakage, seeded
sampling), conflict handling, failure traces, and the verbatim MCP policy are in
[`docs/methodology.md`](docs/methodology.md).

```bash
python -m eval.runner --dataset longmemeval --variant s --local --midas-no-rerank --max-questions 40
python -m eval.runner --dataset beam --beam-tier 100K --local --dumb-reader   # frontier benchmark
```

## Privacy & license

Local-first: every memory lives in a SQLite file on your machine, recall returns the exact stored
text, and capture/recall/forget make **no network calls**. No account, API key, or telemetry. The
only outbound traffic is a one-time embedding-model download (for the `local` backend) and the
package install. Full details in [`PRIVACY.md`](PRIVACY.md) · [MIT](LICENSE).
