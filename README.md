# Midas

**Local-first, eval-first memory for long-horizon AI agents — no LLM at ingest.**

[![tests](https://github.com/vornicx/Midas/actions/workflows/ci.yml/badge.svg)](https://github.com/vornicx/Midas/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/midas-memory)](https://pypi.org/project/midas-memory/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<p align="center">
  <img src="docs/demo.gif" alt="Midas remembers facts with no LLM, then recalls them later by meaning — local and source-traceable" width="800">
</p>

Midas is a small Python SDK (and an MCP server) that gives AI agents durable memory across long,
multi-session work — coding agents, research agents, assistants — *without* sending every turn through
an LLM to "extract" facts. It runs on your machine, costs nothing per message, and every recalled memory
is traceable to its source.

- **No LLM at ingest or query** → **$0 API spend, zero data egress**, fast local ops (no per-turn network round-trip; ingest is embed-bound, ~tens of ms).
- **Auditable provenance** → recall returns the **source turns**, not LLM-rewritten facts.
- **Stays current and bounded** → belief revision, selective forgetting + tiers, dedup — all no-LLM.
- **Embeddable + store-agnostic** → a library, not a SaaS. Bring your own embedder/store.
- **Eval-first** → every claim has a reproducible benchmark ([BENCHMARKS.md](BENCHMARKS.md)).

> **Status:** early. The API may change. Built narrow and measured-first.

---

## How it works (in plain English)

Your AI assistant forgets everything between sessions — every new chat starts from zero. Midas is a
**memory that lives next to your AI, on your computer.** It does four simple things:

1. **Notices what matters.** As you work, Midas saves the durable stuff — a decision, a fact about you, a
   preference, a deadline — and ignores small talk. It judges *"does this matter?"* by reading the words
   (names, numbers, dates make a turn important) — **without calling another AI**.
2. **Hands the right notes back.** Before the AI answers, Midas finds the handful of past notes related
   to your question — by **meaning**, not exact keywords — and slips them into the prompt.
3. **Keeps the notebook honest and tidy.** When something changes ("actually, use Postgres now") it
   **updates** the old note instead of keeping both; it **merges duplicates**; and it **forgets** old,
   unimportant trivia so memory never bloats.
4. **Stays yours.** Everything is a local file — no cloud, no per-message AI bill — and every note links
   back to the exact moment it came from, so you can always check *why* the AI "knows" something.

The trick that makes it cheap, private, and local: Midas never sends your conversation to an AI to
"process" it. It uses fast local math (*embeddings* — turning text into vectors and comparing them). The
only AI involved is the one you're already talking to.

**Why "no LLM at ingest" matters:** other memory tools call an LLM to summarize every session — you pay
in tokens forever, in latency, and by sending every turn to a provider. Midas trades that for cheap,
local, auditable retrieval.

---

## Install

You need **Python 3.11+**. Check with `python --version` (or `python3 --version`). If you don't have it:
[python.org/downloads](https://www.python.org/downloads/), or `winget install Python.Python.3.12`
(Windows) · `brew install python@3.12` (macOS) · your package manager (Linux). The easiest installer for
everything below is [**uv**](https://docs.astral.sh/uv/) (one line: see its site), but `pip`/`pipx` work
too.

### A) To plug Midas into an AI tool (Claude Code, Cursor, …) — install the `midas-mcp` command

This puts a `midas-mcp` program on your PATH that any MCP client can launch — **one line, no clone**:

```bash
uv tool install "midas-memory[mcp,local]"     # recommended (Windows, macOS, Linux)
# …or:  pipx install "midas-memory[mcp,local]"
```

> **No-install option:** Midas is also on the [official MCP registry](https://registry.modelcontextprotocol.io/)
> as `io.github.vornicx/midas`. Clients that read the registry can discover it automatically, or you can
> launch the same server on demand with `uvx midas-memory-mcp` — nothing to install.

Where the command lands (you'll need this path for some clients):

| OS | `midas-mcp` location | Find it with |
|---|---|---|
| **Linux / macOS** | `~/.local/bin/midas-mcp` | `which midas-mcp` |
| **Windows** | `%USERPROFILE%\.local\bin\midas-mcp.exe` | `where midas-mcp` |

### B) To use Midas as a Python library

```bash
pip install "midas-memory[all]"     # SDK + local embeddings + MCP + LangGraph
# smaller: `pip install midas-memory` (core, zero deps) · `"…[local]"` (embeddings) · `"…[mcp]"`
```

*(Want the source / to contribute? `git clone https://github.com/vornicx/Midas && cd Midas && pip
install -e ".[all,dev]"`.)*

> **First run** downloads the embedding model once (~90 MB, `bge-base` ONNX), then works **fully
> offline**. No API key, ever.

**Verify:**

```bash
which midas-mcp || where midas-mcp                       # the server command is installed
python -c "import midas; print('Midas', midas.__version__, 'OK')"
python quickstart.py                                     # tiny end-to-end demo: remember → recall
```

---

## Connect it to your coding agent

Midas is a standard **MCP server**. Every MCP client launches the same command — `midas-mcp` — and
passes a few environment variables. **The only thing that differs between tools is *where* you put the
config.** Use this block everywhere (swap in your real home path):

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

> ⚠️ **The #1 gotcha:** GUI apps don't share your terminal's `PATH`, so they may not find `midas-mcp`.
> If a client says *"command not found"*, replace `"command": "midas-mcp"` with the **absolute path**
> from `which midas-mcp` (macOS/Linux) or `where midas-mcp` (Windows, e.g.
> `"C:/Users/you/.local/bin/midas-mcp.exe"` — use forward slashes or `\\` in JSON). On Windows, write the
> DB path with forward slashes too: `C:/Users/you/.midas/memory.sqlite3`.

### Claude Code

Use the CLI (no file editing) — this is the exact command, verified:

```bash
claude mcp add midas -s user \
  -e MIDAS_MCP_EMBEDDER=local \
  -e MIDAS_MCP_DB="$HOME/.midas/memory.sqlite3" \
  -e MIDAS_MCP_MAX_RECORDS=50000 \
  -e MIDAS_MCP_MIN_IMPORTANCE=2 \
  -- midas-mcp

claude mcp list        # → midas: midas-mcp - ✓ Connected
```

`-s user` = available in **all** your projects · `-s project` = writes a shareable `.mcp.json` in the
repo · `-s local` = just you, this project. Remove with `claude mcp remove midas -s user`.

### Cursor

Edit **`~/.cursor/mcp.json`** (all projects) or **`.cursor/mcp.json`** (this project) and paste the JSON
block above. Then Cursor → **Settings → MCP** should show `midas`. Restart Cursor after changing `env`.

### Claude Desktop

**Settings → Developer → Edit Config** opens the file (or edit it directly):

| OS | Path |
|---|---|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

Paste the JSON block, save, and **restart Claude Desktop**.

### Codex CLI

Codex uses **TOML**, not JSON. Either run `codex mcp add midas -- midas-mcp`, or add this to
**`~/.codex/config.toml`**:

```toml
[mcp_servers.midas]
command = "midas-mcp"
args = []
env = { MIDAS_MCP_EMBEDDER = "local", MIDAS_MCP_DB = "/home/you/.midas/memory.sqlite3", MIDAS_MCP_MAX_RECORDS = "50000", MIDAS_MCP_MIN_IMPORTANCE = "2" }
```

Start a session and run **`/mcp`** to confirm it's connected.

### Windsurf

Edit the config (Cascade → **MCP icon → Configure** opens it), paste the JSON block, refresh:

| OS | Path |
|---|---|
| macOS / Linux | `~/.codeium/windsurf/mcp_config.json` |
| Windows | `%USERPROFILE%\.codeium\windsurf\mcp_config.json` |

### Anything else (VS Code, Cline, Zed, OpenAI Agents SDK…)

Same pattern: point it at command `midas-mcp` with those env vars (JSON clients reuse the block above).

### What happens once it's connected

On connect, Midas **injects a short memory policy into the agent** (via the MCP `instructions`): *recall
relevant memory first, then `capture` durable facts / decisions / preferences / constraints /
corrections as they come up.* The agent captures freely; **Midas decides what's actually kept** — it
scores importance (no LLM), drops trivia below `MIDAS_MCP_MIN_IMPORTANCE` and skips duplicates, and keeps
memory bounded via `MIDAS_MCP_MAX_RECORDS` (forgetting low-value items, protecting durable facts).
Restart the client (or run `/mcp`) after editing config so it picks up the server.

**Tools it exposes:** `remember`, `capture` (policy-gated auto-store), `recall` (source-traceable),
`build_context` (budgeted prompt block), `maintain` (dedup + forgetting, returns a deletion audit),
`stats` (counts + short/medium/long tiers), `forget` / `forget_all`. **Env knobs:** `MIDAS_MCP_DB`
(persist to a SQLite file), `MIDAS_MCP_EMBEDDER` (`local` or `hashing`), `MIDAS_MCP_MAX_RECORDS`,
`MIDAS_MCP_MIN_IMPORTANCE`.

---

## Use it from Python (the SDK)

```python
from midas import Memory, LocalEmbedder, ContentImportance

# Real semantic memory, fully local. (Or just `Memory()` for a zero-setup offline hashing embedder.)
mem = Memory(embedder=LocalEmbedder(), importance_scorer=ContentImportance())

mem.remember("Decision: the primary database is PostgreSQL.", kind="constraint", importance=5)
mem.remember("The launch date moved to September 14.", kind="fact", importance=5)
mem.remember("haha yeah sounds good")  # filler — auto-scored low-importance, first to be forgotten

# Budgeted, prompt-ready context — highest-value first, dated, source-traceable:
print(mem.assemble("When do we launch?", token_budget=128))

# Or structured, ranked hits, each traceable to its source:
for hit in mem.recall("which database did we pick?", limit=3):
    print(f"{hit.score:.2f}  {hit.record.content}")

# Auto-capture: forward a turn; Midas keeps it only if it clears the relevance policy (no LLM).
mem.capture("My deploy key expires on 2027-03-01.", kind="fact")   # -> stored
mem.capture("lol ok cool")                                          # -> skipped (below the floor)
```

### Staying current and bounded — the long-horizon core

A multi-day agent's memory must stay **current** (no stale beliefs) and **bounded** (can't grow forever):

```python
from midas.nli import LocalNLI

# Belief revision — a turn that CONTRADICTS an old belief supersedes it (local NLI, not keywords):
mem = Memory(embedder=LocalEmbedder(), supersede=True, supersede_conversational=True, nli=LocalNLI())

mem.forget_decayed(max_records=50_000)      # evict lowest value (importance × recency); protects facts
mem.consolidate(similarity_threshold=0.95)  # collapse near-duplicate restatements (keeps provenance)
mem.tier(record)                            # 'short' (≤1d) | 'medium' (≤1w) | 'long'
```

Forgetting returns the removed ids as a **deletion audit trail** and never drops the durable tier
(facts/preferences/constraints, high importance). **Durable storage:** `Memory(store=SQLiteStore(
"memory.db"), embedder=LocalEmbedder())` — a local file, no native extension.

### Use with LangGraph

Back LangGraph's long-term memory with Midas (`pip install ".[langgraph]"`):

```python
from midas.integrations.langgraph_store import MidasStore

store = MidasStore()  # offline by default; pass Memory(embedder=LocalEmbedder(), ...) for semantic
store.put(("user", "123"), "pref", {"text": "prefers dark mode and concise answers"})
hits = store.search(("user", "123"), query="ui preferences")
```

---

## Benchmarks

Midas leads on the **reader-independent** axes that isolate a memory layer's quality (full methodology +
reproduce commands in [BENCHMARKS.md](BENCHMARKS.md)):

| | baseline (recency window) | **Midas** |
|---|---:|---:|
| **Retrieval** — LongMemEval-`s` recall@k (evidence buried among distractors, n=40) | 0.03 | **0.95** |
| **Retrieval** — LoCoMo recall@k (5 conversations, n=50) | 0.02 | **0.85** |
| **Answer** — LongMemEval-`s` correctness (reader = gpt-4.1-mini, n=40) | 0.05 | **0.82** |
| **Ingest cost** | — | **0 LLM calls · $0 API · 0 data egress** |

We lead with **retrieval and cost** (deterministic, reader-independent) because end-to-end correctness on
these benchmarks is dominated by the *reader* LLM, not the memory layer. **Head-to-head, same reader:**
with `gpt-4o`, Midas scores **0.84** on LongMemEval-`s` — **matching** the LLM-ingest SOTA (Observational
Memory) while doing **no LLM at ingest** — and on a ~500-session haystack (~4,944 turns) it assembles a
bounded ~480-token context (recall@k 0.78), where keep-every-observation-in-context designs do not fit
by construction. (Same-reader, within-harness comparison — not a leaderboard rank; see BENCHMARKS.md.)

## The eval harness

`eval/` (dev-only) runs Midas and competitors through LoCoMo / LongMemEval with deterministic `recall@k`,
cost/latency instrumentation, an optional local-or-hosted LLM judge, and a retention/forgetting measure:

```bash
python -m eval.runner --dataset longmemeval --variant s --local --midas-no-rerank --max-questions 15
python -m eval.retention --dataset locomo --max-convs 1 --local --derive-importance
```

## Design concept

[`docs/long-horizon-memory.md`](docs/long-horizon-memory.md) — the north-star: the **4 C's**
(Complete · Clean · Current · Calibrated), why multi-day accuracy is a *belief-management* problem, and
the honest, measured state of each piece (including the open frontiers).

## Layout

```
midas/      # the SDK (importable; zero core dependencies)
  memory.py       # Memory: remember / capture / recall / build_context · forget_decayed · consolidate · tier
  importance.py   # ContentImportance — no-LLM per-turn salience   ·   policy.py — MemoryPolicy + auto-memory prompt
  nli.py          # LocalNLI — local entailment/contradiction (belief revision + abstention)
  embeddings.py   # Hashing / Local (bge) / OpenAI · DiskCachedEmbedder · LocalReranker
  store.py · sqlite_store.py · ann.py   # in-memory cosine · persistent SQLite · IVF index
  mcp_server.py   # the MCP server
eval/       # dev-only benchmark harness (datasets · adapters · metrics · runner · retention)
```

## License

[MIT](LICENSE).
