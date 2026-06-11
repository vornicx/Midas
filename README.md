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

**See it remember across sessions** — session 1 stores decisions; a fresh session 2 recalls them *by
meaning*:

<p align="center">
  <img src="docs/demo-claude-code.gif" alt="Across two sessions: Midas stores decisions in session 1, and a fresh session 2 recalls them by meaning" width="820">
</p>

<p align="center"><sub>Claude Code-style demo — the recalled lines (in green) are the real output Midas returned across two separate processes sharing one on-disk store.</sub></p>

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

### No Python? `npx` it (TypeScript port, experimental)

A Node-native port lives at [`packages/midas-ts`](packages/midas-ts): `npx -y midas-memory-mcp`
with the **same tools, env knobs, injected policy, and SQLite schema** — a TS server and a Python
server can even share one DB file live (bit-comparable hashing embedder; verified both ways in
tests). Caveat: no semantic ONNX embeddings yet — the Python server stays the reference.

### What happens once it's connected

On connect, Midas **injects a short memory policy into the agent** (via the MCP `instructions`): *recall
relevant memory first, then `capture` durable facts / decisions / preferences / constraints /
corrections as they come up.* Every captured memory is tagged with provenance:
`planning`, `action`, `observation`, or `user_confirmation`. The agent captures freely; **Midas decides
what's actually kept** — it scores importance (no LLM), drops trivia below `MIDAS_MCP_MIN_IMPORTANCE`
and skips duplicates, keeps memory current via typed belief revision, and keeps memory bounded via
`MIDAS_MCP_MAX_RECORDS` (forgetting low-value items, protecting durable facts). Restart the client (or
run `/mcp`) after editing config so it picks up the server.

**Guard boundary:** memory can guide planning, but it cannot by itself authorize external or destructive
actions. Before relying on memory to act outside the chat, call `check_memory_use` with
`intended_use="external_action"` or `"destructive_action"`. Those actions require
`user_confirmation` provenance; otherwise the agent must ask the user to confirm in the current turn.

**One memory, many clients.** Point Claude Code, Claude Desktop, Cursor, etc. at the **same**
`MIDAS_MCP_DB` file and they share one live memory: each server detects the others' writes (SQLite
`data_version`) and refreshes, so a fact captured in your IDE is recallable from your chat app
moments later — no restarts. Use `MIDAS_MCP_NAMESPACE` (or the per-call `namespace` argument every
tool accepts) to keep projects, agents, or users scoped inside that shared DB.

<p align="center">
  <img src="docs/demo-multi-client.gif" alt="Two live processes share one Midas DB: a recall that finds nothing, a capture from a different process, then the same never-restarted session recalls it" width="820">
</p>

<p align="center"><sub>Real run, reconstructed chrome: the recall/capture lines are the verbatim output of two separate processes sharing one SQLite file — the second recall succeeds without restarting anything.</sub></p>

**Tools it exposes:** `remember`, `capture` (policy-gated auto-store), `recall` (source-traceable),
`build_context` (compact budgeted prompt block, dated and anchored to today so the agent can resolve
relative time; use `recall`/`inspect_memory` for full provenance), `check_memory_use` (Guard provenance boundary), `memory_policy` (exact injected
policy text), `maintain` (dedup + forgetting, returns a deletion audit), `stats` (counts +
provenance + short/medium/long tiers + namespaces), `forget` (chain-safe single delete),
`forget_matching` (topic-level erasure: dry-run preview by default, then delete with a full audit),
`forget_all`. **Env knobs:**
`MIDAS_MCP_DB` (persist to a SQLite file), `MIDAS_MCP_EMBEDDER` (`local`, `hashing`,
`multilingual` — for non-English memory, where the English-only default silently degrades — or any
fastembed model id), `MIDAS_MCP_MAX_RECORDS`, `MIDAS_MCP_MIN_IMPORTANCE`, `MIDAS_MCP_NAMESPACE`
(default scope for this server's reads/writes), `MIDAS_MCP_ANN=1` (sub-linear IVF search for very
large stores), `MIDAS_MCP_SUPERSEDE=0` to disable typed belief
revision, `MIDAS_MCP_SUPERSEDE_CONVO=1` to allow strict-cue chat revision, `MIDAS_MCP_NLI=1` to gate
revision with the local NLI model.

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

# Provenance guard: observed memory is fine for planning, but not enough to deploy.
mem.remember("Deploy target is staging.", kind="constraint", provenance="observation")
decision = mem.guard_reliance("deploy target", intended_use="external_action")
assert not decision.allowed  # ask the user to confirm before acting
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

# Topic-level erasure (right-to-be-forgotten): preview, then delete — returns the audit trail.
mem.forget_matching("the user's home address", dry_run=True)   # what WOULD be deleted
mem.forget_matching("the user's home address")                 # delete it (bypasses protections)

# Scoped memory: share one store across projects/users without cross-talk.
mem.remember("the gateway is Kong", kind="fact", metadata={"namespace": "proj-a"})
mem.recall("api gateway", metadata_filter={"namespace": "proj-a"})
```

Forgetting returns the removed ids as a **deletion audit trail** and never drops the durable tier
(facts/preferences/constraints, high importance) — while `forget_matching` deliberately bypasses
those protections, because an explicit erasure request outranks retention. **Durable storage:**
`Memory(store=SQLiteStore("memory.db"), embedder=LocalEmbedder())` — a local file, no native
extension, safe to share across threads and processes (writers are lock-guarded; other processes'
writes are picked up live via SQLite's `data_version`).

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
reproduce commands in [BENCHMARKS.md](BENCHMARKS.md); anti-cheating checklist, failure cases, and
verbatim MCP policy in [docs/methodology.md](docs/methodology.md)):

| | baseline (recency window) | **Midas** |
|---|---:|---:|
| **Retrieval** — LongMemEval-`s` recall@k (FULL set: 500 questions, 246,750 turns) | 0.01 | **0.92** |
| **Retrieval** — LoCoMo recall@k (full public set: 10 conversations, n=1,540) | 0.05 | **0.73** |
| **Answer** — LongMemEval-`s` correctness (reader = gpt-4.1-mini, n=40) | 0.05 | **0.82** |
| **Ingest cost** | — | **0 LLM calls · $0 API · 0 data egress** |

We lead with **retrieval and cost** (deterministic, reader-independent) because end-to-end correctness on
these benchmarks is dominated by the *reader* LLM, not the memory layer. **Head-to-head, same reader:**
with `gpt-4o`, Midas scores **0.84** on LongMemEval-`s` — **matching** the LLM-ingest SOTA (Observational
Memory) while doing **no LLM at ingest** — and on a ~500-session haystack (~4,944 turns) it assembles a
bounded ~480-token context (recall@k 0.78), where keep-every-observation-in-context designs do not fit
by construction. (Same-reader, within-harness comparison — not a leaderboard rank; see BENCHMARKS.md.)

### Where Midas stands

Compared by *architecture class*, since that is what fixes the cost/privacy/auditability structure
(examples of each class: Mem0, Mastra Observational Memory, Hindsight — LLM-at-ingest; Zep —
hosted graph memory):

| | **Midas** | LLM-at-ingest memory | hosted memory services |
|---|---|---|---|
| LLM calls at ingest/query | **0** | ≥1 per session | varies |
| Marginal cost per message | **$0** | $/token, forever | subscription + egress |
| Conversation leaves the box | **never** | yes (to the LLM) | yes (to the service) |
| Recall traceable to source turns | **yes, verbatim** | no (LLM-rewritten facts) | partial |
| Extraction can hallucinate | **no extraction step** | yes, silently | yes, at ingest |
| Bounded memory (selective forgetting + audit) | **yes, no LLM** | LLM compaction | provider-managed |
| Action guard on provenance | **yes** (`check_memory_use`) | — | — |

The honest trade: LLM-at-ingest systems buy *curated observations* with those tokens, which helps
the strongest readers squeeze out a few more points on benchmarks (OM 0.95 vs Midas 0.87–0.89 with
gpt-5-mini). Midas's bet is that **$0/message, zero egress, and auditable recall** is the right
default for a memory that runs forever next to your agent.

## The eval harness

`eval/` (dev-only) runs Midas and competitors through LoCoMo / LongMemEval / **multiday** /
**conflicts-v1** with deterministic `recall@k` and `precision@k`, cost/latency instrumentation, an
optional local-or-hosted LLM judge, a deterministic **dumb-reader ablation** (`--dumb-reader` — proves
the numbers aren't reader-inflated), an **adversarial conflicts benchmark** (near-duplicates +
temporal conflicts), and a retention/forgetting measure with per-question success/failure traces:

```bash
python -m eval.runner --dataset longmemeval --variant s --local --midas-no-rerank --max-questions 40
python -m eval.runner --dataset longmemeval --variant s --local --dumb-reader --max-questions 40
python -m eval.runner --dataset multiday --dumb-reader                    # ctx_stale on leaderboard
python -m eval.runner --dataset conflicts --dumb-reader --midas-supersede
python -m eval.multiday --dataset conflicts --context-only --ab-supersede --midas-only
python -m eval.retention --dataset multiday --trace
python -m eval.retention --dataset multiday --trace --value-rank-only     # forgetting failure mode
```

How the eval avoids the usual memory-stack cheats (no query rewriting, no LLM at ingest, no gold
leakage, seeded sampling), how conflicting memories are handled, and the exact MCP-injected policy
text — with real failure cases — are documented in [`docs/methodology.md`](docs/methodology.md).

## Design concept

[`docs/long-horizon-memory.md`](docs/long-horizon-memory.md) — the north-star: the **4 C's**
(Complete · Clean · Current · Calibrated), why multi-day accuracy is a *belief-management* problem, and
the honest, measured state of each piece (including the open frontiers).

[`docs/methodology.md`](docs/methodology.md) — how the eval avoids the usual memory-stack cheats,
the dumb-reader ablation, conflicts-v1 stress tests, forgetting failure traces, supersession mechanics,
and the exact MCP-injected policy text (for external review / Reddit-style scrutiny).

## Layout

```
midas/      # the SDK (importable; zero core dependencies)
  memory.py       # Memory: remember / capture / recall / build_context · forget_decayed · consolidate · tier
  guard.py        # Guard + Armorer: provenance tags · check_memory_use policy boundary
  importance.py   # ContentImportance — no-LLM per-turn salience   ·   policy.py — MemoryPolicy + auto-memory prompt
  nli.py          # LocalNLI — local entailment/contradiction (belief revision + abstention)
  embeddings.py   # Hashing / Local (bge) / OpenAI · DiskCachedEmbedder · LocalReranker
  store.py · sqlite_store.py · ann.py   # in-memory cosine · persistent SQLite · IVF index
  mcp_server.py   # the MCP server
eval/       # dev-only benchmark harness (datasets · adapters · metrics · runner · multiday · retention)
docs/       # long-horizon-memory.md (design) · methodology.md (eval anti-cheating) · research-notes.md
```

## Privacy

Midas is local-first: every memory lives in a SQLite file on your own machine, recall returns the
exact stored text, and capture/recall/forget make **no network calls** — your memories never leave
your computer. The developer collects no data; there is no account, API key, or telemetry. The only
outbound traffic is infrastructure (a one-time embedding-model download for the `local` backend, and
package install from PyPI), never your data. Full details: [`PRIVACY.md`](PRIVACY.md).

## License

[MIT](LICENSE).
