<h1 align="center">Midas</h1>

<p align="center"><b>The local memory layer for long-horizon AI agents — remembers across sessions, keeps what's current, and won't act on stale memory.</b><br/>No LLM at ingest · $0 per message · fully local · every recall traces to its source.</p>

<p align="center">
  <a href="https://github.com/vornicx/Midas/actions/workflows/ci.yml"><img src="https://github.com/vornicx/Midas/actions/workflows/ci.yml/badge.svg" alt="tests"></a>
  <a href="https://pypi.org/project/midas-memory/"><img src="https://img.shields.io/pypi/v/midas-memory" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/midas-memory-mcp"><img src="https://img.shields.io/npm/v/midas-memory-mcp?label=npm" alt="npm"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License: Apache-2.0"></a>
</p>

Your AI assistant forgets everything between sessions. **Midas is the memory that lives next to it, on
your machine.** Your coding agent remembers the decisions, conventions, and bugs from three sessions
ago — without piping every message through an LLM to "extract" facts. It costs **nothing per message**,
**nothing leaves your computer**, every memory **traces back to the exact turn it came from**, and it
**won't let an agent act on memory that's stale or never confirmed**.

```bash
uv tool install "midas-memory[mcp,local]"   # install
midas init                                  # create the shared memory + wire up your MCP clients
# or, no Python:    npx -y midas-memory-mcp     # TypeScript port
# or, as a library: pip install "midas-memory[local]"
```

<p align="center">
  <a href="#connect-it-to-your-coding-agent"><b>Install in your agent</b></a>
  ·
  <a href="#how-it-does-on-the-benchmarks"><b>See the benchmarks</b></a>
  ·
  <a href="docs/MIDAS.md"><b>Complete overview</b></a>
  ·
  <a href="https://github.com/vornicx/Midas/issues/new?title=Team%2FEnterprise%20Midas%20conversation"><b>Team / Enterprise</b></a>
</p>

---

## Why Midas

Most memory tools call an LLM to summarize every session — so you pay in tokens forever, add latency,
ship every turn to a provider, and get back *rewritten* facts you can't audit. Midas makes the opposite
bet, and that bet is what makes it cheap, private, and trustworthy:

- **$0 and private by construction.** No LLM at ingest or query → no API spend, nothing leaves your
  machine, fast local ops (~tens of ms, no per-turn network round-trip).
- **You can trust what it recalls.** Recall returns the **verbatim source turn**, not an LLM rewrite —
  so there's no extraction step that can silently hallucinate a "fact" you never said.
- **It stays current on its own.** Typed belief revision supersedes the old value instead of piling up
  duplicates; selective forgetting keeps it bounded — all with no LLM.
- **It's safe to build on.** A provenance **guard** lets memory inform planning but **blocks
  memory-justified external or destructive actions** unless you explicitly confirmed them — and a
  *superseded* memory can't authorize an action at all.
- **One file, many tools.** Point Claude Code, Cursor, and your chat app at one SQLite file and they
  share one live memory.
- **Proven, not asserted.** Every claim has a reproducible benchmark — *including the experiments that
  failed.*

## How Midas compares

Every Midas number below is measured and reproducible from this repo; the LLM-at-ingest column
reflects the structural properties of that design class (Mem0, Zep, Hindsight) and the figures
documented in [BENCHMARKS.md](BENCHMARKS.md).

|  | **Midas** | LLM-at-ingest systems (Mem0, Zep, Hindsight) |
|---|---|---|
| LLM calls at ingest | **0** | ≥1 per session |
| Cost per message | **$0** | per-token API spend, forever |
| Data egress at ingest | **None** | every turn leaves the box |
| Ingest latency | **~16–116 ms**, local, embed-bound | ~668 ms + API round-trip |
| Recall returns | **verbatim source turn**, traceable | LLM-rewritten facts (source `recall@k` not computable) |
| Deterministic & reproducible | **yes — every number, one command** | no |
| Works fully offline | **yes** (measured end-to-end with a local Ollama reader) | no |
| LongMemEval-`s` judged answer (gpt-4o) | **0.84** | 0.84 — Observational Memory, with LLM ingest |
| Whole-conversation aggregation / summarization | ❌ **by design** — top-k retrieval can't cover it ([documented](BENCHMARKS.md)) | ✅ their structural edge |

The last row is deliberate: Midas trades whole-conversation abilities for $0, privacy, and
auditability, and publishes the measurements that show exactly where that trade bites.

## More than recall: a memory you can govern

Finding a buried fact is table stakes. A long-horizon coding agent needs memory it can **act on
safely** and **resume from cleanly** — which is where similarity search alone falls short:

| You ask… | Midas answers with | Why top-k recall can't |
|---|---|---|
| *"Can I run this destructive migration?"* | **Guard**: allowed only if **you** confirmed it, and only if that confirmation is still current | provenance + currency aren't a similarity match |
| *"What's the current state of project Apollo?"* | **`memory_state`**: the live, non-superseded decisions / constraints / facts | a broad "current state" query matches no single turn |
| *"What changed since our last session?"* | **`memory_diff`**: beliefs added, and beliefs revised (old → new) | "what's new" isn't a content query at all |
| *"How do I speed up the transactions list?"* | the **prior fix** resurfaces, so the agent doesn't re-diagnose it | — |

These properties are measured, not asserted — the **[agent-memory bench suite](docs/agent-memory-benches.md)**
scores action-safety, decision-adherence, repeated-mistake avoidance, **resume fidelity**, **conflict
detection/precision** (live contradictions between agents found without over-flagging), and adversarial
**memory-safety** across scripted multi-session projects. The safety eval blocks **10 / 10 adversarial attacks** (ASR
**0.00**) — including a planted confirmation next to a prohibition, a confirmation for a *different* action,
a provenance-laundering supersession, and a cross-namespace approval — with **no over-blocking** (benign-pass
**1.00**). Deterministic, $0, no LLM. **Reproduce every number with one command:**

```bash
uv run python -m eval.benches      # the whole governance suite — or `midas bench` from a checkout
```

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

All of it at **0 LLM calls, $0, and 0 data egress** at ingest. Full numbers, per-category breakdowns,
reproduce commands, and the head-to-head vs Mem0/Zep/Mastra are in **[BENCHMARKS.md](BENCHMARKS.md)**.

> **Eval-first means we publish the misses too.** Hybrid retrieval, reranking, thread-diversification,
> dual-granularity indexing, and *naive distillation* were all measured to **not** help (or to hurt) and
> are documented as such. That honesty is the point — see BENCHMARKS.md and
> [`docs/frontier-2026.md`](docs/frontier-2026.md).

---

## Connect it to your coding agent

**One command wires up everything:**

```bash
midas init        # creates the shared memory + configures every MCP client it finds
midas status      # check what's wired   ·   run `midas init --dry-run` to preview first
```

Both take **`--json`** to emit a machine-readable *client wiring receipt* — which memory each client got
wired to, under which scope/policy, and which clients were skipped (config paths only, never memory
contents). Paste it into a bug report, or let another agent verify the setup without scraping prose.

`midas init` creates **one shared memory** (`~/.midas/memory.sqlite3`) and points the MCP clients it
detects — **Claude Code, Codex, Cursor, Claude Desktop, Windsurf, VS Code, Gemini CLI, Cline, Zed** —
at it. So all your agents read and write the **same** memory, autonomously, with no per-client paths to
keep in sync.

Prefer a single endpoint over per-client launches? Run one server and give your clients an **MCP URL**:

```bash
midas serve --http        # → http://127.0.0.1:7077/mcp   (one server, one memory, every client shares it)
midas serve --http --token <secret>   # require `Authorization: Bearer <secret>` on every request
```

Keep Midas current with **`midas update`**. See your memory anytime with **`midas inspect`**.

Already carrying agent memory in files? **`midas import --from claude-md CLAUDE.md`** (or
`--from cursorrules`, `--from jsonl`, `--from mem0`, `--from zep`) turns those rules and exports into
first-class, recallable, governable memories — tagged with where they came from, idempotent on re-run.

Want memory even when the agent never calls `capture`? **`midas init --claude-hook`** installs a
Claude Code SessionEnd hook that offers each session's user turns to memory — Midas's no-LLM policy
still decides what is actually kept.

<details>
<summary><b>Manual setup</b> — any client, or to customize (click to expand)</summary>

Midas is a standard MCP server: point any client at the **`midas-mcp`** command. It uses the shared store
by default — no path needed. The universal block:

```json
{ "mcpServers": { "midas": { "command": "midas-mcp", "env": { "MIDAS_MCP_EMBEDDER": "local" } } } }
```

| Client | Where the config goes |
|---|---|
| **Claude Code** | `claude mcp add midas -s user -e MIDAS_MCP_EMBEDDER=local -- midas-mcp` |
| **Cursor** | `~/.cursor/mcp.json` — paste the JSON block |
| **Claude Desktop** | Settings → Developer → Edit Config (`claude_desktop_config.json`) — paste, restart |
| **Codex CLI** | `codex mcp add midas -- midas-mcp` |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` — paste the block |
| **VS Code** | user `mcp.json` (`servers` key, `"type": "stdio"`) — `midas init` writes it |
| **Gemini CLI** | `~/.gemini/settings.json` (`mcpServers` key) — `midas init` writes it |
| **Cline** | `cline_mcp_settings.json` in VS Code global storage — `midas init` writes it |
| **Zed** | `settings.json` → `context_servers` — `midas init` writes it |
| **Anything else** | point it at command `midas-mcp` |
| **No Python** | `npx -y midas-memory-mcp` — the [TypeScript port](packages/midas-ts) (experimental; semantic embeddings via optional `@huggingface/transformers`) |

Override per client with env: **`MIDAS_MCP_DB`** (default `~/.midas/memory.sqlite3`; `:memory:` = ephemeral)
· `MIDAS_MCP_MAX_RECORDS` · `MIDAS_MCP_MIN_IMPORTANCE` · `MIDAS_MCP_NAMESPACE`.

> ⚠️ **GUI apps don't share your shell `PATH`.** If a client says *"command not found"*, use the absolute
> path from `which midas-mcp`. On Windows use forward slashes in JSON paths.

</details>

**Once connected**, Midas injects a short policy into the agent (*recall first, then capture durable
facts/decisions/preferences/constraints/corrections*). The agent captures freely; **Midas decides what's
kept** — it scores importance (no LLM), drops trivia, skips duplicates, revises stale beliefs, and forgets
the low-value tail to stay bounded. Before any memory-justified external or destructive action, the agent
calls `check_memory_use` and is **blocked unless you confirmed it** (and that confirmation is still
current).

### One memory, many clients

By default every client shares **one live memory** (`~/.midas/memory.sqlite3`) — each detects the others'
writes (SQLite `data_version`) and refreshes, so a fact captured in your IDE is recallable from your chat
app seconds later, no restarts.

Want **per-project** separation instead? **`midas init --project-scoped`** (or `MIDAS_MCP_NAMESPACE=auto`)
gives each project its own partition in the same store — the scope is derived from the git repo / cwd the
server runs in. Or scope it manually per project/agent/user with `MIDAS_MCP_NAMESPACE`.

<p align="center">
  <img src="docs/demo-multi-client.gif" alt="Two live processes share one Midas SQLite file: a recall that finds nothing, a capture from a different process, then the same never-restarted session recalls it" width="820">
</p>

<p align="center"><sub>Real run, reconstructed chrome — the capture/recall lines are verbatim output of two separate processes sharing one file.</sub></p>

<details>
<summary><b>All tools & env knobs</b></summary>

**Tools:** `remember`, `capture` (policy-gated auto-store), `recall` (source-traceable), `build_context`
(compact, dated, today-anchored prompt block), `resume` (the one-call session-onboarding pack: pinned +
state + changes + open loops + conflicts), `memory_state` (current project state), `memory_diff`
(what changed since), `memory_conflicts` (live beliefs that contradict each other, ranked),
`open_loops` / `remember_commitment` / `close_loop` (promised work that survives sessions),
`check_memory_use` (guard), `memory_policy`, `maintain` (TTL + dedup + forgetting, returns a deletion
audit), `stats`, `forget` (chain-safe), `forget_matching` (topic-level erasure, dry-run by default),
`forget_all`. Prompts: `memory_session`, `distill`.

**Env:** `MIDAS_MCP_DB` · `MIDAS_MCP_EMBEDDER` (`local` / `hashing` / `multilingual` / any fastembed id) ·
`MIDAS_MCP_MAX_RECORDS` · `MIDAS_MCP_MIN_IMPORTANCE` · `MIDAS_MCP_NAMESPACE` (`=auto` → per-project scope) · `MIDAS_MCP_ANN=1` (sub-linear
IVF for huge stores) · `MIDAS_MCP_SUPERSEDE` · `MIDAS_MCP_NLI=1` (NLI-gated revision) ·
`MIDAS_MCP_AUTO_MAINTAIN=<min>` (idle-time upkeep) · `MIDAS_MCP_PINNED` (pin standing directives) ·
`MIDAS_MCP_TTL` (per-kind retention, e.g. `chat=30,note=90`) · `MIDAS_MCP_TOKEN` (HTTP bearer auth) ·
`MIDAS_MCP_KEY` (SQLCipher encryption at rest — `pip install "midas-memory[encrypted]"`).

</details>

### Troubleshooting

Something not wired right? **`midas doctor`** is the one-command diagnosis — it checks `midas-mcp` is on
`PATH`, that your store opens, whether the local embedder is available, and which clients are actually
wired, with a fix hint per failed check. It reads config paths and versions only — **no memory contents**,
so its output is safe to paste into a bug report.

```bash
midas doctor          # ✓/⚠ per check, with a hint for each failure
midas status          # what's wired + the store's record count
```

| Symptom | Likely cause & fix |
|---|---|
| Client says **"command not found"** | GUI apps don't inherit your shell `PATH`. Use the absolute path from `which midas-mcp` in the client config. |
| **Recall feels weak / lexical** | The offline hashing embedder is in use. Install the local embedder: `uv tool install "midas-memory[mcp,local]"` (or `pip install "midas-memory[local]"`). `midas doctor` flags this. |
| A client **doesn't see** another's memory | Confirm both point at the same store — `midas status` shows the path; the wiring receipt (`midas status --json`) shows each client's exact command + env. |
| **MCP server won't start** | The SDK is installed but the `[mcp]` extra isn't — `pip install "midas-memory[mcp]"`. `midas doctor` calls this out specifically. |

Still stuck? Open a [bug report](https://github.com/vornicx/Midas/issues/new?template=bug_report.yml)
(it pre-fills the `midas doctor` block) or ask in [Discussions](https://github.com/vornicx/Midas/discussions).

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
<summary><b>Project state & diff · belief revision · forgetting · namespaces · bitemporal · LangGraph</b></summary>

```python
from midas import Memory, LocalEmbedder
from midas.nli import LocalNLI
from midas.sqlite_store import SQLiteStore
from midas.state import memory_state, memory_diff   # the control-plane views

# Durable, shareable, no native extension. Safe across threads & processes (live data_version refresh).
mem = Memory(store=SQLiteStore("memory.db"), embedder=LocalEmbedder(),
             supersede=True, nli=LocalNLI())   # a turn that CONTRADICTS an old belief supersedes it

# Control-plane: the current state of a project, and what changed since a point in time (no LLM):
memory_state(mem, scope={"project": "apollo"})          # live, non-superseded decisions/constraints/facts
memory_diff(mem, since=last_session_epoch)              # {added: [...], revised: [(old, new), ...]}

mem.forget_decayed(max_records=50_000)         # evict lowest value (importance × recency); protects facts
mem.recall("when is the launch?", as_of=1_700_000_000)   # bitemporal: "what did we believe on date X"

# Right-to-be-forgotten — preview, then erase, with an audit trail:
mem.forget_matching("the user's home address", dry_run=True)
mem.forget_matching("the user's home address")

# Back LangGraph's long-term memory with Midas:
from midas.integrations.langgraph_store import MidasStore
store = MidasStore(); store.put(("user", "123"), "pref", {"text": "prefers dark mode"})
```

</details>

---

## See &amp; control your memory — `midas inspect`

Most memory is a black box of LLM-rewritten facts. Midas is **glass-box**: run a **local** inspector over
your store and see exactly what your agent remembers, why, and from what source — then correct, pin, or
forget it.

```bash
midas inspect --db ~/.midas/memory.sqlite3      # opens http://localhost:7777 — local only, zero egress
# before install:  python -m midas.inspector --db <your.sqlite3> --embedder hashing
```

- **Overview** — counts, attributability, a 30-day activity chart, and kind/provenance/recency breakdowns,
  each kind and provenance color-coded *consistently across every view* (a fixed categorical palette,
  validated for colorblind-safe contrast in both themes — never color-only, every value keeps its label).
- **Browse + search** every memory (verbatim, with provenance + source), filterable by kind, provenance,
  and sort order.
- **Belief history + time-travel** — what you believed, what it superseded, and when.
- **Project state** (decisions / bugs / forbidden) and **what changed** since a date.
- **Governance** — would memory authorize an action, and why (the audit trail); **forget** with a receipt.
- **Conflicts** and **Open loops** — the same control-plane views from `memory_conflicts`/`open_loops`,
  with one-click resolve/close from the UI.
- **Audit log** — the hash chain's verification status and its most recent entries.
- Light + dark themes (a real second theme, not an inverted dark one), keyboard shortcuts (**⌘K** to jump
  anywhere or search, **/** to focus search), and a responsive layout down to phone width.

And every mutation (write / revise / forget) appends to a **tamper-evident, hash-chained audit log**
inside the store — hashes only, never content. `midas audit` shows it; `midas audit --json` verifies
the whole chain and reports the first broken entry if anyone rewrote history.

No LLM, no account, runs on your file. The thing a black-box memory can't show.

## Commercial path

The core stays open source under **Apache-2.0** — local SQLite memory, MCP tools, SDKs, and the bench
suite are free to use, fork, and embed. Midas is **dev / enterprise-led**: paid work is what *teams* and
*regulated orgs* need around that core. It does **not** monetize by closing the memory core.

| Edition | For | Includes | Status |
|---|---|---|---|
| **OSS** | every dev | local core, SDK, MCP, the [bench suite](docs/agent-memory-benches.md) | Available now |
| **Team** | agent teams | hosted MCP, RBAC namespaces, admin + audit trail, SSO | Founding customers |
| **Enterprise / VPC** | regulated | on-prem/VPC, provable forgetting, audit-completeness, data residency, SSO/SAML, SLA, DPA | By arrangement |
| **[Agent-Memory Audit](docs/agent-memory-audit.md)** | memory buyers / builders | benchmark *your* stack vs the suite + `recall@k`, with failure traces + recommendations | By arrangement |

The differentiator isn't recall — it's **memory an agent can be trusted to act on**, proven by the benches
and the audit. Strategy and editions in full: **[docs/gtm.md](docs/gtm.md)**.

## Honest status

Midas is **early** but built narrow and measured-first. Where it stands, plainly:

- **Retrieval is its strength and is essentially maxed** for a no-LLM design — confirmed by our own A/Bs
  *and* by the frontier papers (the retriever is not the bottleneck). The benchmark numbers above are the
  result.
- **The frontier's extra lever is structure-preserving extraction — and it needs a capable model Midas
  deliberately won't run at ingest.** We built the judged harness and measured it on BEAM's summarization
  category: a small local extractor doesn't help (raw 0.28 vs replace 0.07 rubric coverage), and the lift
  is gated on a strong model — so it belongs to *the agent's* model, not Midas's. The optional distillation
  dial ships **off by default**; we don't claim it as a win. (Details: [`docs/frontier-2026.md`](docs/frontier-2026.md) §2b.)
- **Where it's heading:** from recall to a **governed memory control-plane** — `memory_state` / `memory_diff`,
  the provenance guard that won't act on stale or unconfirmed memory, and the
  [Agent Continuity Bench](eval/continuity.py) that measures those properties. Local, auditable, and
  honest about what's proven.

## The eval harness

`eval/` (dev-only) runs Midas and competitors through synthetic / LoCoMo / LongMemEval / multiday /
conflicts-v1 / **BEAM** with deterministic `recall@k` + `precision@k`, cost/latency instrumentation, a
**dumb-reader ablation** (proves the numbers aren't reader-inflated), and an optional local-or-hosted LLM
judge. The anti-cheating checklist (no query rewriting, no LLM at ingest, no gold leakage, seeded sampling),
conflict handling, failure traces, and the verbatim MCP policy are in
[`docs/methodology.md`](docs/methodology.md).

```bash
python -m eval.runner --dataset longmemeval --variant s --local --midas-no-rerank --max-questions 40
python -m eval.runner --dataset beam --beam-tier 100K --local --dumb-reader   # frontier benchmark
python -m eval.continuity                                                      # Agent Continuity Bench
```

## Privacy & license

Local-first: every memory lives in a SQLite file on your machine, recall returns the exact stored text,
and capture/recall/forget make **no network calls**. No account, API key, or telemetry. The only outbound
traffic is a one-time embedding-model download (for the `local` backend) and the package install.
Optional **encryption at rest**: set `MIDAS_MCP_KEY` with the `[encrypted]` extra and the store is a
SQLCipher database — unreadable without the key (and Midas fails closed rather than silently writing
plaintext). Full details in [`PRIVACY.md`](PRIVACY.md) · [Apache-2.0](LICENSE).
