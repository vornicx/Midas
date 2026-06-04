# Midas

**Local-first, eval-first memory for long-horizon AI agents — no LLM at ingest.**

Midas is a small Python SDK (and an MCP server) that gives agents durable memory across long,
multi-session work — coding agents, research agents, assistants — *without* sending every turn through
an LLM to "extract" facts. It is deliberately different from LLM-heavy memory platforms:

- **No LLM at ingest or query.** Midas uses local embeddings + ranking, not an LLM to extract/summarize.
  That means **$0 API spend and zero data egress at ingest**, millisecond-scale operations, and no
  conversation turns leaving your infrastructure.
- **Auditable provenance.** Recall returns the **source turns**, traceable to their exact origin — not
  LLM-rewritten facts that can silently hallucinate at extraction time.
- **Stays current, and bounded.** Local **belief revision** (supersede contradicted facts), **selective
  forgetting + temporal tiers**, and **extractive consolidation** keep memory current and stop it from
  growing without bound — all with **no LLM**.
- **Embeddable + store-agnostic.** A library, not a server. Bring your own embedder and store.
- **Eval-first.** Every claim is backed by a reproducible benchmark — see [BENCHMARKS.md](BENCHMARKS.md).

> **Status:** early (v0.0.1). The API may change. Built narrow and measured-first.

## Why "no LLM at ingest" matters

LLM-at-ingest memory systems call an LLM to extract/summarize facts on every ingested session. They pay
for it three times: **$ per token forever at scale, the latency of LLM inference, and every conversation
turn sent to an LLM provider.** For production agents, that cost/privacy structure — not a few points of
benchmark accuracy — is often what decides build-vs-buy. Midas trades the heavyweight extraction step
for cheap, local, auditable retrieval, and puts the only LLM (the *reader* that answers) under your
control.

## Install

```bash
git clone https://github.com/vornicx/Midas && cd Midas
pip install -e .                 # core: zero third-party dependencies
pip install -e ".[local]"        # + local semantic embeddings (fastembed/ONNX — no API key, no torch)
pip install -e ".[local,mcp]"    # + the MCP server
```

## Quickstart

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
```

## Staying current and bounded — the long-horizon core

A multi-day agent's memory must stay **current** (no stale/contradictory beliefs) and **bounded** (it
can't grow forever). Midas does both with **no LLM**:

```python
from midas.nli import LocalNLI

# 1) Belief revision — a new turn that CONTRADICTS an old belief supersedes it (local NLI, not keywords)
mem = Memory(embedder=LocalEmbedder(), supersede=True, supersede_conversational=True, nli=LocalNLI())

# 2) Selective forgetting + temporal tiers — bound storage, protect the durable tier, audit deletions
forgotten_ids = mem.forget_decayed(max_records=50_000)   # evict lowest value (importance × recency)
mem.tier(record)                                          # 'short' (≤1d) | 'medium' (≤1w) | 'long'

# 3) Extractive consolidation — collapse near-duplicate restatements to one copy (keeps provenance)
mem.consolidate(similarity_threshold=0.95)
```

Forgetting **protects** the durable tier (facts/preferences/constraints, high importance) and never
orphans a supersession chain; it returns the removed ids as a **deletion audit trail**. Importance can
be supplied by you or derived from content (`ContentImportance`, no LLM). All of this is measured against
a reproducible retention harness — see [`eval/retention.py`](eval/retention.py) and the design doc.

**Durable storage:** swap the in-memory store for SQLite — `Memory(store=SQLiteStore("memory.db"),
embedder=LocalEmbedder())` — a local file, no native extension; search stays fast (records mirrored in
memory and scanned vectorised).

## Use as an MCP server

Expose Midas to any [MCP](https://modelcontextprotocol.io) client (Claude Desktop, Cursor, Windsurf,
agent frameworks) — memory tools with **no LLM and nothing leaving the machine**:

```bash
pip install -e ".[mcp,local]"
python -m midas.mcp_server          # or the `midas-mcp` console script
```

Register it (e.g. Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "midas": {
      "command": "python",
      "args": ["-m", "midas.mcp_server"],
      "env": {
        "MIDAS_MCP_EMBEDDER": "local",
        "MIDAS_MCP_DB": "/home/you/.midas/memory.sqlite3",
        "MIDAS_MCP_MAX_RECORDS": "50000"
      }
    }
  }
}
```

Tools: `remember` (auto-derives importance), `recall` (**source-traceable** hits), `build_context`
(budgeted, prompt-ready block), `maintain` (no-LLM retention: dedup + selective forgetting, returns the
**deletion audit**), `stats` (counts + short/medium/long tier distribution), `forget` / `forget_all`.
- `MIDAS_MCP_DB` persists memory across restarts (local SQLite, no native extension).
- `MIDAS_MCP_MAX_RECORDS` keeps memory **bounded out of the box** — over the cap, the lowest-value tail
  is auto-forgotten (durable facts protected).

## Use with LangGraph

Back LangGraph's long-term memory with Midas — semantic, local, source-traceable recall behind the
standard `BaseStore` (`pip install -e ".[langgraph]"`):

```python
from midas.integrations.langgraph_store import MidasStore

store = MidasStore()  # offline by default; pass Memory(embedder=LocalEmbedder(), ...) for semantic
store.put(("user", "123"), "pref", {"text": "prefers dark mode and concise answers"})
hits = store.search(("user", "123"), query="ui preferences")
```

## Benchmarks

Midas leads on the **reader-independent** axes that actually isolate a memory layer's quality (full
methodology, numbers, and reproduce commands in [BENCHMARKS.md](BENCHMARKS.md)):

| | baseline (recency window) | **Midas** |
|---|---:|---:|
| **Retrieval** — LongMemEval-`s` recall@k (evidence buried among distractors, n=40) | 0.03 | **0.95** |
| **Retrieval** — LoCoMo recall@k (5 conversations, n=50) | 0.02 | **0.85** |
| **Answer** — LongMemEval-`s` correctness (reader = gpt-4.1-mini, n=40) | 0.05 | **0.82** |
| **Ingest cost** | — | **0 LLM calls · $0 API · 0 data egress** |

We deliberately lead with **retrieval and cost** (deterministic, reader-independent) rather than
end-to-end answer accuracy — because correctness on these benchmarks is dominated by the *reader* LLM,
not the memory layer. That's the honest way to measure a memory system; caveats are in BENCHMARKS.md.

**Head-to-head, same reader, zero LLM at ingest.** With `gpt-4o` as reader, Midas scores **0.84** on
LongMemEval-`s` — **matching** the LLM-ingest SOTA (Observational Memory, 0.84 @ gpt-4o) while doing
**no LLM at ingest**. And it **scales**: on a 500-session haystack (~4,944 turns) Midas assembles a
bounded ~480-token context, where keep-everything-in-context designs overflow.

## The eval harness

`eval/` is a dev-only benchmark harness (kept out of the shipped wheel) that runs Midas and competitors
through the same datasets (LoCoMo, LongMemEval) with deterministic `recall@k`, cost/latency
instrumentation, an optional (local or hosted) LLM judge, and a retention/forgetting measurement.

```bash
# deterministic retrieval benchmark — no API key needed
python -m eval.runner --dataset longmemeval --variant s --local --midas-no-rerank \
  --max-questions 15 --limit 20 --seed 0

# retention: does no-LLM forgetting keep recall while bounding storage? (vs recency/random controls)
python -m eval.retention --dataset locomo --max-convs 1 --local --derive-importance
```

## Design concept

[`docs/long-horizon-memory.md`](docs/long-horizon-memory.md) — the north-star: the **4 C's**
(Complete · Clean · Current · Calibrated), why multi-day accuracy is a *belief-management* problem (not
only retrieval), and the honest, measured state of each piece (including the open frontiers).

## Layout

```
midas/      # the SDK (importable; zero core dependencies)
  embeddings.py   # Embedder protocol · Hashing / Local / OpenAI · DiskCachedEmbedder · LocalReranker
  store.py        # InMemoryStore (cosine) · sqlite_store.py (persistent) · ann.py (IVF index)
  memory.py       # Memory: remember / recall / build_context · forget_decayed · consolidate · tier
  importance.py   # ContentImportance — no-LLM per-turn salience
  nli.py          # LocalNLI — local entailment/contradiction (belief revision + abstention)
  mcp_server.py   # the MCP server
  types.py        # MemoryRecord, RecallHit
eval/       # dev-only benchmark harness (datasets · adapters · metrics · runner · retention)
```

## License

[MIT](LICENSE).
