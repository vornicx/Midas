# Roadmap

Midas is **eval-first**: the roadmap is shaped by what the measurements say, not by a feature wishlist.
This is a living document — open a [Discussion](https://github.com/vornicx/Midas/discussions) or an
issue to push on any of it.

## The contract (from 1.0.0)

The public surface — the `midas.Memory` SDK (remember / recall / consolidate / supersession / forget),
the guard (`guard_reliance`, `decide_memory_use`, the coding `is_forbidden` gate), the MCP tools, and
the `midas` CLI — follows [semver](https://semver.org). **Breaking changes only land in a major.**
Additive, no-LLM, measured improvements are always welcome within a minor.

## Design invariants (these will not change)

These are the bet, not a backlog. They are load-bearing for the cost/privacy/auditability story:

- **No LLM at ingest or query.** $0 per message, zero data egress, deterministic. New *no-LLM*
  mechanisms are very welcome; an LLM in the ingest/query path is out of scope by design.
- **Local-first.** Memory is a SQLite file on your machine. No account, no telemetry, works offline.
- **Source-traceable recall.** Recall returns the verbatim stored turn, never an LLM rewrite.
- **Governed, not just retrieved.** Provenance + currency gate what an agent may act on.

## Shipped (1.0.0)

The retrieval core (LongMemEval-`s` `recall@k` 0.92, LoCoMo 0.73, BEAM to 10M tokens), typed belief
revision, selective forgetting, the provenance guard + memory-safety bench (ASR 0.00 / benign-pass
1.00), the coding-agent vocabulary, per-project scoping, the MCP server + `midas` CLI (`init` / `serve`
/ `status` / `doctor` / `inspect` / `export` / `import`), the machine-readable wiring receipt, and the
TypeScript port. Full detail in [CHANGELOG.md](CHANGELOG.md) and [BENCHMARKS.md](BENCHMARKS.md).

## Near-term (help wanted)

Ordered by leverage, not certainty. Several are [`good first issue`](https://github.com/vornicx/Midas/labels/good%20first%20issue)-sized.

- **Third-party head-to-heads.** Run Mem0 / Letta OSS through `eval/` on LongMemEval-`s` and LoCoMo at
  the same embedder budget and seeds, and publish the table *with the cost column* — including any row
  where Midas loses. This is the single highest-credibility measurement left to make.
- **Judged answer-rate at scale.** The current gpt-4o judged number is `n=40` (cost-bound and honestly
  flagged). A cheaper judge at `n≥200 × 3 seeds` turns it from anecdote into a defensible number.
- **More client guides.** One page per client (Cursor, Claude Code, Codex, Windsurf, Zed, …) covering
  wiring, shared-memory setup, and verifying with `midas doctor`.
- **Dataset mirrors & repro polish.** Make every `BENCHMARKS.md` number reproducible from a clean
  checkout with a single documented command and a pinned dataset.
- **Footprint at scale.** Extend the measured RAM/disk figures to 100k / 1M records.

## Explicitly *not* planned

Kept here so nobody has to guess — each is a documented, measured ceiling, not an oversight:

- **LLM-summarized / abstractive memory at ingest.** Measured to hurt (naive distillation 0.37 → 0.08)
  and it breaks the $0 / zero-egress / traceable invariants. See [`docs/frontier-2026.md`](docs/frontier-2026.md).
- **Whole-conversation aggregation / summarization queries.** Top-k retrieval structurally can't cover
  these; documented as the trade Midas makes. See [BENCHMARKS.md](BENCHMARKS.md).
- **A cloud service, account system, or telemetry.** Midas is local-first by design: the memory is a
  SQLite file on your machine and it never phones home. Everything in this repo is and stays free
  under Apache-2.0.
