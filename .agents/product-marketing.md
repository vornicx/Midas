# Product Marketing Context

*Last updated: 2026-07-08*

## Product Overview
**One-liner:** The local memory layer for long-horizon AI agents — remembers across sessions, keeps what's current, and won't act on stale memory.
**What it does:** Midas gives coding agents (Claude Code, Cursor, Codex, and any MCP client) one shared, persistent memory that lives in a SQLite file on the user's machine. It ingests with no LLM (local embeddings only — $0 per message, zero data egress), returns verbatim source-traceable recall, revises stale beliefs instead of piling up duplicates, and mechanically blocks agents from acting on memory that is stale, unconfirmed, or forbidden.
**Product category:** Agent memory / AI agent infrastructure (searched as "AI agent memory", "MCP memory server", "Claude Code memory", "Mem0 alternative").
**Product type:** Open-source developer tool (Python SDK + MCP server + TypeScript port). MIT, fully free — no paid tiers, no telemetry, no account.
**Business model:** None — free and open source. The maintainer's return is reputation, portfolio, and community. (Positioning must never imply a sales motion.)

## Target Audience
**Target users:** Individual developers and small teams running AI coding agents daily; builders of long-horizon / autonomous agents; privacy-conscious devs and those in regulated environments who can't ship conversation data to third parties.
**Decision-makers:** The developer themself (self-serve, `uv tool install` + `midas init` in under a minute).
**Primary use case:** Give a coding agent durable memory across sessions — decisions, conventions, fixed bugs, forbidden actions — without paying per-message LLM extraction or leaking conversations to a provider.
**Jobs to be done:**
- "Stop my agent from re-asking my conventions and re-introducing bugs it fixed three sessions ago."
- "Share one memory across Claude Code, Cursor, and my chat app without syncing anything."
- "Let my agent act on memory *safely* — never on stale or unconfirmed beliefs."
**Use cases:**
- Multi-session coding projects (architecture decisions, project state, open loops surviving restarts)
- Multi-agent setups sharing one store (conflict detection between agents' beliefs)
- Importing existing CLAUDE.md / .cursorrules / Mem0 / Zep memory into something governed

## Problems & Pain Points
**Core problem:** Agents are amnesiac — the context window resets every session, so agents re-learn architecture, re-ask conventions, and repeat fixed mistakes.
**Why alternatives fall short:**
- Mainstream memory tools (Mem0, Zep, Letta, Hindsight) call an LLM at ingest: per-token cost forever, added latency, every turn leaves the box, and the extraction step can hallucinate "facts" you can't audit.
- CLAUDE.md / rules files are static, unsearchable, manually curated, and shared by paste.
**What it costs them:** API spend on every message, privacy exposure, time re-explaining context, bugs reintroduced.
**Emotional tension:** Distrust — "what did my agent actually remember about me, and is it even true?"; fear of an autonomous agent acting on wrong or stale memory.

## Competitive Landscape
**Direct:** Mem0, Zep, Letta, Hindsight, Mastra OM — LLM-at-ingest memory APIs/SaaS. Fall short on: cost per message, data egress, unauditable rewritten facts, no action-safety governance.
**Secondary:** CLAUDE.md / .cursorrules / static context files — no recall, no revision, no scale; and RAG-over-notes setups — no currency, no governance.
**Indirect:** "Just use a bigger context window" — measured to collapse beyond ~1M tokens (BEAM 10M tier), pays full-context cost every turn, and still has no cross-session persistence.

## Differentiation
**Key differentiators:**
- **No LLM at ingest or query** — $0 per message, ~16–116 ms local ops, works fully offline.
- **Verbatim, source-traceable recall** — never an LLM rewrite; every hit points to its source turn.
- **Governed memory** — a provenance guard mechanically blocks memory-justified external/destructive actions unless user-confirmed and still current; measured ASR 0.00 with benign-pass 1.00.
- **Eval-first honesty** — every claim has a one-command repro, and the measured failures are published too.
- **One file, many tools** — 9 MCP clients wired by one command against one live SQLite store.
**Why customers choose it:** It ties LLM-ingest SOTA on judged answers (0.84 LongMemEval-s, gpt-4o) at literally $0 ingest with nothing leaving the machine — and it's the only memory layer that can *prove* why an agent was allowed to act.

## Objections
| Objection | Response |
|-----------|----------|
| "No LLM at ingest must mean worse memory" | Judged answer 0.84 ties the LLM-ingest SOTA (same gpt-4o judge); recall@k 0.92 on full LongMemEval-s. One command reproduces every number. |
| "What's the catch / where does it lose?" | Whole-conversation summarization — documented as a structural trade in BENCHMARKS.md, stated in Midas's own comparison table. The honesty *is* the answer. |
| "Another memory tool, why trust it?" | Fully local (read the SQLite file yourself), `midas inspect` shows everything, hash-chained audit log, MIT, no account/telemetry. |
| "Is it maintained / production-ready?" | 1.0.0 under semver, 346-test suite, multi-OS CI, Python + TypeScript. |

**Anti-persona:** Teams that want whole-conversation summaries/aggregation as the primary feature; products wanting a hosted memory API with zero self-hosting.

## Switching Dynamics
**Push:** API bills per remembered message; privacy/compliance blockers; agents acting on wrong memory.
**Pull:** $0, private, one-command setup, works with the tools they already use, provable behavior.
**Habit:** CLAUDE.md files already work "well enough"; inertia of existing Mem0/Zep integration. (Counter: `midas import --from claude-md|mem0|zep` is idempotent and takes one command.)
**Anxiety:** "Will a local no-LLM design be good enough?" (counter with the tied SOTA number + reproduce command, and the published misses that make the wins credible).

## Customer Language
**How they describe the problem:**
- "My agent forgets everything between sessions."
- "Claude keeps re-asking about my stack / reintroducing the same bug."
- "I don't want my conversations sent to some memory API."
**Words to use:** local-first, no LLM at ingest, $0 per message, source-traceable, verbatim, provenance, belief revision, reproducible, "won't act on stale memory", glass-box.
**Words to avoid:** revolutionary, game-changing, AI-powered memory (it's pointedly *not* LLM-powered at ingest), enterprise-grade, "best" without a number, any claim without a repro command.
**Glossary:**
| Term | Meaning |
|------|---------|
| Provenance | How a memory came to be: planning / action / observation / user_confirmation |
| Supersession | Belief revision — the old record points to its replacement |
| Guard | The mechanical gate that decides what memory may justify an action |
| recall@k | Fraction of gold supporting turns retrieved — reader-independent |
| ASR | Attack-success-rate in the adversarial memory-safety eval (target 0) |

## Brand Voice
**Tone:** Plainspoken, technical, measured. Confident because the numbers are reproducible, never hypey.
**Style:** Lead with the measured truth; name the trade-offs unprompted; cite the repro command.
**Personality:** Honest, rigorous, local-first/independent, quietly opinionated.

## Proof Points
**Metrics:** recall@k 0.92 LongMemEval-s (500q, 246,750 turns) · 0.73 LoCoMo · BEAM holds 100K→10M tokens (0.56→0.32) · judged 0.84 ties LLM-ingest SOTA at $0 ingest · ingest ~16–116 ms vs ~668 ms Mem0-class · memory-safety ASR 0.00 / benign-pass 1.00 · fully-offline e2e run measured.
**Value themes:**
| Theme | Proof |
|-------|-------|
| Free & private | $0/message, zero egress, offline e2e measured, MIT |
| Trustworthy recall | verbatim source turns, hash-chained audit log, `midas inspect` |
| Safe to act on | guard + memory-safety bench (10/10 attacks blocked, no over-blocking) |
| Honest engineering | published negatives (naive distillation 0.37→0.08, etc.) |

## Goals
**Business goal:** Adoption and reputation: GitHub stars, installs, external benchmark reproductions, contributors. Portfolio value for the maintainer.
**Conversion action:** `uv tool install "midas-memory[mcp,local]" && midas init` (or `npx -y midas-memory-mcp`), then a ⭐ on GitHub.
**Current metrics:** v1.0.0 just shipped to PyPI + npm (2026-07-08); early-stage repo (2 external contributions to date).
