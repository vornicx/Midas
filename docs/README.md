# Midas documentation

Midas is the local memory layer for long-horizon AI agents — no LLM at ingest, $0 per message,
fully local, every recall traceable to its source. **Free and open source under Apache-2.0.**

New here? The [main README](../README.md) has the 2-minute install (`midas init` wires up every MCP
client it finds) and the headline numbers. This page is the map of everything else.

## Start here

| Doc | What it answers |
|---|---|
| [README](../README.md) | Install, connect your coding agent, quickstart, troubleshooting |
| [**MIDAS.md** — the complete picture](MIDAS.md) | What Midas is, the design bets, the architecture, everything measured, honest limits |
| [ROADMAP](../ROADMAP.md) | The semver contract, design invariants, what's next, what's deliberately *not* planned |

## The numbers

Every claim Midas makes has a reproduce command — including the experiments that failed.

| Doc | What it covers |
|---|---|
| [**BENCHMARKS.md**](../BENCHMARKS.md) | All results: LongMemEval / LoCoMo / BEAM retrieval, cost & latency, scaling, forgetting, governance — with one-command repros |
| [Methodology](methodology.md) | The anti-cheating checklist, failure-case traces, and the verbatim MCP policy |
| [The agent-memory bench suite](agent-memory-benches.md) | Beyond `recall@k`: action-safety, decision-adherence, repeated-mistake, adversarial memory-safety — deterministic, $0, runnable against any memory layer |

## Research & design notes

The reasoning behind the design, written for people building or evaluating memory systems.

| Doc | What it covers |
|---|---|
| [The agent-memory frontier (2026)](frontier-2026.md) | The landscape — what the frontier systems do, what Midas adopts, what it rejects, and why |
| [Notes from building an honest benchmark](research-notes.md) | Measured lessons: why most published memory numbers measure the reader, not the memory |
| [Long-horizon memory design](long-horizon-memory.md) | The design doc: retention, forgetting, belief revision — including the kept negatives |

## Project

| Doc | What it covers |
|---|---|
| [CONTRIBUTING](../CONTRIBUTING.md) | The eval-first bar for core changes, and the lighter bar for docs & examples |
| [CHANGELOG](../CHANGELOG.md) | Every release, in detail |
| [PRIVACY](../PRIVACY.md) | Exactly what Midas does and does not do with your data (short version: it never leaves your machine) |
| [LICENSE](../LICENSE) | Apache-2.0 — use it, fork it, embed it in commercial products |
