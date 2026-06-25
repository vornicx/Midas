# Agent-Memory Audit

*Can your agent be trusted to **act** on what it remembers? We measure it.*

Everyone benchmarks memory on `recall@k` — *"can it find the fact?"*. That is not the question that
decides whether you can ship an autonomous agent. The questions that matter:

- Will it **act on a stale or unconfirmed** belief?
- Does a **revised decision** surface its current value, or the old one?
- Can **adversarial memory** (injected, forgotten, superseded) **authorize an unsafe action**?
- Does a prior **bug/failure resurface**, or does the agent repeat it?

The Agent-Memory Audit answers these against **your** memory stack — Midas, Mem0, Zep, a vector DB, or
something homegrown — with a deterministic, $0-to-run, reproducible suite.

## What we measure

| Axis | Question | Metric |
|---|---|---|
| **Retrieval** | Does the evidence even surface? | `recall@k` / `precision@k` vs a recency baseline |
| **Action-safety** | Can stale/unconfirmed/injected memory authorize an action? | **Attack-Success-Rate** (target 0) + a benign-pass floor |
| **Decision-adherence** | Does a revised decision surface its live value? | adherence rate |
| **Repeated-mistake** | Do prior fixes/failures resurface? | resurfacing rate |
| **Forbidden-action** | Are prohibited actions caught — without over-blocking? | accuracy + false-positive rate |

These run on the open [agent-memory bench suite](agent-memory-benches.md) (`eval/`); the methodology and
the anti-cheating checklist are public ([BENCHMARKS.md](../BENCHMARKS.md), [docs/methodology.md](methodology.md)).

## What you get

- A **report card** across every axis, with a clear pass/fail verdict (the benign/allow floors mean a
  trivially-safe "do nothing" stack can't fake a pass).
- **Failure traces** — the concrete cases your stack got wrong (the most useful part), not just averages.
- A **vs-baseline comparison** — your stack against the no-LLM-ingest floor and against recency.
- **Recommendations** — where to harden (provenance, currency, supersession, forgetting), grounded in the
  failures, not generic advice.

## Why us

We built the benches and we publish our **own** misses (naive distillation, a query-adapter, three
semantic-gate attempts) — so the audit is honest, reproducible, and not a sales funnel dressed as a
benchmark. It is deterministic and $0 to run: you can re-run every number yourself.

## Who it's for

- Teams shipping **long-horizon / coding agents** who need to trust memory before letting an agent act.
- Anyone **choosing a memory vendor** who wants a trust comparison, not a recall@k leaderboard.
- **Regulated / enterprise** teams who must *prove* an agent's memory behaves safely.

## How to start

Run it yourself: `uv run python -m eval.benches` (deterministic, $0). Or have us run a full audit on your
stack with traces + recommendations — open a
[Team/Enterprise conversation](https://github.com/vornicx/Midas/issues/new?title=Agent-Memory%20Audit).
