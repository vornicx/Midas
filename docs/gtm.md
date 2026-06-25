# Midas — Go-To-Market (dev / enterprise-led)

*Positioning, who we sell to, the wedge, editions, and the staged plan. Written the way the product is:
honest and early. Midas is pre-1.0 — this is the **founding-customer / design-partner** phase, not a
billing launch.*

## Positioning

**Midas is the governed memory & trust plane for long-horizon AI agents** — local, source-traceable, no
LLM at ingest. Not "another vector store / memory API": the differentiator is **memory an agent can be
trusted to act on** — it won't act on stale, unconfirmed, or forbidden memory, and it can *prove why*.

- One-liner (dev): *"Give your coding agent a memory it can act on — local, $0, auditable."*
- One-liner (enterprise): *"Governed, on-prem agent memory: provenance, audit trail, provable forgetting."*

## Who we sell to (ICP)

- **Primary — teams building long-horizon / coding agents** (Claude Code / Cursor / Codex-style tools,
  internal dev-agents, agent startups). Pain: agents forget across sessions; LLM-at-ingest memory is
  costly, leaky, and unauditable. Motion: adopt the OSS via MCP, pay for **Team** (shared namespaces +
  RBAC + audit) as they scale.
- **Secondary — regulated / enterprise** (fintech, banks — the *Santander AI Lab* signal: a bank's lab
  is building this exact governance category). Pain: can't let an agent act on memory without provenance,
  an audit trail, provable forgetting, and data residency. Motion: **Enterprise / VPC** + compliance.
- **The wedge buyer — anyone choosing or shipping a memory layer** who needs to *measure* whether it's
  safe to act on → the **Agent-Memory Audit**.

## The wedge: the Agent-Memory Bench / Audit

The cheapest way in and the most defensible. Everyone measures `recall@k`; **nobody measures whether
memory is safe to act on.** We publish the standard ([the bench suite](agent-memory-benches.md):
Continuity + Memory-Safety + Coding) and offer to run it on a customer's stack (the
[Agent-Memory Audit](agent-memory-audit.md)). It builds the brand (eval-first), generates qualified leads
(teams choosing a memory vendor), and feeds the core.

## Editions

| Edition | For | Includes (real, shipped features) | Model |
|---|---|---|---|
| **OSS** | every dev | local core, SDK, MCP, the **bench suite** | Free (Apache-2.0) |
| **Team** | agent teams | hosted MCP, **RBAC namespaces**, admin + **audit trail**, SSO | per-seat / mo |
| **Enterprise / VPC** | regulated | on-prem/VPC, **provable forgetting**, **audit-completeness**, data residency, SSO/SAML, SLA, DPA, support | annual contract |
| **Agent-Memory Audit** | memory buyers/builders | benchmark *your* stack vs the suite + `recall@k`, with failure traces + recommendations | service / report |
| *Pro (sync/backups)* | *prosumer* | *encrypted multi-machine sync* | *optional, later — only if the funnel pulls for it* |

Principle (keep it): **we don't monetize by closing the memory core.** We monetize operational trust,
team controls, compliance, and benchmark-grade evaluation.

## Channels & motion

**Bottom-up → top-down.** Devs adopt the OSS (MCP ecosystem + the benches) → teams pull for hosted/RBAC/
audit → enterprise/regulated pay for compliance + VPC. Channels:

- The **MCP ecosystem** (Claude Code / Cursor / Codex / Windsurf) — Midas is MCP-native; meet agents where
  they are.
- **Show HN / dev & agent communities** — lead with the benchmark + the published negatives.
- **The benchmark as a credibility artifact** — invite others (even competitors) to run it.
- **X / social** — the consumer video (`prompt-video-storytelling.md`) as a *top-of-funnel awareness*
  hook that funnels to the dev story (reframe its CTA toward developers; it is not the core GTM).

## Proof assets (lead with these, not hype)

- **SOTA-tie at $0 ingest** — LongMemEval-`s` judged 0.84, matching an LLM-at-ingest SOTA with zero egress.
- **Three Midas-native benches** — governance / safety / continuity; nobody else publishes them.
- **The published negatives** — naive distillation, query-adapter, semantic-auth (×3), governance-levels-
  not-needed. *"We publish what didn't work"* earns trust with a technical audience.
- **The Santander signal** — a bank's AI lab building the same governance category validates the market.

## Staged plan

1. **Now** — OSS adoption + the benchmark wedge + a "talk to us" path for Team/Enterprise. **No billing
   system yet.** Goal: land **1–3 design partners** (one coding-agent team + one regulated org). Run a few
   free Agent-Memory Audits to seed the wedge and the case studies.
2. **Next** — package **Team** (hosted MCP + RBAC + an audit dashboard) once 1–2 teams pull for it.
3. **Later** — self-serve billing; an optional Pro/consumer tier *iff* the funnel demands it.

## What we won't do

- **Close the memory core** — it's the adoption engine and the moat.
- **Chase B2C / ship a consumer product** — off-moat (consumers don't value audit/RBAC/governance) and a
  crowded, expensive-acquisition market.
- **Claim general SOTA, traction, or revenue we don't have** — the eval-first brand *is* the honesty.
