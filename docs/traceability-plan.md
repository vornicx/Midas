# Traceability & Projects — plan

*Goal: make every memory fully traceable (who produced it, from where, from what turn, in which project)
and make **projects** first-class — so a team can answer "where did this belief come from, in which
project, and what acted on it?" This deepens the governance moat. Measured by the Inspector's
**attributable %** (today ~9% on a sample → target ≥90% for agent-written memory).*

## Today — the gap

Records already carry the **fields**: `id, content, kind, importance, provenance, actor, source,
metadata, created_at, updated_at, superseded_by`. So traceability is *possible*. But:

- **Attributability is low.** Most `capture`d memories lack `source` and `actor`, so they can't be traced
  back. `audit_completeness` (fraction with *both*) was ~**9%** on the demo. The fields exist — they're
  just not being filled.
- **"Project" is only a namespace** + `metadata.project` (the coding vocab). There's no first-class
  project object and no per-project Inspector beyond the existing Project tab.

The fix is two moves: **fill the trace on every write**, and **lift projects to first-class**.

## Phase 1 — Make everything attributable (the core, highest value)

Principle: **every write records who produced it, from where, and from what turn.**

- **Source** — stamp a structured source on every MCP write: `mcp:<client>:<session>:<turn>` (the client
  id, the session, and the turn/message index). The server already has the session; add client + turn.
- **Actor** — always set the real client id (`claude-code` / `codex` / `cursor`) from the MCP context;
  `user` only for explicit confirmations. (Today it defaults to the generic `midas-mcp`.)
- **Origin (optional, local)** — stamp `metadata.origin` = the git commit / branch / cwd the agent was in
  when it captured. Cheap, local, and links a memory to the **code state** it came from.
- **Enforce** — `capture` auto-fills source+actor (never silently drops them); `audit_completeness`
  becomes a first-class health metric, surfaced in the Inspector Overview (already there) and `midas
  doctor`.
- **Measure** — an eval that asserts agent-written memory is **≥90% attributable**; track the % over a
  fresh run.

*Deliverable: a fresh agent session produces memory that is ≥90% traceable to client+session+turn (+origin).*

## Phase 2 — Projects as first-class

- A lightweight **Project** = `{name, root (git/cwd), created_at, description}` — derived automatically
  (the `--project-scoped` / `MIDAS_MCP_NAMESPACE=auto` work already names it) and recorded once.
- **CLI:** `midas projects` (list, with counts + last-active), `midas project <name>` (its state).
- **Inspector:** a **Projects home** — pick a project → its state (decisions / bugs / forbidden), its
  recent activity, its attributable %, its governance log. A per-project Overview.
- **Link memory → action/commit:** when the agent acts or captures a `bug_fixed` / `command_worked`, stamp
  the commit/PR, so the memory traces to the code change it produced.

## Phase 3 — Governance & audit, per project

- A per-project governance view in the Inspector: what was decided / blocked / forbidden in this project,
  with the audit trail and the per-actor breakdown (`belief_history` + `audit_use` already exist — surface
  them scoped).
- This is exactly what the paid **Team dashboard** is, multi-user, over the same data — so Phase 3 doubles
  as the Team product's UI.

## Build order & metrics

1. **Phase 1 — attributability** (source/actor/origin on every write). Metric: attributable % → **≥90%**.
2. **Phase 2 — first-class projects** (object + CLI + Inspector home).
3. **Phase 3 — per-project governance views** (also the Team dashboard).

Each phase ships independently; **Phase 1 is the highest-value** — it *is* "make everything traceable",
it's measurable, and it reinforces the governance brand.

## What we won't do

- **No LLM** to infer provenance — it stays mechanical and honest (we measured semantic provenance as
  unreliable).
- **No cloud** — all local; the Team dashboard is the paid multi-user layer over this same traceable data.
