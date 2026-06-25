# Midas — The Complete Picture

*The local, governed memory & trust plane for long-horizon coding agents — no LLM at ingest, $0 per
message, fully local, every recall traceable to its source.*

> This document is the single, complete reference for what Midas **is**, what has been **researched and
> measured**, what has been **achieved**, and what **remains**. It is written the way Midas is built:
> lead with the measured truth, name the bets, and publish the misses. Numbers cite their source files
> (`BENCHMARKS.md`, `docs/frontier-2026.md`, `docs/overnight-experiments.md`) so nothing here is
> assertion-only.

**Contents**
1. [The one-paragraph version](#1-the-one-paragraph-version)
2. [The concept — problem, bet, vision](#2-the-concept)
3. [Design principles (the theses)](#3-design-principles-the-theses)
4. [Architecture](#4-architecture)
5. [What has been researched & measured](#5-what-has-been-researched--measured)
6. [What has been achieved](#6-what-has-been-achieved)
7. [Honest weaknesses & structural limits](#7-honest-weaknesses--structural-limits)
8. [The roadmap — what remains](#8-the-roadmap--what-remains)
9. [Strategic frame & positioning](#9-strategic-frame--positioning)
10. [Method & ethos](#10-method--ethos)
11. [Appendix — module map, glossary, reproduce](#11-appendix)

---

## 1. The one-paragraph version

Your AI agent forgets everything between sessions. Midas is the memory that lives next to it, **on your
machine** — it remembers the durable decisions, conventions, and bugs across long, multi-session work
**without** piping every turn through an LLM to "extract" facts. It costs **nothing per message**,
**nothing leaves the box**, and **every recalled memory traces back to the exact turn it came from**.
On top of that retrieval core sits the part that makes Midas *more than a vector store*: a **mechanical
governance layer** — memory may inform planning, but an agent is **blocked from acting on memory that is
stale, unconfirmed, or never user-approved**, and it can prove why. Midas is built **eval-first**: every
claim has a reproducible benchmark, *including the experiments that failed.*

---

## 2. The concept

### 2.1 The problem

Two problems, really:

1. **Agents are amnesiac.** A coding agent re-learns your architecture, re-asks your conventions, and
   re-introduces bugs you fixed three sessions ago, because the context window is the only memory it has
   and it resets every session.
2. **The popular fix is expensive, leaky, and unauditable.** The mainstream answer (Mem0, Zep, Letta,
   Hindsight, Mastra OM) calls an LLM **at ingest** to summarize/extract facts from every session. That
   buys answer-quality headroom but pays for it forever: **$/token at ingest, latency, every turn
   leaving the box to a provider, and a rewritten "fact" you cannot audit** (the extraction step can
   silently hallucinate). At the scale where agent memory actually matters, that cost structure — not a
   few points of benchmark accuracy — decides build-vs-buy.

### 2.2 The bet

Midas makes the opposite bet on every axis:

- **No LLM at ingest or query** → **$0 API spend, zero data egress**, fast local ops (tens of ms,
  embed-bound, no per-turn network round-trip).
- **Source-traceable** → recall returns the **verbatim source turn**, not an LLM rewrite. No extraction
  step that can hallucinate.
- **Mechanical governance** → trust is enforced by deterministic gates (provenance + currency), not by
  asking a model to behave.
- **Eval-first** → every default is A/B-measured on cached offline benchmarks; negatives are documented,
  not buried.

### 2.3 The vision (what "more than memory" means)

Retrieval is table stakes and — for a no-LLM design — essentially **maxed** (Section 5). The growth is
not "more search." It is to go **up a layer** and **deep into a vertical**:

> **Midas is the governed memory & trust plane for long-horizon coding agents.** Not *"what the agent
> remembers"* but *"what the agent may believe, may act on, and must be able to prove."*

Three layers, in delivery order:

- **(A) Governance / audit plane** — the depth. provenance + the `check_memory_use` guard + the currency
  rule + bitemporal `as_of` + source-trace = an **audit trail for autonomous agents**: *why* it acted,
  on *which confirmed source*, and *when*. Externally validated: Banco Santander AI Lab ships
  `mech-gov-framework` — "governance for high-stakes LLM decisions in banking" — i.e. a regulated bank's
  own lab is building exactly this category.
- **(B) Coding-agent vertical** — the focus. Memory specialized for code work: architecture decisions,
  bug-already-fixed, conventions, forbidden actions, project state. Proven by the **Agent Continuity
  Bench**. Natural design partner: Apollo.
- **(C) The benchmark as public wedge** — the brand. The **Agent Continuity Bench** + **Memory-Safety
  eval** measure what nobody else does (action-safety, decision-adherence, repeated-mistake,
  attack-success-rate). They make Midas the *standard for measuring agent memory* — a cheap,
  defensible, honesty-aligned wedge that feeds the core.

---

## 3. Design principles (the theses)

1. **No LLM in Midas at ingest/query.** The only LLM in the loop is the host agent's, or a pluggable
   *reader*. Midas never calls one to store or recall. (An optional, off-by-default distillation tier
   exists for non-agentic ingest — Section 4.10 — explicitly fenced.)
2. **Verbatim over rewritten.** Store the source turn; never replace it with a lossy summary. (Measured:
   replacing raw turns with summaries is *catastrophic* — Section 5.5.)
3. **Mechanical gates over model goodwill.** Governance is deterministic (provenance + currency +
   actor), not a prompt asking the model to be safe.
4. **Bounded by construction.** Long-horizon memory must stay finite: selective forgetting by value,
   with an audit trail.
5. **Bitemporal truth.** Every belief carries event time and a supersession link, so Midas can answer
   "what did we believe on date X" and retire stale values without deleting history.
6. **Measure every default; publish the misses.** The moat is partly the *honesty* — see Section 10.

---

## 4. Architecture

### 4.1 The data model (`midas/types.py`)

A `MemoryRecord` is: `id`, `content` (verbatim), `kind` (`note | chat | mission | fact | preference |
constraint`), `importance` (1–5), `source`, **`provenance`** (`planning | action | observation |
user_confirmation`), `actor`, `metadata` (free-form scope: namespace/project/user/agent), `created_at`
(**event time**, not load order), `updated_at`, `embedding` (float32), and **`superseded_by`** (the id
of the belief that replaced it). The provenance + supersession fields are what make the governance and
bitemporal layers possible.

### 4.2 Ingestion (`Memory.remember` / `Memory.capture`, `midas/importance.py`)

- `remember(...)` stores directly. `capture(...)` is **policy-gated**: a no-LLM importance scorer
  (`StructuralImportance` / `ContentImportance`) scores salience, drops trivia below a floor, skips
  near-duplicates, and (optionally) revises stale beliefs. The agent captures freely; **Midas decides
  what is kept.**
- Standing-directive detection (cue regex, no LLM, user-voiced only) tags durable rules ("from now on,
  reply in Spanish") so `build_context` can **pin** them — the no-LLM version of Letta core memory.
- `remember_many` batches; ingest is pure local ONNX embedding.

### 4.3 Retrieval (`Memory.recall`)

Candidates are scored by a blend: **`score = relevance + 0.3·importance_norm + 0.2·recency`** (recency
decays relative to a query reference time `now`). On top:
- **Supersession resolution** — superseded records are demoted/hidden so recall surfaces the *current*
  belief.
- **Scale-free parsimony floor** (`min_relevance_ratio`, default 0.3) — drop hits below 0.3× the query's
  own top hit. ~2× precision, 30–40% fewer tokens, **zero gold evicted** (Section 5).
- **Time-aware grounding** — event-time + a "today" anchor, lifting temporal recall 0.86 → 0.95, all
  regex/relative-date math, no LLM.
- **Bitemporal `as_of`** — "what did memory say on date X": later records excluded, supersession chains
  resolve to the version valid then.
- Optional, off by default: `hybrid` (BM25+RRF), `pool`, `anchor_boost`, `mmr_lambda`, `thread_cap`.

### 4.4 Context assembly (`Memory.build_context` / `assemble`)

Produces a prompt-ready, **dated, today-anchored, bounded** context block: relevance-ranked recall +
a reserved **pinned** channel for standing directives + a token budget + `max_record_chars` clipping.
This is the cheap reader-facing surface; audit tools stay separate.

### 4.5 Belief revision (`midas/nli.py`, supersession)

When a new turn contradicts an old belief, the old record is **superseded** (not duplicated). Gating is
on **entity overlap + ambiguity margin** (and optionally a local int8 ONNX NLI contradiction check), so
entity-confusable pairs (Apollo/PostgreSQL vs Artemis/MySQL) are **not** wrongly collapsed.

### 4.6 Selective forgetting (`Memory.forget*`, `eval/retention.py`)

Evicts by `memory_value = importance × recency` to a budget, protecting the durable tier.
`forget_matching` is topic-level erasure with a **dry-run-by-default audit trail** (right-to-be-
forgotten). Measured: importance-ranked forgetting beats recency beats random at every budget.

### 4.7 The governance layer (`midas/guard.py`, `Memory.guard_reliance`)

The part that makes Midas a *trust plane*. A `MemoryUse` is one of `planning | answer | external_action
| destructive_action`. `decide_memory_use` is a **mechanical gate**:

| Use | Allowed provenance |
|---|---|
| `planning` | any (it changes nothing in the world) |
| `answer` | observation, action, user_confirmation (not internal plans) |
| `external_action` | **only** `user_confirmation` |
| `destructive_action` | **only** `user_confirmation` |

Plus two hardening rules: a **cross-agent** rule (agent B can't act on agent A's autonomous memory) and
the **currency rule** — a **superseded** belief cannot justify an answer or action *on its own*, even
one that was user-confirmed when current (planning may still weigh history). Surfaced via the MCP
`check_memory_use` tool. `Armorer` stamps provenance/actor/source at capture.

### 4.8 The control-plane views (`midas/state.py`)

Two deterministic, no-LLM views that top-k recall is bad at (the query doesn't resemble any one turn):
- **`memory_state(scope)`** — the live, non-superseded decisions/constraints/facts of a project,
  newest-first. For onboarding into a project and planning.
- **`memory_diff(since)`** — what was *added* and what was *revised* (old→new) since a timestamp. The
  "what changed since our last session" view. Both exposed as read-only MCP tools.

### 4.8b The coding-agent vertical (`midas/coding.py`)

The Fase-B layer that makes Midas concretely *for code agents* — a **non-invasive** vocabulary over the
core (each code memory maps to a `MemoryKind` + a `code_kind`/`project` tag, so recall, supersession,
forgetting, and the Guard all keep working unchanged):
- **`code_kind`s** — `architecture_decision`, `dependency_choice`, `convention`, `bug_fixed`,
  `recurring_failure`, `forbidden_action`, `command_worked/failed`; captured via `remember_code` (+ named
  helpers), exposed as the MCP `remember_code` tool.
- **`project_state(project)`** — the live, grouped onboarding view (what's decided / fixed / forbidden),
  revised decisions showing their current value. MCP tool.
- **`is_forbidden(action, project)`** — a no-LLM gate: a proposed action under a live `forbidden_action`
  rule is flagged on **either** lexical overlap (high precision) **or** embedding cosine (recall on
  paraphrases — semantic-authorization, a *soft* signal, §5.6). MCP `check_forbidden_action`. The A×B tie:
  a forbidden action is a user-confirmed constraint, so the agent won't do it and can cite the rule.
- The injected MCP capture policy is **code-aware** (point 5): capture decisions/bugs/rules, onboard with
  `project_state`, check `check_forbidden_action` before acting.

### 4.9 Storage & scale (`store.py`, `sqlite_store.py`, `index.py`, `ann.py`, `turbovec_*`, `vector_source.py`)

- **`InMemoryStore`** (default) — exact cached cosine scan, fast in absolute terms.
- **`SQLiteStore`** — durable, multi-process, **one file many clients** (live `data_version` refresh),
  embeddings as float32 BLOBs. No native extension.
- **`IVFStore`** (`ann.py`) — numpy-only inverted-file index, sub-linear search, opt-in for large
  read-heavy corpora (~20× at 1M records at recall ~0.90).
- **`TurboVecStore`** (`turbovec_*`, opt-in) — 2/4-bit compressed candidate generation + exact-cosine
  rerank; recall = exact while index RAM shrinks **~15×** (10M×768 ≈ 31 GB → ~2 GB). A
  `SQLiteVectorSource` keeps full vectors on disk so steady-state RAM is the compressed index alone.
- The `MemoryStore` / `VectorIndex` **Protocols** + an `allowed_ids` allowlist pushdown make scope
  filtering O(1)/in-kernel for native backends.

### 4.10 Embedders & optional retrieval (`embeddings.py`, `bm25.py`, `sparse.py`, `colbert.py`, `distill.py`)

- **Embedders**: `LocalEmbedder` (bge-base ONNX, default), `multilingual` (paraphrase-MiniLM),
  `HashingEmbedder` (zero-setup offline), model2vec (ultra-fast static tier).
- **Optional, measured, opt-in**: hybrid BM25, learned-sparse BM42/SPLADE (`sparse.py`), ColBERT
  reranker (`colbert.py`), Matryoshka dim-truncation. *Each kept opt-in because it didn't generalize —
  Section 5.*
- **Distillation tier** (`distill.py`, the explicitly-fenced "crosses the no-LLM line" path for
  non-agentic ingest): `OllamaDistiller` / `HTTPDistiller`, **off by default**, `keep_raw=True` when on.

### 4.11 Interfaces

- **Python SDK** (`from midas import Memory, LocalEmbedder`).
- **MCP server** (`midas/mcp_server.py`) — tools: `remember`, `capture`, `recall`, `build_context`,
  `memory_state`, `memory_diff`, `check_memory_use`, `memory_policy`, `maintain`, `stats`, `forget`,
  `forget_matching`, `forget_all`; prompts: `memory_session`, `distill`. Configurable via `MIDAS_MCP_*`
  env vars.
- **TypeScript port** (`packages/midas-ts`, `npx midas-memory-mcp`) — same tools/schema (experimental:
  no semantic embeddings yet).
- **LangGraph store** (`midas/integrations/langgraph_store.py`) — back LangGraph long-term memory with
  Midas.

---

## 5. What has been researched & measured

Midas's evaluation harness (`eval/`, dev-only) runs Midas and competitors through synthetic / LoCoMo /
LongMemEval / multiday / conflicts / **BEAM** with deterministic `recall@k`/`precision@k`, cost/latency
instrumentation, a **dumb-reader ablation** (proves the numbers aren't reader-inflated), and an optional
local-or-hosted LLM judge. Methodology (anti-cheating: no query rewriting, no LLM at ingest, no gold
leakage, seeded sampling): `docs/methodology.md`.

### 5.1 Retrieval quality (deterministic, reader-independent) — the headline wins

| Benchmark (FULL public set) | baseline | **Midas** |
|---|---:|---:|
| **LongMemEval-`s`** — 500 questions, 246,750 turns | 0.01 | **0.92** |
| **LoCoMo** — 10 conversations, n=1,540 | 0.05 | **0.73** |
| **BEAM** — 100K → 500K → 1M → **10M tokens** | 0.00 | **0.56 → 0.51 → 0.40 → 0.32** |

LongMemEval per-category: fact 0.97 · knowledge-update 0.93 · temporal 0.91 · multi-session 0.89 ·
preference 0.89. The recency baseline finds essentially **none** of the buried evidence. The BEAM wedge
**holds across a 100× scale-up**; at the 10M tier (where no context window fits) Midas still retrieves a
third of the gold, with knowledge-update at 0.68.

### 5.2 Judged answer-rate (the cross-system metric) — SOTA-class at $0 ingest

- **LongMemEval-`s`** (gpt-4o reader/judge): Midas **0.84 — *ties* Mastra Observational Memory's SOTA**,
  which gets the same number by running an LLM Observer+Reflector at ingest. Midas pays **$0 at ingest**.
  (gpt-5-mini: Midas 0.87–0.89 vs OM 0.95 — curated observations help a strong reader more than raw
  turns.)
- **BEAM-100K** (gpt-4o judge, n=400): Midas answer **0.40 vs a context-stuffing baseline 0.05** (8×) —
  the *no-LLM-ingest floor*.

### 5.3 Cost / latency — the structural edge

| | ingest ms/event | memory-layer LLM | API $ | egress |
|---|---:|---|---|---|
| **Midas** | ~116 cold · ~0 cached | **0** | **$0** | **none** |
| Mem0 (LLM-at-ingest class) | ~668 | ≥1 call/session | yes/token | yes/turn |

`remember` ~16 ms p50, `build_context` ~51 ms p50 over 2,000 records; ~3.6 MB / 1,000 records (float32).
**Demonstrated end-to-end fully offline** (Midas + local llama3.2:1b reader via Ollama): LongMemEval-`s`
recall@k 0.80, **0 API calls / $0 / nothing leaves the box**.

### 5.4 The mechanisms that *survived* measurement (now defaults)

- **Scale-free parsimony floor (0.3)** — ~2× precision, 30–40% fewer tokens, zero gold evicted; a no-op
  on bge (the safety property of a *relative* floor). 0.4+ is too aggressive (documented boundary).
- **Time-aware grounding** — temporal recall@k 0.86 → 0.95, no LLM.
- **Supersession (belief revision)** — conflicts-v1: retires stale copies (ctx_stale 0.86) with **zero
  wrongful entity supersessions** (all four confusable pairs keep recall 1.00).
- **Pinned standing directives** — instruction-following recall@k 0.26 → 0.44 (pinned=2) → 0.51
  (pinned=4). The first detector (which also pinned assistant advice) *hurt* — kept as the honest
  negative that set the user-voice-only rule.
- **Importance-ranked forgetting** — beats recency beats random at every budget; `StructuralImportance`
  adds +0.10–0.13 recall under forgetting.
- **Float32 embeddings** — ~7× smaller than `list[float]`, recall unchanged.

### 5.5 The misses — measured negatives we publish (this is the moat)

- **Naive distillation** (replace raw turns with LLM summaries): judged **0.37 → 0.08** (catastrophic);
  augmenting recovers to 0.32 (still below raw). Default: distillation **off**, `keep_raw` when on.
- **Structure-preserving extraction on summarization** (Fase 1): built the judged rubric-coverage
  harness (`eval/summarization_ab.py`; summarization had *never* been judgeable before). Local (qwen2.5:3b):
  raw **0.28** vs replace **0.07**. **Hosted with a capable extractor (gpt-4o, judge gpt-4o, n=8): raw
  0.45 · augment 0.49 (+0.04, within n=8 noise) · replace 0.34 (−0.11).** A strong model makes the cards
  far better (replace 0.07 → 0.34) but **still does not beat raw** — so the summarization ceiling is
  **structural** (broad queries under-retrieve by similarity; any abstraction loses the verbatim detail
  the rubric rewards), **not** an extractor-quality gap. Distillation is not claimed as a summarization
  win at any extractor tier — the "use a capable model" objection is now measured and removed.
- **Query linear-adapter** (this session, Fase 5; inspired by Santander's `linear-adapter-trainer`):
  numpy-only, torch kept out even of the experiment. BEAM held-out: train recall@10 **+0.13** but
  **test −0.13** — classic overfit, does **not** generalize. Not shipped.
- **Hybrid BM25** — negative on conversational data (multi-session 0.97 → 0.81); opt-in only.
- **Learned-sparse BM42** — +0.10 on BEAM but **+0.00 on LongMemEval** and O(N)/query slow at scale;
  dataset-specific, opt-in only. SPLADE++ OOMs.
- **ColBERT reranker** — −0.10 and 51 min; don't ship.
- **Cross-encoder reranker** — ~80× latency, **no recall change** (0.88 → 0.88); off on large haystacks.
- **Embedder upgrade** — mxbai-large +0.03 at 6× latency; marginal. The ceiling is **not** the embedder.
- **Dual-granularity indexing** — overall recall 0.55 → 0.48; near-duplicate rounds compete for slots.
- **The meta-conclusion** (`docs/overnight-experiments.md`, `frontier-2026.md`): *no retrieval-component
  lever generalizes reliably.* The remaining gaps are **structural** (summarization/aggregation); the
  scale story is **capacity** (TurboVec), not swapping retrieval parts.

### 5.6 Governance & safety evals (this session — the new IP)

- **Agent Continuity Bench** (`eval/continuity.py`) — deterministic, $0, no-LLM. Scores **action_safety**
  (guard allows/blocks by provenance + currency, with an allow-discriminator so a "block-everything"
  policy fails), **decision_adherence** (a revised decision surfaces its live value), **repeated_mistake**
  (a prior fix/failure resurfaces). Midas **1.00 / 1.00 / 1.00**.
- **Memory-Safety eval** (`eval/memory_safety.py`, framing from Santander's autoguardrails) — adversarial:
  6 attacks (superseded / unconfirmed / cross-agent / **injected-content** / **forgotten** memory trying
  to authorize a use) + 3 benign. Metrics **ASR** (target 0) + **benign_pass** (target 1). Midas **0.00 /
  1.00**.
- **Coding-agent memory bench** (`eval/coding_bench.py`, Fase B×C) — deterministic, no-LLM:
  **decision_currency** (a revised architecture decision surfaces its live value via `project_state`),
  **repeated_mistake** (a prior bug/failure resurfaces), **forbidden_accuracy** (violations flagged, benign
  actions not — a benign floor). Midas **1.00 / 1.00 / 1.00**.
- **Semantic forbidden-action matching** (labelled eval, `eval/forbidden_eval.py`, bge-base, n=24) — the
  verdict the n=6 spot-check hid: paraphrase and benign cosine distributions **overlap** (paraphrases
  0.56–0.72, benign 0.49–0.69), so there is **no clean threshold**. Best semantic F1 **0.79** (recall 0.92,
  precision **0.69 ⇒ ~31% false positives**). We also tested an **NLI entailment** matcher (action → the
  prohibited act): it separates the *typical* case far more sharply (entailment median 0.72 for positives
  vs 0.07 for benign) but its **tails** sink it — some paraphrases entail ~0, one near-miss benign entails
  0.88 — so its best F1 is **0.72 < cosine's 0.79**. **Neither matcher is a reliable gate**, so semantic
  stays **advisory**: the MCP gate returns `forbidden` (lexical → refuse) vs `possibly_forbidden` (semantic
  → ask the user). A clean case of eval-first catching an overclaim — the larger eval reversed the small
  one, and the "better matcher" didn't help either. (A cosine∧NLI combination is unmeasured future work.)

### 5.7 Scaling research

- **IVF/ANN** — sub-linear, numpy-only; ~20× at 1M at recall ~0.90; exact↔IVF crossover ~10k records.
- **TurboVec** — recall = exact at 2/4-bit, RAM 15× smaller; long-horizon memory on a laptop.

### 5.8 What we learned from Banco Santander AI Lab (verified org)

Analyzed in code, not by reputation (all repos are clean v0.1.0/Jun-2026, *early not proven*): the value
isn't code to import, it's two patterns — **mech-gov-framework**'s "mechanical hard gate, not model
goodwill" (Midas's guard already is one) and **autoguardrails**' "ASR vs a fixed suite + benign-pass
floor" (adopted in the Memory-Safety eval) — plus one experiment, **linear-adapter-trainer** (tested,
measured negative). `sota-stressed-datasets` is a tabular credit set (not memory); `ralph-vault-skill` /
`llm_bridge` are solid but Apollo-side / already covered.

---

## 6. What has been achieved

- **Retrieval that is SOTA-class at $0 ingest** — 0.92 / 0.73 recall@k on full LongMemEval/LoCoMo, the
  BEAM wedge holding to 10M tokens, and a **judged 0.84 that ties LLM-ingest SOTA** with $0 egress.
- **A complete, honest eval harness** — deterministic metrics, dumb-reader ablation, reproduce commands,
  and a documented trail of negatives (Section 5.5).
- **Bitemporal belief machinery** — supersession with zero wrongful entity collapses, `as_of` time
  travel, time-aware temporal grounding.
- **Bounded memory** — value-ranked forgetting with audit; right-to-be-forgotten with dry-run.
- **The governance / trust plane** — provenance, the mechanical guard, the **currency rule** (stale
  beliefs can't authorize actions), `check_memory_use`, and the control-plane views `memory_state` /
  `memory_diff`.
- **Three Midas-native benchmarks nobody else has** — the Agent Continuity Bench, the Memory-Safety eval
  (ASR 0.00 / benign_pass 1.00), and the Coding-agent bench (1.00 across decision-currency / repeated-mistake
  / forbidden-accuracy).
- **The coding-agent vertical (Fase B)** — a `code_kind` memory vocabulary, `project_state` onboarding,
  **forbidden-action enforcement** (lexical + a first semantic signal), and a code-aware capture policy —
  i.e. *"remembers decisions, won't act on stale or forbidden memory, and proves it"* made into product.
- **Capacity to 10M-on-a-laptop** — TurboVec 15× RAM compression with recall intact.
- **Shipping surfaces** — Python SDK, MCP server (one file, many clients), TS port, LangGraph store;
  Apache-2.0; **247 tests green**.

---

## 7. Honest weaknesses & structural limits

- **Aggregation / summarization is a structural weakness** — top-k returns turns, not abstracts; at 10M
  the aggregation categories collapse (instruction-following 0.00, event-ordering 0.02, summarization
  0.03). This is *the* ceiling, and it's gated on a capable extractor Midas won't run at ingest.
- **Governance is mostly mechanical; semantics is hard** — the guard trusts a `user_confirmation` stamp
  (provenance *integrity* — forging it — is a capture-time concern) and can't tell a confirmation
  authorizes a *different* action than asked. Semantic-authorization is **started** for forbidden actions
  (lexical + embedding) but a labelled eval (n=24, §5.6) measured it **advisory, not reliable** (F1 0.79,
  ~31% false positives, overlapping distributions), so the gate treats lexical as a hard refuse and
  semantic as "verify with the user". A reliable semantic gate is genuinely open work.
- **Correctness is reader-dominated** — a bigger reader moves the headline more than the memory does, so
  we lead with reader-independent `recall@k`.
- **TypeScript port lacks semantic embeddings** — parity is partial (ONNX embeddings pending).
- **GTM**: the governance/enterprise value is real but B2B/regulated sales are slow; the consumer
  "memory passport" angle has a hard go-to-market.

---

## 8. The roadmap — what remains

Ordered by the vision (governance A → coding-agent vertical B → benchmark wedge C).

### Near-term
1. ~~Close Fase 1 honestly~~ — **done**: hosted gpt-4o summarization test showed **no lift** (augment +0.04
   in-noise, replace −0.11); the ceiling is structural, not extractor-quality (§5.5).
2. ~~Coding-agent vertical (B)~~ — **done (foundation)**: `code_kind` vocabulary, `project_state`,
   forbidden-action enforcement, code-aware capture policy, the Coding bench (§4.8b, §5.6).
3. ~~Validate semantic-authorization~~ — **done** (`eval/forbidden_eval.py`, n=24): **bounded as
   advisory** (cosine F1 0.79, ~31% FP, overlapping distributions), so the MCP gate splits hard-lexical
   from advisory-semantic. **NLI entailment was also measured (F1 0.72 < cosine) — doesn't help.** A
   *reliable* semantic gate remains open: a trained classifier, or an unmeasured cosine∧NLI combination.
4. **Governance levels (mech-gov-inspired)** — formalize `check_memory_use` into explicit levels — **only
   where the safety evals show a gap** (measure first, don't add levels speculatively).
5. **Package the benches as the public standard (C)** — Continuity + Memory-Safety + Coding bench as the
   way to measure agent memory; the eval-as-wedge.

### Mid-term
5. **Structure-preserving extraction by the host agent** — if Fase 1 hosted confirms it, ship the
   "no-*Midas*-LLM" path (the agent's model writes entity/attribute/time cards; Midas validates,
   versions, traces, governs).
6. **Enterprise/regulated hardening** — audit-completeness metric, RBAC/namespaces, data-residency,
   provable forgetting — the compliance story `mech-gov-framework` validates.
7. **TS parity** — local ONNX semantic embeddings in `packages/midas-ts`.
8. **Entity index (`midas/entity.py`)** — a no-LLM nod to graph memory, **only if** it measures a win on
   multi-hop.

### Deferred / parked (with honest reasons)
- Low-rank query adapter + massive cross-dataset triplets (the only path after Fase 5's overfit).
- Genetic-algorithm policy tuning (premature; risks optimizing a bad metric).
- Generic vector-DB / RAG-platform scope (commodity, against the thesis).

### Commercial path
OSS (core, free) · Pro (encrypted sync, backups) · Team (shared namespaces, hosted MCP, admin/audit) ·
Enterprise/VPC (on-prem, SSO/SAML, RBAC, SLA, DPA) · **Eval Pack** (benchmark a memory stack against
BEAM/LongMemEval + the Continuity/Safety benches with raw outputs). Midas **does not monetize by closing
the memory core** — it monetizes operational trust, deployment, team controls, and benchmark-grade eval.

---

## 9. Strategic frame & positioning

**The category**: not "another vector store with memory," but **the local, governed memory & trust plane
for long-horizon coding agents.** The moat is the combination — *local + source-traceable + bitemporal +
mechanical governance + a measurement culture* — none of which the LLM-at-ingest incumbents offer.

**Versus the field**: Mem0/Zep/Letta/Hindsight/Mastra OM compete on *recall@k and judged answer* bought
with LLM-at-ingest cost/egress/opacity. Midas competes on a different axis: **$0, private, auditable,
governed, and measured** — and *ties* their SOTA answer-rate at gpt-4o with $0 ingest. We do **not**
chase a few points of judged answer by adopting their cost structure.

**What we deliberately won't do**: LLM at ingest in the core; generic RAG platform; an agent framework;
selling "graph memory" before it measures; shipping any lever that doesn't generalize.

---

## 10. Method & ethos

- **Eval-first**: A/B-measure every default on cached offline benchmarks before shipping.
- **Document the negatives**: the published misses (Section 5.5) are a feature — they are why the wins
  are credible, and they save the next person the dead ends.
- **Reader-independent headlines**: lead with deterministic `recall@k`/`precision@k` + structural cost;
  treat answer-correctness as a secondary, wide-bar signal.
- **No hype**: claims cite a reproduce command; "we don't know yet" and "this didn't work" are
  first-class outputs.

---

## 11. Appendix

### 11.1 Module map

**Core** (`midas/`): `types` (data model) · `memory` (the engine: remember/capture/recall/build_context/
guard_reliance/forget) · `store` / `sqlite_store` (backends) · `index` / `ann` / `turbovec_index` /
`turbovec_store` / `vector_source` (scale) · `embeddings` · `importance` · `bm25` / `sparse` / `colbert`
(optional retrieval) · `nli` (contradiction/entailment) · `guard` (governance) · `state` (control-plane
views) · `coding` (coding-agent vertical: code_kind / project_state / is_forbidden) · `policy` (MCP policy
text) · `distill` (optional tier) · `entity` (experimental) · `mcp_server`
· `integrations/langgraph_store`.

**Eval** (`eval/`): `runner` · `datasets` · `schema` · `metrics` · `adapters/*` (midas / baseline_raw /
mem0) · `llm` · `retention` · `multiday` · `bench_perf` / `bench_ann` · `distill_ab` · **`summarization_ab`**
(Fase 1) · **`continuity`** (Continuity Bench) · **`memory_safety`** (Safety eval) · **`coding_bench`**
(Coding bench) · **`forbidden_eval`** (labelled semantic forbidden-action eval) · **`retrieval_adapter`**
(Fase 5) · `midas_sweep`.

**Docs**: `BENCHMARKS.md` (numbers) · `methodology.md` (anti-cheating, traces) · `frontier-2026.md`
(landscape, what to adopt/reject) · `overnight-experiments.md` (the retrieval sweep) ·
`long-horizon-memory.md` / `research-notes.md` (design notes) · this file.

### 11.2 Glossary

- **Provenance** — how a memory came to be: `planning | action | observation | user_confirmation`.
- **Currency** — whether a belief is still live (not superseded). The guard checks it.
- **Supersession** — belief revision: the old record points to the new via `superseded_by`.
- **Bitemporal / `as_of`** — query the state of memory *as it was* on a past date.
- **Parsimony floor** — drop hits below 0.3× the top hit (precision, fewer tokens).
- **ASR / benign-pass** — attack-success-rate and the over-blocking floor in the Memory-Safety eval.

### 11.3 Reproduce (no API key needed for the deterministic metrics)

```bash
python -m eval.runner --dataset longmemeval --variant s --local --midas-no-rerank --max-questions 40
python -m eval.runner --dataset beam --beam-tier 100K --local --dumb-reader   # frontier benchmark
python -m eval.continuity        # Agent Continuity Bench (action-safety, decision-adherence, repeated-mistake)
python -m eval.memory_safety     # Memory-Safety eval (ASR + benign-pass)
python -m eval.coding_bench       # Coding-agent bench (decision-currency, repeated-mistake, forbidden-accuracy)
```

---

*Midas is early (the API may change) but built narrow and measured-first. The bet: as agents get more
autonomous, the scarce thing isn't recall — it's **memory you can trust enough to let an agent act on**,
locally and provably. That is what Midas is becoming.*
