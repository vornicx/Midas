# Eval methodology — where the bodies are (not) buried

External reviewers of memory-stack benchmarks rightly look for three places systems usually cheat:
**eval contamination, hidden query rewriting, and a heroic reader** recovering answers that weak
retrieval never really surfaced. This page documents, with file references and reproduce commands,
exactly what the Midas pipeline does and does not do — plus the failure cases and the verbatim
memory policy injected over MCP.

All metrics below are **deterministic and $0** (no API key) unless explicitly marked as judged.

## 1. What the pipeline does NOT do

- **No query rewriting.** Benchmark questions are passed verbatim to the memory layer:
  `adapter.query(question.text)` in `eval/runner.py` and `build_context(question, ...)` in
  `eval/adapters/midas_adapter.py`. There is no LLM (or heuristic) reformulation step anywhere
  between the dataset question and retrieval.
- **No LLM at ingest.** Ingest is local ONNX embedding only (`BAAI/bge-base-en-v1.5`, or an offline
  hashing embedder). The runner's cost table reports `memory-layer LLM: 0 (none)` for Midas because
  there is literally no model call to hide.
- **No gold leakage.** Gold supporting turns for LongMemEval come from the dataset's own
  `has_answer=true` annotations inside `answer_session_ids` (`eval/datasets.py`); LoCoMo gold comes
  from each QA's `evidence` ids. The memory layer never sees gold ids — they are only compared
  against the retrieved ids after the fact.
- **Seeded, representative sampling.** When `--max-questions` caps a run, questions are shuffled
  with a fixed seed (`--seed`, default 0) so the cap takes a representative spread, and the exact
  sample reproduces.
- **Verbatim storage.** Midas stores source turns, not LLM-rewritten "facts", so every retrieved
  context line is auditable back to a turn id (`recall@k`/`precision@k` are computable at all —
  fact-synthesizing systems cannot offer this).

## 2. Disentangling retrieval quality from answer quality

The metrics form a ladder from fully reader-independent to fully reader-bound. We lead with the
bottom of the ladder and treat the top as a secondary, noisy signal:

| metric | reader involved? | what it isolates |
|---|---|---|
| `recall@k` | none | did the gold supporting turns get retrieved at all |
| `precision@k` | none | false-positive stress: how many retrieved turns are distractors |
| `answer_recoverable` | none | is the gold answer string present anywhere in the context |
| **`answer_dumb`** (`--dumb-reader`) | **deterministic extractive, no LLM** | could a reader with zero reasoning answer from this context |
| `answer` (`--judge`) | full LLM reader + judge | end-to-end correctness (reader-dominated) |

### The dumb reader (`--dumb-reader`)

The dumb reader (`eval/metrics.py: extractive_answer`) picks the single retrieved turn with the
highest content-word overlap with the question and scores it against gold (1.0 if the gold answer
appears verbatim in that turn, else token F1). It cannot reason, combine turns, or paraphrase — so
its score can only move with retrieval quality. **If `answer_dumb` tracks `recall@k`, no heroic
reader recovery is inflating the numbers.** Measured (hashing embedder, fully offline):

| dataset | adapter | recall@k | answer_dumb |
|---|---|---:|---:|
| synthetic-v0 | baseline-raw | 0.50 | 0.38 |
| synthetic-v0 | **midas** | **1.00** | **1.00** |
| conflicts-v1 | baseline-raw | 0.78 | 0.49 |
| conflicts-v1 | **midas** | **1.00** | 0.82 |

`answer_dumb` rises and falls with `recall@k` for both systems — the gap between the two systems is
retrieval, not the reader. The per-question trace also exposes a real failure case worth publishing:
on the two temporal-conflict questions (`q-review`, `q-logs`) the dumb reader scores 0.18 even with
recall@k 1.00, because it extracts the **stale repetition** ("Reminder: the quarterly security
review is scheduled for March 3") — the stale copy lexically mirrors the question better than the
buried update does. That is precisely the "bad memory is worse than no memory" trap: when both
values are in context, a zero-reasoning consumer quotes the prominent stale one. Supersession
shrinks that exposure (section 3); the residue is measured, not hidden.

```bash
python -m eval.runner --dataset synthetic --dumb-reader
python -m eval.runner --dataset conflicts --dumb-reader --midas-supersede
python -m eval.runner --dataset longmemeval --variant s --local --max-questions 40 --dumb-reader
```

## 3. Adversarial near-duplicates and temporal conflicts (`conflicts-v1`)

Local embedding memory often looks great until the corpus contains *"same fact, different date"* or
a repeated plan with one constraint changed. `conflicts-v1` (`eval/datasets.py: conflicts`) encodes
exactly those traps:

- **Same fact, different date** — the old value is stated 2–3× with near-identical, prominent
  phrasing; the update appears once, later, lower importance ("security review March 3 → March 17").
- **Same plan, one constraint changed** — a multi-clause deploy plan restated verbatim except one
  number (abort at 2% → 1% error rate); cosine similarity between the copies is ~1.
- **Entity-confusable near-duplicates** — Apollo uses PostgreSQL / Artemis uses MySQL; Luna's vet
  visit is Friday / Nova's is Monday. **Both sides are simultaneously true and both are asked**, so
  wrongly superseding one with the other fails a question.

Deterministic results (hashing embedder, `--context-only`, no LLM anywhere):

| adapter | ctx_current | ctx_stale | ctx_contradict | avg_tokens |
|---|---:|---:|---:|---:|
| midas (supersede=on) | 1.00 | **0.86** | **0.86** | **94** |
| midas (supersede=off) | 1.00 | 1.00 | 1.00 | 113 |

Reading it: with belief revision **off**, every updated fact drags its stale twin into the context
(ctx_stale 1.00 — the adversarial construction works). Turning supersession **on** starts retiring
stale copies (0.86) and shrinks the context, **without ever losing a current value (ctx_current
stays 1.00) and without one wrongful entity supersession** — all four confusable questions keep
recall@k 1.00 and their correct value in context, because supersession is gated on entity overlap
and ambiguity margin, not cosine similarity alone. The remaining 0.86 is the honest gap: several
stale copies survive (see the failure cases below).

```bash
python -m eval.multiday --dataset conflicts --context-only --ab-supersede --midas-only
python -m eval.runner --dataset conflicts --dumb-reader --midas-supersede   # full leaderboard
```

`ctx_stale` / `ctx_contradict` also appear directly in the main runner leaderboard for any dataset
that annotates outdated values (`stale_answer`), so staleness is not hidden in a side harness.

## 4. How conflicting memories are handled today

The direct answer to "how do you handle conflicting memories right now":

1. **Typed belief revision via supersession chains** (`midas/memory.py: _maybe_supersede`). A new
   memory may mark an old one as superseded (`superseded_by` points old → new). The old record is
   **not deleted** — chains are auditable, and recall resolves a stale hit to its current head.
2. **Gating, because over-supersession is worse than staleness.** A revision only happens if ALL of:
   embedding similarity above a threshold (lowered when an explicit update cue like "actually /
   moved to / no longer" is present); shared proper entities (Apollo never supersedes Artemis);
   a content-word anchor; an ambiguity margin between the top two candidate heads; and optionally a
   local NLI contradiction check (`midas/nli.py`, no LLM). Chat does not revise chat by default —
   an earlier cue-only heuristic regressed temporal recall@k 0.95 → 0.76 and was rolled back
   (documented in `docs/long-horizon-memory.md`).
3. **Superseded records are excluded from assembled context** (and protected from forgetting while
   they anchor a chain), so the agent sees the current belief, with the history still queryable.
4. **What it does not do (yet):** no semantic merge of partially-overlapping facts, and no
   resolution of conflicts that arrive *simultaneously* with equal evidence — those surface as
   `ctx_contradict` and are left to the reader, which is measured, not hidden.

In the eval harness supersession is **off by default** and enabled explicitly (`--midas-supersede`,
`--ab-supersede`) so retrieval numbers are never silently flattered by belief revision.

## 5. Selective forgetting: where it worked, where it failed

`eval/retention.py --trace` prints a per-question audit after eviction — which gold turns survived,
which were evicted, and the value-vs-fifo diff. From a real run (multiday, hashing embedder):

**Where it worked** (value kept the gold turn fifo evicted):

```text
keep 25%, q-budget: fifo dropped d2-1 ('The monthly infrastructure budget ceiling is 2000 euros.')
keep 25%, q-db:     fifo dropped d1-2 ('The primary database is PostgreSQL.')
keep 50%, q-budget: fifo dropped d2-1 (same fact)
keep 50%, q-db:     fifo dropped d1-2 (same fact)
keep 75%, q-db:     fifo dropped d1-2 (same fact)
```

Age-based eviction drops old-but-durable facts; importance×recency keeps them.

**Where it failed** (pure value-*rank* eviction, durable-tier protections off via
`--value-rank-only`):

```text
keep 25%, q-launch: value dropped d6-1 ("Heads up: we'll actually go live the first week of October instead.")
keep 25%, q-lead:   value dropped d7-1 ("Diego has taken over as team lead from Mara.")
```

This is the exact trap the dataset encodes: the *updates* were stated once, casually, with low
importance — so a pure importance×recency rank evicts precisely the freshest beliefs at a tight
budget. The shipping configuration protects the durable/high-importance tier (and supersession
re-anchors updates), which is why the default `value` policy keeps them — but the failure mode is
real, measured, and worth knowing about: **a memory layer whose forgetting is value-ranked can
silently prefer a confident stale fact over a quiet fresh one.**

```bash
python -m eval.retention --dataset multiday --trace                      # shipping behaviour
python -m eval.retention --dataset multiday --trace --value-rank-only    # the failure mode
```

## 6. The exact memory policy injected over MCP

This is the verbatim `instructions` text the Midas MCP server injects into a connecting agent
(`midas/policy.py: AGENT_MEMORY_INSTRUCTIONS`; also exposed at runtime via the `memory_policy`
tool):

> You have a persistent Midas memory: local, source-traceable, with no LLM at ingest. Use it on
> every task.
>
> 1) RECALL FIRST. Before answering or acting, call `build_context` (or `recall`) with the user's
> goal to load what you already know about them: prior decisions, stated preferences, established
> facts, hard constraints, and past corrections. Use it silently to stay consistent.
>
> 2) CAPTURE AS YOU GO. Call `capture` to save anything durable and reusable across sessions. You do
> NOT need to judge relevance perfectly — Midas scores each item and skips trivia and duplicates for
> you, telling you what it kept. When in doubt, capture. Especially capture:
>    - facts the user states about themselves, their project, or their environment  (kind="fact")
>    - decisions and their rationale  (kind="note")
>    - the user's stated preferences  (kind="preference")
>    - hard requirements / constraints  (kind="constraint")
>    - corrections — when the user overrides or changes something said earlier
>
>    Tag provenance on every `remember`/`capture` call:
>    - provenance="planning" for internal plans, hypotheses, or proposed next steps.
>    - provenance="action" for completed agent/tool actions and their observed result.
>    - provenance="observation" for passive observations from files, tools, logs, or retrieved data.
>    - provenance="user_confirmation" only when the user explicitly confirms the content.
>
>    Skip pure small talk and acknowledgements; Midas enforces the relevance floor regardless.
>
> 3) GUARD MEMORY BEFORE ACTION. Memory can guide planning, but it cannot by itself authorize
> external or destructive actions. Before relying on recalled memory to act outside the chat, call
> `check_memory_use` with intended_use="external_action" or "destructive_action". If the decision
> is not allowed, ask the user to confirm in the current turn. External/destructive actions may rely
> only on user_confirmation provenance.
>
> Everything is stored verbatim with its source, so recall is auditable, and memory is bounded
> automatically (low-value, stale items are forgotten) — so capturing freely is safe and cheap.

The machine-enforced half (importance floor, accepted kinds, dedup threshold) lives in
`midas/policy.py: MemoryPolicy`; the Guard provenance matrix (planning may use anything; answers
may not cite internal plans; external/destructive actions require `user_confirmation`) lives in
`midas/guard.py` and is enforced by `check_memory_use`, not by trusting the prompt.

## 7. Reproducing everything on this page

```bash
# dumb-reader ablation (offline, deterministic)
python -m eval.runner --dataset synthetic --dumb-reader
python -m eval.runner --dataset conflicts --dumb-reader --midas-supersede

# adversarial conflicts benchmark, supersession A/B (offline, deterministic)
python -m eval.multiday --dataset conflicts --context-only --ab-supersede --midas-only

# forgetting audit with success/failure traces (offline, deterministic)
python -m eval.retention --dataset multiday --trace
python -m eval.retention --dataset multiday --trace --value-rank-only

# LongMemEval-s with the dumb reader (downloads dataset + local bge model on first run)
# curl -L -o data/longmemeval_s.json \
#   https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
python -m eval.runner --dataset longmemeval --variant s --local --max-questions 40 --dumb-reader --midas-no-rerank --seed 0
```
