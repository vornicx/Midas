# Long-Horizon Agent Memory — Design Concept

> The north-star for **Midas**: how to keep an agent accurate when the context is long and the
> session spans **days**. Grounded in this repo's own eval findings, not just theory.
>
> Status: concept · 2026-05-31 · working name "Midas".

---

## 1. The problem, precisely

An LLM does not hallucinate simply *because the context is long*. It hallucinates because its
context window is a **tiny, lossy, leaky view** of a history that grows every day. Over a
multi-day agent session that window fails in four distinct ways — and each needs a different fix:

| Failure | What happens | Where we stand |
|---|---|---|
| **Missing** | the fact is in history but isn't retrieved | retrieval — solved-ish (recall@k **0.83**) |
| **Cluttered** | too many turns → distractors → dilution | *demonstrated here*: more budget → **worse** correctness |
| **Stale / contradictory** | a fact changed; old + new versions both linger | **the distinctive multi-day failure** |
| **Ungrounded** | no source/recency → the model fills gaps by guessing | abstention/provenance — rarely addressed |

The job of long-horizon memory: **turn "everything that happened" into the minimal, sufficient,
*current*, *grounded* context the model needs right now — and know when it doesn't know.**

---

## 2. The thesis (what our own data forced us to accept)

**Solving multi-day hallucination is not a retrieval problem.** In this repo's eval we reached
**recall@k 0.83** while judged correctness stalled at **~0.33** — retrieving the right turn did
**not** fix the answers. The bottleneck is **belief management**: keeping a small, *current*,
*consistent*, *grounded* set of beliefs as the world changes, and knowing the boundary of what's
known.

Retrieval is necessary but not the differentiator. The differentiators are
**consolidation + temporal belief revision + grounded abstention.**

---

## 3. The frame: the 4 C's of good context

What the memory layer must deliver each turn. Good context is:

- **Complete** — it contains what's needed to answer.  ← retrieval / recall
- **Clean** — minimal distractors, no padding.  ← selection / parsimony
- **Current** — no stale or contradictory beliefs.  ← temporal versioning / belief revision
- **Calibrated** — grounded in sources, and signals when to abstain.  ← provenance / abstention

Each failure in §1 is a violation of one C. The architecture below is the machine that produces
all four. Crucially, **Complete is necessary but the other three are where multi-day accuracy is
won or lost** — and they are mostly ignored by today's memory layers.

---

## 4. The architecture (six pieces)

**1. Dual store — episodic vs semantic.**
Recent raw turns/events (episodic, verbatim) vs distilled facts/beliefs (semantic, durable). The
backbone of 2026 consolidation architectures. *Produces: Complete.*

**2. Consolidation over time (episodic → semantic).**
The `memory_tree` idea (hour → day → … → identity), **vindicated by our budget sweep**: raw volume
causes context rot; distillation wins at scale. Keep recent detail raw; compress old episodes into
dense facts/summaries. Bounds growth *and* reduces dilution. *Produces: Clean + Current.*

**3. Temporal versioning + belief revision — THE multi-day core.**
Every memory carries *temporal validity*. When new info **contradicts/updates** an existing fact,
the system **supersedes** it (mark the old one stale, link new → old) — it does not just append.
Without this, retrieval surfaces the old *and* the new → contradiction → hallucination. This is the
"LLMs reset and can't revise beliefs" failure, and what temporal knowledge graphs (Zep/Graphiti)
attack. **The hardest and most important piece for "several days".** *Produces: Current.*

**4. Selective, minimal-sufficient assembly.**
Retrieve the *smallest* set that suffices; tight budget; critical info first/last (anti
lost-in-the-middle). **Our strongest empirical finding**: more context hurt correctness. Resist
filling the window. *Produces: Clean.*

**5. Grounding + abstention — the direct anti-hallucination lever.**
Each memory carries source + timestamp; the assembled block is provenance-tagged; and — key — when
retrieval confidence is **low**, signal the model to **abstain** ("I don't have that") instead of
guessing. 2025–26 work (behavioral calibration, "Rewarding Doubt") shows rewarding abstention cuts
hallucination as much as better recall. **Almost no memory system does this** — differentiation
lives here. *Produces: Calibrated.*

**6. Selective forgetting + temporal tiers — THE bounded-growth core.** *(built + measured 2026-06-04)*
Memory cannot grow forever: an agent that runs for weeks accumulates a hoard that costs storage *and*
dilutes retrieval (the budget sweep showed volume hurts answers). The fix is a **value** on every
memory — `importance × recency`, the same query-free signals `recall()` already ranks by — and a
no-LLM eviction (`Memory.forget_decayed`) that drops the lowest-value tail while **protecting the
durable semantic tier** (facts/preferences/constraints, high importance) and never orphaning a
supersession chain. This also *is* the **short → medium → multi-day → long** tiering the design needs:
`Memory.tier()` names a memory's horizon by age (short ≤ 1d, medium ≤ 1w, long/multi-day older), while
durability decides what survives forgetting. 2026 evidence: *selective* forgetting beats both unlimited
retention and blind compression — and it stays on-wedge because it is **extractive/structural, not LLM
summarisation** (zero LLM at ingest preserved). *Maintains: Clean + Current; adds: Bounded.*

---

## 5. Where Midas is today  *(updated 2026-06-03)*

- **Complete + Clean** — *built and strong.* Semantic recall + neighbour-window + parsimony, all in
  `midas.Memory`, no LLM at ingest/query. On LongMemEval-`s` (n=40) **recall@k 0.95** vs a recency
  baseline's 0.03. With the same reader as the SOTA (gpt-4o) this converts to **answer 0.84 — matching
  Observational Memory's 0.84 — while doing ZERO LLM at ingest** (they run an LLM per conversation).
- **Current** — *two no-LLM mechanisms; ordering is the safe default.*
  **(a) Time-aware recency ordering** surfaces the latest belief first, so the reader answers with the
  *current* value — on LongMemEval-`s` knowledge-update this gives **answer 1.00**, at no recall cost.
  This is the safe, default conversational-currency mechanism.
  **(b) Hard supersession** (remove the old). On typed facts it works directly. On conversations the
  first cut (strict revision-*cue* gate) was safe on LoCoMo but **regressed large diverse haystacks**
  (LongMemEval temporal 0.95→0.76 — "similar + cue" is often a *distinct* fact). **Fixed with a local
  NLI model** (`midas/nli.py`, a ~70 MB int8 ONNX MNLI cross-encoder — no LLM, no API): a chat turn
  revises an earlier belief only when NLI scores it an actual **CONTRADICTION** (not mere similarity).
  This **restores LongMemEval recall (temporal back to 0.95, current 1.00)** while staying precise on
  real updates (unit tests: "I switched to Rust" *contradicts* "main language is Python" → revise;
  distinct facts don't). The §7 "cheap no-LLM contradiction detection" open problem is now **closed**:
  the answer is a small local NLI cross-encoder. Opt-in (`--midas-supersede-convo --midas-nli`).
- **Calibrated** — *measured, and the honest hard frontier (unsolved no-LLM).* We now measure
  abstention on LongMemEval's unanswerable questions — a metric the harness previously skipped. Three
  findings, all measured:
  1. Midas **abstains 0.83 vs a recency baseline's 0.97** — *worse*: Midas's strong retrieval surfaces
     topically-near distractors that *tempt* the reader to confabulate (baseline's recency window just
     lacks the topic, so the reader abstains).
  2. Bi-encoder **cosine does NOT separate** answerable (mean 0.70) from unanswerable (mean 0.64).
  3. The **cross-encoder reranker DOES separate the score** (answerable median 0.93 vs unanswerable
     0.17) — but **no intervention built on it beat plain Midas (0.83)**, across four measured variants:
     rerank-on + soft warning → 0.73 (reorder surfaces distractors); separate calibration-rerank (no
     reorder) + firm warning → 0.70 (telling the reader "answer only if explicit" makes it scrutinise
     and confabulate); separate-cal + hard-prune the context at a conservative floor → 0.83 (no-op,
     too few cases score that low). A higher floor would prune more unanswerable but also drop
     answerable evidence (the score distributions still overlap at the tails).
  4. We then built the **local NLI** check itself ("does any retrieved turn *entail* an answer?"). The
     signal separates in aggregate (answerable entailment median **0.98** vs unanswerable **0.37** —
     far better than cosine or relevance), but gating abstention on it is still **neutral** (0.77–0.80
     vs plain 0.83): the specific near-miss distractors that make the *reader* confabulate also score
     high on NLI, so they aren't pruned. The same distractor fools both.
  5. Reader-side calibration (a firm answerer clause): also neutral (abstention 0.77). Eight
     pre-generation/prompt attempts; none beat 0.83.
  6. **Post-hoc answer-grounding — verify the ANSWER, not the question** (`--answer-verify-nli`):
     after the reader answers, override to "I don't know" if no retrieved turn *entails that answer*.
     It looked like a win on the hosted reader (gpt-4o 0.83 → 0.87, answerable held 0.84) — **but the
     reproducible deterministic-reader A/B shows NO effect (0.37 → 0.37)**, so the 0.87 was noise (the
     floor was non-monotonic: 0.5→0.87, 0.6→0.80, impossible unless the hosted reader's answers drift).
  **ROOT CAUSE (the real finding): the reader's confabulation is drawn FROM a retrieved distractor, so
  that distractor *entails* the wrong answer** — which is why cosine, the cross-encoder, NLI-entailment,
  reader prompts, AND post-hoc grounding all fail: they inspect the very turn that genuinely supports
  the confabulation. 10 reproducibly-tested approaches; none beat plain Midas (0.83). Calibration needs
  *entity-grained* verification (does the turn answer **about the same entity** the question asks?) or a
  reasoning reader that catches the mismatch — not a relevance/entailment gate. The contribution: the
  abstention *metric*, this rigorously-mapped negative, and the root-cause diagnosis. The
  abstention *metric* and this rigorously-mapped negative are the contribution; the moat is reporting
  them truthfully. (NLI's contradiction half, by contrast,
  cleanly *solved* Current — see §5.)

---

## 6. How we validate it (eval-first)

You can't solve what you don't measure. Today's harness measures recall + answer-correctness on a
*single* conversation. Multi-day failures need new metrics + scenarios:

- **Contradiction rate** — does the answer contradict a previously established belief?
- **Staleness rate** — a fact updated on day 2: does the agent answer with the old or new value on day 5?
- **Abstention accuracy** — when the fact isn't in memory, does it abstain or confabulate?

Scenarios: multi-session streams with **evolving** facts (LongMemEval has temporal reasoning; 2026
incremental multi-turn benchmarks exist). **Plan: extend the harness to *detect* these three
failures first, then build pieces 3 & 5 against that metric** — same discipline that got retrieval
honest.

---

## 7. Open problems (unvarnished)

- **Cheap contradiction/supersession detection** — ✅ **SOLVED (2026-06-03).** A small local int8 ONNX
  MNLI cross-encoder (`midas/nli.py`, ~70 MB, no torch/API) scores CONTRADICTION between a new turn and
  an existing belief; gating revision on it fixed the cue-heuristic regression (LongMemEval recall
  0.76→0.95) and is precise on real updates. Mem0 pays an LLM per write; Midas does this **locally,
  $0**. Remaining: tune the contradiction threshold per domain.
- **A no-LLM importance signal for raw turns** — *first version built + measured 2026-06-04.*
  `midas/importance.py::ContentImportance` scores a turn 1–5 from content alone (content-word density,
  numbers/dates, proper-noun-like tokens, anti-backchannel) — no LLM; `Memory(importance_scorer=…)`
  auto-applies it to turns that arrive without one. Measured on LoCoMo (where forgetting was previously
  stuck at recency): **used as a *protection* it lifts recall@k under forgetting from 0.10 (recency) to
  0.18** — Midas sheds ~34% filler but refuses to drop fact-bearing turns. **As a pure eviction *rank*
  (equal budget) it helps only at moderate compression** (50% keep: 0.13 vs 0.10) and is too noisy at
  aggressive cuts (25%: 0.08 vs 0.10). So the coarse content heuristic belongs as a *protection/tier*
  signal, not the sole rank. The next lever, **novelty-vs-store**, was built and **tested 2026-06-04 —
  and it is a measured NEGATIVE.** `Memory(novelty_weight=…)` blends importance with `1 − max-cosine-to-
  store` (no LLM, 7 unit tests, off by default), on the hypothesis that a *new* fact should outrank a
  *repeated* one. At equal budget it is **neutral on LoCoMo/synthetic recall@k and HARMFUL on multiday**
  (value-rank recall@k **1.00 → 0.60** at 50% keep, 0.60 → 0.40 at 25%). Why: novelty demotes a repeated
  fact as "redundant", but in real memory **repetition usually signals importance** (reinforcement), and
  when the gold evidence is itself restated, demoting the repeats *evicts* it. So novelty is the **wrong
  signal for eviction-ranking** — its correct home is **consolidation** (merge true duplicates, keep one
  copy), which `Memory.consolidate` already does. We then built the **inverse — reinforcement**
  (`Memory(reinforce=True)`: a restated turn *boosts* the matched memory's importance + recency, 7 unit
  tests, off by default) — and it is **also a measured NEGATIVE**: at equal budget recall@k *drops*
  (LoCoMo 0.13→0.11 @50%, 0.08→**0.03** @25%; multiday 0.60→0.40 @25%). Why the symmetric failure is the
  real lesson: **on raw conversation, repetition tracks *commonness*, not importance** — generic phrasings
  recur while a key fact is often stated once. So novelty drops repeated *facts* (multiday) and
  reinforcement boosts repeated *filler* (LoCoMo); **the repetition-vs-store signal — in either direction
  — is a poor no-LLM proxy for importance and does not improve forgetting.** Conclusion: stop chasing
  repetition-based importance; `ContentImportance` used as a *protection* (not a rank) remains the best
  no-LLM signal, and sharpening it needs a different axis (e.g. entity/structure), not how often a turn
  recurs. Both knobs ship off by default as tested, reproducible negatives. We then tried a genuinely
  *better* signal — **structural importance** (`StructuralImportance`: content salience + boost for an
  *assertion of a durable attribute*, demote questions/meta; it provably fixes a ContentImportance blind
  spot — "My favorite food is sushi" now outranks "Where is the best sushi in Tokyo?", which content
  scored backwards) — and it is **neutral on LoCoMo/multiday** (value-rank 0.13→0.14 @50%, 0.08→0.06
  @25%; multiday unchanged), but those benchmarks **cannot isolate importance** (gold is diverse, so
  recall doesn't gate on it, or saturated). So — rule #1, the moat is measuring — we **built the right
  eval (multi-sample LongMemEval-`s`, real evidence buried among distractors, `eval/retention.py` now
  averages over all samples) and the signal RESOLVES POSITIVE — strongly, at n=40.** Under forgetting,
  **importance-ranked eviction beats recency at every level** (value > fifo > random, consistent): e.g.
  keep-50% value-rank recall@k **0.43 (content) / 0.56 (structural) vs fifo 0.36 vs random 0.25**;
  keep-25% **0.26 / 0.36 vs fifo 0.19 vs random 0.12**. And **structural beats content by +0.10 to +0.13
  recall@k** under forgetting (no longer noise) — plus +0.05 on the no-forget recall ranking (0.59 →
  0.64). **Conclusion: the three earlier "neutrals" were the EVAL, not the signal — the right
  measurement vindicates both importance-as-rank (where importance actually matters: buried facts) and
  the structural refinement, substantially.** Importance also wins as a **protection** (0.10→0.18).
  `StructuralImportance` ships as the default scorer for `capture`/auto-memory (measured-better).

- **Calibrated — entity-grained abstention built + offline-validated (2026-06-05).** A new no-LLM lever
  for the abstention frontier (`midas/entity.py` + `eval/metrics.py::_entity_grounds_answer`): abstain
  when the answer's **source turn is about a *different entity* than the question asks** — the diagnosed
  confab-from-distractor root cause. It is **orthogonal to cosine/NLI/entailment** (which all score the
  fooling distractor high, because it genuinely entails the wrong answer). The trick: drop the recurring
  *attribute* words ("favorite", "name") so the focus is the **entity** noun. On the diagnosed failure
  cases it separates confab from real **8/8**, including the pairs that broke a naive check — *"favorite
  colour" vs "favorite food"* → abstain; *"What city…" vs "I live in Barcelona"* → keep (shared
  predicate). 11 tests. **Honest limit: these are crafted cases** — the end-to-end win on LongMemEval's
  unanswerable set needs a *capable* reader (the local 1B model hallucinates rather than confabulating
  from a distractor, so it can't exhibit the target; OpenRouter credits = $0). Lesson (again): I
  predicted this would be another near-miss and was wrong — **measure, don't assume**.
- **Lossy compression without losing the critical detail** — exactly the thing a user notices when it
  breaks. *Extractive consolidation built + measured 2026-06-04* (`Memory.consolidate`): it collapses
  near-**duplicate** restatements to the single highest-value copy — extractive (drops whole redundant
  records, never LLM-rewrites them, so provenance survives) and conservative (highest-value-first
  representative, cosine ≥ threshold only, chains kept). Measured **safe** — recall@k held (LoCoMo
  0.27→0.26 dropping 10 dups at 0.92; multiday held 1.00 at 0.85). Honest limit: **yield is modest at
  safe thresholds** (2–8% here) because LoCoMo/multiday restate facts by *paraphrase* (cosine 0.7–0.9),
  not verbatim; merging paraphrases needs a lower threshold and the same caution as supersession (the
  recall@k guard says where it's safe). Yield grows with literal redundancy → with scale (see §8 / the
  LongMemEval-m stress test). Abstractive (LLM) summarisation stays off the table — it would break the
  zero-LLM-ingest wedge.
- **Calibration/abstention is genuinely hard** — getting the model to say "I don't know" instead of
  confabulating, with the *right* threshold.

---

## 8. Status & next steps

- [x] **Complete + Clean** (retrieval): built, measured, beats Mem0 at tight budget.
- [x] **Eval — multi-day detector built** (`eval/multiday.py`: `current_acc` / `stale_rate` /
  `abstention_acc` over an evolving-fact multi-session set). With an **adversarial** staleness scenario
  (old value prominent/repeated/imp5 + keyword "launch"; new value stated once, late, paraphrased
  "go live", imp3, buried) the detector bites (n=8, budget 256, Kimi judge):

  | adapter | current_acc | stale_rate | abstention_acc |
  |---|---|---|---|
  | baseline-raw | 1.00 | 0.00 | 1.00 |
  | midas | **0.40** | **0.50** | 1.00 |
  | mem0 | 0.80 | 0.50 | 0.67 |

  **Key finding — Midas's selectivity has a failure mode.** It confidently retrieves the prominent,
  important, keyword-matching OLD value and *drops the subtly-worded NEW update* → 50% stale. This is
  the flip side of "selection beats volume": here selection **hurts** currency. **The fix is
  supersession (belief revision), not more retrieval.** baseline's 1.00 is a small-scale artifact
  (all 24 events fit the budget → the LLM saw the full chronology and reasoned "instead / taken over")
  and won't scale. Abstention again favours Midas (1.00 vs Mem0 0.67). **This empirically motivates the
  *Current* piece below — and gives it a metric to be built against.**
- [~] **Current — belief revision (supersession) BUILT + validated** in `midas.Memory`. On the
  adversarial multi-day detector it fixes staleness: stale_rate 0.50 → **0.00**, ctx_stale/contradict
  **0.00**, abstention 1.00, ~13× fewer tokens. **Gated to typed durable facts (`same_kind_only=True`)
  after a regression check caught that ungated supersession DESTROYS conversational retrieval
  (LoCoMo recall@k 0.62 → 0.00 — bge's high baseline cosine + a keyword heuristic over-supersede
  all-chat data). The gate makes it a safe no-op on chat (recall back to 0.62).** Open limits: it
  therefore does *nothing* on raw conversations yet (needs fact-typing or a hybrid LLM-confirm — the
  documented hard problem); and the keyword heuristic is untested on real fact-heavy multi-session
  data (→ LongMemEval). Toggle: `--midas-supersede` on the runner.
- [x] **Bounded — tiered memory + selective forgetting BUILT + measured** (`Memory.forget_decayed` /
  `memory_value` / `tier`, no LLM; 9 unit tests, suite 57 green). Measured with a deterministic
  retention harness (`eval/retention.py`) that compares eviction policies at the **same** retained
  budget — the honest control, not "forget vs not":
  - **On data with an importance signal** (synthetic, multiday) value-based forgetting **holds recall@k
    1.00 at 50% and 25% retention** while the recency control (fifo) falls to **0.60 / 0.33** and random
    to **0.40 / 0.17**: it protects the durable, answer-bearing tier and sheds the rest (and refuses to
    evict below the protected floor — visible in the store count).
  - **On uniform-importance conversational data** (LoCoMo, every event `importance=1`) value-based
    forgetting **correctly reduces to recency** (value == fifo, exactly), cuts context tokens **~3×**
    (87 → 27), and still beats random under aggressive cuts (0.10 vs 0.03 at 25%). Regression-safe:
    no-forget LoCoMo recall@k holds at **0.62** (rerank on) — the change is purely additive, retrieval
    is untouched. Honest limit: with no
    importance signal it cannot protect gold, so recall decays with the cut — **deriving importance from
    raw turns without an LLM is the next step** (see §7).
- [~] **Long-horizon at scale — LongMemEval-m (500 sessions) structurally proven 2026-06-04.** One
  haystack is **~4 944 turns**; Midas ingests it and assembles a **budgeted ~480-token** context —
  *bounded regardless of history size*. This is the structural answer to Observational Memory's failure
  mode: OM keeps **all** observations in context and **overflows** at this scale; Midas retrieves +
  forgets, so context stays flat. Extractive consolidation at scale strips **5 % (282 records)** of
  near-duplicate restatements at 0.92 — vs 2 % on a single LoCoMo conversation, confirming **redundancy
  yield grows with scale** as predicted (§7). **Multi-question recall@k now measured (2026-06-05, n=3
  via `eval.runner --variant m --midas-only`): recall@k 0.78 (fact 1.00, temporal 0.67)** — the earlier
  n=1 0.00 was a single hard temporal-3-gold question, not representative; at 500-session scale retrieval
  is healthy. The structural token-bound + this recall@k + the consolidation-yield are the measured wins
  (full multi-question is still CPU-bound on a modest box). Data: `data/longmemeval_m.json`;
  `eval/retention.py --dataset longmemeval --variant m`.
- [ ] **Calibrated**: provenance in the assembled block + a low-confidence abstention signal.

---

## References

- Memory in the Age of AI Agents: A Survey — https://github.com/Shichun-Liu/Agent-Memory-Paper-List
- Memory for Autonomous LLM Agents (arXiv 2603.07670) — https://arxiv.org/abs/2603.07670
- Mem0: Scalable Long-Term Memory (arXiv 2504.19413) — https://arxiv.org/pdf/2504.19413
- Graph-based Agent Memory (arXiv 2602.05665) — https://arxiv.org/html/2602.05665v1
- Evaluating Memory via Incremental Multi-Turn Interactions (arXiv 2507.05257) — https://arxiv.org/html/2507.05257v3
- Mitigating Hallucination via Behaviorally Calibrated RL (arXiv 2512.19920) — https://arxiv.org/abs/2512.19920
- Conformal Abstention (arXiv 2405.01563) — https://arxiv.org/pdf/2405.01563
- Hallucination Detection & Mitigation, State of the Art 2026 — https://zylos.ai/research/2026-01-27-llm-hallucination-detection-mitigation
