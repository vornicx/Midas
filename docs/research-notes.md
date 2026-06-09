# Notes from building an honest agent-memory benchmark

Short, opinionated lessons from measuring Midas against the agent-memory field. The theme: **most
published "memory" accuracy numbers measure the reader LLM more than the memory.** If you want to know
whether a *memory layer* is good, you have to measure the parts that don't depend on the reader.

## 1. End-to-end correctness is reader-dominated, not memory-dominated

On LongMemEval-style benchmarks, the final answer accuracy is driven mostly by the reader model:

- Public SOTA reports roughly **+45 points from the memory system** (full-context vs their memory, same
  reader) but **another ~8 points from simply swapping in a bigger reader** (open model → frontier
  model), same architecture. A large chunk of the headline is the reader, not the memory.
- In our own runs, Midas retrieved the evidence at **recall@k ≈ 0.9** on buried-evidence questions yet
  end-to-end correctness stayed low — because the bottleneck was the reader *reasoning over* retrieved
  evidence (temporal math, cross-session counting), not finding it.

**Takeaway:** report retrieval quality (`recall@k`) and cost as primary; treat end-to-end correctness
as reader-bound and secondary.

## 2. Hosted MoE judges are not reproducible across sessions

We graded answers with a hosted MoE model at temperature 0. Identical inputs scored **~0.46 one day
and ~0.07–0.20 the next** — a swing far larger than the effect sizes we were trying to measure. MoE
routing (and silent server-side model updates) make API judges non-deterministic even at temp 0.

Fixes we found:
- `recall@k` is computed from local embeddings → **fully deterministic**; it reproduces exactly. Make
  it the primary metric.
- For correctness, a **local, seed-pinned judge** helps — but with a catch (#3).
- Subtle: running judge calls concurrently against a local server (Ollama) **re-introduced
  non-determinism**, because concurrent requests get batched and batching changes per-request outputs.
  Serializing the judge restored identical runs.

## 3. A weak local reader cannot measure memory quality

Swapping the hosted judge for a small **local** model made runs reproducible — but a 1.5B model was too
weak to be a useful *reader*: a no-memory recency baseline scored **≥** the high-recall memory system,
because the small model couldn't exploit the better context and a no-evidence baseline still "guessed"
answers the weak judge accepted. Reproducible ≠ informative. **Credible correctness needs a reader
that is both strong and fixed.**

## 4. "No LLM at ingest" is a measurable cost/privacy wedge

The dominant cost driver in agent memory is *when you spend an LLM call*. LLM-at-ingest systems call a
model to extract/summarize facts on every ingested session. Measured per event (illustrative, one
machine): an LLM-at-ingest system spent **~668 ms/event** (plus API $ and data egress), versus
**~116 ms/event of pure local embedding** for a no-LLM design — with **$0 API and nothing leaving the
box**. The latency ratio is hardware-dependent; the structural facts (0 LLM calls, $0, no egress) are
not. At production scale, that cost structure often matters more than a few points of accuracy.

## 5. Reranking is not always on the retrieval-quality path

A cross-encoder reranker is standard advice, but on large haystacks (LongMemEval-`s`, CPU) it added
**~80× query latency with no `recall@k` change**: it reorders the records that already fit the budget
(which can help the reader) but doesn't change *which* evidence fits. Measure before assuming a
component helps — and measure its cost, not just its quality.

## 6. The retrieval ceiling is the embedder, not the budget

On LongMemEval-`s`, bi-encoder `recall@k` went 0.88 (tight 2k-token budget) → 0.92 (4k budget) →
**0.92 (8k budget, top-100)** — it plateaus. So ~4% of the misses were budget truncation (recoverable
with more context, at ~2× tokens), but the remaining ~8% is **retrieval-bound**: the bge-base
bi-encoder simply doesn't rank those gold turns highly, and no amount of budget fixes it. The lever to
go past 0.92 is a **better retriever** — a stronger embedding model, or a **hybrid lexical+semantic**
index (BM25 fused with cosine), which stays inside the no-LLM design. (This is what hybrid systems like
Hindsight's TEMPR do for retrieval — minus the per-turn LLM extraction.)

## 7. Hybrid lexical+semantic didn't beat the bi-encoder here (a clean negative)

So we built it: BM25 fused with semantic recall (`Memory.recall(hybrid=True)`), the no-LLM version of
what hybrid systems do. On LongMemEval-`s` it **did not help**:

- With the **full candidate pool** (top-100), hybrid `recall@k` = 0.92 = semantic-only. That means
  **BM25 surfaced no gold the embedder's top-100 missed** — if lexical found unique evidence this would
  exceed 0.92. So the residual ~8% is **inference-bound** (the gold shares neither lexical nor semantic
  surface with the query), not lexical-recoverable.
- At a **tight budget** (top-20, the deployment config), naive `max(cosine, BM25)` fusion actually
  *hurt* (0.88 → 0.83): lexical false-positives outscored and displaced true semantic gold.

Conclusion: on conversational multi-session data, the bi-encoder alone is the right retriever. Hybrid
stays **off by default** (kept as an option — it may help on lexical/keyword/code corpora, untested).
The honest lever for the last ~8% is better *understanding* of the query (reasoning/expansion), not
adding retrieval channels. Reporting a negative result cleanly is the point of eval-first.

**Update (we tested the *proper* fusion).** A SOTA retrieval-centric system (True Memory, *"Storage Is
Not Memory"*, 2026) fuses lexical+dense with **RRF** (reciprocal-rank fusion), not the naive
`max(cosine, bm25)` we'd used — so we implemented RRF and re-ran (LongMemEval-`s`, n=40, top-20):
**dense 0.93 · RRF 0.89 · max 0.88**. RRF is gentler than max (rank-based, no scale mismatch) but still
**below pure dense** — the conclusion holds against the better technique. Why it can't win here: at full
pool BM25 surfaces no gold the embedder missed, so fusion only *reorders*, and any lexical
false-positive displaces true semantic gold at a tight budget. Systems that benefit from hybrid likely
(a) mix in lexical-friendly datasets, (b) score end-to-end accuracy (exact-match helps the *reader*
find names/numbers) rather than recall@k, and (c) stack a reranker on top. We keep RRF as the hybrid
fusion (strictly ≥ max) but hybrid stays off for conversational recall. Plus a cost note: hybrid was
~7× slower per query (86–95 ms vs 13 ms) — it rebuilds BM25 over the corpus each call.

## 8. Where the default store tops out (~10K memories)

Midas v0 ships a brute-force cosine store (O(N) per query). Measured recall latency vs memory count
(this is the cosine scan, so it's embedder-independent):

| memories | recall latency |
|---:|---:|
| 1K | 15 ms |
| 10K | 149 ms |
| 50K | 804 ms |
| 100K | 1.6 s |

Linear, ~16 µs/record/query. So the default store is comfortable to **~10K memories** (<150 ms) and
wants a faster path beyond ~30–50K. Ingest stays cheap and linear (the store add is ~0.04 ms/item;
real embedding dominates ingest). Two honest implications: (1) per-agent / per-project memory
(typically thousands of items) is well within range today; (2) a shared 100K+ corpus needs either a
vectorised scan (embeddings are already L2-normalised, so cosine is a single matrix·vector) or an ANN
index — both behind the same store interface, since `Memory` is store-agnostic.

**Update:** the vectorised scan now **caches the embedding matrix** (rebuilt only when records change),
so repeated queries skip the array build entirely: **0.2 / 2.2 / 11.4 ms at 1K / 10K / 50K** — ~70×
faster than the pure-Python scan (and ~33× faster than rebuilding the matrix per query). Recall is now
~0.2 µs/record, which pushes the comfortable in-memory ceiling from ~10K to **~1M memories** (≈230 ms
at 1M). A pure-Python fallback plus parity and cache-invalidation tests keep it correct; an ANN index
is only needed beyond that. Lesson: the bottleneck wasn't the cosine math (BLAS is fast) but rebuilding
the array each query — caching it was the real ~33× win.

## 9. Past the exact scan: a numpy-only IVF index, and the crossover that says when to use it

Note #8 ends with "an ANN index is only needed beyond ~1M," so we built one — and kept it
**dependency-free**. `IVFStore` clusters the corpus into ≈√N cells (k-means) and a query probes only
the `nprobe` nearest, scanning ≈ `nprobe·√N` candidates instead of N — no faiss/hnswlib native
dependency, just numpy. Measured on 36k **real** bge-base embeddings (k=10, held-out queries):

| nprobe | recall@10 vs exact | speedup |
|---:|---:|---:|
| 4 | 0.82 | 6× |
| 8 | 0.91 | 3× |
| 16 | 0.95 | 1.5× |

Two honest findings. (1) **There's a crossover**: below ~10k records the exact cached scan *wins*,
because IVF's cluster-probe overhead outweighs the savings — so exact stays the default and IVF is
opt-in. (2) **The win grows with N**: candidates scale as √N while exact scales as N, so the speedup
rises with corpus size — extrapolating the candidate count, ~20× at 1M (recall ~0.90). The recurring
lesson: don't add machinery until you've measured the point where it actually pays — and even the
"scale" feature can stay zero-native-dependency, in keeping with the rest of the library.

---

## 10. Dumb reader + adversarial conflicts — isolating retrieval from the reader

External reviewers of memory benchmarks ask three fair questions: is there query rewriting, is there
eval contamination, and is a capable reader doing "heroic recovery"? Midas now publishes explicit
answers in [`methodology.md`](methodology.md):

- **No query rewriting** — questions pass verbatim to `adapter.query(question.text)`.
- **`answer_dumb` (`--dumb-reader`)** — a deterministic extractive reader (no LLM) whose score can only
  move with retrieval. On synthetic-v0, Midas gets recall@k 1.00 and answer_dumb 1.00; on conflicts-v1,
  recall@k 1.00 and answer_dumb 0.82 (the gap is the dumb reader quoting a stale repetition that
  lexically matches the question — a publishable failure case, not hidden heroic recovery).
- **`conflicts-v1`** — adversarial near-duplicates and temporal conflicts ("same fact, different date",
  "same plan, one constraint changed", entity-confusable pairs). Supersession drops ctx_stale from
  1.00 → 0.86 without losing ctx_current; the residual 0.86 is the honest gap.

**Takeaway:** lead with `recall@k` / `precision@k` / `answer_dumb`; publish traces and failure cases,
not just averages. See [`methodology.md`](methodology.md) and [`BENCHMARKS.md`](../BENCHMARKS.md).

---

*All claims here are backed by reproducible runs; see [BENCHMARKS.md](../BENCHMARKS.md) for numbers and
commands, and [methodology.md](methodology.md) for the full anti-cheating write-up. Methodology bias:
we deliberately prefer deterministic, reader-independent metrics over headline accuracy.*
