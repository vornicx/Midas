"""Retention / selective-forgetting measurement — the eval-first proof for bounded long-horizon memory.

Long-horizon memory cannot grow forever: storage cost and retrieval dilution both scale with the
hoard. The question this harness answers, deterministically and for $0 (no LLM, no judge): **when you
must cap memory, does Midas's no-LLM decay-based forgetting (`Memory.forget_decayed`) keep the right
things?** It compares eviction policies at the SAME retained budget — the honest comparison (a control,
per the project's eval-first rule), not "forgetting vs not" (which trivially loses recall):

  - none          : keep everything (the recall@k ceiling + the regression anchor)
  - value (Midas) : evict lowest memory_value (importance x recency), protecting the durable tier
  - fifo          : evict oldest first (recency-only control)
  - random        : evict at random (the floor)

Reported per policy: recall@k (gold supporting turns retrieved), ctx_current (gold ANSWER present in
the assembled context), avg context tokens, and store size. Add `--trace` for a per-question forgetting
audit (which gold turns survived eviction, value-vs-fifo wins and failures) — publishable markdown.

    uv run --no-sync python -m eval.retention --dataset locomo --max-convs 1 --local
    uv run --no-sync python -m eval.retention --dataset multiday --local --no-rerank
    uv run --no-sync python -m eval.retention --dataset multiday --trace
    uv run --no-sync python -m eval.retention --dataset longmemeval --variant s --local --no-rerank \
        --local-max-text-chars 600 --max-questions 15
"""
from __future__ import annotations

import argparse
import random

from .adapters.midas_adapter import MidasAdapter
from .datasets import locomo, longmemeval, multiday, synthetic
from .metrics import recall_at_k
from .multiday import context_quality


def _load(args: argparse.Namespace):
    if args.dataset == "locomo":
        return locomo(max_conversations=args.max_convs)
    if args.dataset == "multiday":
        return multiday()
    if args.dataset == "synthetic":
        return synthetic()
    if args.dataset == "longmemeval":
        return longmemeval(max_questions=args.max_questions, variant=args.variant)
    raise ValueError(f"unknown dataset: {args.dataset}")


def _now_ref(sample) -> float | None:
    """Reference 'now' for recency/value: the latest event time, so 'recent' means the freshest turn.
    None when the dataset carries no event timestamps (recency then falls back to ingest order)."""
    times = [e.metadata.get("timestamp") for e in sample.events if e.metadata.get("timestamp")]
    return max(times) if times else None


def _evict_oldest(store, target: int) -> None:
    """FIFO control: drop oldest-by-event-time first, keep the newest `target`."""
    recs = sorted(store.all(), key=lambda r: r.created_at)
    for r in recs[: max(0, len(recs) - target)]:
        store.delete(r.id)


def _evict_random(store, target: int, seed: int) -> None:
    recs = list(store.all())
    random.Random(seed).shuffle(recs)
    for r in recs[: max(0, len(recs) - target)]:
        store.delete(r.id)


def _measure(adapter: MidasAdapter, sample, *, budget: int) -> dict[str, float]:
    pairs = [(q, adapter.query(q.text, token_budget=budget)) for q in sample.questions]
    recalls = [r for q, res in pairs if (r := recall_at_k(res, q)) is not None]
    cq = context_quality(pairs)
    return {
        "store": float(adapter.store_size),
        "recall@k": sum(recalls) / len(recalls) if recalls else float("nan"),
        "ctx_current": cq["ctx_current"],
        "avg_tokens": cq["avg_tokens"],
    }


def _run_policy(make_adapter, sample, *, budget: int, policy: str, frac: float, seed: int,
                rank_only: bool = False, trace_sink: list | None = None) -> dict:
    adapter = make_adapter()
    adapter.reset()
    adapter.ingest(sample.events)
    now = _now_ref(sample)
    n = adapter.store_size
    target = max(1, round(frac * n))
    if policy == "value":
        # Default = the SHIPPING behaviour: protect the durable/high-importance tier, so forgetting
        # sheds filler but refuses to drop fact-bearing turns (may keep > target — shown in `store`).
        # `--value-rank-only` instead evicts purely by value-RANK to the exact target: the apples-to-
        # apples ranking test vs fifo/random (isolates "is importance×recency a better rank than age").
        if rank_only:
            adapter.forget_decayed(now=now, max_records=target, protect_importance=6, protect_kinds=())
        else:
            adapter.forget_decayed(now=now, max_records=target)
    elif policy == "fifo":
        _evict_oldest(adapter._mem.store, target)
    elif policy == "random":
        _evict_random(adapter._mem.store, target, seed)
    # "none": no eviction
    if trace_sink is not None:
        trace_sink.extend(_trace_rows(adapter, sample, policy=policy, frac=frac, budget=budget))
    return _measure(adapter, sample, budget=budget)


def _trace_rows(adapter: MidasAdapter, sample, *, policy: str, frac: float, budget: int) -> list[dict]:
    """Per-question forgetting audit AFTER eviction: did the gold supporting turns survive, and did
    retrieval still find them? These are the publishable success/failure examples — 'selective
    forgetting kept the fact fifo dropped' or 'forgetting evicted the gold' — not just an average."""
    survived = {
        r.metadata.get("event_id") for r in adapter._mem.store.all() if r.metadata.get("event_id")
    }
    rows: list[dict] = []
    for q in sample.questions:
        if not q.gold_event_ids:
            continue
        result = adapter.query(q.text, token_budget=budget)
        evicted = [g for g in q.gold_event_ids if g not in survived]
        recall = recall_at_k(result, q)
        rows.append(
            {
                "policy": policy,
                "keep": frac,
                "sample": sample.id,
                "qid": q.id,
                "gold": list(q.gold_event_ids),
                "evicted_gold": evicted,
                "recall": recall,
                "verdict": "GOLD EVICTED" if evicted else ("ok" if (recall or 0) > 0 else "kept but not retrieved"),
            }
        )
    return rows


def _run_policy_multi(make_adapter, samples, *, trace_sink: list | None = None, **kw) -> dict:
    """Average a policy's metrics over many samples (e.g. each LongMemEval instance = one buried-fact
    question) — so recall@k under forgetting is measured over N questions, not a single noisy one."""
    runs = [_run_policy(make_adapter, s, trace_sink=trace_sink, **kw) for s in samples]
    out = {}
    for key in runs[0]:
        vals = [r[key] for r in runs if r[key] == r[key]]  # drop NaN (questions without gold)
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def _print_trace(trace_rows: list[dict], samples) -> None:
    """Markdown forgetting trace: the per-question audit table plus the value-vs-fifo diff —
    concrete examples of where selective forgetting worked and where it failed."""
    event_content = {e.id: e.content for s in samples for e in s.events}

    print("\nFORGETTING TRACE (per question, after eviction)")
    print("\n| policy | keep | question | gold | evicted gold | recall@k | verdict |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in trace_rows:
        gold = ", ".join(row["gold"][:4])
        evicted = ", ".join(row["evicted_gold"][:4]) or "-"
        recall = "n/a" if row["recall"] is None else f"{row['recall']:.2f}"
        print(
            f"| {row['policy']} | {int(row['keep'] * 100)}% | {row['qid']} | {gold} | "
            f"{evicted} | {recall} | {row['verdict']} |"
        )

    # The value-vs-fifo diff at each retained fraction: where did the importance×recency rank
    # keep a fact that age-based eviction dropped (win), and where did value evict gold (failure)?
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in trace_rows:
        by_key.setdefault((row["keep"], row["sample"], row["qid"]), {})[row["policy"]] = row
    wins, losses = [], []
    for (keep, _sample_id, qid), policies in sorted(by_key.items()):
        value, fifo = policies.get("value"), policies.get("fifo")
        if not value or not fifo:
            continue
        if not value["evicted_gold"] and fifo["evicted_gold"]:
            wins.append((keep, qid, fifo["evicted_gold"]))
        if value["evicted_gold"]:
            losses.append((keep, qid, value["evicted_gold"]))
    if wins:
        print("\nWhere selective forgetting WORKED (value kept the gold turn fifo evicted):")
        for keep, qid, evicted in wins:
            for gid in evicted[:2]:
                snippet = " ".join(event_content.get(gid, "?").split())[:110]
                print(f"  - keep {int(keep * 100)}%, {qid}: fifo dropped {gid} ('{snippet}')")
    if losses:
        print("\nWhere selective forgetting FAILED (value itself evicted a gold turn):")
        for keep, qid, evicted in losses:
            for gid in evicted[:2]:
                snippet = " ".join(event_content.get(gid, "?").split())[:110]
                print(f"  - keep {int(keep * 100)}%, {qid}: value dropped {gid} ('{snippet}')")
    if not wins and not losses:
        print("\nNo value-vs-fifo divergence on gold turns at these fractions (both kept the gold).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Midas retention / selective-forgetting measurement")
    parser.add_argument("--dataset", choices=["locomo", "multiday", "synthetic", "longmemeval"], default="locomo")
    parser.add_argument("--max-convs", type=int, default=1, help="LoCoMo: conversations to include")
    parser.add_argument("--max-questions", type=int, default=None, help="LongMemEval: questions/instances")
    parser.add_argument("--variant", type=str, default="s", choices=["s", "m", "oracle"])
    parser.add_argument("--local", action="store_true", help="local bge embeddings (no key); else hashing")
    parser.add_argument("--no-rerank", action="store_true", help="disable cross-encoder rerank (faster on big haystacks)")
    parser.add_argument("--min-relevance", type=float, default=None, help="parsimony floor (LoCoMo canonical: 0.75)")
    parser.add_argument("--budget", type=int, default=None, help="token budget (per-dataset default if unset)")
    parser.add_argument("--limit", type=int, default=8, help="top-k recalled")
    parser.add_argument("--local-max-text-chars", type=int, default=2000)
    parser.add_argument("--fractions", type=str, default="0.75,0.5,0.25", help="retained fractions to sweep")
    parser.add_argument("--derive-importance", action="store_true", help="derive per-turn importance from content (no LLM) so forgetting can beat recency on uniform-importance chat")
    parser.add_argument("--structural-importance", action="store_true", help="use StructuralImportance (content salience + assertion/attribute structure) instead of ContentImportance")
    parser.add_argument("--novelty-weight", type=float, default=0.0, help="blend novelty-vs-store into derived importance (0..1); a new fact outranks a repeated one. Pairs with --derive-importance")
    parser.add_argument("--reinforce", action="store_true", help="reinforcement: a restated/near-duplicate turn boosts the matched memory's importance + recency (repetition ⇒ salience)")
    parser.add_argument("--value-rank-only", action="store_true", help="evict purely by value-rank to the exact target (protections off) — the apples-to-apples ranking test vs fifo")
    parser.add_argument("--consolidate", type=float, default=0.0, help="instead of the eviction sweep, measure extractive consolidation (dedup) at this cosine threshold, e.g. 0.95")
    parser.add_argument("--trace", action="store_true", help="per-question forgetting audit: which gold turns survived/were evicted under each policy, plus value-vs-fifo win/failure examples (markdown)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    embedder = None
    embedder_label = "hashing (offline)"
    if args.local:
        from midas.embeddings import DiskCachedEmbedder, LocalEmbedder

        embedder = DiskCachedEmbedder(LocalEmbedder(max_text_chars=args.local_max_text_chars))
        embedder_label = f"local:bge (cache={embedder.path})"

    # Per-dataset defaults that match the canonical runs (so the recall@k baseline is comparable).
    budget = args.budget or {"locomo": 1024, "multiday": 256, "longmemeval": 512, "synthetic": 512}[args.dataset]
    min_relevance = args.min_relevance
    if min_relevance is None and args.dataset == "locomo" and args.local:
        min_relevance = 0.75
    rerank = not args.no_rerank and embedder is not None

    scorer = None
    if args.structural_importance:
        from midas import StructuralImportance

        scorer = StructuralImportance()
    elif args.derive_importance:
        from midas import ContentImportance

        scorer = ContentImportance()

    def make_adapter() -> MidasAdapter:
        return MidasAdapter(
            embedder=embedder, limit=args.limit, rerank=rerank,
            min_relevance=min_relevance, time_aware=True, importance_scorer=scorer,
            novelty_weight=args.novelty_weight, reinforce=args.reinforce,
        )

    dataset = _load(args)
    sample = dataset.samples[0]
    fractions = [float(x) for x in args.fractions.split(",")]

    if args.consolidate > 0:
        cols = ["store", "recall@k", "ctx_current", "avg_tokens"]
        print(f"\nDataset: {dataset.name}  |  events: {len(sample.events)}  |  questions: {len(sample.questions)}")
        print(f"Budget: {budget}  |  top-k: {args.limit}  |  embedder: {embedder_label}  |  "
              f"rerank: {'on' if rerank else 'off'}  |  consolidate threshold: {args.consolidate}")
        print("\n| policy | " + " | ".join(cols) + " |")
        print("| --- | " + " | ".join("---" for _ in cols) + " |")
        a0 = make_adapter(); a0.reset(); a0.ingest(sample.events)
        base = _measure(a0, sample, budget=budget)
        print("| none | " + " | ".join(_fmt(base[c]) for c in cols) + " |")
        a1 = make_adapter(); a1.reset(); a1.ingest(sample.events)
        dropped = a1.consolidate(similarity_threshold=args.consolidate, now=_now_ref(sample))
        m = _measure(a1, sample, budget=budget)
        print(f"| consolidate@{args.consolidate} | " + " | ".join(_fmt(m[c]) for c in cols) + " |")
        pct = len(dropped) * 100 // max(1, int(base["store"]))
        print(f"\nDropped {len(dropped)} near-duplicate records ({pct}% of store). Win = store & "
              "avg_tokens down while recall@k holds (the dropped copies were redundant restatements).")
        return

    samples = dataset.samples  # average the eviction sweep over ALL loaded samples (N questions)
    n_q = sum(len(s.questions) for s in samples)
    print(f"\nDataset: {dataset.name}  |  samples: {len(samples)}  |  questions: {n_q}  "
          f"(recall@k averaged over them)")
    print(f"Budget: {budget}  |  top-k: {args.limit}  |  embedder: {embedder_label}  |  "
          f"rerank: {'on' if rerank else 'off'}  |  min_relevance: {min_relevance}")
    cols = ["store", "recall@k", "ctx_current", "avg_tokens"]
    print("\n| policy | keep | " + " | ".join(cols) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in cols) + " |")

    trace_rows: list[dict] | None = [] if args.trace else None
    base = _run_policy_multi(make_adapter, samples, budget=budget, policy="none", frac=1.0, seed=args.seed)
    print(f"| none | 100% | " + " | ".join(_fmt(base[c]) for c in cols) + " |")
    for frac in fractions:
        for policy in ("value", "fifo", "random"):
            m = _run_policy_multi(make_adapter, samples, budget=budget, policy=policy, frac=frac,
                                  seed=args.seed, rank_only=args.value_rank_only,
                                  trace_sink=trace_rows)
            label = "value (Midas)" if policy == "value" else policy
            print(f"| {label} | {int(frac*100)}% | " + " | ".join(_fmt(m[c]) for c in cols) + " |")

    if trace_rows is not None:
        _print_trace(trace_rows, samples)

    print(
        "\nReading: recall@k & ctx_current higher=better; avg_tokens lower=better; store = records kept "
        "('none' = ceiling + regression anchor). Default `value` PROTECTS the durable/high-importance "
        "tier, so it may keep > target (visible in `store`) — it sheds filler, not facts. Add "
        "`--derive-importance` to give raw chat a no-LLM salience; `--value-rank-only` for the equal-"
        "budget ranking test vs fifo. value==fifo without an importance signal is expected and honest."
    )


def _fmt(x: float) -> str:
    return f"{x:.2f}" if x == x else "n/a"  # x!=x catches NaN


if __name__ == "__main__":
    main()
