"""Judged A/B: does agent-style distillation lift Midas's BEAM answer-rate above the raw-turn floor?

For each conversation we build two memories from the SAME turns:
  - RAW: ingest the turns verbatim (the no-LLM-ingest floor measured in BENCHMARKS).
  - DISTILLED: an LLM (the agent's-model stand-in) compresses the turns into compact facts; we
    ingest those. Midas itself still calls no LLM — the distiller is the pluggable tier-3 dial.
Then the SAME reader+judge (gpt-4o judge, the competitors' protocol) answers every question from
each memory's retrieved context, and we compare judged answer-rates.

recall@k is N/A for the distilled side (facts are not source turns — like Mem0) so the cross-system
metric here is the judged ANSWER, exactly as the leaderboards use.

Run (needs an OpenRouter key; ~$1 for a few conversations):
    JUDGE_PROVIDER=openrouter OPENROUTER_API_KEY=sk-or-... \
    python -m eval.distill_ab --convs 5 --batch 40
"""
from __future__ import annotations

import argparse
import os

from midas import HTTPDistiller, LocalEmbedder, Memory, StructuralImportance

from .datasets import beam
from .llm import ChatLLM, from_env
from .metrics import llm_answer_and_judge


def _distiller(model: str) -> HTTPDistiller:
    return HTTPDistiller(
        model,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )


def _build_raw(turns: list[str], embedder) -> Memory:
    mem = Memory(embedder=embedder, importance_scorer=StructuralImportance())
    mem.remember_many([{"content": t, "kind": "chat"} for t in turns])
    return mem


def _build_distilled(turns: list[str], embedder, distiller, batch: int, keep_raw: bool = False) -> Memory:
    mem = Memory(embedder=embedder, importance_scorer=StructuralImportance(), distiller=distiller)
    for i in range(0, len(turns), batch):
        try:
            mem.distill(turns[i : i + batch], kind="fact", importance=4, keep_raw=keep_raw)
        except Exception as exc:  # one bad batch shouldn't kill the run
            print(f"  (distill batch {i} failed: {exc})")
    return mem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="100K")
    ap.add_argument("--convs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=40, help="turns per distillation call")
    ap.add_argument("--distill-model", default="openai/gpt-4.1-mini")
    ap.add_argument("--reader-model", default="openai/gpt-4.1-mini")
    ap.add_argument("--judge-model", default="openai/gpt-4o")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--keep-raw", action="store_true", help="distilled facts AUGMENT raw turns (not replace)")
    args = ap.parse_args()

    embedder = LocalEmbedder()
    distiller = _distiller(args.distill_model)
    reader: ChatLLM = from_env(args.reader_model)
    judge: ChatLLM = from_env(args.judge_model)

    dataset = beam(tier=args.tier, max_conversations=args.convs)
    raw_hits: dict[str, list[float]] = {}
    dist_hits: dict[str, list[float]] = {}
    n_facts = n_turns = 0

    for sample in dataset.samples:
        turns = [e.content for e in sample.events]
        n_turns += len(turns)
        raw = _build_raw(turns, embedder)
        dist = _build_distilled(turns, embedder, distiller, args.batch, keep_raw=args.keep_raw)
        n_facts += len(dist.store.all())
        print(f"conv {sample.id}: {len(turns)} turns -> {len(dist.store.all())} distilled facts")

        for q in sample.questions:
            if not q.answer:  # skip abstention/rubric-only
                continue
            for label, mem, bucket in (("raw", raw, raw_hits), ("dist", dist, dist_hits)):
                ctx = mem.assemble(q.text, token_budget=args.budget, now=q.metadata.get("asked_at"))
                _, score = llm_answer_and_judge(reader, ctx, q, judge_llm=judge)
                if score is not None:
                    bucket.setdefault(q.category, []).append(score)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    cats = sorted(set(raw_hits) | set(dist_hits))
    print(f"\n{n_turns} turns -> {n_facts} distilled facts ({n_facts / max(1, n_turns):.0%} of volume)\n")
    print(f"{'category':<26} {'n':>3}  {'RAW':>5}  {'DISTILLED':>9}")
    all_raw, all_dist = [], []
    for c in cats:
        r, d = raw_hits.get(c, []), dist_hits.get(c, [])
        all_raw += r
        all_dist += d
        print(f"{c:<26} {len(d):>3}  {_mean(r):>5.2f}  {_mean(d):>9.2f}")
    print(f"{'OVERALL (answerable)':<26} {len(all_dist):>3}  {_mean(all_raw):>5.2f}  {_mean(all_dist):>9.2f}")


if __name__ == "__main__":
    main()
