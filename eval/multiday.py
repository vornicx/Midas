"""Multi-day failure detector — the eval-first step for long-horizon memory.

Single-conversation correctness misses the failures that define *multi-day* sessions. This
detector runs a synthetic multi-session stream with EVOLVING facts and measures:

  - ctx_current / ctx_stale / ctx_contradict  — deterministic context diagnostics (--context-only)
  - current_acc / stale_rate / abstention_acc — LLM-judged (needs key from .env)

Datasets:
  - multiday   — evolving facts, stale answers, unanswerable Qs (`datasets.multiday`)
  - conflicts  — adversarial near-duplicates + temporal conflicts (`datasets.conflicts`)

    uv run --no-sync python -m eval.multiday --local
    uv run --no-sync python -m eval.multiday --dataset conflicts --context-only --ab-supersede --midas-only

Use `--context-only` for a fully offline, deterministic run (no judge). `--ab-supersede` runs Midas
with belief revision on and off in the same table.
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

from .adapters.base import RetrievalResult
from .adapters.baseline_raw import BaselineRawAdapter
from .adapters.mem0_adapter import Mem0Adapter
from .adapters.midas_adapter import MidasAdapter
from .datasets import conflicts, multiday
from .llm import from_env, load_dotenv
from .metrics import contains_answer as _contains, has_stale_conflict as _has_stale_conflict
from .schema import Question

_ANSWER_SYS = (
    "You answer the user's question using ONLY the provided context. Be concise. "
    "If the context does not contain the answer, reply exactly: I don't know."
)


def _last(text: str, options: list[str]) -> str | None:
    found = re.findall(r"\b(" + "|".join(options) + r")\b", text.upper())
    return found[-1] if found else None


def _answer(llm, context: str, question: str) -> str:
    if not context.strip():
        return "I don't know."
    return llm.complete(
        _ANSWER_SYS,
        f"Context:\n{context or '(no context)'}\n\nQuestion: {question}",
        max_tokens=256,
    )


def _classify_value(llm, predicted: str, current: str, outdated: str | None) -> str:
    out = outdated if outdated is not None else "(no outdated value)"
    verdict = llm.complete(
        "You compare a candidate answer to a CURRENT value and an OUTDATED value. "
        "Reason briefly, then end with one word: CURRENT, OUTDATED, or OTHER.",
        f"Current value: {current}\nOutdated value: {out}\nCandidate: {predicted}\n\n"
        "Does the candidate match the current value, the outdated value, or neither?",
        max_tokens=200,
    )
    return _last(verdict, ["CURRENT", "OUTDATED", "OTHER"]) or "OTHER"


def _classify_abstain(llm, predicted: str) -> str:
    if predicted.strip().lower() in {"i don't know", "i don't know."}:
        return "ABSTAIN"
    verdict = llm.complete(
        "Decide whether a candidate answer ABSTAINS (says it doesn't know / no information / "
        "can't find it) or makes a factual CLAIM. Reason briefly, then end with one word: "
        "ABSTAIN or CLAIM.",
        f"Candidate: {predicted}\n\nABSTAIN or CLAIM?",
        max_tokens=200,
    )
    return _last(verdict, ["ABSTAIN", "CLAIM"]) or "CLAIM"


def _verdict(llm, question: Question, context: str) -> tuple[str, str]:
    predicted = _answer(llm, context, question.text)
    if question.category == "unanswerable":
        return ("unanswerable", _classify_abstain(llm, predicted))
    return (question.category, _classify_value(llm, predicted, question.answer or "", question.stale_answer))


def context_quality(pairs: list[tuple[Question, RetrievalResult]]) -> dict[str, float]:
    """Deterministic context diagnostics before spending LLM-judge calls."""
    answerable = current_hits = 0
    updated = stale_hits = contradictions = 0
    unanswerable = empty_unknown = 0
    token_costs: list[int] = []

    for question, result in pairs:
        context = result.context
        token_costs.append(result.tokens)
        if question.category == "unanswerable":
            unanswerable += 1
            empty_unknown += not context.strip()
            continue

        answerable += 1
        has_current = _contains(context, question.answer)
        current_hits += has_current
        if question.stale_answer is not None:
            updated += 1
            has_stale = _has_stale_conflict(context, question.answer, question.stale_answer)
            stale_hits += has_stale
            contradictions += has_current and has_stale

    avg_tokens = sum(token_costs) / len(token_costs) if token_costs else 0.0
    return {
        "ctx_current": current_hits / answerable if answerable else 0.0,
        "ctx_stale": stale_hits / updated if updated else 0.0,
        "ctx_contradict": contradictions / updated if updated else 0.0,
        "ctx_empty_unknown": empty_unknown / unanswerable if unanswerable else 0.0,
        "avg_tokens": avg_tokens,
    }


def context_details(pairs: list[tuple[Question, RetrievalResult]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for question, result in pairs:
        has_current = _contains(result.context, question.answer)
        has_stale = _has_stale_conflict(result.context, question.answer, question.stale_answer)
        rows.append(
            {
                "id": question.id,
                "category": question.category,
                "current": has_current,
                "stale": has_stale,
                "contradict": has_current and has_stale,
                "empty": not result.context.strip(),
                "tokens": result.tokens,
            }
        )
    return rows


def run_adapter(adapter, sample, *, token_budget: int, llm, workers: int = 6) -> dict[str, float]:
    adapter.reset()
    adapter.ingest(sample.events)
    # Retrieval first (sequential — avoids concurrent store access), then judge in parallel.
    pairs = [(q, adapter.query(q.text, token_budget=token_budget)) for q in sample.questions]
    metrics = context_quality(pairs)

    def work(pair):
        q, res = pair
        return q, _verdict(llm, q, res.context)

    answerable = correct = update_total = update_stale = 0
    abst_total = abst_correct = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for q, (cat, verdict) in pool.map(work, pairs):
            if cat == "unanswerable":
                abst_total += 1
                abst_correct += verdict == "ABSTAIN"
            else:
                answerable += 1
                correct += verdict == "CURRENT"
                if q.stale_answer is not None:  # an updated fact
                    update_total += 1
                    update_stale += verdict == "OUTDATED"
    metrics.update(
        {
            "current_acc": correct / answerable if answerable else 0.0,
            "stale_rate": update_stale / update_total if update_total else 0.0,
            "abstention_acc": abst_correct / abst_total if abst_total else 0.0,
        }
    )
    return metrics


def run_adapter_context_only(adapter, sample, *, token_budget: int) -> dict[str, float]:
    adapter.reset()
    adapter.ingest(sample.events)
    pairs = [(q, adapter.query(q.text, token_budget=token_budget)) for q in sample.questions]
    return context_quality(pairs)


def collect_context_details(adapter, sample, *, token_budget: int) -> list[dict[str, object]]:
    adapter.reset()
    adapter.ingest(sample.events)
    pairs = [(q, adapter.query(q.text, token_budget=token_budget)) for q in sample.questions]
    return context_details(pairs)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Midas multi-day failure detector")
    parser.add_argument("--dataset", choices=["multiday", "conflicts"], default="multiday", help="multiday = evolving facts; conflicts = adversarial near-duplicates + temporal conflicts")
    parser.add_argument("--budget", type=int, default=256, help="token budget (tight, to create retrieval pressure)")
    parser.add_argument("--local", action="store_true", help="use local fastembed embeddings for Midas")
    parser.add_argument("--no-supersede", action="store_true", help="disable Midas belief-revision (for A/B)")
    parser.add_argument("--ab-supersede", action="store_true", help="run Midas twice — supersede on AND off — in the same table (the belief-revision A/B)")
    parser.add_argument("--context-only", action="store_true", help="skip LLM judge; report deterministic context diagnostics")
    parser.add_argument("--midas-only", action="store_true", help="only run Midas; useful for fast R&D loops")
    parser.add_argument("--details", action="store_true", help="print per-question context diagnostics")
    args = parser.parse_args()

    embedder = None
    embedder_label = "hashing (offline)"
    if args.local:
        from midas.embeddings import LocalEmbedder, HashingEmbedder

        embedder = LocalEmbedder()
        embedder_label = f"local:{embedder.model_name}"

    llm = None if args.context_only else from_env()
    dataset = conflicts() if args.dataset == "conflicts" else multiday()
    sample = dataset.samples[0]

    def make_midas(supersede: bool, name: str = "midas") -> MidasAdapter:
        adapter = MidasAdapter(
            embedder=embedder,
            limit=8,
            min_relevance=0.15 if isinstance(embedder, type(None)) or isinstance(embedder, HashingEmbedder) else 0.65,
            neighbor_min_importance=2,
            supersede=supersede,
            supersede_same_kind_only=False,
            exclude_superseded_from_context=True,
        )
        adapter.name = name
        return adapter

    if args.ab_supersede:
        midas_variants = [make_midas(True, "midas(supersede=on)"), make_midas(False, "midas(supersede=off)")]
    else:
        midas_variants = [make_midas(not args.no_supersede)]
    adapters = midas_variants if args.midas_only else [BaselineRawAdapter(), *midas_variants, Mem0Adapter()]
    if args.context_only:
        results = {
            a.name: run_adapter_context_only(a, sample, token_budget=args.budget)
            for a in adapters
        }
    else:
        results = {
            a.name: run_adapter(a, sample, token_budget=args.budget, llm=llm)
            for a in adapters
        }

    cols = ["ctx_current", "ctx_stale", "ctx_contradict", "ctx_empty_unknown", "avg_tokens"]
    if not args.context_only:
        cols += ["current_acc", "stale_rate", "abstention_acc"]
    print(f"\nDataset: {dataset.name}  |  events: {len(sample.events)}  |  questions: {len(sample.questions)}")
    supersede_label = "A/B (on vs off)" if args.ab_supersede else ("off" if args.no_supersede else "on")
    print(
        f"Budget: {args.budget}  |  embedder: {embedder_label}  |  "
        f"midas supersede: {supersede_label}  |  "
        f"judge: {'skipped' if args.context_only else llm.model}\n"
    )
    print("| adapter | " + " | ".join(cols) + " |")
    print("| --- " * (len(cols) + 1) + "|")
    for name, m in results.items():
        print(f"| {name} | " + " | ".join(f"{m[c]:.2f}" for c in cols) + " |")
    if args.details:
        print("\nContext details:")
        print("| adapter | question | category | current | stale | contradict | empty | tokens |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for adapter in adapters:
            for row in collect_context_details(adapter, sample, token_budget=args.budget):
                print(
                    f"| {adapter.name} | {row['id']} | {row['category']} | "
                    f"{int(row['current'])} | {int(row['stale'])} | "
                    f"{int(row['contradict'])} | {int(row['empty'])} | {row['tokens']} |"
                )
    print(
        "\nReading: ctx_current higher=better (current answer present in context);\n"
        "ctx_stale and ctx_contradict LOWER=better (old value / old+new conflict present);\n"
        "ctx_empty_unknown higher=better (unanswerable queries produced no context).\n"
        "LLM metrics, when enabled: current_acc higher=better; stale_rate lower=better; "
        "abstention_acc higher=better."
    )


if __name__ == "__main__":
    main()
