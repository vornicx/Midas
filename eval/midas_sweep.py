"""Sweep Midas retrieval/assembly knobs on LoCoMo without LLM judge.

This is an R&D tool: use deterministic recall/precision + answer-string recoverability to find
cheap candidate settings, then validate only the promising ones with `eval.runner --judge`.
"""
from __future__ import annotations

import argparse

from .adapters.midas_adapter import MidasAdapter
from .datasets import locomo
from .runner import _resample_questions, run_adapter


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Midas LoCoMo settings")
    parser.add_argument("--max-convs", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local", action="store_true", help="use local fastembed embeddings")
    parser.add_argument("--budgets", default="256,512,1024")
    parser.add_argument("--min-relevances", default="none,0.55,0.65,0.75")
    parser.add_argument("--max-record-chars", default="240,360,600")
    args = parser.parse_args()

    embedder = None
    embedder_label = "hashing"
    if args.local:
        from midas import LocalEmbedder

        embedder = LocalEmbedder()
        embedder_label = f"local:{embedder.model_name}"

    dataset = locomo(max_conversations=args.max_convs)
    _resample_questions(dataset, seed=args.seed)

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    min_relevances: list[float | None] = [
        None if x.strip().lower() == "none" else float(x.strip())
        for x in args.min_relevances.split(",")
        if x.strip()
    ]
    max_record_chars = [int(x.strip()) for x in args.max_record_chars.split(",") if x.strip()]

    print(
        f"\nDataset: {dataset.name} | convs: {args.max_convs} | "
        f"questions: {args.max_questions} | embedder: {embedder_label}\n",
        flush=True,
    )
    print(
        "| budget | min_rel | max_chars | recall@k | precision@k | answer_proxy | avg_tokens | "
        "eff(ans/1k_tok) |",
        flush=True,
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |", flush=True)
    for budget in budgets:
        for min_rel in min_relevances:
            for chars in max_record_chars:
                adapter = MidasAdapter(
                    embedder=embedder,
                    limit=100,
                    min_relevance=min_rel,
                    max_record_chars=chars,
                )
                metrics = run_adapter(
                    adapter,
                    dataset,
                    token_budget=budget,
                    max_questions=args.max_questions,
                )
                print(
                    f"| {budget} | {_fmt(min_rel)} | {chars} | {_fmt(metrics['recall@k'])} | "
                    f"{_fmt(metrics['precision@k'])} | {_fmt(metrics['answer'])} | "
                    f"{_fmt(metrics['avg_tokens'])} | "
                    f"{_fmt(metrics['eff(ans/1k_tok)'])} |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
