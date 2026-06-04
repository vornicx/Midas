"""Run the adapter benchmark and print a leaderboard.

    uv run python -m eval.runner                                      # synthetic, offline
    uv run --no-sync python -m eval.runner --dataset locomo --local   # local semantic embeddings
    uv run --no-sync python -m eval.runner --dataset locomo --local --judge          # + LLM-judge
    uv run --no-sync python -m eval.runner --dataset locomo --local --judge --mem0   # + Mem0 competitor

`--judge` uses an LLM (key from .env) to answer from each adapter's context and grade
vs gold — the real, cross-system correctness metric. `--mem0` adds the Mem0 competitor.
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

from .adapters.midas_adapter import MidasAdapter
from .adapters.mem0_adapter import Mem0Adapter
from .adapters.base import MemoryAdapter
from .adapters.baseline_raw import BaselineRawAdapter
from .datasets import locomo, longmemeval, synthetic
from .llm import from_env, load_dotenv
from .metrics import answer_recoverable, llm_answer_and_judge, recall_at_k, score_abstention
from .schema import Dataset

_COLUMNS = ["recall@k", "answer", "avg_tokens", "eff(ans/1k_tok)"]

_DEFAULTS = {
    "synthetic": {"budget": 110, "limit": 5},
    "locomo": {"budget": 1024, "limit": 100},
    "longmemeval": {"budget": 2048, "limit": 50},
}


def run_adapter(
    adapter: MemoryAdapter,
    dataset: Dataset,
    *,
    token_budget: int,
    llm=None,
    judge_llm=None,
    verify_nli=None,
    verify_floor: float = 0.0,
    verify_llm=None,
    max_questions: int | None = None,
    trace_questions: bool = False,
    trace_snippets: bool = False,
) -> dict[str, float | None]:
    tracks_recall = getattr(adapter, "tracks_event_ids", True)
    recalls: list[float] = []
    answers: list[float] = []
    answers_grounded: list[float] = []  # answerable correctness AFTER the verifier (its cost)
    abstentions: list[float] = []  # Calibrated: did the reader abstain on unanswerable questions?
    abstentions_grounded: list[float] = []  # + NLI source-grounding override (same answers, no noise)
    token_costs: list[int] = []
    judged: list[tuple] = []  # (question, context) pairs to grade with the LLM
    question_traces: list[dict[str, object]] = []
    asked = 0
    ingest_seconds = 0.0
    query_seconds = 0.0
    n_ingested = 0
    
    # For per-category breakdown
    category_recalls: dict[str, list[float]] = {}
    category_answers: dict[str, list[float]] = {}
    category_tokens: dict[str, list[int]] = {}
    
    for sample in dataset.samples:
        adapter.reset()
        _t_ingest = time.perf_counter()
        adapter.ingest(sample.events)
        ingest_seconds += time.perf_counter() - _t_ingest
        n_ingested += len(sample.events)
        event_by_id = {event.id: event.content for event in sample.events}
        for question in sample.questions:
            if max_questions is not None and asked >= max_questions:
                break
            _t_query = time.perf_counter()
            result = adapter.query(
                question.text, token_budget=token_budget, now=question.metadata.get("asked_at")
            )
            query_seconds += time.perf_counter() - _t_query
            asked += 1
            token_costs.append(result.tokens)
            recall = None
            if tracks_recall:
                recall = recall_at_k(result, question)
                if recall is not None:
                    recalls.append(recall)
            answer = None
            trace_entry = None
            if trace_questions:
                trace_entry = {
                    "id": question.id,
                    "category": question.category,
                    "gold": question.gold_event_ids,
                    "retrieved": result.retrieved_event_ids,
                    "recall@k": recall,
                    "answer": None,
                    "predicted": None,
                    "tokens": result.tokens,
                }
                if trace_snippets:
                    trace_entry["gold_snippets"] = _event_snippets(
                        question.gold_event_ids, event_by_id, limit=3
                    )
                    trace_entry["retrieved_snippets"] = _event_snippets(
                        result.retrieved_event_ids, event_by_id, limit=3
                    )
                question_traces.append(trace_entry)
            if llm is not None:
                judged.append((question, result.context, trace_entry, result.top_texts))
            else:
                answer = answer_recoverable(result, question)
                if answer is not None:
                    answers.append(answer)
                if trace_entry is not None:
                    trace_entry["answer"] = answer
            
            # Track per-category metrics
            category = question.category
            if category not in category_recalls:
                category_recalls[category] = []
                category_answers[category] = []
                category_tokens[category] = []
            
            if tracks_recall:
                if recall is not None:
                    category_recalls[category].append(recall)
            
            if llm is not None:
                # Will be populated later
                pass
            else:
                if answer is not None:
                    category_answers[category].append(answer)
            
            category_tokens[category].append(result.tokens)
            
        if max_questions is not None and asked >= max_questions:
            break

    # Retrieval done; grade with the LLM in parallel (independent calls, httpx is
    # thread-safe). Retrieval stayed sequential to avoid concurrent store access.
    if llm is not None and judged:
        def _judge(pair: tuple):
            question, context, _, top_texts = pair
            # Unanswerable (answer=None) -> measure ABSTENTION (Calibrated); else grade correctness.
            if question.answer is None:
                predicted, base, grounded = score_abstention(
                    llm, context, question, nli=verify_nli, verify_floor=verify_floor,
                    top_texts=top_texts, verify_llm=verify_llm,
                )
                return predicted, (base, grounded), "abstention"
            predicted, base, grounded = llm_answer_and_judge(
                llm, context, question, judge_llm=judge_llm, nli=verify_nli, verify_floor=verify_floor,
                top_texts=top_texts, verify_llm=verify_llm, return_grounded=True,
            )
            return predicted, (base, grounded), "answer"

        with ThreadPoolExecutor(max_workers=getattr(llm, "max_workers", 6)) as pool:
            judged_results = list(pool.map(_judge, judged))
            for (question, _, trace_entry, _tt), (predicted, a, kind) in zip(judged, judged_results):
                if kind == "abstention":
                    if a is not None:
                        abstentions.append(a[0])           # base (reader alone)
                        abstentions_grounded.append(a[1])  # + NLI grounding (same answer, no noise)
                    a = a[1]  # for the trace
                elif a is not None:
                    answers.append(a[0])           # base correctness (no verifier override)
                    answers_grounded.append(a[1])  # after verifier override (same answer, no noise)
                    # Track per-category for LLM judge
                    category = question.category
                    if category not in category_answers:
                        category_answers[category] = []
                    category_answers[category].append(a[0])
                    a = a[0]
                if trace_entry is not None:
                    trace_entry["answer"] = a
                    trace_entry["predicted"] = predicted

    avg_answer = _mean(answers)
    avg_tokens = _mean(token_costs)
    efficiency = (avg_answer / avg_tokens * 1000.0) if avg_tokens else 0.0
    
    # Build category breakdown
    category_breakdown = {}
    for category in set(category_recalls.keys()) | set(category_answers.keys()):
        cat_recalls = category_recalls.get(category, [])
        cat_answers = category_answers.get(category, [])
        cat_tokens = category_tokens.get(category, [])
        category_breakdown[category] = {
            "count": len(cat_tokens),
            "recall@k": _mean(cat_recalls) if cat_recalls else None,
            "answer": _mean(cat_answers) if cat_answers else None,
            "avg_tokens": _mean(cat_tokens) if cat_tokens else 0.0,
        }
    
    return {
        "recall@k": _mean(recalls) if tracks_recall else None,
        "answer": avg_answer,
        "answer_grounded": _mean(answers_grounded) if answers_grounded else None,
        "abstention": _mean(abstentions) if abstentions else None,
        "abstention_grounded": _mean(abstentions_grounded) if abstentions_grounded else None,
        "n_abstention": len(abstentions),
        "avg_tokens": avg_tokens,
        "eff(ans/1k_tok)": efficiency,
        "ingest_ms/event": (ingest_seconds / n_ingested * 1000.0) if n_ingested else 0.0,
        "query_ms/q": (query_seconds / asked * 1000.0) if asked else 0.0,
        "uses_internal_llm": getattr(adapter, "uses_internal_llm", False),
        "categories": category_breakdown,
        "traces": question_traces,
    }


def _mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def leaderboard(results: dict[str, dict[str, float | None]]) -> str:
    header = "| adapter | " + " | ".join(_COLUMNS) + " |"
    sep = "| --- " * (len(_COLUMNS) + 1) + "|"
    rows = [header, sep]
    for name, metrics in results.items():
        cells = " | ".join(_fmt(metrics[c]) for c in _COLUMNS)
        rows.append(f"| {name} | {cells} |")
    return "\n".join(rows)


def cost_latency_table(results: dict[str, dict[str, float | None]]) -> str:
    """Memory-layer cost/latency — the axis LLM-heavy systems (Mem0, Hindsight) don't
    publish. Excludes the shared reader/judge LLM, which is identical across systems."""
    header = "| adapter | ingest ms/event | query ms/query | ctx tokens | memory-layer LLM |"
    sep = "| --- | --- | --- | --- | --- |"
    rows = [header, sep]
    for name, m in results.items():
        llm = "internal (>=1/session)" if m.get("uses_internal_llm") else "0 (none)"
        rows.append(
            f"| {name} | {_fmt(m.get('ingest_ms/event'))} | {_fmt(m.get('query_ms/q'))} "
            f"| {_fmt(m.get('avg_tokens'))} | {llm} |"
        )
    return "\n".join(rows)


def category_breakdown_table(results: dict[str, dict[str, float | None]]) -> str:
    """Generate a table showing per-category metrics for all adapters."""
    if not results:
        return ""
    
    # Collect all categories across all adapters
    all_categories = set()
    for adapter_results in results.values():
        if "categories" in adapter_results:
            all_categories.update(adapter_results["categories"].keys())
    
    if not all_categories:
        return ""
    
    all_categories = sorted(all_categories)
    
    # Build header
    header_cells = ["category", "count"] + [f"{name}_recall" for name in results.keys()] + [f"{name}_answer" for name in results.keys()]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "| --- " * (len(header_cells) + 1) + "|"
    
    rows = [header, sep]
    
    # Build rows
    for category in all_categories:
        cells = [category]
        count = 0
        for adapter_name, adapter_results in results.items():
            cat_data = adapter_results.get("categories", {}).get(category, {})
            count = cat_data.get("count", 0)
            cells.append(str(count) if adapter_name == list(results.keys())[0] else "")
        
        for adapter_name, adapter_results in results.items():
            cat_data = adapter_results.get("categories", {}).get(category, {})
            recall = cat_data.get("recall@k")
            cells.append(_fmt(recall))
        
        for adapter_name, adapter_results in results.items():
            cat_data = adapter_results.get("categories", {}).get(category, {})
            answer = cat_data.get("answer")
            cells.append(_fmt(answer))
        
        rows.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(rows)


def category_breakdown_by_adapter(results: dict[str, dict[str, float | None]]) -> str:
    """Generate breakdown tables per adapter."""
    tables = []
    for adapter_name, metrics in results.items():
        categories = metrics.get("categories", {})
        if not categories:
            continue
        
        header = f"\n#### {adapter_name} — category breakdown"
        table_header = "| category | count | recall@k | answer | avg_tokens |"
        sep = "| --- | --- | --- | --- | --- |"
        rows = [header, "", table_header, sep]
        
        for category, cat_metrics in sorted(categories.items()):
            count = cat_metrics.get("count", 0)
            recall = cat_metrics.get("recall@k")
            answer = cat_metrics.get("answer")
            avg_tokens = cat_metrics.get("avg_tokens")
            rows.append(
                f"| {category} | {count} | {_fmt(recall)} | {_fmt(answer)} | {_fmt(avg_tokens)} |"
            )
        
        tables.append("\n".join(rows))
    
    return "\n".join(tables) if tables else ""


def question_trace_by_adapter(results: dict[str, dict[str, float | None]]) -> str:
    """Generate compact per-question trace tables for debugging failures."""
    tables = []
    for adapter_name, metrics in results.items():
        traces = metrics.get("traces", [])
        if not traces:
            continue
        include_snippets = any(
            trace.get("gold_snippets") or trace.get("retrieved_snippets") for trace in traces
        )

        rows = [
            f"\n#### {adapter_name} - question trace",
            "",
        ]
        headers = ["id", "category", "recall@k", "answer", "tokens", "predicted", "gold", "retrieved"]
        if include_snippets:
            headers.extend(["gold_snippets", "retrieved_snippets"])
        rows.append("| " + " | ".join(headers) + " |")
        rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for trace in traces:
            gold_ids = trace.get("gold", [])
            retrieved_ids = trace.get("retrieved", [])
            gold = ", ".join(gold_ids[:4]) if isinstance(gold_ids, list) else "-"
            retrieved = ", ".join(retrieved_ids[:6]) if isinstance(retrieved_ids, list) else "-"
            if isinstance(retrieved_ids, list) and len(retrieved_ids) > 6:
                retrieved += f" (+{len(retrieved_ids) - 6})"
            predicted = _short_cell(trace.get("predicted"))
            cells = [
                str(trace.get("id", "-")),
                str(trace.get("category", "-")),
                _fmt(trace.get("recall@k")),
                _fmt(trace.get("answer")),
                _fmt(trace.get("tokens")),
                predicted,
                gold or "-",
                retrieved or "-",
            ]
            if include_snippets:
                cells.extend(
                    [
                        _short_cell(trace.get("gold_snippets"), max_chars=140),
                        _short_cell(trace.get("retrieved_snippets"), max_chars=140),
                    ]
                )
            rows.append("| " + " | ".join(cells) + " |")
        tables.append("\n".join(rows))

    return "\n".join(tables) if tables else ""


def _short_cell(value: object, *, max_chars: int = 90) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        text = " || ".join(" ".join(str(item).split()) for item in value)
    else:
        text = " ".join(str(value).split())
    text = text.replace("|", "/")
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text or "-"


def _event_snippets(
    ids: list[str],
    event_by_id: dict[str, str],
    *,
    limit: int,
    max_chars: int = 120,
) -> list[str]:
    snippets = []
    seen: set[str] = set()
    for event_id in ids:
        if event_id in seen:
            continue
        seen.add(event_id)
        content = event_by_id.get(event_id)
        if not content:
            continue
        snippets.append(f"{event_id}: {' '.join(content.split())[:max_chars]}")
        if len(snippets) >= limit:
            break
    return snippets


def _resample_questions(dataset: Dataset, *, seed: int) -> None:
    """Shuffle each sample's questions (seeded) so a later cap takes a representative
    spread across sessions, not just the early ones — which unfairly favours retrieval
    over the recency baseline."""
    rng = random.Random(seed)
    for sample in dataset.samples:
        rng.shuffle(sample.questions)


def _load(
    dataset: str,
    *,
    max_convs: int,
    max_questions: int | None = None,
    variant: str = "s",
    longmemeval_path: str | None = None,
    abstention_only: bool = False,
) -> Dataset:
    if dataset == "synthetic":
        return synthetic()
    if dataset == "locomo":
        return locomo(max_conversations=max_convs)
    if dataset == "longmemeval":
        return longmemeval(
            path=longmemeval_path, max_questions=max_questions, variant=variant,
            abstention_only=abstention_only,
        )
    raise ValueError(f"unknown dataset: {dataset}")


def _make_embedder(args: argparse.Namespace) -> tuple[object | None, str]:
    if args.openai:
        from midas.embeddings import OpenAIEmbedder

        return OpenAIEmbedder(), "openai"
    if args.local:
        from midas.embeddings import DiskCachedEmbedder, LocalEmbedder

        emb = LocalEmbedder(
            batch_size=args.local_batch_size,
            max_text_chars=args.local_max_text_chars,
        )
        label = f"local:{emb.model_name}, chars={emb.max_text_chars}, batch={emb.batch_size}"
        if args.no_local_embedding_cache:
            return emb, label
        cached = DiskCachedEmbedder(emb, path=args.local_embedding_cache_path)
        return cached, f"{label}, disk_cache={cached.path}"
    return None, "hashing (offline)"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Midas agentic-memory eval harness")
    parser.add_argument("--dataset", choices=["synthetic", "locomo", "longmemeval"], default="synthetic")
    parser.add_argument("--budget", type=int, default=None, help="token budget (per-dataset default if unset)")
    parser.add_argument("--limit", type=int, default=None, help="top-k (per-dataset default if unset)")
    parser.add_argument("--max-convs", type=int, default=2, help="LoCoMo: conversations to include")
    parser.add_argument("--max-questions", type=int, default=None, help="cap questions (default 25 when --judge)")
    parser.add_argument("--variant", type=str, default="s", choices=["s", "m", "oracle"], help="LongMemEval: variant to use (s, m, oracle)")
    parser.add_argument("--longmemeval-path", type=str, default=None, help="Path to LongMemEval JSON file (overrides variant default)")
    parser.add_argument("--longmemeval-abstention", action="store_true", help="LongMemEval: load only the abstention (unanswerable) questions, to measure Calibrated/abstention")
    parser.add_argument("--local", action="store_true", help="use local fastembed semantic embeddings (no key)")
    parser.add_argument("--local-batch-size", type=int, default=32, help="local embedder batch size")
    parser.add_argument("--local-max-text-chars", type=int, default=2000, help="truncate local embedding inputs")
    parser.add_argument("--no-local-embedding-cache", action="store_true", help="disable local SQLite embedding cache")
    parser.add_argument("--local-embedding-cache-path", type=str, default=None, help="path for local SQLite embedding cache")
    parser.add_argument("--openai", action="store_true", help="use OpenAI embeddings (needs OPENAI_API_KEY)")
    parser.add_argument("--judge", action="store_true", help="use an LLM to answer + grade vs gold (key from .env)")
    parser.add_argument("--judge-ollama", action="store_true", help="use a LOCAL Ollama model as reader+judge (reproducible: temp 0 + seed)")
    parser.add_argument("--judge-model", type=str, default=None, help="judge model name (Ollama model when --judge-ollama)")
    parser.add_argument("--reader-model", type=str, default=None, help="reader/answerer model (default: judge model); set to fix the judge and vary the reader")
    parser.add_argument("--judge-workers", type=int, default=None, help="judge/reader concurrency (default: model's; lower for flaky gateways)")
    parser.add_argument("--mem0", action="store_true", help="add the Mem0 competitor (needs mem0ai + groq + key)")
    parser.add_argument("--seed", type=int, default=0, help="seed for representative question sampling")
    parser.add_argument("--midas-min-relevance", type=float, default=None, help="drop Midas hits below this relevance")
    parser.add_argument("--midas-max-record-chars", type=int, default=600, help="truncate Midas record bodies")
    parser.add_argument("--midas-only", action="store_true", help="only run Midas")
    parser.add_argument("--midas-supersede", action="store_true", help="enable Midas belief-revision (regression test)")
    parser.add_argument("--midas-supersede-convo", action="store_true", help="conversational belief revision: chat may revise chat on a strict cue (implies --midas-supersede)")
    parser.add_argument("--midas-abstain-floor", type=float, default=0.0, help="Calibrated abstention: emit a firm 'don't guess' signal when top-hit relevance < FLOOR (use with rerank on, e.g. 0.5)")
    parser.add_argument("--midas-nli", action="store_true", help="use a local NLI model (no LLM): contradiction-gated belief revision; pairs with --midas-supersede-convo")
    parser.add_argument("--midas-abstain-nli", type=float, default=0.0, help="Calibrated abstention via local NLI: abstain when no retrieved turn ENTAILS an answer at >= FLOOR (e.g. 0.5)")
    parser.add_argument("--answer-verify-nli", type=float, default=0.0, help="Calibrated: post-hoc NLI grounding — override the reader's answer to 'I don't know' if no retrieved turn entails it at >= FLOOR")
    parser.add_argument("--answer-verify-llm", action="store_true", help="Calibrated: LOCAL reasoning verification — the reader self-checks if its answer is supported by the context (local, $0)")
    parser.add_argument("--midas-no-rerank", action="store_true", help="disable Midas cross-encoder reranker")
    parser.add_argument("--midas-no-time", action="store_true", help="ablation: ignore dataset event time (recency = ingest order, not real timestamps)")
    parser.add_argument("--midas-hybrid", action="store_true", help="hybrid retrieval: fuse BM25 (lexical) with semantic recall (no LLM)")
    parser.add_argument("--midas-hybrid-fusion", type=str, default="rrf", choices=["rrf", "max"], help="hybrid fusion method (default rrf; max = legacy)")
    parser.add_argument(
        "--midas-context-order",
        choices=["relevance", "chronological", "recency"],
        default="recency",
        help="order selected Midas records in the assembled context",
    )
    parser.add_argument("--trace-questions", action="store_true", help="print per-question recall/answer trace")
    parser.add_argument("--trace-snippets", action="store_true", help="include short gold/retrieved event snippets in question traces")
    args = parser.parse_args()

    defaults = _DEFAULTS[args.dataset]
    budget = args.budget if args.budget is not None else defaults["budget"]
    limit = args.limit if args.limit is not None else defaults["limit"]
    max_q = args.max_questions if args.max_questions is not None else (25 if args.judge else None)

    dataset = _load(
        args.dataset,
        max_convs=args.max_convs,
        max_questions=args.max_questions,
        variant=args.variant,
        longmemeval_path=args.longmemeval_path,
        abstention_only=args.longmemeval_abstention,
    )
    if max_q is not None:
        _resample_questions(dataset, seed=args.seed)
    embedder, embedder_label = _make_embedder(args)
    midas_min_relevance = args.midas_min_relevance
    if midas_min_relevance is None and (args.local or args.openai) and args.dataset == "locomo":
        midas_min_relevance = 0.75

    llm = None          # reader/answerer
    judge_llm = None    # grader (may differ from reader, to fix the judge across reader sweeps)
    answer_mode = "substring proxy"
    if args.judge:
        if args.judge_ollama:
            from .llm import from_ollama
            llm = from_ollama(model=args.judge_model or "sam860/dolphin3-qwen2.5:1.5b")
            judge_llm = llm
        else:
            judge_llm = from_env(args.judge_model)
            # Reader defaults to the judge model; --reader-model decouples them (fixed judge, varied
            # reader) — the apples-to-apples setup published leaderboards use (e.g. gpt-4o judge).
            llm = from_env(args.reader_model) if args.reader_model else judge_llm
        if args.judge_workers:
            llm.max_workers = args.judge_workers  # lower for flaky/overloaded hosted gateways
            judge_llm.max_workers = args.judge_workers
        answer_mode = (
            f"LLM-judge (reader={llm.model}, judge={judge_llm.model})"
            if llm is not judge_llm
            else f"LLM-judge ({llm.model})"
        )

    verify_nli = None
    if args.answer_verify_nli > 0:
        from midas.nli import LocalNLI

        verify_nli = LocalNLI()  # local, $0 — post-hoc grounding of the reader's answer

    midas_rerank = not args.midas_no_rerank and embedder is not None
    midas = MidasAdapter(
        embedder=embedder,
        limit=limit,
        rerank=midas_rerank,
        min_relevance=midas_min_relevance,
        max_record_chars=args.midas_max_record_chars,
        supersede=args.midas_supersede or args.midas_supersede_convo,
        supersede_conversational=args.midas_supersede_convo,
        use_nli=args.midas_nli,
        abstention_relevance_floor=args.midas_abstain_floor,
        abstention_entailment_floor=args.midas_abstain_nli,
        context_order=args.midas_context_order,
        hybrid=args.midas_hybrid,
        hybrid_fusion=args.midas_hybrid_fusion,
        time_aware=not args.midas_no_time,
    )
    adapters: list[MemoryAdapter] = [midas] if args.midas_only else [BaselineRawAdapter(), midas]
    if args.mem0 and not args.midas_only:
        adapters.append(Mem0Adapter())

    results = {
        a.name: run_adapter(
            a,
            dataset,
            token_budget=budget,
            llm=llm,
            judge_llm=judge_llm,
            verify_nli=verify_nli,
            verify_floor=args.answer_verify_nli,
            verify_llm=llm if args.answer_verify_llm else None,
            max_questions=max_q,
            trace_questions=args.trace_questions,
            trace_snippets=args.trace_snippets,
        )
        for a in adapters
    }

    n_events = sum(len(s.events) for s in dataset.samples)
    n_questions = sum(len(s.questions) for s in dataset.samples)
    print(
        f"\nDataset: {dataset.name}  |  samples: {len(dataset.samples)}  |  "
        f"events: {n_events}  |  questions: {n_questions}"
        + (f"  (random sample of {max_q}, seed {args.seed})" if max_q else "")
    )
    print(f"Budget: {budget} tokens  |  Midas top-k: {limit}  |  embedder: {embedder_label}")
    print(
        f"Midas knobs: min_relevance={midas_min_relevance}  |  "
        f"max_record_chars={args.midas_max_record_chars}  |  "
        f"rerank={'on' if midas_rerank else 'off'}  |  "
        f"context_order={args.midas_context_order}"
    )
    if embedder is not None and hasattr(embedder, "hits") and hasattr(embedder, "misses"):
        print(f"Embedding cache: hits={embedder.hits}  |  misses={embedder.misses}")
    print(f"answer metric: {answer_mode}\n")
    print(leaderboard(results))

    # Calibrated (abstention): on UNANSWERABLE questions, did the reader abstain instead of confabulate?
    if any(r.get("abstention") is not None for r in results.values()):
        print("\nABSTENTION (Calibrated) — higher = abstains instead of confabulating on unanswerable Qs")
        for name, m in results.items():
            if m.get("abstention") is not None:
                g = m.get("abstention_grounded")
                extra = f"  ->  +verifier: {_fmt(g)}" if g is not None and g != m["abstention"] else ""
                print(f"  {name} abstention: {_fmt(m['abstention'])}  (n={m.get('n_abstention', 0)}){extra}")
            ag = m.get("answer_grounded")
            if ag is not None and m.get("answer") is not None and ag != m["answer"]:
                print(f"  {name} answer (verifier cost): {_fmt(m['answer'])}  ->  {_fmt(ag)}")

    print("\n" + "=" * 60)
    print("COST / LATENCY  (memory layer only - excludes shared reader/judge LLM)")
    print("=" * 60)
    print(cost_latency_table(results))
    print(
        "\nMidas & baseline make ZERO LLM calls in ingest/query (local embeddings only).\n"
        "LLM-based memory (Mem0, Hindsight) pays an LLM call per ingested session - the\n"
        "cost/latency/privacy axis where the no-LLM design wins. The reader/judge LLM is\n"
        "identical across systems, so it is excluded here."
    )

    # Show per-category breakdown
    has_categories = any("categories" in r and r["categories"] for r in results.values())
    if has_categories:
        print("\n" + "=" * 60)
        print("CATEGORY BREAKDOWN")
        print("=" * 60)
        print(category_breakdown_by_adapter(results))

    if args.trace_questions:
        print("\n" + "=" * 60)
        print("QUESTION TRACE")
        print("=" * 60)
        print(question_trace_by_adapter(results))
    
    print(
        "\nReading it: higher recall@k and answer = better; lower avg_tokens = cheaper.\n"
        "recall@k is '-' for Mem0 (it returns synthesized facts, not source turn ids);\n"
        "judged 'answer' is the cross-system metric."
    )


if __name__ == "__main__":
    main()
