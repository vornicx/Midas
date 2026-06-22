"""Judged A/B on BEAM **summarization** — the one category where Midas's deterministic recall@k sits
at ~0.18 and the existing distillation A/B can't even score (summarization ships no reference string,
only a rubric). The open question from `docs/frontier-2026.md`: does *structure-preserving* extraction
(LIGHT-style entity/attribute/value/time cards, written by the agent's model — never Midas's) break
the summarization ceiling, where *naive* distillation already measured a wash?

Arms — all built from the SAME turns, judged by the SAME fixed reader+judge (only the memory differs):
  raw            : verbatim turns (the production floor).
  naive          : raw + naive "compress into one fact" cards (midas.distill's shipped prompt — the
                   already-measured-negative; included so STRUCT is compared against it, not just raw).
  struct         : raw + structure-preserving cards (entity/attribute/value/time, changes kept).
  struct_replace : structure cards ONLY (keep_raw=False) — opt-in; replace was catastrophic on the
                   reference categories, but summarization may differ (raw turns dilute the budget).
naive/struct use keep_raw=True (augment, never replace — measured-safe default).

Grading reproduces BEAM's rubric protocol: the candidate summary is marked against each rubric bullet
("response should contain X"); the score is the FRACTION of bullets covered (rubric recall) in [0,1].
recall@k is N/A here (cards are not source turns, and summarization has no gold turns to hit cleanly)
— the rubric coverage IS the metric, exactly as the BEAM leaderboard grades this category.

Everything runs on LOCAL Ollama, seed-pinned + temp 0 (reproducible, $0, zero egress). Midas still
calls no LLM at ingest/query; the distiller is the agent-model stand-in — the "no *Midas* LLM" line,
not "no LLM". This is the experiment, not a shipped default: nothing here changes Midas's behaviour.

Run, local ($0, needs `ollama serve` + the models pulled; minutes/conv on a 7B, serialized):
    uv run python -m eval.summarization_ab --convs 2 --arms raw,naive,struct \
        --distill-model qwen2.5:3b --reader-model mistral:7b --judge-model mistral:7b

Run, hosted (the real confirm/falsify — a CAPABLE extractor; put a key in .env, e.g.
OPENROUTER_API_KEY=sk-or-… ; defaults: distill/reader gpt-4.1-mini, judge gpt-4o; ~$1–5 for a few convs):
    uv run python -m eval.summarization_ab --backend hosted --convs 8 \
        --arms raw,naive,struct,struct_replace --batch 30
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

from midas import LocalEmbedder, Memory, StructuralImportance
from midas.distill import DISTILL_PROMPT, OllamaDistiller, _parse_facts

from .datasets import beam
from .llm import ChatLLM, from_env, from_ollama, load_dotenv

# The hypothesis, made concrete. Unlike the naive "one compact fact" prompt, this preserves the
# project's evolving STATE: entity -> attribute -> value -> when, every technical specific kept, and
# crucially BOTH sides of any change retained (the temporal/changed-value detail a lossy summary drops
# — the diagnosed reason naive distillation failed on knowledge-update/temporal questions).
STRUCT_PROMPT = (
    "You convert raw conversation turns into STRUCTURED MEMORY CARDS that capture the project's "
    "evolving state. For every concrete fact, decision, implementation detail, problem, or resolution "
    "in the turns, output ONE card on its own line, in exactly this form:\n"
    "  [<entity/component>] <attribute or action> = <specific value/detail> (when: <time or step if "
    "stated, else unspecified>)\n"
    "Rules:\n"
    "- One card per distinct fact. KEEP every technical specific: library/API names, methods, "
    "parameters, versions, numbers, file names. Do NOT generalize or merge distinct facts.\n"
    "- Preserve CHANGES: if a value was updated or a problem later resolved, emit BOTH the earlier "
    "card and the later one — never drop the prior value.\n"
    "- Each card must stand alone (name the entity and the value). No preamble, no numbering, no "
    "commentary. Skip pure small talk.\n"
    "- NEVER copy code, HTML, or markup verbatim — describe what it DOES as a fact "
    "(e.g. '[auth] login route = POST /login validates the password hash').\n\n"
    "Turns:\n{turns}\n\nCards:"
)

_SUMM_SYS = (
    "You answer the user's request using ONLY the provided context (dated memory entries from earlier "
    "conversations). Produce a thorough, specific summary that covers ALL relevant points present in "
    "the context — keep concrete technical details (names, methods, parameters, numbers). Do not invent "
    "anything that is not in the context."
)
_RUBRIC_SYS = (
    "You check whether a candidate summary covers one specific required point. Accept paraphrases and "
    "equivalent content; the point counts as covered only if the summary actually conveys its "
    "substance (not merely the topic). Reason in one short sentence if needed, then end your reply "
    "with a single word: YES or NO."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_YESNO_RE = re.compile(r"\b(YES|NO)\b")
_PREAMBLE = (
    "here", "great", "sure", "it looks", "let's", "okay", "ok,", "below", "in summary",
    "this summary", "these are", "the following", "based on",
)


def _strip_think(text: str) -> str:
    """Drop a reasoning model's <think>…</think> so it doesn't leak into the graded summary."""
    return _THINK_RE.sub("", text).strip()


def _is_card(line: str) -> bool:
    """Keep only lines carrying the card SIGNATURE — a `[entity]` head or the `(when: …)` temporal
    slot. This enforces the exact structure under test AND rejects what a weak extractor leaks on
    code-heavy conversations: raw code/HTML/markup (which contains `=` but never the temporal slot
    or an `[entity]` head), markdown headers, and conversational preamble. If a model can't produce
    the signature, its 'cards' are correctly dropped — that null is a real result, not a bug."""
    s = line.strip()
    if not s or s.startswith("#") or s.endswith(":") or s.lower().startswith(_PREAMBLE):
        return False
    return ("(when:" in s.lower()) or bool(re.match(r"\[[^\]]+\]\s*\S", s))


class StructuredDistiller:
    """Host-agent stand-in that emits structure-preserving cards (not a naive summary). Local Ollama
    `/api/generate`, seed-pinned + temp 0 so the A/B is reproducible. Implements the `Distiller`
    protocol (`.distill(texts) -> list[str]`), so `Memory(distiller=…)` accepts it unchanged."""

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 0,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout

    def distill(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        prompt = STRUCT_PROMPT.format(turns="\n".join(texts))
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature, "seed": self.seed},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())["response"]
        return [c for c in _parse_facts(_strip_think(out)) if _is_card(c)]


class _ChatLLMDistiller:
    """Hosted distiller over any OpenAI-compatible `ChatLLM` (the `--backend hosted` path) — the
    capable host-agent model the local 7B couldn't be (≈100 s/batch on CPU). Same prompts as the local
    tier, so only the model changes: `structure=True` runs the structure-preserving prompt + card-
    signature filter; `structure=False` is the naive baseline. Midas still calls no LLM — this is the
    agent's model doing the extraction."""

    def __init__(self, llm: ChatLLM, prompt: str, *, structure: bool) -> None:
        self.llm = llm
        self.prompt = prompt
        self.structure = structure

    def distill(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        out = self.llm.complete(
            "Output only the requested lines, nothing else — no preamble, no commentary.",
            self.prompt.format(turns="\n".join(texts)),
            max_tokens=1500,
        )
        cards = _parse_facts(_strip_think(out))
        return [c for c in cards if _is_card(c)] if self.structure else cards


def _build_raw(turns: list[str], embedder) -> Memory:
    mem = Memory(embedder=embedder, importance_scorer=StructuralImportance())
    mem.remember_many([{"content": t, "kind": "chat"} for t in turns])
    return mem


def _build_distilled(turns, embedder, distiller, batch: int, arm: str, keep_raw: bool) -> Memory:
    mem = Memory(embedder=embedder, importance_scorer=StructuralImportance(), distiller=distiller)
    for i in range(0, len(turns), batch):
        try:
            mem.distill(
                turns[i : i + batch], kind="fact", importance=4,
                keep_raw=keep_raw, metadata={"arm": arm},
            )
        except Exception as exc:  # one bad batch shouldn't kill the run
            print(f"  (distill batch {i} [{arm}] failed: {exc})")
    return mem


def _rubric_points(rubric) -> list[str]:
    """BEAM rubric bullets look like 'LLM response should contain: <point>'. Strip the boilerplate
    prefix so the judge sees just the required point."""
    points: list[str] = []
    for bullet in rubric or []:
        text = str(bullet)
        if ":" in text:
            text = text.split(":", 1)[1]
        text = text.strip()
        if text:
            points.append(text)
    return points


def rubric_coverage(judge: ChatLLM, summary: str, points: list[str]) -> float | None:
    """Fraction of rubric points the summary covers — BEAM's summarization metric, one judge call per
    point with a FIXED judge (so the score moves with the summary, not judge drift)."""
    if not points:
        return None
    covered = 0
    for point in points:
        verdict = judge.complete(
            _RUBRIC_SYS,
            f"Candidate summary:\n{summary or '(empty)'}\n\nRequired point: {point}\n\n"
            "Does the summary cover this point?",
            max_tokens=256,
        )
        found = _YESNO_RE.findall(verdict.upper())
        if found and found[-1] == "YES":
            covered += 1
    return covered / len(points)


_ARMS = ("raw", "naive", "struct", "struct_replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="100K")
    ap.add_argument("--convs", type=int, default=2)
    ap.add_argument("--arms", default="raw,naive,struct", help=f"comma list of {_ARMS}")
    ap.add_argument("--batch", type=int, default=40, help="turns per distillation call")
    ap.add_argument("--backend", choices=("ollama", "hosted"), default="ollama",
                    help="ollama = local, seed-pinned, $0; hosted = OpenAI-compatible via a .env key "
                         "(from_env) — the capable extractor for the real confirm/falsify of the leap.")
    ap.add_argument("--distill-model", default=None, help="default: qwen2.5:3b (ollama) / gpt-4.1-mini (hosted)")
    ap.add_argument("--reader-model", default=None, help="default: qwen2.5:3b (ollama) / gpt-4.1-mini (hosted)")
    ap.add_argument("--judge-model", default=None, help="default: qwen2.5:3b (ollama) / gpt-4o (hosted)")
    ap.add_argument("--budget", type=int, default=4000, help="per-question context token budget")
    ap.add_argument("--limit", type=int, default=100, help="max records recalled before the budget caps")
    ap.add_argument("--min-ratio", type=float, default=0.0,
                    help="parsimony floor (0 disables). Off by default here: broad summary queries "
                         "don't match specific turns, so the production 0.3 floor starves the context.")
    ap.add_argument("--max-tokens", type=int, default=900, help="reader summary length cap")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in _ARMS]
    if bad:
        ap.error(f"unknown arm(s) {bad}; choose from {_ARMS}")

    embedder = LocalEmbedder()
    if args.backend == "hosted":
        load_dotenv()  # pull the provider key from .env (OPENROUTER_API_KEY / GROQ_API_KEY / …)
        d_model = args.distill_model or "openai/gpt-4.1-mini"
        r_model = args.reader_model or "openai/gpt-4.1-mini"
        j_model = args.judge_model or "openai/gpt-4o"  # fixed strong judge, competitors' protocol
        reader, judge = from_env(r_model), from_env(j_model)
        naive_d = _ChatLLMDistiller(from_env(d_model), DISTILL_PROMPT, structure=False)
        struct_d = _ChatLLMDistiller(from_env(d_model), STRUCT_PROMPT, structure=True)
    else:
        d_model = args.distill_model or "qwen2.5:3b"
        r_model = args.reader_model or "qwen2.5:3b"
        j_model = args.judge_model or "qwen2.5:3b"
        reader, judge = from_ollama(model=r_model), from_ollama(model=j_model)
        naive_d = OllamaDistiller(d_model, timeout=600.0)
        struct_d = StructuredDistiller(d_model)

    def _make(arm: str, turns: list[str]) -> Memory:
        if arm == "raw":
            return _build_raw(turns, embedder)
        if arm == "naive":
            return _build_distilled(turns, embedder, naive_d, args.batch, arm, keep_raw=True)
        if arm == "struct":
            return _build_distilled(turns, embedder, struct_d, args.batch, arm, keep_raw=True)
        if arm == "struct_replace":
            return _build_distilled(turns, embedder, struct_d, args.batch, arm, keep_raw=False)
        raise ValueError(arm)

    dataset = beam(tier=args.tier, max_conversations=args.convs)
    scores: dict[str, list[float]] = {a: [] for a in arms}
    n_summ = 0

    for sample in dataset.samples:
        turns = [e.content for e in sample.events]
        summ_qs = [
            q for q in sample.questions
            if q.category == "summarization" and _rubric_points(q.metadata.get("rubric"))
        ]
        if not summ_qs:
            continue
        mems = {a: _make(a, turns) for a in arms}
        sizes = {a: len(m.store.all()) for a, m in mems.items()}
        print(f"\nconv {sample.id}: {len(turns)} turns -> records {sizes}; {len(summ_qs)} summ Qs")

        for q in summ_qs:
            n_summ += 1
            points = _rubric_points(q.metadata["rubric"])
            row = []
            for arm in arms:
                ctx = mems[arm].assemble(
                    q.text, token_budget=args.budget, limit=args.limit,
                    min_relevance_ratio=args.min_ratio, now=q.metadata.get("asked_at"),
                )
                summary = _strip_think(
                    reader.complete(
                        _SUMM_SYS, f"Context:\n{ctx}\n\nRequest: {q.text}", max_tokens=args.max_tokens
                    )
                )
                cov = rubric_coverage(judge, summary, points)
                if cov is not None:
                    scores[arm].append(cov)
                row.append(f"{arm}={cov:.2f}" if cov is not None else f"{arm}=NA")
            print(f"  [{len(points)} pts] {q.text[:70]!r}  " + "  ".join(row))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n=== BEAM-{args.tier} summarization rubric coverage  [{args.backend}] "
          f"({n_summ} questions, distill={d_model}, reader={r_model}, judge={j_model}) ===")
    print(f"{'arm':<16}{'n':>4}{'coverage':>10}")
    base = _mean(scores.get("raw", []))
    for arm in arms:
        m = _mean(scores[arm])
        delta = f"  ({m - base:+.2f} vs raw)" if arm != "raw" and "raw" in arms else ""
        print(f"{arm:<16}{len(scores[arm]):>4}{m:>10.2f}{delta}")


if __name__ == "__main__":
    main()
