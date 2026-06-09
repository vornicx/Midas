from __future__ import annotations

import re

from .adapters.base import RetrievalResult
from .schema import Question


def recall_at_k(result: RetrievalResult, question: Question) -> float | None:
    """Fraction of gold supporting events that made it into the retrieved set.
    None when the question has no gold annotations."""
    if not question.gold_event_ids:
        return None
    retrieved = set(result.retrieved_event_ids)
    hit = sum(1 for gid in question.gold_event_ids if gid in retrieved)
    return hit / len(question.gold_event_ids)


def precision_at_k(result: RetrievalResult, question: Question) -> float | None:
    """Fraction of retrieved source events that are gold supporting events.

    This is the false-positive stress metric: high recall with low precision means the memory layer
    found the answer but also packed distractors that can mislead an agent loop.
    """
    if not question.gold_event_ids:
        return None
    retrieved = list(dict.fromkeys(result.retrieved_event_ids))
    if not retrieved:
        return 0.0
    gold = set(question.gold_event_ids)
    hit = sum(1 for rid in retrieved if rid in gold)
    return hit / len(retrieved)


def answer_recoverable(result: RetrievalResult, question: Question) -> float | None:
    """Offline proxy for answer correctness: is the gold answer string present in
    the assembled context at all? If it isn't, no downstream LLM could answer from
    this context. Cheap but undercounts paraphrased gold — `llm_judge_correct` is
    the real metric when a key is configured."""
    if not question.answer:
        return None
    return 1.0 if question.answer.lower() in result.context.lower() else 0.0


# --- Staleness diagnostics (deterministic, no LLM) ---------------------------------------------


def contains_answer(text: str, needle: str | None) -> bool:
    return bool(needle) and needle.lower() in text.lower()


def has_stale_conflict(context: str, current: str | None, stale: str | None) -> bool:
    """True when some context line asserts the OUTDATED value without the current one beside it —
    the line an agent could quote as if it were still true."""
    if not stale:
        return False
    for line in context.splitlines():
        if contains_answer(line, stale) and not contains_answer(line, current):
            return True
    return False


# --- Dumb reader: a deterministic, no-LLM reader ablation -------------------------------------
# Reviewers of memory systems rightly ask whether a capable reader is doing "heroic recovery" that
# makes weak retrieval look strong. The dumb reader answers that: it picks the single retrieved turn
# with the highest content-word overlap with the question and scores it against gold. It cannot
# reason, combine turns, or paraphrase — so its score moves with retrieval quality alone. If
# `answer_dumb` tracks recall@k, the published numbers are not reader-inflated.

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DUMB_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have how i in is it its me my of on "
    "or our s t that the their them they this to was we were what when where which who whom why "
    "will with you your".split()
)


def _content_words(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in _DUMB_STOPWORDS]


def extractive_answer(result: RetrievalResult, question: Question) -> str:
    """The dumb reader's 'answer': the retrieved turn with the highest content-word overlap with
    the question. Prefers `top_texts` (relevance-ranked turns); falls back to context lines. Ties
    keep the earlier (higher-ranked) turn, so the score is deterministic."""
    turns = [t for t in (result.top_texts or []) if t.strip()]
    if not turns:
        turns = [
            ln for ln in result.context.split("\n") if ln.strip() and not ln.lstrip().startswith("#")
        ]
    if not turns:
        return ""
    q_words = set(_content_words(question.text))
    if not q_words:
        return turns[0]

    def _overlap(turn: str) -> int:
        return len(q_words & set(_content_words(turn)))

    return max(turns, key=_overlap)


def token_f1(predicted: str, gold: str) -> float:
    """SQuAD-style token F1 between two strings (multiset overlap of word tokens)."""
    pred_tokens = _TOKEN_RE.findall(predicted.lower())
    gold_tokens = _TOKEN_RE.findall(gold.lower())
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    common = 0
    for tok in gold_tokens:
        if pred_counts.get(tok, 0) > 0:
            pred_counts[tok] -= 1
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def dumb_reader_score(result: RetrievalResult, question: Question) -> float | None:
    """Score the dumb reader's extracted turn against gold: 1.0 when the gold answer appears
    verbatim in the chosen turn, else token F1 (partial credit for paraphrase overlap).
    None when the question has no gold answer (unanswerable)."""
    if not question.answer:
        return None
    extracted = extractive_answer(result, question)
    if question.answer.lower() in extracted.lower():
        return 1.0
    return token_f1(extracted, question.answer)


_ANSWER_SYS = (
    "You answer questions using ONLY the provided context: dated memory entries, each tagged with the "
    "date it happened (and how long ago), plus a 'Today is <date>' line.\n"
    "Work in two steps:\n"
    "1) Pick out the specific entries (with their dates) that bear on the question. For multi-part or "
    "'which happened first / in what order' questions, list each relevant entry and its date.\n"
    "2) For 'how long ago' / 'how many days/weeks/months' / 'between' questions, compute from those "
    "dates and the Today date — do the arithmetic explicitly before answering.\n"
    "Then finish with a line in exactly this form:\n"
    "FINAL ANSWER: <answer>\n"
    "Keep <answer> short (a few words or a number, no restatement of the question). "
    "If an entry states the answer (even if paraphrased), give it. But if the entries only contain "
    "information *related* to the topic without actually stating the specific thing asked, do NOT "
    "infer or guess from it — finish with exactly: FINAL ANSWER: I don't know."
)
_JUDGE_SYS = (
    "You grade whether a candidate answer matches the gold answer for a question. "
    "Accept paraphrases and equivalent facts; reject wrong or 'I don't know' answers. "
    "Reason briefly if you must, then end your reply with a single word: YES or NO."
)

_FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*", re.IGNORECASE)


def _extract_final_answer(text: str) -> str:
    """Pull the answer after the last 'FINAL ANSWER:' marker, so a reasoning model's chain-of-thought
    doesn't leak into the graded answer. Falls back to the whole reply if the marker is absent (e.g. a
    non-reasoning model that answered directly, or a reply truncated before the marker)."""
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    return (text[matches[-1].end():] if matches else text).strip()


def llm_answer_and_judge(
    llm, context: str, question: Question, judge_llm=None, nli=None, verify_floor: float = 0.0,
    top_texts=None, verify_llm=None, return_grounded: bool = False,
):
    """Return the reader model's answer and whether the judge accepts it.

    `llm` reads/answers; `judge_llm` grades (defaults to `llm`). Decoupling them lets a benchmark
    FIX the judge (e.g. gpt-4o, as published leaderboards do) while varying the reader — which removes
    the judge-variance confound, since the same answers graded by different judges score differently.

    The reader may be a reasoning model that thinks in the content stream; rather than suppress the
    reasoning (which TANKED correctness in testing: 0.46->0.13), we give it room and require a
    'FINAL ANSWER:' marker we extract robustly. NOTE: absolute correctness is still reader/judge
    dependent; recall@k is the stable, reader-independent metric."""
    if not question.answer:
        return None, None
    judge_llm = judge_llm or llm
    raw = llm.complete(
        _ANSWER_SYS,
        f"Context:\n{context or '(no context)'}\n\nQuestion: {question.text}",
        max_tokens=2048,  # reasoning readers need room to finish CoT *and* write the answer
    )
    predicted = _extract_final_answer(raw)
    verdict = judge_llm.complete(
        _JUDGE_SYS,
        f"Question: {question.text}\nGold answer: {question.answer}\n"
        f"Candidate answer: {predicted}\n\nDoes the candidate match the gold?",
        max_tokens=512,
    )
    # Robust to verbose/reasoning models: take the LAST standalone YES/NO in the reply.
    found = re.findall(r"\b(YES|NO)\b", verdict.upper())
    base = 1.0 if (found and found[-1] == "YES") else 0.0
    if not return_grounded:
        return predicted, base
    # `grounded`: a verifier (local LLM or NLI) that rejects a CORRECT answer makes the agent wrongly
    # abstain on an answerable question -> correctness drops to 0 there. This is the abstention/answer
    # TRADE measured cleanly on the same answers (the cost of the Calibrated lever).
    grounded = base
    if base > 0.0:
        if verify_llm is not None and not _llm_verifies_answer(verify_llm, context, question, predicted):
            grounded = 0.0
        elif nli is not None and verify_floor > 0 and not _nli_grounds_answer(
            nli, context, question, predicted, verify_floor, top_texts
        ):
            grounded = 0.0
    return predicted, base, grounded


# Phrases a reader uses to abstain. Calibrated memory should make the reader say one of these on an
# UNANSWERABLE question (the fact was never stored) instead of confabulating from spurious neighbours.
_ABSTAIN_RE = re.compile(
    r"i\s*don'?t\s*know|do\s*not\s*know|don'?t\s*have|no\s+(information|record|mention|data|details)"
    r"|not\s+(mention|stated|provided|specified|available|found|in\s+the\s+(context|memory|conversation))"
    r"|did\s*n'?o?t?\s*(mention|provide|state|specify)|cannot\s+(find|determine|answer)"
    r"|unable\s+to\s+(find|determine|answer)|no\s+such|isn'?t\s+(mentioned|stated|provided|in)"
    r"|haven'?t\s+(mentioned|told|shared)|wasn'?t\s+(mentioned|stated|provided)",
    re.IGNORECASE,
)


def _nli_grounds_answer(
    nli, context: str, question: Question, answer: str, floor: float, top_texts=None
) -> bool:
    """Post-hoc grounding via CONTRADICTION of the SOURCE turn — the principled Calibrated lever.
    Find the turn the answer was drawn from (the one that lexically contains the answer), then ask NLI:
    does that turn *contradict* "the answer to Q is <answer>"? A confabulation drawn from a wrong-entity
    distractor does ("I have a CAT named Luna" contradicts "the HAMSTER is Luna" -> CON≈1.0); a real
    answer's source does not ("my favorite food is sushi" does not contradict "...is sushi" -> CON≈0).
    Checking only the SOURCE (not every retrieved turn) is essential — the answer-framing makes
    *unrelated* turns read as contradictions, so a max-over-all over-fires. If the answer appears in no
    turn (computed/paraphrased), we keep it (don't override). Local, no LLM."""
    if not answer or _ABSTAIN_RE.search(answer):
        return True  # already abstaining
    turns = list(top_texts or [])
    if not turns:  # fall back to context lines if the adapter didn't surface ranked turns
        turns = [ln for ln in context.split("\n") if ln.strip() and not ln.lstrip().startswith("#")]
    ans_words = set(re.findall(r"[a-zA-Z0-9]{3,}", answer.lower()))
    if not turns or not ans_words:
        return True

    def _overlap(turn: str) -> int:
        return len(ans_words & set(re.findall(r"[a-zA-Z0-9]{3,}", turn.lower())))

    source = max(turns, key=_overlap)
    if _overlap(source) == 0:
        return True  # answer not stated in any retrieved turn (computed/paraphrased) -> don't override
    hyp = f'The answer to the question "{question.text}" is: {answer}'
    return nli.scores(source, hyp).get("CONTRADICTION", 0.0) < floor


def _entity_grounds_answer(question, answer: str, context: str, top_texts=None) -> bool:
    """Entity grounding (no LLM): the answer's SOURCE turn must be about the entity the question asks
    about. Find the turn the answer was drawn from (max lexical overlap with the answer), then check it
    shares the question's focus entity — a confab drawn from a wrong-entity distractor does not. This is
    orthogonal to NLI/cosine (which the fooling distractor also scores high on). Offline-validated on the
    diagnosed failure cases (see midas/entity.py + tests); the end-to-end win still needs a capable
    reader. `question` may be a Question or a raw string."""
    from midas.entity import entity_grounded

    q_text = getattr(question, "text", question)
    if not answer or _ABSTAIN_RE.search(answer):
        return True
    turns = list(top_texts or [])
    if not turns:
        turns = [ln for ln in context.split("\n") if ln.strip() and not ln.lstrip().startswith("#")]
    ans_words = set(re.findall(r"[a-zA-Z0-9]{3,}", answer.lower()))
    if not turns or not ans_words:
        return True

    def _overlap(turn: str) -> int:
        return len(ans_words & set(re.findall(r"[a-zA-Z0-9]{3,}", turn.lower())))

    source = max(turns, key=_overlap)
    if _overlap(source) == 0:
        return True  # answer not stated in any retrieved turn (computed) -> don't override
    return entity_grounded(q_text, source)


_VERIFY_SYS = (
    "You check whether a proposed answer is EXPLICITLY supported by the provided context. "
    "Information that is merely related to the topic but does NOT state the specific thing asked "
    "is NOT support. Reply with exactly one word: YES or NO."
)


def _llm_verifies_answer(verify_llm, context: str, question: Question, answer: str) -> bool:
    """LOCAL reasoning verification (the 'reasoning reader verifies' path): a small local LLM judges
    whether the answer is explicitly supported by the context. Local + $0; catches confabulations the
    cheap NLI/relevance signals miss (real confabs are varied, not just lexical entity-mismatches)."""
    if not answer or _ABSTAIN_RE.search(answer):
        return True
    raw = verify_llm.complete(
        _VERIFY_SYS,
        f"Context:\n{context or '(no context)'}\n\nQuestion: {question.text}\n"
        f"Proposed answer: {answer}\n\nIs the proposed answer explicitly supported by the context?",
        max_tokens=8,
    )
    found = re.findall(r"\b(YES|NO)\b", raw.upper())
    return not (found and found[-1] == "NO")  # grounded unless the verifier says NO


def score_abstention(
    reader_llm, context: str, question: Question, nli=None, verify_floor: float = 0.0, top_texts=None,
    verify_llm=None,
) -> tuple[str, float, float]:
    """For an UNANSWERABLE question (answer=None): does the reader abstain (good) or confabulate (bad)?
    Returns (predicted, 1.0 if abstained else 0.0). Deterministic check on the reader's FINAL ANSWER —
    no judge call. This is the *Calibrated* C: knowing the boundary of what's in memory."""
    raw = reader_llm.complete(
        _ANSWER_SYS,
        f"Context:\n{context or '(no context)'}\n\nQuestion: {question.text}",
        max_tokens=2048,
    )
    predicted = _extract_final_answer(raw)
    base = 1.0 if _ABSTAIN_RE.search(predicted) else 0.0
    # Grounded score: also abstain if the reader's (non-abstaining) answer is contradicted by its
    # source turn. Reported alongside `base` from the SAME answer, so the grounding A/B has ZERO reader
    # noise (the local/hosted reader isn't bit-deterministic, which swamps the effect across two runs).
    grounded = base
    if base == 0.0:
        if verify_llm is not None and not _llm_verifies_answer(verify_llm, context, question, predicted):
            grounded = 1.0  # local LLM says the answer isn't supported -> abstain
        elif nli is not None and verify_floor > 0 and not _nli_grounds_answer(
            nli, context, question, predicted, verify_floor, top_texts
        ):
            grounded = 1.0
    return predicted, base, grounded


def llm_judge_correct(llm, context: str, question: Question) -> float | None:
    """End-to-end correctness: answer the question from the adapter's assembled
    context, then have the judge compare that answer to the gold (paraphrase-tolerant).
    This measures the full retrieve->answer pipeline, the way memory benchmarks do."""
    _, score = llm_answer_and_judge(llm, context, question)
    return score
