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


def answer_recoverable(result: RetrievalResult, question: Question) -> float | None:
    """Offline proxy for answer correctness: is the gold answer string present in
    the assembled context at all? If it isn't, no downstream LLM could answer from
    this context. Cheap but undercounts paraphrased gold — `llm_judge_correct` is
    the real metric when a key is configured."""
    if not question.answer:
        return None
    return 1.0 if question.answer.lower() in result.context.lower() else 0.0


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
