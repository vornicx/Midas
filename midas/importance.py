"""No-LLM per-turn importance scoring — the signal selective forgetting needs on raw conversation.

`Memory.forget_decayed` protects the high-importance/durable tier, and the measured win is decisive
*when an importance signal exists* (synthetic/multiday: recall@k 1.00 vs recency 0.33-0.60 at half the
store). But raw chat turns arrive undifferentiated (every turn `importance=1`), so on conversational
data forgetting reduces to recency. `ContentImportance` recovers a salience from the turn's CONTENT
alone — no LLM, no API, deterministic — so a concrete factual turn ("I adopted a dog named Max in
2019") outranks chit-chat ("haha yeah sounds good") and survives forgetting while the filler decays.

The features are deliberately cheap and transparent (the project's eval-first ethos — an auditable
heuristic you can reason about, not an opaque model):
  - **content-word density** — salient (non-stopword, >3-char) words; backchannels have almost none.
  - **specifics** — numbers/dates (a token with a digit) and proper-noun-like tokens (capitalised
    mid-sentence), the markers of durable facts (names, places, amounts, deadlines).
  - **anti-backchannel floor** — very short turns with no specifics are pinned to the minimum.

It is a *salience* heuristic, not fact extraction: it separates fact-bearing turns from filler, which
is exactly what forgetting needs to keep the answer-bearing tier. Tune or replace via `Memory(
importance_scorer=...)`.
"""
from __future__ import annotations

import re

# Common words that carry no salience — kept small and generic (no domain/benchmark-specific terms,
# the same discipline as the supersession stopwords: a hand-tuned list overfits one dataset).
_STOP = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can", "we", "our", "us", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "they", "them", "my", "your", "his", "her",
    "their", "me", "him", "so", "just", "yeah", "yes", "no", "ok", "okay", "haha", "lol", "hey", "hi",
    "oh", "well", "really", "very", "too", "also", "then", "than", "as", "if", "not", "what", "when",
    "how", "why", "who", "about", "from", "up", "out", "get", "got", "like", "good", "nice", "cool",
    "thanks", "thank", "please", "sure", "okay",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_HAS_DIGIT = re.compile(r"\d")
_SENT_END = re.compile(r"[.!?]$")


class ContentImportance:
    """Score a turn's importance in 1..5 from its content alone (no LLM). Callable: `scorer(text)`."""

    def __init__(self, *, min_importance: int = 1, max_importance: int = 5) -> None:
        self.min_importance = min_importance
        self.max_importance = max_importance

    def score(self, text: str) -> int:
        text = (text or "").strip()
        if not text:
            return self.min_importance
        tokens = text.split()
        n_tokens = len(tokens)

        content: set[str] = set()
        for w in tokens:
            m = _WORD.fullmatch(w.strip(".,;:!?\"'()[]"))
            if m and len(m.group()) > 3 and m.group().lower() not in _STOP:
                content.add(m.group().lower())
        n_content = len(content)

        n_digit = sum(1 for w in tokens if _HAS_DIGIT.search(w))

        # Proper-noun-like: a capitalised alpha token (>2 chars, not a stopword) that does NOT start a
        # sentence — the first word of every sentence is capitalised too, so we skip those to cut
        # false positives. Tracks whether the previous token ended a sentence.
        n_proper = 0
        prev_ends_sentence = True  # the very first token starts a sentence
        for w in tokens:
            core = w.strip(".,;:!?\"'()[]")
            if (
                not prev_ends_sentence
                and len(core) > 2
                and core[:1].isupper()
                and core.isalpha()
                and core.lower() not in _STOP
            ):
                n_proper += 1
            prev_ends_sentence = bool(_SENT_END.search(w))

        # Anti-backchannel: a short turn with no specifics is filler ("haha yeah sounds good").
        if n_tokens <= 4 and n_digit == 0 and n_proper == 0 and n_content <= 1:
            return self.min_importance

        score = 1
        if n_content >= 3:
            score += 1
        if n_content >= 7:
            score += 1
        if n_digit >= 1:
            score += 1  # a number/date/amount — the spine of a durable fact
        if n_proper >= 1:
            score += 1  # a named entity — person, place, product
        return max(self.min_importance, min(self.max_importance, score))

    __call__ = score


class StructuralImportance(ContentImportance):
    """Content salience **plus structure** — boost a turn that *asserts a durable attribute* (especially
    about the user: "my X is Y", "I'm a…", "I prefer…"), and demote questions / meta / backchannel.

    The discriminator the bag-of-features content score misses: an *assertion of a personal fact* vs a
    *question* that merely mentions the same salient words (a question and a fact about "sushi in Tokyo"
    score alike on content features, but only the assertion is worth remembering). Still no LLM — cheap,
    generic English cues (no dataset-specific terms), composed on top of `ContentImportance`.
    """

    _FIRST_PERSON = re.compile(
        r"\b(i'?m|i\s+am|my|mine|i\s+have|i'?ve|i\s+(?:prefer|like|love|hate|need|use|own|drive|speak|"
        r"work|live)|i\s+was\s+born)\b",
        re.I,
    )
    _DURABLE = re.compile(
        r"\b(prefer|allergic|favou?rite|deadline|due|born|named|called|decided|always|never|must|"
        r"require|policy|rule)\b",
        re.I,
    )
    _COPULA = re.compile(r"\b(is|are|was|were|named|called)\b", re.I)
    _META = re.compile(
        r"\b(let'?s|let\s+me|can\s+you|could\s+you|should\s+we|shall\s+we|please|thanks|thank\s+you|"
        r"sounds?\s+good|got\s+it|no\s+problem)\b",
        re.I,
    )

    def score(self, text: str) -> int:
        text = (text or "").strip()
        base = super().score(text)
        is_question = text.endswith("?")
        boost = 0
        if self._FIRST_PERSON.search(text):
            boost += 1  # a fact about the user — the durable stuff an assistant must keep
        if self._DURABLE.search(text):
            boost += 1  # preference / constraint / deadline markers
        if not is_question and self._COPULA.search(text):
            boost += 1  # a declarative attribute assertion (not a question)
        penalty = 0
        if is_question:
            penalty += 1  # questions are rarely the durable fact to remember
        elif boost == 0 and self._META.search(text):
            penalty += 1  # pure meta / acknowledgement
        score = base + min(boost, 2) - penalty
        return max(self.min_importance, min(self.max_importance, score))

    __call__ = score
