"""No-LLM entity-grounded abstention — a candidate lever for the *Calibrated* frontier.

Diagnosed root cause (see docs §5/§7): the reader confabulates an answer drawn from a retrieved
distractor that is about a DIFFERENT entity than the question asks about (e.g. Q asks the *hamster's*
name, the retrieved turn is "I have a *cat* named Luna" → the reader answers "Luna"). Relevance, cosine
and NLI-entailment all fail here because that distractor genuinely *entails* the wrong answer.

This checks something orthogonal: is the answer's **source turn about the entity the question asks
about?** We pull the question's focus nouns and test whether the source mentions them. Pure regex, no
LLM. This module is **measured offline** before any end-to-end claim (the real test needs a capable
reader; the local 1B model exhibits hallucination, not confab-from-distractor — see the benchmark notes).
"""
from __future__ import annotations

import re

# Generic scaffolding that is NOT the entity a question is about: question words, copulas, and the
# attribute/relation words that recur across confab pairs ("favorite", "name") and so must not count
# as a shared entity.
_STOP = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "is", "are", "was", "were",
    "the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "with", "and", "or", "but", "do",
    "does", "did", "have", "has", "had", "his", "her", "their", "my", "your", "our", "its", "this",
    "that", "user", "users", "you", "i", "we", "they", "he", "she", "it", "name", "named", "call",
    "called", "favorite", "favourite", "kind", "type", "about", "tell", "me", "please", "they're",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def entities(text: str) -> set[str]:
    """Salient content words (non-stopword, >=3 chars), lowercased — a turn's rough entity fingerprint.
    Possessives are normalised to the base noun ("user's" -> "user") so scaffolding is filtered."""
    out: set[str] = set()
    for m in _WORD.finditer(text or ""):
        w = m.group().lower()
        if w.endswith("'s"):
            w = w[:-2]  # possessive -> base noun
        if len(w) >= 3 and w not in _STOP:
            out.add(w)
    return out


def question_focus(question: str) -> set[str]:
    """The entity/entities a question is about — its salient nouns minus the generic scaffolding."""
    return entities(question)


def entity_grounded(question: str, source: str, *, min_overlap: int = 1) -> bool:
    """Is the source turn about the entity the question asks about? True if they share >= `min_overlap`
    salient words. A confab drawn from a wrong-entity distractor shares none of the question's focus."""
    return len(question_focus(question) & entities(source)) >= min_overlap
