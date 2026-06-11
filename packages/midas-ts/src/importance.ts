/** No-LLM per-turn importance scoring — a faithful port of `midas/importance.py`.
 * ContentImportance: content-word density + specifics (digits, proper-noun-likes) with an
 * anti-backchannel floor. StructuralImportance adds assertion-vs-question structure. */

const STOP = new Set([
  "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is",
  "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
  "would", "could", "should", "may", "might", "must", "can", "we", "our", "us", "it", "its", "this",
  "that", "these", "those", "i", "you", "he", "she", "they", "them", "my", "your", "his", "her",
  "their", "me", "him", "so", "just", "yeah", "yes", "no", "ok", "okay", "haha", "lol", "hey", "hi",
  "oh", "well", "really", "very", "too", "also", "then", "than", "as", "if", "not", "what", "when",
  "how", "why", "who", "about", "from", "up", "out", "get", "got", "like", "good", "nice", "cool",
  "thanks", "thank", "please", "sure",
]);
const WORD = /^[A-Za-z][A-Za-z'-]*$/;
const HAS_DIGIT = /\d/;
const SENT_END = /[.!?]$/;
const PUNCT = /^[.,;:!?"'()[\]]+|[.,;:!?"'()[\]]+$/g;

export type ImportanceScorer = (text: string) => number;

export function contentImportance(text: string): number {
  text = (text ?? "").trim();
  if (!text) return 1;
  const tokens = text.split(/\s+/);

  const content = new Set<string>();
  for (const w of tokens) {
    const core = w.replace(PUNCT, "");
    if (WORD.test(core) && core.length > 3 && !STOP.has(core.toLowerCase())) {
      content.add(core.toLowerCase());
    }
  }
  const nContent = content.size;
  const nDigit = tokens.filter((w) => HAS_DIGIT.test(w)).length;

  let nProper = 0;
  let prevEndsSentence = true;
  for (const w of tokens) {
    const core = w.replace(PUNCT, "");
    if (
      !prevEndsSentence &&
      core.length > 2 &&
      core[0] >= "A" &&
      core[0] <= "Z" &&
      /^[A-Za-z]+$/.test(core) &&
      !STOP.has(core.toLowerCase())
    ) {
      nProper += 1;
    }
    prevEndsSentence = SENT_END.test(w);
  }

  if (tokens.length <= 4 && nDigit === 0 && nProper === 0 && nContent <= 1) return 1;

  let score = 1;
  if (nContent >= 3) score += 1;
  if (nContent >= 7) score += 1;
  if (nDigit >= 1) score += 1;
  if (nProper >= 1) score += 1;
  return Math.max(1, Math.min(5, score));
}

const FIRST_PERSON =
  /\b(i'?m|i\s+am|my|mine|i\s+have|i'?ve|i\s+(?:prefer|like|love|hate|need|use|own|drive|speak|work|live)|i\s+was\s+born)\b/i;
const DURABLE =
  /\b(prefer|allergic|favou?rite|deadline|due|born|named|called|decided|always|never|must|require|policy|rule)\b/i;
const COPULA = /\b(is|are|was|were|named|called)\b/i;
const META =
  /\b(let'?s|let\s+me|can\s+you|could\s+you|should\s+we|shall\s+we|please|thanks|thank\s+you|sounds?\s+good|got\s+it|no\s+problem)\b/i;

export function structuralImportance(text: string): number {
  text = (text ?? "").trim();
  const base = contentImportance(text);
  const isQuestion = text.endsWith("?");
  let boost = 0;
  if (FIRST_PERSON.test(text)) boost += 1;
  if (DURABLE.test(text)) boost += 1;
  if (!isQuestion && COPULA.test(text)) boost += 1;
  let penalty = 0;
  if (isQuestion) penalty += 1;
  else if (boost === 0 && META.test(text)) penalty += 1;
  const score = base + Math.min(boost, 2) - penalty;
  return Math.max(1, Math.min(5, score));
}
