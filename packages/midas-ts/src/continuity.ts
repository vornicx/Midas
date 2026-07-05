/** Continuity control-plane (TypeScript port of `midas/continuity.py`) — resume a session in one
 * call, surface live contradictions, and keep commitments across sessions. Deterministic, no LLM.
 * The conflict detector is the heuristic tier only (the local NLI gate stays Python-side): same
 * gates as the Python fallback — value swap on one slot, numeric drift, negation flip. */
import { contentWords, properEntities, DURABLE_KINDS, Memory, approxTokens, formatRecord, fmtDate } from "./memory.js";
import type { MemoryRecord } from "./types.js";

const NUM_RE = /\d+(?:[.,:]\d+)*/g;
const NEG_RE =
  /\b(?:not|no longer|never|don'?t|doesn'?t|isn'?t|aren'?t|won'?t|can'?t|cannot|shouldn'?t|stop(?:ped)?|forbidden|banned|disallowed|deprecated)\b/i;
// Sentence-initial capitalised cue words that the entity regex picks up as noise.
const ENTITY_NOISE = new Set(
  ("the a an this that these those we i you it they he she update updated now note reminder new " +
    "also actually however meanwhile today yesterday tomorrow from after before once finally").split(" "),
);

const entities = (text: string): Set<string> => {
  const out = new Set<string>();
  for (const e of properEntities(text)) if (!ENTITY_NOISE.has(e)) out.add(e);
  return out;
};
const numbers = (text: string): Set<string> => new Set(text.match(NUM_RE) ?? []);
const norm = (text: string): string => text.split(/\s+/).join(" ").toLowerCase();
const diff = (a: Set<string>, b: Set<string>): Set<string> =>
  new Set([...a].filter((x) => !b.has(x)));
const inter = (a: Set<string>, b: Set<string>): Set<string> =>
  new Set([...a].filter((x) => b.has(x)));
const setEq = (a: Set<string>, b: Set<string>): boolean =>
  a.size === b.size && [...a].every((x) => b.has(x));

export interface Conflict {
  a: MemoryRecord;
  b: MemoryRecord;
  similarity: number;
  signal: "value" | "numeric" | "negation";
  strength: number;
}

function contradiction(a: MemoryRecord, b: MemoryRecord, similarity: number): Conflict["signal"] | null {
  if (norm(a.content) === norm(b.content)) return null; // restatements agree
  const shared = inter(contentWords(a.content), contentWords(b.content));
  if (shared.size < 2) return null;
  const na = numbers(a.content), nb = numbers(b.content);
  const ea = entities(a.content), eb = entities(b.content);
  const da = diff(ea, eb), db = diff(eb, ea);
  const frameA = diff(diff(contentWords(a.content), ea), na);
  const frameB = diff(diff(contentWords(b.content), eb), nb);
  if (da.size === 1 && db.size === 1 && setEq(na, nb) && inter(frameA, frameB).size >= 2) {
    return "value";
  }
  if (da.size && db.size) return null; // each names its own thing → two facts, not one slot
  if (na.size && nb.size && !setEq(na, nb)) return "numeric";
  if (NEG_RE.test(a.content) !== NEG_RE.test(b.content)) return "negation";
  return null;
}

function scopeMatch(record: MemoryRecord, scope: Record<string, unknown> | null | undefined): boolean {
  if (!scope) return true;
  return Object.entries(scope).every(([k, v]) => record.metadata?.[k] === v);
}

export function memoryConflicts(
  mem: Memory,
  opts: {
    scope?: Record<string, unknown> | null;
    minSimilarity?: number;
    pool?: number;
    neighbors?: number;
    limit?: number;
  } = {},
): Conflict[] {
  const minSimilarity = opts.minSimilarity ?? 0.35;
  const live = mem.store
    .all()
    .filter(
      (r) =>
        r.supersededBy === null &&
        r.embedding !== null &&
        (DURABLE_KINDS as readonly string[]).includes(r.kind) &&
        scopeMatch(r, opts.scope),
    )
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, opts.pool ?? 500);
  const liveIds = new Set(live.map((r) => r.id));

  const conflicts: Conflict[] = [];
  const seen = new Set<string>();
  for (const record of live) {
    const near = mem.store.search(record.embedding!, {
      limit: opts.neighbors ?? 10,
      predicate: (r) => r.id !== record.id && liveIds.has(r.id),
    });
    for (const [score, other] of near) {
      if (score < minSimilarity) break; // sorted by similarity desc
      const pair = record.id < other.id ? `${record.id}|${other.id}` : `${other.id}|${record.id}`;
      if (seen.has(pair)) continue;
      seen.add(pair);
      const signal = contradiction(record, other, score);
      if (signal) conflicts.push({ a: record, b: other, similarity: score, signal, strength: score });
    }
  }
  conflicts.sort((a, b) => b.strength - a.strength);
  return conflicts.slice(0, opts.limit ?? 20);
}

// ---- open loops (commitments) -------------------------------------------------------------------

export async function rememberCommitment(
  mem: Memory,
  content: string,
  opts: { project?: string; due?: string; actor?: string; source?: string } = {},
): Promise<MemoryRecord> {
  const metadata: Record<string, unknown> = {};
  if (opts.project) metadata.project = opts.project;
  if (opts.due) metadata.due = opts.due;
  return mem.remember(content, {
    kind: "commitment",
    importance: 4,
    provenance: "planning",
    actor: opts.actor ?? null,
    source: opts.source ?? null,
    metadata,
  });
}

export function openLoops(
  mem: Memory,
  opts: { scope?: Record<string, unknown> | null; limit?: number } = {},
): MemoryRecord[] {
  const records = mem.store
    .all()
    .filter(
      (r) =>
        r.supersededBy === null &&
        r.kind === "commitment" &&
        !r.metadata?.closes && // a resolution record is the close, not a new loop
        scopeMatch(r, opts.scope),
    )
    .sort((a, b) => a.createdAt - b.createdAt); // oldest = most overdue first
  return opts.limit ? records.slice(0, opts.limit) : records;
}

export async function closeLoop(
  mem: Memory,
  loopId: string,
  resolution: string,
  opts: { actor?: string } = {},
): Promise<MemoryRecord> {
  const target = mem.store.get(loopId);
  if (!target || target.kind !== "commitment") {
    throw new Error(`no open commitment with id '${loopId}'`);
  }
  const metadata: Record<string, unknown> = { closes: loopId };
  for (const k of ["project", "namespace"]) if (target.metadata?.[k]) metadata[k] = target.metadata[k];
  const closing = await mem.remember(`Done: ${resolution}`, {
    kind: "commitment",
    importance: 2,
    provenance: "action",
    actor: opts.actor ?? null,
    metadata,
  });
  target.supersededBy = closing.id;
  target.metadata = { ...target.metadata, superseded_at: closing.createdAt };
  mem.store.put(target);
  return closing;
}

// ---- resume: the one-call session-onboarding pack --------------------------------------------------

export interface ResumePack {
  generatedAt: number;
  since: number;
  pinned: MemoryRecord[];
  forbidden: MemoryRecord[];
  state: MemoryRecord[];
  added: MemoryRecord[];
  revised: Array<[MemoryRecord, MemoryRecord]>;
  openLoops: MemoryRecord[];
  conflicts: Conflict[];
  context: string;
  truncated: boolean;
}

export function resume(
  mem: Memory,
  opts: {
    scope?: Record<string, unknown> | null;
    project?: string;
    since?: number;
    tokenBudget?: number;
    now?: number;
  } = {},
): ResumePack {
  const now = opts.now ?? Date.now() / 1000;
  const since = opts.since ?? now - 7 * 86_400;
  const scope = opts.project ? { ...(opts.scope ?? {}), project: opts.project } : opts.scope ?? null;
  const budget = opts.tokenBudget ?? 900;

  const all = mem.store.all();
  const live = all.filter((r) => r.supersededBy === null && scopeMatch(r, scope));
  const pinned = live
    .filter((r) => r.metadata?.standing || r.kind === "mission")
    .sort((a, b) => b.updatedAt - a.updatedAt);
  const forbidden = live
    .filter((r) => r.metadata?.code_kind === "forbidden_action")
    .sort((a, b) => b.updatedAt - a.updatedAt);
  const skip = new Set([...pinned, ...forbidden].map((r) => r.id));
  const state = live
    .filter((r) => (DURABLE_KINDS as readonly string[]).includes(r.kind) && !skip.has(r.id))
    .sort((a, b) => b.updatedAt - a.updatedAt);

  // memory_diff: added = new current beliefs; revised = (old, new) pairs since `since`.
  const byId = new Map(all.map((r) => [r.id, r]));
  const supersedingIds = new Set(all.map((r) => r.supersededBy).filter((x): x is string => x !== null));
  const added: MemoryRecord[] = [];
  const revised: Array<[MemoryRecord, MemoryRecord]> = [];
  for (const r of all) {
    if (!scopeMatch(r, scope)) continue;
    if (r.supersededBy !== null) {
      const nw = byId.get(r.supersededBy);
      if (nw && nw.createdAt >= since) revised.push([r, nw]);
    } else if (r.createdAt >= since && !supersedingIds.has(r.id)) {
      added.push(r);
    }
  }
  added.sort((a, b) => b.createdAt - a.createdAt);
  revised.sort((a, b) => b[1].createdAt - a[1].createdAt);

  const loops = openLoops(mem, { scope });
  const conflicts = memoryConflicts(mem, { scope, limit: 5 });

  const lines = [`RESUME (today is ${fmtDate(now)}; changes since ${fmtDate(since)})`];
  let used = approxTokens(lines[0]);
  let truncated = false;
  const emit = (header: string, rows: string[]): void => {
    if (!rows.length) return;
    for (const line of [header, ...rows]) {
      const cost = approxTokens(line);
      if (used + cost > budget) {
        truncated = true;
        return;
      }
      lines.push(line);
      used += cost;
    }
  };
  const fmt = (r: MemoryRecord): string => formatRecord(r, { now });
  emit("PINNED:", pinned.map(fmt));
  emit("FORBIDDEN:", forbidden.map(fmt));
  emit("CHANGED (added):", added.map(fmt));
  emit("CHANGED (revised):",
    revised.map(([o, n]) => `${fmt(o)}\n    -> now: ${n.content.split(/\s+/).join(" ").slice(0, 300)}`));
  emit("CURRENT STATE:", state.map(fmt));
  emit("OPEN LOOPS:", loops.map(fmt));
  emit("CONFLICTS (unresolved — verify before relying on either):",
    conflicts.map((c) =>
      `- [${c.signal}] "${c.a.content.split(/\s+/).join(" ").slice(0, 200)}"  vs  "${c.b.content.split(/\s+/).join(" ").slice(0, 200)}"`));

  return {
    generatedAt: now, since, pinned, forbidden, state, added, revised,
    openLoops: loops, conflicts, context: lines.join("\n"), truncated,
  };
}
