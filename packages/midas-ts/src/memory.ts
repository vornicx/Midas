/** Midas core — a faithful TypeScript port of `midas/memory.py`'s shipping behaviour:
 * recall ranks by relevance × importance × recency with the scale-free parsimony floor
 * (`minRelevanceRatio` 0.3, measured-safe), optional BM25+RRF hybrid (cached index),
 * typed belief revision (supersession chains), policy-gated `capture`, budgeted lean
 * `buildContext` with the dated "Today is …" anchor, and no-LLM selective forgetting. */
import { randomUUID } from "node:crypto";

import { BM25 } from "./bm25.js";
import { cosine, HashingEmbedder, type Embedder } from "./embeddings.js";
import { contentImportance, type ImportanceScorer } from "./importance.js";
import { InMemoryStore } from "./store.js";
import type { MemoryKind, MemoryProvenance, MemoryRecord, RecallHit } from "./types.js";
import { MEMORY_PROVENANCE } from "./types.js";

export const RECENCY_HALF_LIFE_DAYS = 30.0;
export const DEFAULT_RECALL_LIMIT = 6;
export const DURABLE_KINDS: readonly MemoryKind[] = ["fact", "preference", "constraint"];

// Update/change cues for typed belief revision — same pattern set as the Python module.
const UPDATE_RE =
  /\b(actually|instead|now|currently|updated|changed|change|new|replaced|replace|taken over|take over|handed over|hand over|signed off|resigned|stepped down|stepped up|postponed|delayed|moved to|moved from|rescheduled|cancelled|no longer|not anymore|never)\b|\bwas\b.*\bbut now\b/i;

const SUPERSEDE_STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is",
  "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
  "would", "could", "should", "may", "might", "must", "can", "we", "our", "us", "it", "its",
  "this", "that", "these", "those", "i", "you", "he", "she", "they", "them",
]);
const ENTITY_STOPWORDS = new Set([
  "project", "user", "client", "team", "ticket", "issue", "task", "feature", "release", "launch",
  "date", "database", "service", "server", "environment",
  "january", "february", "march", "april", "may", "june", "july", "august", "september",
  "october", "november", "december",
]);
const PROPER_ENTITY = /\b[A-Z][A-Za-z0-9_-]{2,}\b/g;
const PROVENANCE_RANK: Record<MemoryProvenance, number> = {
  planning: 0,
  observation: 1,
  action: 2,
  user_confirmation: 3,
};

function contentWords(text: string): Set<string> {
  const out = new Set<string>();
  for (const w of text.split(/\s+/)) {
    const lw = w.toLowerCase();
    if (w.length > 3 && !SUPERSEDE_STOPWORDS.has(lw) && /^[a-z0-9]+$/i.test(w)) out.add(lw);
  }
  return out;
}

function properEntities(text: string): Set<string> {
  const out = new Set<string>();
  for (const m of text.matchAll(PROPER_ENTITY)) {
    const lw = m[0].toLowerCase();
    if (!ENTITY_STOPWORDS.has(lw)) out.add(lw);
  }
  return out;
}

function intersects(a: Set<string>, b: Set<string>): boolean {
  for (const x of a) if (b.has(x)) return true;
  return false;
}

export interface MemoryPolicy {
  minImportance: number;
  acceptKinds: readonly string[];
  dedupThreshold: number;
}

export const DEFAULT_POLICY: MemoryPolicy = {
  minImportance: 2,
  acceptKinds: ["note", "chat", "fact", "preference", "constraint", "mission"],
  dedupThreshold: 0.95,
};

export interface CaptureResult {
  stored: boolean;
  reason: string;
  record: MemoryRecord | null;
  importance: number;
}

export interface RememberOptions {
  kind?: MemoryKind;
  importance?: number | null;
  source?: string | null;
  provenance?: MemoryProvenance;
  actor?: string | null;
  metadata?: Record<string, unknown>;
  createdAt?: number | null;
}

export interface RecallOptions {
  limit?: number;
  kind?: MemoryKind | null;
  minImportance?: number | null;
  minRelevance?: number | null;
  minRelevanceRatio?: number | null;
  pool?: number | null;
  hybrid?: boolean;
  now?: number | null;
  metadataFilter?: Record<string, unknown> | null;
}

export interface ContextBlock {
  text: string;
  records: MemoryRecord[];
  tokens: number;
}

export interface MemoryOptions {
  store?: InMemoryStore;
  embedder?: Embedder;
  wRelevance?: number;
  wImportance?: number;
  wRecency?: number;
  supersede?: boolean;
  supersedeThreshold?: number;
  supersedePool?: number;
  supersedeMargin?: number;
  supersedeLowerThreshold?: number;
  minRelevanceRatio?: number;
  importanceScorer?: ImportanceScorer | null;
  policy?: MemoryPolicy;
  now?: () => number;
}

export class Memory {
  readonly store: InMemoryStore;
  readonly embedder: Embedder;
  private wRelevance: number;
  private wImportance: number;
  private wRecency: number;
  private supersede: boolean;
  private supersedeThreshold: number;
  private supersedePool: number;
  private supersedeMargin: number;
  private supersedeLowerThreshold: number;
  private minRelevanceRatio: number;
  private importanceScorer: ImportanceScorer | null;
  private policy: MemoryPolicy;
  private now: () => number;
  private bm25Cache: [number, BM25] | null = null;

  constructor(opts: MemoryOptions = {}) {
    this.store = opts.store ?? new InMemoryStore();
    this.embedder = opts.embedder ?? new HashingEmbedder();
    this.wRelevance = opts.wRelevance ?? 1.0;
    this.wImportance = opts.wImportance ?? 0.3;
    this.wRecency = opts.wRecency ?? 0.2;
    this.supersede = opts.supersede ?? false;
    this.supersedeThreshold = opts.supersedeThreshold ?? 0.55;
    this.supersedePool = opts.supersedePool ?? 20;
    this.supersedeMargin = opts.supersedeMargin ?? 0.05;
    this.supersedeLowerThreshold = opts.supersedeLowerThreshold ?? 0.3;
    this.minRelevanceRatio = Math.max(0, opts.minRelevanceRatio ?? 0.3);
    this.importanceScorer = opts.importanceScorer ?? null;
    this.policy = opts.policy ?? DEFAULT_POLICY;
    this.now = opts.now ?? (() => Date.now() / 1000);
  }

  async remember(content: string, opts: RememberOptions = {}): Promise<MemoryRecord> {
    content = (content ?? "").trim();
    if (!content) throw new Error("Memory.remember: `content` is required");
    const embedding = await this.embedder.embed(content);
    const ts = opts.createdAt ?? this.now();
    const importance = this.deriveImportance(opts.importance ?? null, content);
    const provenance = opts.provenance ?? "observation";
    if (!MEMORY_PROVENANCE.includes(provenance)) {
      throw new Error(`invalid memory provenance '${provenance}'`);
    }
    const record: MemoryRecord = {
      id: randomUUID(),
      content,
      kind: opts.kind ?? "note",
      importance,
      source: opts.source ?? null,
      provenance,
      actor: opts.actor ?? null,
      metadata: opts.metadata ?? {},
      createdAt: ts,
      updatedAt: ts,
      embedding,
      supersededBy: null,
    };
    this.store.put(record);
    if (this.supersede) this.maybeSupersede(record);
    return record;
  }

  async capture(content: string, opts: RememberOptions = {}): Promise<CaptureResult> {
    content = (content ?? "").trim();
    if (!content) return { stored: false, reason: "empty content", record: null, importance: 0 };
    const scorer = this.importanceScorer ?? contentImportance;
    const importance = clampImportance(scorer(content));
    const kind = opts.kind ?? "chat";
    if (!this.policy.acceptKinds.includes(kind)) {
      return { stored: false, reason: `kind '${kind}' not accepted by policy`, record: null, importance };
    }
    if (importance < this.policy.minImportance) {
      return {
        stored: false,
        reason: `below relevance floor (importance ${importance} < ${this.policy.minImportance})`,
        record: null,
        importance,
      };
    }
    if (this.policy.dedupThreshold > 0) {
      const near = this.store.search(await this.embedder.embed(content), { limit: 1 });
      if (near.length && near[0][0] >= this.policy.dedupThreshold) {
        const existing = near[0][1];
        const upgraded = this.upgradeProvenance(existing, opts);
        return {
          stored: false,
          reason: "near-duplicate of an existing memory" + (upgraded ? " (provenance upgraded)" : ""),
          record: existing,
          importance,
        };
      }
    }
    const record = await this.remember(content, { ...opts, kind, importance });
    return { stored: true, reason: "stored", record, importance };
  }

  private upgradeProvenance(record: MemoryRecord, opts: RememberOptions): boolean {
    const provenance = opts.provenance ?? "observation";
    if (PROVENANCE_RANK[provenance] <= PROVENANCE_RANK[record.provenance]) return false;
    record.provenance = provenance;
    record.actor = opts.actor ?? record.actor;
    record.source = opts.source ?? record.source;
    record.metadata = { ...record.metadata, ...(opts.metadata ?? {}) };
    record.updatedAt = opts.createdAt ?? this.now();
    this.store.put(record);
    return true;
  }

  private deriveImportance(importance: number | null, content: string): number {
    if (importance !== null && importance !== undefined && importance > 0) {
      return clampImportance(importance);
    }
    return clampImportance(this.importanceScorer ? this.importanceScorer(content) : 1);
  }

  async recall(query: string, opts: RecallOptions = {}): Promise<RecallHit[]> {
    const limit = opts.limit ?? DEFAULT_RECALL_LIMIT;
    const qEmb = await this.embedder.embed(query);
    const now = opts.now ?? this.now();
    const predicate = (r: MemoryRecord): boolean => {
      if (opts.kind && r.kind !== opts.kind) return false;
      if (opts.minImportance != null && r.importance < opts.minImportance) return false;
      if (opts.metadataFilter) {
        for (const [k, v] of Object.entries(opts.metadataFilter)) {
          if (r.metadata?.[k] !== v) return false;
        }
      }
      return true;
    };

    const pool = opts.pool ?? Math.max(limit * 5, limit);
    let candidates: Array<[number, MemoryRecord]>;
    let relevanceMap: Map<string, number> | null = null;
    if (opts.hybrid) {
      [candidates, relevanceMap] = this.hybridCandidates(query, qEmb, pool, predicate);
    } else {
      candidates = this.store.search(qEmb, { limit: pool, predicate });
    }
    if (this.supersede) candidates = this.resolveCandidates(candidates);
    if (!candidates.length) return [];

    const relevances = candidates.map(([cos, r]) =>
      relevanceMap ? relevanceMap.get(r.id) ?? Math.max(0, cos) : Math.max(0, cos),
    );

    const ratio = opts.minRelevanceRatio ?? this.minRelevanceRatio;
    const ratioFloor = ratio > 0 && relevances.length ? ratio * Math.max(...relevances) : null;

    const hits: RecallHit[] = [];
    for (let i = 0; i < candidates.length; i++) {
      const [, record] = candidates[i];
      const relevance = relevances[i];
      if (opts.minRelevance != null && relevance < opts.minRelevance) continue;
      if (ratioFloor !== null && relevance < ratioFloor) continue;
      const importanceNorm = (record.importance - 1) / 4;
      const ageDays = Math.max(0, (now - record.updatedAt) / 86_400);
      const recency = Math.exp(-ageDays / RECENCY_HALF_LIFE_DAYS);
      hits.push({
        record,
        score: this.wRelevance * relevance + this.wImportance * importanceNorm + this.wRecency * recency,
        relevance,
        importanceNorm,
        recency,
      });
    }
    hits.sort((a, b) => b.score - a.score || b.record.updatedAt - a.record.updatedAt);
    return hits.slice(0, limit);
  }

  private hybridCandidates(
    query: string,
    qEmb: Float32Array,
    pool: number,
    predicate: (r: MemoryRecord) => boolean,
  ): [Array<[number, MemoryRecord]>, Map<string, number>] {
    const semantic = this.store.search(qEmb, { limit: pool, predicate });
    const records = this.store.all().filter(predicate);
    const allowed = new Map(records.map((r) => [r.id, r]));
    const bmScores = new Map(
      [...this.bm25Index().scores(query)].filter(([rid]) => allowed.has(rid)),
    );
    const bmRanked = [...bmScores.keys()].sort((a, b) => bmScores.get(b)! - bmScores.get(a)!);

    const byId = new Map<string, [number, MemoryRecord]>(semantic.map(([cos, r]) => [r.id, [cos, r]]));
    for (const rid of bmRanked.slice(0, pool)) {
      if (!byId.has(rid)) {
        const rec = allowed.get(rid)!;
        const cos = rec.embedding ? cosine(qEmb, rec.embedding) : 0;
        byId.set(rid, [cos, rec]);
      }
    }
    // Reciprocal-rank fusion (k=60), normalised to [0,1] — same as the Python implementation.
    const K = 60;
    const denseRank = new Map(semantic.map(([, r], i) => [r.id, i]));
    const bmRank = new Map(bmRanked.map((rid, i) => [rid, i]));
    const raw = new Map<string, number>();
    let top = 0;
    for (const rid of byId.keys()) {
      const d = denseRank.has(rid) ? 1 / (K + denseRank.get(rid)! + 1) : 0;
      const b = bmRank.has(rid) ? 1 / (K + bmRank.get(rid)! + 1) : 0;
      raw.set(rid, d + b);
      top = Math.max(top, d + b);
    }
    const relevanceMap = new Map<string, number>();
    for (const [rid, s] of raw) relevanceMap.set(rid, top > 0 ? s / top : 0);
    return [[...byId.values()], relevanceMap];
  }

  private bm25Index(): BM25 {
    const version = this.store.version;
    if (!this.bm25Cache || this.bm25Cache[0] !== version) {
      this.bm25Cache = [version, new BM25(this.store.all())];
    }
    return this.bm25Cache[1];
  }

  private maybeSupersede(newRecord: MemoryRecord): void {
    if (!newRecord.embedding) return;
    if (newRecord.kind === "chat") return; // chat never revises (the safe Python default)
    const hasCue = UPDATE_RE.test(newRecord.content);
    const effectiveThreshold = hasCue ? this.supersedeLowerThreshold : this.supersedeThreshold;
    const candidates = this.store.search(newRecord.embedding, {
      limit: this.supersedePool,
      predicate: (r) => r.id !== newRecord.id && r.kind === newRecord.kind,
    });
    const byHead = new Map<string, [number, MemoryRecord]>();
    for (const [score, candidate] of candidates) {
      if (score < effectiveThreshold) continue;
      const head = this.resolveHead(candidate);
      if (head.id === newRecord.id || head.supersededBy !== null) continue;
      if (normalize(head.content) === normalize(newRecord.content)) continue;
      const ne = properEntities(newRecord.content);
      const he = properEntities(head.content);
      if (ne.size && he.size && !intersects(ne, he)) continue;
      if (score < this.supersedeThreshold && !intersects(contentWords(newRecord.content), contentWords(head.content))) {
        continue;
      }
      const prev = byHead.get(head.id);
      if (!prev || score > prev[0]) byHead.set(head.id, [score, head]);
    }
    if (!byHead.size) return;
    const ranked = [...byHead.values()].sort((a, b) => b[0] - a[0]);
    if (ranked.length > 1 && ranked[0][0] - ranked[1][0] < this.supersedeMargin) return;
    const head = ranked[0][1];
    head.supersededBy = newRecord.id;
    this.store.put(head);
  }

  resolveHead(record: MemoryRecord): MemoryRecord {
    const seen = new Set<string>();
    let current = record;
    while (current.supersededBy !== null && !seen.has(current.id)) {
      seen.add(current.id);
      const next = this.store.get(current.supersededBy);
      if (!next) break;
      current = next;
    }
    return current;
  }

  private resolveCandidates(candidates: Array<[number, MemoryRecord]>): Array<[number, MemoryRecord]> {
    const best = new Map<string, [number, MemoryRecord]>();
    for (const [score, record] of candidates) {
      const head = this.resolveHead(record);
      const prev = best.get(head.id);
      if (!prev || score > prev[0]) best.set(head.id, [score, head]);
    }
    return [...best.values()];
  }

  async buildContext(
    query: string,
    opts: RecallOptions & { tokenBudget?: number; header?: boolean; maxRecordChars?: number } = {},
  ): Promise<ContextBlock> {
    const budget = opts.tokenBudget ?? 512;
    const now = opts.now ?? this.now();
    const hits = await this.recall(query, { ...opts, now });
    let records = hits.map((h) => h.record);
    if (this.supersede) records = records.filter((r) => this.resolveHead(r).id === r.id);

    const chosen: Array<[MemoryRecord, string, number]> = [];
    let used = 0;
    for (const record of records) {
      const line = formatRecord(record, { maxChars: opts.maxRecordChars ?? 600, now });
      const cost = approxTokens(line);
      if (used + cost > budget) break;
      chosen.push([record, line, cost]);
      used += cost;
    }
    if (!chosen.length) return { text: "", records: [], tokens: 0 };

    let body = chosen.map(([, line]) => line).join("\n");
    if (opts.header !== false) {
      let head = "# Memory - relevant context (use silently to stay consistent)";
      head += `\n# Today is ${fmtDate(now)}`;
      body = head + "\n" + body;
    }
    return { text: body, records: chosen.map(([r]) => r), tokens: used };
  }

  forget(recordId: string): boolean {
    const target = this.store.get(recordId);
    if (!target) return false;
    for (const r of this.store.all()) {
      if (r.supersededBy === recordId) {
        r.supersededBy = target.supersededBy;
        this.store.put(r);
      }
    }
    return this.store.delete(recordId);
  }

  async forgetMatching(
    query: string,
    opts: { minRelevance?: number; limit?: number; metadataFilter?: Record<string, unknown> | null; dryRun?: boolean } = {},
  ): Promise<MemoryRecord[]> {
    const minRelevance = opts.minRelevance ?? 0.5;
    const hits = await this.recall(query, {
      limit: opts.limit ?? 20,
      minRelevance,
      metadataFilter: opts.metadataFilter ?? null,
    });
    const matched = hits.filter((h) => h.relevance >= minRelevance).map((h) => h.record);
    if (!opts.dryRun) for (const r of matched) this.forget(r.id);
    return matched;
  }

  memoryValue(record: MemoryRecord, now?: number): number {
    now = now ?? this.now();
    const importanceNorm = (record.importance - 1) / 4;
    const ageDays = Math.max(0, (now - record.updatedAt) / 86_400);
    const recency = Math.exp(-ageDays / RECENCY_HALF_LIFE_DAYS);
    const denom = this.wImportance + this.wRecency;
    return denom > 0 ? (this.wImportance * importanceNorm + this.wRecency * recency) / denom : 0;
  }

  tier(record: MemoryRecord, now?: number): "short" | "medium" | "long" {
    now = now ?? this.now();
    const ageDays = Math.max(0, (now - record.updatedAt) / 86_400);
    if (ageDays <= 1) return "short";
    if (ageDays <= 7) return "medium";
    return "long";
  }

  forgetDecayed(opts: { maxRecords?: number | null; minValue?: number | null; protectImportance?: number } = {}): string[] {
    const now = this.now();
    const records = this.store.all();
    const pointedTo = new Set(records.map((r) => r.supersededBy).filter((x): x is string => x !== null));
    const protectImportance = opts.protectImportance ?? 4;
    const protectedRec = (r: MemoryRecord) =>
      DURABLE_KINDS.includes(r.kind) ||
      r.importance >= protectImportance ||
      r.supersededBy !== null ||
      pointedTo.has(r.id);

    const candidates = records.filter((r) => !protectedRec(r));
    const value = new Map(candidates.map((r) => [r.id, this.memoryValue(r, now)]));
    candidates.sort((a, b) => value.get(a.id)! - value.get(b.id)!);

    const toDrop: string[] = [];
    const dropped = new Set<string>();
    if (opts.minValue != null) {
      for (const r of candidates) {
        if (value.get(r.id)! < opts.minValue) {
          toDrop.push(r.id);
          dropped.add(r.id);
        }
      }
    }
    if (opts.maxRecords != null) {
      let surviving = records.length - dropped.size;
      for (const r of candidates) {
        if (surviving <= opts.maxRecords) break;
        if (dropped.has(r.id)) continue;
        toDrop.push(r.id);
        dropped.add(r.id);
        surviving -= 1;
      }
    }
    return toDrop.filter((rid) => this.store.delete(rid));
  }
}

function normalize(text: string): string {
  return text.split(/\s+/).join(" ").toLowerCase();
}

function clampImportance(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.max(1, Math.min(5, Math.round(value)));
}

export function fmtDate(ts: number): string {
  return new Date(ts * 1000).toISOString().slice(0, 10); // UTC, same as Python
}

function relativeAge(createdAt: number, now: number): string {
  const days = Math.floor((now - createdAt) / 86_400);
  if (days <= 0) return "today";
  if (days < 14) return `${days}d ago`;
  if (days < 60) return `${Math.floor(days / 7)}w ago`;
  if (days < 730) return `${Math.floor(days / 30)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
}

export function formatRecord(
  record: MemoryRecord,
  opts: { maxChars?: number; now?: number | null } = {},
): string {
  let date = fmtDate(record.createdAt);
  if (opts.now != null) date = `${date}, ${relativeAge(record.createdAt, opts.now)}`;
  const body = record.content.split(/\s+/).join(" ").slice(0, opts.maxChars ?? 600);
  return `- [${record.kind} | ${date}] ${body}`;
}

export function approxTokens(text: string): number {
  return Math.max(1, Math.floor(text.split(/\s+/).length * 1.3));
}
