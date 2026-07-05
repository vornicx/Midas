/** Stores — `InMemoryStore` (exact cosine scan) and `SQLiteStore` (node:sqlite persistence).
 *
 * SQLiteStore uses the SAME table schema and float32-blob encoding as the Python
 * `midas/sqlite_store.py`, and the same `PRAGMA data_version` staleness probe — so a TypeScript
 * MCP server and a Python one can share a single memory file, live, in both directions. */
import { createHash } from "node:crypto";
import { DatabaseSync } from "node:sqlite";

import type { MemoryRecord } from "./types.js";
import { cosine } from "./embeddings.js";

export type Predicate = (record: MemoryRecord) => boolean;

export class InMemoryStore {
  protected records = new Map<string, MemoryRecord>();
  /** Monotonic change counter — lets Memory cache derived indexes (BM25) cheaply. */
  version = 0;

  put(record: MemoryRecord): void {
    this.records.set(record.id, record);
    this.version += 1;
  }

  get(recordId: string): MemoryRecord | null {
    return this.records.get(recordId) ?? null;
  }

  delete(recordId: string): boolean {
    const existed = this.records.delete(recordId);
    if (existed) this.version += 1;
    return existed;
  }

  all(): MemoryRecord[] {
    return [...this.records.values()];
  }

  clear(): void {
    this.records.clear();
    this.version += 1;
  }

  search(
    embedding: Float32Array,
    opts: { limit: number; predicate?: Predicate },
  ): Array<[number, MemoryRecord]> {
    const scored: Array<[number, MemoryRecord]> = [];
    for (const r of this.records.values()) {
      if (r.embedding === null) continue;
      if (opts.predicate && !opts.predicate(r)) continue;
      scored.push([cosine(embedding, r.embedding), r]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, opts.limit);
  }
}

interface Row {
  id: string;
  content: string;
  kind: string;
  importance: number;
  source: string | null;
  provenance: string;
  actor: string | null;
  metadata_json: string;
  created_at: number;
  updated_at: number;
  superseded_by: string | null;
  embedding: Uint8Array | null;
}

const AUDIT_GENESIS = "0".repeat(64);

function auditHash(seq: number, at: number, op: string, recordId: string, contentSha: string, prevHash: string): string {
  // `at` is canonicalised to integer microseconds (floor) — identical to the Python formula, so
  // either runtime verifies chains written by the other.
  return createHash("sha256")
    .update(`${seq}|${Math.floor(at * 1_000_000)}|${op}|${recordId}|${contentSha}|${prevHash}`, "utf8")
    .digest("hex");
}

function contentSha(record: MemoryRecord): string {
  return createHash("sha256")
    .update(`${record.id}\x00${record.content}\x00${record.supersededBy ?? ""}`, "utf8")
    .digest("hex");
}

export interface AuditEntry {
  seq: number;
  at: number;
  op: string;
  record_id: string;
  content_sha: string;
  prev_hash: string;
  hash: string;
}

export class SQLiteStore extends InMemoryStore {
  private db: DatabaseSync;
  private dataVersion: number;
  private auditEnabled: boolean;

  constructor(path: string, opts: { audit?: boolean } = {}) {
    super();
    this.auditEnabled = opts.audit ?? true;
    this.db = new DatabaseSync(path);
    this.db.exec("PRAGMA journal_mode=WAL");
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        kind TEXT NOT NULL,
        importance INTEGER NOT NULL,
        source TEXT,
        provenance TEXT NOT NULL DEFAULT 'observation',
        actor TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        superseded_by TEXT,
        embedding BLOB
      )
    `);
    // Tamper-evident mutation log — same additive table and hash chain as the Python store.
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS audit_log (
        seq INTEGER PRIMARY KEY,
        at REAL NOT NULL,
        op TEXT NOT NULL,
        record_id TEXT NOT NULL,
        content_sha TEXT NOT NULL,
        prev_hash TEXT NOT NULL,
        hash TEXT NOT NULL
      )
    `);
    this.load();
    this.dataVersion = this.currentDataVersion();
  }

  private audit(op: string, recordId: string, sha: string): void {
    const row = this.db.prepare("SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1").get() as
      | { seq: number; hash: string }
      | undefined;
    const [prevSeq, prevHash] = row ? [Number(row.seq), String(row.hash)] : [0, AUDIT_GENESIS];
    const seq = prevSeq + 1;
    const at = Date.now() / 1000;
    this.db
      .prepare(
        "INSERT INTO audit_log (seq, at, op, record_id, content_sha, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
      )
      .run(seq, at, op, recordId, sha, prevHash, auditHash(seq, at, op, recordId, sha, prevHash));
  }

  /** The mutation history (append-only): hashes only, never memory content. */
  auditLog(opts: { limit?: number } = {}): AuditEntry[] {
    const rows = this.db
      .prepare("SELECT seq, at, op, record_id, content_sha, prev_hash, hash FROM audit_log ORDER BY seq")
      .all() as unknown as AuditEntry[];
    const entries = rows.map((r) => ({ ...r, seq: Number(r.seq), at: Number(r.at) }));
    return opts.limit ? entries.slice(-opts.limit) : entries;
  }

  /** Walk the whole chain recomputing every hash — any edit/removal/reorder breaks it. */
  verifyAuditLog(): { ok: boolean; entries: number; first_invalid_seq: number | null } {
    const entries = this.auditLog();
    let prevHash = AUDIT_GENESIS;
    let expected = 1;
    for (const e of entries) {
      const good =
        e.seq === expected &&
        e.prev_hash === prevHash &&
        e.hash === auditHash(e.seq, e.at, e.op, e.record_id, e.content_sha, e.prev_hash);
      if (!good) return { ok: false, entries: entries.length, first_invalid_seq: e.seq };
      prevHash = e.hash;
      expected += 1;
    }
    return { ok: true, entries: entries.length, first_invalid_seq: null };
  }

  private currentDataVersion(): number {
    const row = this.db.prepare("PRAGMA data_version").get() as { data_version: number };
    return Number(row.data_version);
  }

  /** Reload the mirror when ANOTHER connection (e.g. the Python server) wrote the file. */
  private refreshIfStale(): void {
    const v = this.currentDataVersion();
    if (v === this.dataVersion) return;
    this.dataVersion = v;
    this.records.clear();
    this.version += 1;
    this.load();
  }

  private load(): void {
    const rows = this.db
      .prepare(
        "SELECT id, content, kind, importance, source, provenance, actor, metadata_json, " +
          "created_at, updated_at, superseded_by, embedding FROM memories",
      )
      .all() as unknown as Row[];
    for (const row of rows) {
      super.put(rowToRecord(row));
    }
  }

  override put(record: MemoryRecord): void {
    this.refreshIfStale();
    super.put(record);
    const blob = record.embedding
      ? new Uint8Array(record.embedding.buffer.slice(0), record.embedding.byteOffset, record.embedding.byteLength)
      : null;
    this.db
      .prepare(
        `INSERT INTO memories (id, content, kind, importance, source, provenance, actor,
                               metadata_json, created_at, updated_at, superseded_by, embedding)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           content=excluded.content, kind=excluded.kind, importance=excluded.importance,
           source=excluded.source, provenance=excluded.provenance, actor=excluded.actor,
           metadata_json=excluded.metadata_json,
           created_at=excluded.created_at, updated_at=excluded.updated_at,
           superseded_by=excluded.superseded_by, embedding=excluded.embedding`,
      )
      .run(
        record.id,
        record.content,
        record.kind,
        record.importance,
        record.source,
        record.provenance,
        record.actor,
        JSON.stringify(record.metadata ?? {}),
        record.createdAt,
        record.updatedAt,
        record.supersededBy,
        blob,
      );
    if (this.auditEnabled) this.audit("put", record.id, contentSha(record));
  }

  override get(recordId: string): MemoryRecord | null {
    this.refreshIfStale();
    return super.get(recordId);
  }

  override all(): MemoryRecord[] {
    this.refreshIfStale();
    return super.all();
  }

  override search(
    embedding: Float32Array,
    opts: { limit: number; predicate?: Predicate },
  ): Array<[number, MemoryRecord]> {
    this.refreshIfStale();
    return super.search(embedding, opts);
  }

  override delete(recordId: string): boolean {
    this.refreshIfStale();
    const target = super.get(recordId);
    const existed = super.delete(recordId);
    if (existed) {
      this.db.prepare("DELETE FROM memories WHERE id = ?").run(recordId);
      if (this.auditEnabled) this.audit("delete", recordId, target ? contentSha(target) : AUDIT_GENESIS);
    }
    return existed;
  }

  override clear(): void {
    super.clear();
    this.db.exec("DELETE FROM memories");
    if (this.auditEnabled) this.audit("clear", "*", AUDIT_GENESIS);
  }

  close(): void {
    this.db.close();
  }
}

function rowToRecord(row: Row): MemoryRecord {
  let embedding: Float32Array | null = null;
  if (row.embedding && row.embedding.byteLength >= 4) {
    const bytes = new Uint8Array(row.embedding); // copy → aligned buffer
    embedding = new Float32Array(bytes.buffer, 0, Math.floor(bytes.byteLength / 4));
  }
  return {
    id: row.id,
    content: row.content,
    kind: row.kind as MemoryRecord["kind"],
    importance: row.importance,
    source: row.source,
    provenance: (row.provenance || "observation") as MemoryRecord["provenance"],
    actor: row.actor,
    metadata: row.metadata_json ? JSON.parse(row.metadata_json) : {},
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    supersededBy: row.superseded_by,
    embedding,
  };
}

