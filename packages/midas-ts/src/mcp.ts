/** Midas as an MCP server (TypeScript) — same tool surface, env knobs, injected policy, and
 * SQLite schema as the Python `midas.mcp_server`, so the two are interchangeable and can even
 * share one DB file live. No LLM and no network at ingest/query. MIDAS_MCP_EMBEDDER=local enables
 * local ONNX semantic embeddings when the optional `@huggingface/transformers` package is
 * installed; otherwise the byte-parity hashing embedder is used (and the fallback is announced). */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

import { HashingEmbedder, LocalEmbedder, type Embedder } from "./embeddings.js";
import { decideMemoryUse, MEMORY_USES, type MemoryUse } from "./guard.js";
import { Memory, DEFAULT_POLICY } from "./memory.js";
import { structuralImportance } from "./importance.js";
import { AGENT_MEMORY_INSTRUCTIONS, parseTtlSpec, policySummary } from "./policy.js";
import {
  closeLoop as closeLoopImpl,
  memoryConflicts,
  openLoops as openLoopsView,
  rememberCommitment as rememberCommitmentImpl,
  resume as resumeView,
} from "./continuity.js";
import { InMemoryStore, SQLiteStore } from "./store.js";
import { MEMORY_PROVENANCE, type MemoryKind, type MemoryProvenance, type MemoryRecord } from "./types.js";

const MAX_RECORDS = Number(process.env.MIDAS_MCP_MAX_RECORDS ?? "0") || null;
const MIN_IMPORTANCE = Number(process.env.MIDAS_MCP_MIN_IMPORTANCE ?? "2");
const ACTOR = process.env.MIDAS_MCP_ACTOR ?? "midas-mcp-ts";
const NAMESPACE = process.env.MIDAS_MCP_NAMESPACE ?? "";
const SUPERSEDE = process.env.MIDAS_MCP_SUPERSEDE !== "0";
const TTL = parseTtlSpec(process.env.MIDAS_MCP_TTL ?? "");

async function buildEmbedder(): Promise<Embedder> {
  const choice = (process.env.MIDAS_MCP_EMBEDDER ?? "hashing").toLowerCase();
  if (choice && choice !== "hashing") {
    try {
      // "local" -> the default bge-small; any other value is treated as a model id (Python parity).
      const embedder = await (choice === "local"
        ? LocalEmbedder.create()
        : LocalEmbedder.create(process.env.MIDAS_MCP_EMBEDDER));
      console.error(`[midas-mcp-ts] semantic embeddings: ${embedder.model} (${embedder.dim}d, local ONNX)`);
      return embedder;
    } catch (err) {
      console.error(
        `[midas-mcp-ts] local embedder unavailable (${(err as Error).message}) — using the offline hashing embedder`,
      );
    }
  }
  return new HashingEmbedder();
}

async function buildMemory(): Promise<Memory> {
  let store: InMemoryStore | undefined;
  const db = process.env.MIDAS_MCP_DB;
  if (db) {
    store = new SQLiteStore(db);
    console.error(`[midas-mcp-ts] persisting memory to ${db}`);
  }
  return new Memory({
    store,
    embedder: await buildEmbedder(),
    importanceScorer: structuralImportance,
    policy: { ...DEFAULT_POLICY, minImportance: MIN_IMPORTANCE },
    supersede: SUPERSEDE,
  });
}

const mem = await buildMemory();

function ns(namespace: string): string {
  return namespace || NAMESPACE;
}

function nsMetadata(namespace: string, session: string): Record<string, unknown> {
  const n = ns(namespace);
  return { session, ...(n ? { namespace: n } : {}) };
}

function nsFilter(namespace: string): Record<string, unknown> | null {
  const n = ns(namespace);
  return n ? { namespace: n } : null;
}

function serializeRecord(record: MemoryRecord) {
  return {
    id: record.id,
    kind: record.kind,
    importance: record.importance,
    provenance: record.provenance,
    actor: record.actor,
    source: record.source,
    metadata: record.metadata,
    created_at: record.createdAt,
    updated_at: record.updatedAt,
    superseded_by: record.supersededBy,
    content: record.content,
  };
}

function boundStore(): void {
  if (MAX_RECORDS && mem.store.all().length > MAX_RECORDS) {
    mem.forgetDecayed({ maxRecords: MAX_RECORDS });
  }
}

function json(data: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
}

function text(data: string) {
  return { content: [{ type: "text" as const, text: data }] };
}

export function createServer(): McpServer {
  const server = new McpServer(
    { name: "midas-memory", version: "0.2.0" }, // keep in sync with package.json on each release
    { instructions: AGENT_MEMORY_INSTRUCTIONS },
  );

  server.registerTool(
    "remember",
    {
      title: "Remember",
      description:
        "Store a memory for later recall. kind: note | chat | fact | preference | constraint | mission. " +
        "importance 1-5 (0 = auto-derive from content, no LLM). provenance: planning | action | observation | " +
        "user_confirmation (external actions may rely only on user_confirmation).",
      inputSchema: {
        content: z.string(),
        kind: z.string().default("note"),
        importance: z.number().int().min(0).max(5).default(0),
        session: z.string().default("default"),
        provenance: z.enum(MEMORY_PROVENANCE as [MemoryProvenance, ...MemoryProvenance[]]).default("observation"),
        actor: z.string().default(""),
        namespace: z.string().default(""),
      },
    },
    async (args) => {
      const rec = await mem.remember(args.content, {
        kind: args.kind as MemoryKind,
        importance: args.importance || null,
        source: `mcp:${args.session}`,
        provenance: args.provenance,
        actor: args.actor || ACTOR,
        metadata: nsMetadata(args.namespace, args.session),
      });
      boundStore();
      return text(`remembered (${rec.id})`);
    },
  );

  server.registerTool(
    "capture",
    {
      title: "Capture turn",
      description:
        "Offer a turn to memory; Midas decides whether to keep it (no LLM): scores importance, enforces the " +
        "relevance floor, skips duplicates, and says why. kind: note | chat | fact | preference | constraint | mission. " +
        "provenance: planning | action | observation | user_confirmation.",
      inputSchema: {
        content: z.string(),
        kind: z.string().default("chat"),
        session: z.string().default("default"),
        provenance: z.enum(MEMORY_PROVENANCE as [MemoryProvenance, ...MemoryProvenance[]]).default("observation"),
        actor: z.string().default(""),
        namespace: z.string().default(""),
      },
    },
    async (args) => {
      const result = await mem.capture(args.content, {
        kind: args.kind as MemoryKind,
        source: `mcp:${args.session}`,
        provenance: args.provenance,
        actor: args.actor || ACTOR,
        metadata: nsMetadata(args.namespace, args.session),
      });
      if (result.stored) boundStore();
      return json({
        stored: result.stored,
        reason: result.reason,
        importance: result.importance,
        id: result.record?.id ?? null,
      });
    },
  );

  server.registerTool(
    "recall",
    {
      title: "Recall",
      description:
        "Retrieve relevant memories with deterministic, source-traceable evidence (exact stored text + " +
        "provenance/source/timestamps + score components). Set hybrid=true for exact-identifier queries.",
      inputSchema: {
        query: z.string(),
        limit: z.number().int().min(1).default(5),
        kind: z.string().default(""),
        min_importance: z.number().int().min(0).default(0),
        hybrid: z.boolean().default(false),
        namespace: z.string().default(""),
      },
    },
    async (args) => {
      const hits = await mem.recall(args.query, {
        limit: args.limit,
        kind: (args.kind || null) as MemoryKind | null,
        minImportance: args.min_importance || null,
        hybrid: args.hybrid,
        metadataFilter: nsFilter(args.namespace),
      });
      return json(
        hits.map((h) => ({
          ...serializeRecord(h.record),
          score: round3(h.score),
          relevance: round3(h.relevance),
          importance_norm: round3(h.importanceNorm),
          recency: round3(h.recency),
        })),
      );
    },
  );

  server.registerTool(
    "inspect_memory",
    {
      title: "Inspect memory",
      description: "Inspect one stored memory by id without search, mutation, or embedding exposure.",
      inputSchema: { memory_id: z.string() },
    },
    async (args) => {
      const record = mem.store.get(args.memory_id);
      return json(record ? { found: true, ...serializeRecord(record) } : { found: false, id: args.memory_id });
    },
  );

  server.registerTool(
    "build_context",
    {
      title: "Build context",
      description:
        "Assemble a budgeted, prompt-ready context block (lean dated lines + a 'Today is' anchor). " +
        "Use recall/inspect_memory for full provenance.",
      inputSchema: {
        query: z.string(),
        token_budget: z.number().int().min(16).default(512),
        limit: z.number().int().min(1).default(6),
        hybrid: z.boolean().default(false),
        namespace: z.string().default(""),
      },
    },
    async (args) =>
      text(
        (await mem.buildContext(args.query, {
          tokenBudget: args.token_budget,
          limit: args.limit,
          hybrid: args.hybrid,
          metadataFilter: nsFilter(args.namespace),
        })).text,
      ),
  );

  server.registerTool(
    "memory_policy",
    {
      title: "Memory policy",
      description: "Return the exact MCP-injected memory policy text and guard parameters.",
      inputSchema: {},
    },
    async () =>
      json({
        instructions: AGENT_MEMORY_INSTRUCTIONS,
        relevance_policy: policySummary({ ...DEFAULT_POLICY, minImportance: MIN_IMPORTANCE }),
        provenance: MEMORY_PROVENANCE,
        guard_uses: MEMORY_USES,
      }),
  );

  server.registerTool(
    "check_memory_use",
    {
      title: "Check memory use",
      description:
        "Decide whether recalled memory may justify the intended use. intended_use: planning | answer | " +
        "external_action | destructive_action (the last two require user_confirmation provenance).",
      inputSchema: {
        query: z.string(),
        intended_use: z.enum(MEMORY_USES as [MemoryUse, ...MemoryUse[]]).default("planning"),
        limit: z.number().int().min(1).default(5),
        acting_agent: z.string().default(""),
        namespace: z.string().default(""),
      },
    },
    async (args) => {
      const hits = await mem.recall(args.query, { limit: args.limit, metadataFilter: nsFilter(args.namespace) });
      return json(
        decideMemoryUse(hits.map((h) => h.record), {
          intendedUse: args.intended_use,
          actingAgent: args.acting_agent || ACTOR,
        }),
      );
    },
  );

  server.registerTool(
    "forget",
    {
      title: "Forget memory",
      description: "Delete a single memory by id (supersession chains are relinked, not orphaned).",
      inputSchema: { memory_id: z.string() },
    },
    async (args) => text(mem.forget(args.memory_id) ? "forgotten" : `no memory with id ${args.memory_id}`),
  );

  server.registerTool(
    "forget_matching",
    {
      title: "Forget matching",
      description:
        "Topic-level erasure with a reviewable audit. DRY RUN by default: returns what WOULD be deleted; " +
        "repeat with dry_run=false to erase. Bypasses durability protections on purpose.",
      inputSchema: {
        query: z.string(),
        min_relevance: z.number().min(0).max(1).default(0.5),
        limit: z.number().int().min(1).default(20),
        dry_run: z.boolean().default(true),
        namespace: z.string().default(""),
      },
    },
    async (args) => {
      const matched = await mem.forgetMatching(args.query, {
        minRelevance: args.min_relevance,
        limit: args.limit,
        metadataFilter: nsFilter(args.namespace),
        dryRun: args.dry_run,
      });
      return json({
        dry_run: args.dry_run,
        matched: matched.length,
        records: matched.map(serializeRecord),
        note: args.dry_run ? "dry run — nothing deleted; repeat with dry_run=false to erase" : "deleted",
      });
    },
  );

  server.registerTool(
    "forget_all",
    {
      title: "Forget all",
      description: "Clear all stored memories (fresh start).",
      inputSchema: {},
    },
    async () => {
      const n = mem.store.all().length;
      mem.store.clear();
      return text(`cleared ${n} memories`);
    },
  );

  server.registerTool(
    "resume",
    {
      title: "Resume session",
      description:
        "START OF SESSION: everything needed to pick up where the last session left off, in ONE call — " +
        "pinned directives, forbidden rules, what changed in the last `hours` (default: a week), current " +
        "state, open commitments, and unresolved conflicts. `context` is prompt-ready and token-budgeted.",
      inputSchema: {
        project: z.string().default(""),
        namespace: z.string().default(""),
        hours: z.number().min(0).default(168),
        token_budget: z.number().int().min(64).default(900),
      },
    },
    async (args) => {
      const pack = resumeView(mem, {
        scope: nsFilter(args.namespace),
        project: args.project || undefined,
        since: Date.now() / 1000 - args.hours * 3600,
        tokenBudget: args.token_budget,
      });
      return json({
        context: pack.context,
        truncated: pack.truncated,
        counts: {
          pinned: pack.pinned.length, forbidden: pack.forbidden.length, state: pack.state.length,
          added: pack.added.length, revised: pack.revised.length,
          open_loops: pack.openLoops.length, conflicts: pack.conflicts.length,
        },
        open_loops: pack.openLoops.map(serializeRecord),
        conflicts: pack.conflicts.map((c) => ({
          signal: c.signal, similarity: round3(c.similarity),
          a: serializeRecord(c.a), b: serializeRecord(c.b),
        })),
      });
    },
  );

  server.registerTool(
    "memory_conflicts",
    {
      title: "Memory conflicts",
      description:
        "Live beliefs that CONTRADICT each other with neither superseding the other — the multi-agent " +
        "failure mode. Ranked candidate pairs (same-slot heuristic: value swap / numbers disagree / one " +
        "side negates). Verify with the user, then forget the wrong one or capture the corrected value.",
      inputSchema: { namespace: z.string().default(""), limit: z.number().int().min(1).default(10) },
    },
    async (args) => {
      const found = memoryConflicts(mem, { scope: nsFilter(args.namespace), limit: args.limit });
      return json({
        count: found.length,
        conflicts: found.map((c) => ({
          signal: c.signal, similarity: round3(c.similarity), strength: round3(c.strength),
          a: serializeRecord(c.a), b: serializeRecord(c.b),
        })),
      });
    },
  );

  server.registerTool(
    "open_loops",
    {
      title: "Open loops",
      description:
        "Unresolved commitments — work someone said WOULD be done and never closed — oldest (most " +
        "overdue) first. Record one with remember_commitment; close it with close_loop.",
      inputSchema: {
        project: z.string().default(""),
        namespace: z.string().default(""),
        limit: z.number().int().min(1).default(20),
      },
    },
    async (args) => {
      const scope: Record<string, unknown> = { ...(nsFilter(args.namespace) ?? {}) };
      if (args.project) scope.project = args.project;
      const records = openLoopsView(mem, {
        scope: Object.keys(scope).length ? scope : null,
        limit: args.limit,
      });
      return json({ count: records.length, open_loops: records.map(serializeRecord) });
    },
  );

  server.registerTool(
    "remember_commitment",
    {
      title: "Remember commitment",
      description:
        "Record a commitment (an OPEN LOOP): promised work that stays visible in open_loops/resume " +
        "until closed with close_loop, so promises survive across sessions.",
      inputSchema: {
        content: z.string(),
        project: z.string().default(""),
        due: z.string().default(""),
        session: z.string().default("default"),
        namespace: z.string().default(""),
      },
    },
    async (args) => {
      const rec = await rememberCommitmentImpl(mem, args.content, {
        project: args.project || undefined,
        due: args.due || undefined,
        actor: ACTOR,
        source: `mcp:${args.session}`,
      });
      rec.metadata = { ...rec.metadata, ...nsMetadata(args.namespace, args.session) };
      mem.store.put(rec);
      return text(`commitment recorded (${rec.id}) — close it with close_loop when done`);
    },
  );

  server.registerTool(
    "close_loop",
    {
      title: "Close loop",
      description:
        "Close an open commitment: records the resolution and supersedes the open loop, keeping the " +
        "promise -> resolution history auditable. Get loop_id from open_loops.",
      inputSchema: { loop_id: z.string(), resolution: z.string() },
    },
    async (args) => {
      try {
        const rec = await closeLoopImpl(mem, args.loop_id, args.resolution, { actor: ACTOR });
        return text(`loop closed (${args.loop_id} -> ${rec.id})`);
      } catch (err) {
        return text((err as Error).message);
      }
    },
  );

  server.registerTool(
    "maintain",
    {
      title: "Maintain memory",
      description:
        "Run a no-LLM memory-maintenance pass (bound to max_records and/or drop below min_value) and return " +
        "the deletion audit. Durable kinds and supersession chains are protected.",
      inputSchema: {
        max_records: z.number().int().min(0).default(0),
        min_value: z.number().min(0).default(0),
        ttl: z.string().default(""),
      },
    },
    async (args) => {
      const before = mem.store.all().length;
      const ttlMap = args.ttl ? parseTtlSpec(args.ttl) : TTL;
      const expired = Object.keys(ttlMap).length ? mem.forgetExpired(ttlMap) : [];
      const forgotten =
        args.max_records || args.min_value
          ? mem.forgetDecayed({
              maxRecords: args.max_records || null,
              minValue: args.min_value || null,
            })
          : [];
      return json({
        before,
        remaining: mem.store.all().length,
        expired: expired.length,
        forgotten: forgotten.length,
        removed_ids: [...expired, ...forgotten],
      });
    },
  );

  server.registerTool(
    "stats",
    {
      title: "Memory stats",
      description:
        "Memory stats: total, by kind/provenance/namespace, and short/medium/long temporal tiers.",
      inputSchema: { namespace: z.string().default("") },
    },
    async (args) => {
      const n = ns(args.namespace);
      const byKind: Record<string, number> = {};
      const byProvenance: Record<string, number> = {};
      const byTier: Record<string, number> = { short: 0, medium: 0, long: 0 };
      const byNamespace: Record<string, number> = {};
      for (const r of mem.store.all()) {
        const rns = (r.metadata?.namespace as string) || "(none)";
        byNamespace[rns] = (byNamespace[rns] ?? 0) + 1;
        if (n && r.metadata?.namespace !== n) continue;
        byKind[r.kind] = (byKind[r.kind] ?? 0) + 1;
        byProvenance[r.provenance] = (byProvenance[r.provenance] ?? 0) + 1;
        byTier[mem.tier(r)] += 1;
      }
      return json({
        total: Object.values(byKind).reduce((a, b) => a + b, 0),
        namespace: n || null,
        by_kind: byKind,
        by_provenance: byProvenance,
        by_tier: byTier,
        by_namespace: byNamespace,
        guard_uses: MEMORY_USES,
      });
    },
  );

  return server;
}

function round3(x: number): number {
  return Math.round(x * 1000) / 1000;
}

export async function main(): Promise<void> {
  const server = createServer();
  await server.connect(new StdioServerTransport());
}
