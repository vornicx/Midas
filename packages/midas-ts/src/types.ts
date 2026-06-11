/** Midas memory types — mirrors `midas/types.py` so records stay portable across the
 * Python/TypeScript split (including sharing one SQLite file between both servers). */

export type MemoryKind = "note" | "chat" | "mission" | "fact" | "preference" | "constraint";
export const MEMORY_KINDS: readonly MemoryKind[] = [
  "note",
  "chat",
  "mission",
  "fact",
  "preference",
  "constraint",
];

export type MemoryProvenance = "planning" | "action" | "observation" | "user_confirmation";
export const MEMORY_PROVENANCE: readonly MemoryProvenance[] = [
  "planning",
  "action",
  "observation",
  "user_confirmation",
];

export interface MemoryRecord {
  id: string;
  content: string;
  kind: MemoryKind;
  importance: number; // 1 (incidental) .. 5 (critical)
  source: string | null;
  provenance: MemoryProvenance;
  actor: string | null;
  metadata: Record<string, unknown>;
  createdAt: number; // epoch seconds (event time)
  updatedAt: number;
  embedding: Float32Array | null;
  supersededBy: string | null; // belief revision chain
}

export interface RecallHit {
  record: MemoryRecord;
  score: number;
  relevance: number;
  importanceNorm: number;
  recency: number;
}
