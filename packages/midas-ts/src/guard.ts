/** Provenance guard — port of `midas/guard.py`'s decision core: low-trust memory may guide
 * planning, but external/destructive actions require explicit user confirmation. */
import type { MemoryProvenance, MemoryRecord } from "./types.js";
import { MEMORY_PROVENANCE } from "./types.js";

export type MemoryUse = "planning" | "answer" | "external_action" | "destructive_action";
export const MEMORY_USES: readonly MemoryUse[] = [
  "planning",
  "answer",
  "external_action",
  "destructive_action",
];

const ALLOWED_BY_USE: Record<MemoryUse, readonly MemoryProvenance[]> = {
  planning: MEMORY_PROVENANCE,
  answer: ["action", "observation", "user_confirmation"],
  external_action: ["user_confirmation"],
  destructive_action: ["user_confirmation"],
};

export interface GuardDecision {
  allowed: boolean;
  intended_use: MemoryUse;
  acting_agent: string | null;
  allowed_ids: string[];
  blocked_ids: string[];
  reason: string;
}

export function decideMemoryUse(
  records: MemoryRecord[],
  opts: { intendedUse?: MemoryUse; actingAgent?: string | null } = {},
): GuardDecision {
  const use = opts.intendedUse ?? "planning";
  const allowedProv = ALLOWED_BY_USE[use];
  if (!allowedProv) throw new Error(`invalid memory use '${use}'`);
  const allowedIds: string[] = [];
  const blockedIds: string[] = [];
  for (const r of records) {
    (allowedProv.includes(r.provenance) ? allowedIds : blockedIds).push(r.id);
  }
  const allowed = allowedIds.length > 0;
  const reason = allowed
    ? `allowed: ${allowedIds.length} memory(ies) satisfy provenance for ${use}`
    : `blocked: ${use} requires provenance in [${allowedProv.join(", ")}]; ` +
      "ask the user to confirm in the current turn";
  return {
    allowed,
    intended_use: use,
    acting_agent: opts.actingAgent ?? null,
    allowed_ids: allowedIds,
    blocked_ids: blockedIds,
    reason,
  };
}
