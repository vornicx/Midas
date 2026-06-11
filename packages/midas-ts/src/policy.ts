/** The injected agent policy and capture parameters — verbatim the same text the Python MCP
 * server injects (token-lean: 198 approx tokens), so behaviour is identical across servers. */
import type { MemoryPolicy } from "./memory.js";

export const AGENT_MEMORY_INSTRUCTIONS =
  "Use Midas memory on every task. It is local, source-traceable, and uses no LLM at ingest/query.\n\n" +
  "1) RECALL FIRST. Call `build_context` with the user's goal; use the returned facts silently. Use " +
  "`recall`/`inspect_memory` only when you need audit details.\n\n" +
  "2) CAPTURE DURABLE SIGNAL. Call `capture` for reusable facts, decisions, preferences, constraints, " +
  "corrections, and completed actions. Skip pure small talk. Midas scores, dedups, and rejects trivia, " +
  "so capture can be brief and does not need an LLM. Set kind/provenance accurately; use " +
  'provenance="user_confirmation" only for explicit user confirmation.\n\n' +
  "3) GUARD ACTIONS. Memory may guide planning, but before external/destructive actions based on memory, " +
  "call `check_memory_use`. If it is not allowed, ask the user to confirm in this turn.\n\n" +
  "4) FORGET ON REQUEST. Use `forget_matching` as a dry-run first, show matches, then repeat with " +
  "dry_run=false after confirmation.\n\n" +
  "Midas stores verbatim source records and bounds memory automatically; compact context is for cheap " +
  "reader prompts, audit tools are for traceability.";

export function policySummary(policy: MemoryPolicy): string {
  return (
    `keep items scoring importance >= ${policy.minImportance}/5, ` +
    `kinds ${JSON.stringify(policy.acceptKinds)}, ` +
    `skip near-duplicates (cosine >= ${policy.dedupThreshold}); ` +
    "guard external/destructive actions to user_confirmation provenance"
  );
}
