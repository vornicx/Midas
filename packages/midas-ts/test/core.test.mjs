import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync } from "node:fs";

import {
  Memory,
  HashingEmbedder,
  SQLiteStore,
  cosine,
  contentImportance,
  structuralImportance,
  decideMemoryUse,
} from "../dist/index.js";

test("hashing embedder is bit-comparable with the Python implementation", () => {
  const fixture = JSON.parse(readFileSync(new URL("./parity_fixture.json", import.meta.url), "utf8"));
  const e = new HashingEmbedder(fixture.dim);
  for (const [text, pyVec] of Object.entries(fixture.vectors)) {
    const tsVec = e.embed(text);
    assert.equal(tsVec.length, pyVec.length, text);
    for (let i = 0; i < pyVec.length; i++) {
      assert.ok(Math.abs(tsVec[i] - pyVec[i]) < 1e-6, `${text} dim ${i}: ${tsVec[i]} vs ${pyVec[i]}`);
    }
  }
});

test("remember -> recall roundtrip ranks the relevant memory first", () => {
  const mem = new Memory();
  mem.remember("Decision: the primary database is PostgreSQL.", { kind: "constraint", importance: 5 });
  mem.remember("We chatted about the weekend weather.", { kind: "chat" });
  const hits = mem.recall("which database did we pick?", { limit: 2 });
  assert.ok(hits.length >= 1);
  assert.match(hits[0].record.content, /PostgreSQL/);
});

test("min_relevance_ratio prunes hits far below the top hit (default 0.3)", () => {
  const mem = new Memory();
  mem.remember("the launch date is September 14", { kind: "fact", importance: 3 });
  mem.remember("completely unrelated quarterly parking rota", { kind: "note", importance: 3 });
  const hits = mem.recall("when is the launch date?", { limit: 5 });
  assert.ok(hits.every((h) => h.relevance >= 0.3 * hits[0].relevance));
  const all = mem.recall("when is the launch date?", { limit: 5, minRelevanceRatio: 0 });
  assert.equal(all.length, 2);
});

test("hybrid recall finds exact identifiers", () => {
  const mem = new Memory();
  mem.remember("ticket ABC-123 tracks the login bug", { kind: "note", importance: 3 });
  mem.remember("the weather is nice today", { kind: "chat" });
  const hits = mem.recall("ABC-123", { hybrid: true, limit: 2 });
  assert.match(hits[0].record.content, /ABC-123/);
});

test("capture enforces the relevance policy and dedups", () => {
  const mem = new Memory({ importanceScorer: structuralImportance });
  const fact = mem.capture("My API rate limit is 5000 requests per hour.", { kind: "fact" });
  assert.equal(fact.stored, true);
  const filler = mem.capture("haha ok cool");
  assert.equal(filler.stored, false);
  assert.match(filler.reason, /floor/);
  const dup = mem.capture("My API rate limit is 5000 requests per hour.", { kind: "fact" });
  assert.equal(dup.stored, false);
  assert.match(dup.reason, /duplicate/);
});

test("supersession retires the stale fact and recall returns the head", () => {
  const mem = new Memory({ supersede: true, supersedeThreshold: 0.3, supersedeLowerThreshold: 0.1 });
  const oldRec = mem.remember("the launch date is September 14", { kind: "fact", importance: 4 });
  mem.remember("actually the launch date moved to October 2", { kind: "fact", importance: 4 });
  assert.notEqual(mem.store.get(oldRec.id).supersededBy, null);
  const hits = mem.recall("when is the launch date?", { limit: 2 });
  assert.match(hits[0].record.content, /October 2/);
});

test("forget repairs supersession chains; forgetMatching dry-run audits", () => {
  const mem = new Memory({ supersede: true, supersedeThreshold: 0.3, supersedeLowerThreshold: 0.1 });
  const a = mem.remember("the launch date is September 14", { kind: "fact", importance: 4 });
  const b = mem.remember("actually the launch date moved to October 2", { kind: "fact", importance: 4 });
  const c = mem.remember("actually the launch date moved again to November 5", { kind: "fact", importance: 4 });
  assert.equal(mem.forget(b.id), true);
  assert.equal(mem.resolveHead(mem.store.get(a.id)).id, c.id);

  const preview = mem.forgetMatching("launch date", { minRelevance: 0.1, dryRun: true });
  assert.ok(preview.length >= 1);
  assert.ok(mem.store.all().length === 2, "dry run must not delete");
});

test("buildContext emits lean dated lines with a Today anchor within budget", () => {
  const mem = new Memory();
  mem.remember("Launch date moved to September 14.", { kind: "fact", importance: 5 });
  const block = mem.buildContext("when do we launch?", { tokenBudget: 100 });
  assert.match(block.text, /# Today is \d{4}-\d{2}-\d{2}/);
  assert.match(block.text, /September 14/);
  assert.ok(!block.text.includes("id:"));
});

test("forgetDecayed evicts low-value records but protects durable kinds", () => {
  const mem = new Memory();
  mem.remember("haha filler note one", { kind: "note", importance: 1 });
  mem.remember("user PII must never be stored in plaintext", { kind: "constraint", importance: 5 });
  const dropped = mem.forgetDecayed({ maxRecords: 1 });
  assert.equal(dropped.length, 1);
  assert.match(mem.store.all()[0].content, /PII/);
});

test("guard blocks external actions without user_confirmation", () => {
  const mem = new Memory();
  mem.remember("Deploy target is staging.", { kind: "constraint", importance: 5, provenance: "observation" });
  const records = mem.recall("deploy target", { limit: 3 }).map((h) => h.record);
  assert.equal(decideMemoryUse(records, { intendedUse: "external_action" }).allowed, false);
  assert.equal(decideMemoryUse(records, { intendedUse: "planning" }).allowed, true);
});

test("importance scoring separates facts from filler", () => {
  assert.equal(contentImportance("haha yeah sounds good"), 1);
  assert.ok(contentImportance("My account number is 4421 and renewal is in 2027.") >= 3);
  assert.ok(structuralImportance("I prefer all answers in metric units.") >= 3);
});

test("SQLiteStore persists across reopen and shares writes live across connections", () => {
  const dir = mkdtempSync(join(tmpdir(), "midas-ts-"));
  const db = join(dir, "mem.db");
  const a = new Memory({ store: new SQLiteStore(db) });
  a.remember("decision: queue runs on Redis Streams", { kind: "fact", importance: 5 });

  const b = new Memory({ store: new SQLiteStore(db) });
  const hits = b.recall("what does the queue run on?", { limit: 1 });
  assert.match(hits[0].record.content, /Redis Streams/);

  // cross-connection live visibility (data_version probe)
  b.remember("note from the other connection", { kind: "note" });
  assert.ok(a.store.all().some((r) => r.content.includes("other connection")));
});

test("cosine of identical text embeddings is ~1", () => {
  const e = new HashingEmbedder();
  assert.ok(Math.abs(cosine(e.embed("hello world test"), e.embed("hello world test")) - 1) < 1e-6);
});
