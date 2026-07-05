import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  Memory,
  SQLiteStore,
  closeLoop,
  memoryConflicts,
  openLoops,
  rememberCommitment,
  resume,
} from "../dist/index.js";

test("memoryConflicts finds the value swap, numeric drift, and negation flip", async () => {
  const mem = new Memory();
  await mem.remember("The primary database is PostgreSQL.", { kind: "constraint", actor: "claude-code" });
  await mem.remember("The primary database is MySQL.", { kind: "constraint", actor: "cursor" });
  const value = memoryConflicts(mem);
  assert.equal(value.length, 1);
  assert.equal(value[0].signal, "value");
  assert.deepEqual(new Set([value[0].a.actor, value[0].b.actor]), new Set(["claude-code", "cursor"]));

  const mem2 = new Memory();
  await mem2.remember("The API rate limit is 100 requests per minute.", { kind: "fact" });
  await mem2.remember("The API rate limit is 500 requests per minute.", { kind: "fact" });
  assert.equal(memoryConflicts(mem2)[0].signal, "numeric");

  const mem3 = new Memory();
  await mem3.remember("We support Internet Explorer in the dashboard.", { kind: "constraint" });
  await mem3.remember("We no longer support Internet Explorer in the dashboard.", { kind: "constraint" });
  assert.equal(memoryConflicts(mem3)[0].signal, "negation");
});

test("memoryConflicts ignores benign look-alikes and superseded pairs", async () => {
  const mem = new Memory();
  await mem.remember("Project Apollo launches on May 3.", { kind: "fact" });
  await mem.remember("Project Artemis launches on June 9.", { kind: "fact" });
  await mem.remember("Deploys run from the main branch.", { kind: "fact" });
  await mem.remember("Deploys run from the main branch.", { kind: "fact" });
  assert.deepEqual(memoryConflicts(mem), []);

  const mem2 = new Memory();
  const old = await mem2.remember("The launch date is September 14.", { kind: "fact" });
  const nw = await mem2.remember("The launch date is September 20.", { kind: "fact" });
  assert.equal(memoryConflicts(mem2).length, 1); // both live → real conflict
  old.supersededBy = nw.id;
  mem2.store.put(old);
  assert.deepEqual(memoryConflicts(mem2), []); // revision resolves it
});

test("open loops lifecycle: record, list oldest-first, close with auditable resolution", async () => {
  const mem = new Memory();
  const loop = await rememberCommitment(mem, "Migrate the sessions table to UUID keys.", { project: "apollo" });
  assert.equal(loop.kind, "commitment");
  assert.deepEqual(openLoops(mem).map((r) => r.id), [loop.id]);
  assert.equal(openLoops(mem, { scope: { project: "apollo" } })[0].id, loop.id);

  const done = await closeLoop(mem, loop.id, "migrated in PR #42");
  assert.deepEqual(openLoops(mem), []);
  assert.equal(mem.store.get(loop.id).supersededBy, done.id);
  assert.equal(done.metadata.closes, loop.id);
  await assert.rejects(closeLoop(mem, "nope", "x"), /no open commitment/);
});

test("resume bundles pinned/state/changes/loops/conflicts under a token budget", async () => {
  const mem = new Memory();
  await mem.remember("From now on, always answer with metric units.", {
    kind: "mission", importance: 5, metadata: { standing: true },
  });
  await mem.remember("Decision: the primary database is PostgreSQL.", { kind: "constraint", importance: 5 });
  await rememberCommitment(mem, "Add the audit log table.");
  const pack = resume(mem);
  assert.match(pack.context, /PINNED:/);
  assert.match(pack.context, /metric units/);
  assert.match(pack.context, /CURRENT STATE:/);
  assert.match(pack.context, /PostgreSQL/);
  assert.match(pack.context, /OPEN LOOPS:/);
  assert.equal(pack.truncated, false);

  for (let i = 0; i < 60; i++) {
    await mem.remember(`Architecture decision number ${i} about service ${i} routing.`, {
      kind: "constraint", importance: 4,
    });
  }
  assert.equal(resume(mem, { tokenBudget: 80 }).truncated, true);
});

test("SQLite audit chain: chained entries, verification, tamper detection", async () => {
  const dir = mkdtempSync(join(tmpdir(), "midas-ts-audit-"));
  const db = join(dir, "m.db");
  const store = new SQLiteStore(db);
  const mem = new Memory({ store });
  const a = await mem.remember("The primary database is PostgreSQL.", { kind: "constraint" });
  await mem.remember("The launch date is September 14.", { kind: "fact" });
  mem.forget(a.id);

  const log = store.auditLog();
  assert.deepEqual(log.map((e) => e.op), ["put", "put", "delete"]);
  assert.equal(log[1].prev_hash, log[0].hash);
  assert.ok(!JSON.stringify(log).includes("PostgreSQL")); // hashes only, never content
  assert.deepEqual(store.verifyAuditLog(), { ok: true, entries: 3, first_invalid_seq: null });

  // an attacker rewrites history directly → the chain breaks at that entry
  const { DatabaseSync } = await import("node:sqlite");
  const raw = new DatabaseSync(db);
  raw.exec("UPDATE audit_log SET op = 'delete' WHERE seq = 2");
  raw.close();
  const tampered = new SQLiteStore(db).verifyAuditLog();
  assert.equal(tampered.ok, false);
  assert.equal(tampered.first_invalid_seq, 2);
});

test("audit can be disabled for perf-sensitive paths", async () => {
  const dir = mkdtempSync(join(tmpdir(), "midas-ts-audit-"));
  const store = new SQLiteStore(join(dir, "m.db"), { audit: false });
  await new Memory({ store }).remember("perf-path write", { kind: "note" });
  assert.deepEqual(store.auditLog(), []);
});
