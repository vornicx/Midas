"""The continuity control-plane: memory_conflicts (live contradictions), open loops (commitments),
resume (the one-call session-onboarding pack), and per-kind TTL retention. All no-LLM, deterministic
with the offline hashing embedder."""
from __future__ import annotations

import time

from midas import HashingEmbedder, Memory, parse_ttl_spec
from midas.continuity import close_loop, memory_conflicts, open_loops, remember_commitment, resume


def _mem() -> Memory:
    return Memory(embedder=HashingEmbedder())  # supersession off → conflicting writes both stay live


# ---- memory_conflicts ---------------------------------------------------------------------------

def test_conflicts_value_swap_across_actors() -> None:
    mem = _mem()
    mem.remember("The primary database is PostgreSQL.", kind="constraint", actor="claude-code")
    mem.remember("The primary database is MySQL.", kind="constraint", actor="cursor")
    found = memory_conflicts(mem)
    assert len(found) == 1
    c = found[0]
    assert c.signal == "value"
    assert {c.a.actor, c.b.actor} == {"claude-code", "cursor"}  # the multi-agent case


def test_conflicts_numeric_mismatch() -> None:
    mem = _mem()
    mem.remember("The launch date is September 14.", kind="fact")
    mem.remember("The launch date is September 20.", kind="fact")
    found = memory_conflicts(mem)
    assert len(found) == 1 and found[0].signal == "numeric"


def test_conflicts_negation_mismatch() -> None:
    mem = _mem()
    mem.remember("We support Internet Explorer in the dashboard.", kind="constraint")
    mem.remember("We no longer support Internet Explorer in the dashboard.", kind="constraint")
    found = memory_conflicts(mem)
    assert len(found) == 1 and found[0].signal == "negation"


def test_conflicts_ignore_agreement_and_unrelated() -> None:
    mem = _mem()
    mem.remember("The primary database is PostgreSQL.", kind="constraint")
    mem.remember("The primary database is PostgreSQL.", kind="constraint")   # restatement, agrees
    mem.remember("Standup happens on Tuesday mornings.", kind="fact")        # unrelated
    assert memory_conflicts(mem) == []


def test_conflicts_different_subjects_are_not_conflicts() -> None:
    mem = _mem()
    mem.remember("Project Apollo launches on May 3.", kind="fact")
    mem.remember("Project Artemis launches on June 9.", kind="fact")
    # different named subjects → different facts, even though the frames and numbers differ
    assert all(c.signal != "numeric" for c in memory_conflicts(mem))


def test_conflicts_skip_superseded() -> None:
    mem = _mem()
    old = mem.remember("The launch date is September 14.", kind="fact")
    new = mem.remember("The launch date is September 20.", kind="fact")
    assert len(memory_conflicts(mem)) == 1  # both live → a real conflict
    rec = mem.store.get(old.id)
    rec.superseded_by = new.id              # belief revision resolves the disagreement…
    mem.store.put(rec)
    assert memory_conflicts(mem) == []      # …so there is nothing left to surface


def test_conflicts_scope_filter() -> None:
    mem = _mem()
    mem.remember("The API rate limit is 100 requests.", kind="fact", metadata={"namespace": "a"})
    mem.remember("The API rate limit is 500 requests.", kind="fact", metadata={"namespace": "b"})
    assert memory_conflicts(mem, scope={"namespace": "a"}) == []  # scoped views don't cross-contaminate
    assert len(memory_conflicts(mem)) == 1                        # the unscoped view still sees it


# ---- open loops ---------------------------------------------------------------------------------

def test_open_loops_lifecycle() -> None:
    mem = _mem()
    loop = remember_commitment(mem, "Migrate the sessions table to UUID keys.", project="apollo")
    assert loop.kind == "commitment" and loop.importance == 4
    assert [r.id for r in open_loops(mem)] == [loop.id]
    assert open_loops(mem, scope={"project": "apollo"})[0].id == loop.id

    done = close_loop(mem, loop.id, "migrated in PR #42")
    assert open_loops(mem) == []                       # closed → no longer open
    assert mem.store.get(loop.id).superseded_by == done.id   # …but the history chain remains
    assert done.metadata["closes"] == loop.id and done.provenance == "action"


def test_close_loop_rejects_non_commitments() -> None:
    mem = _mem()
    fact = mem.remember("The sky is blue.", kind="fact")
    try:
        close_loop(mem, fact.id, "nope")
        raise AssertionError("close_loop must reject non-commitment ids")
    except ValueError:
        pass


def test_open_loops_oldest_first() -> None:
    mem = _mem()
    old = remember_commitment(mem, "Write the migration runbook.")
    old_rec = mem.store.get(old.id)
    old_rec.created_at -= 3600
    mem.store.put(old_rec)
    remember_commitment(mem, "Enable the new CI matrix.")
    assert open_loops(mem)[0].id == old.id  # most overdue first


# ---- resume -------------------------------------------------------------------------------------

def test_resume_bundles_all_sections() -> None:
    mem = _mem()
    mem.remember("From now on, always reply in Spanish.", kind="mission", importance=5)
    from midas.coding import remember_forbidden_action
    remember_forbidden_action(mem, "Never run destructive migrations on prod.", project="apollo")
    mem.remember("The primary database is PostgreSQL.", kind="constraint",
                 metadata={"project": "apollo"})
    remember_commitment(mem, "Add the audit log table.", project="apollo")

    pack = resume(mem, project="apollo")
    ctx = pack["context"]
    assert "FORBIDDEN:" in ctx and "destructive migrations" in ctx
    assert "CURRENT STATE:" in ctx and "PostgreSQL" in ctx
    assert "OPEN LOOPS:" in ctx and "audit log table" in ctx
    assert "CHANGED (added):" in ctx            # everything above was added inside the window
    assert pack["truncated"] is False
    # the pinned mission has no project tag, so the project scope excludes it — the global view has it
    assert "reply in Spanish" in resume(mem)["context"]


def test_resume_respects_token_budget() -> None:
    mem = _mem()
    for i in range(60):
        mem.remember(f"Architecture decision number {i} about service {i} routing.",
                     kind="constraint", importance=4)
    pack = resume(mem, token_budget=80)
    from midas import approx_tokens
    assert pack["truncated"] is True
    assert approx_tokens(pack["context"]) <= 80 * 1.2  # budget respected (approx by construction)


def test_resume_diff_window() -> None:
    mem = _mem()
    old = mem.remember("The old constraint from last month.", kind="constraint")
    rec = mem.store.get(old.id)
    rec.created_at -= 30 * 86_400
    mem.store.put(rec)
    mem.remember("The new constraint from this week.", kind="constraint")
    pack = resume(mem)  # default window: 7 days
    added_ids = {r.id for r in pack["added"]}
    assert old.id not in added_ids                    # old change is outside the diff window…
    assert any("old constraint" in r.content for r in pack["state"])  # …but still in current state


# ---- per-kind TTL retention ----------------------------------------------------------------------

def test_parse_ttl_spec() -> None:
    assert parse_ttl_spec("chat=30,note=90") == {"chat": 30.0, "note": 90.0}
    assert parse_ttl_spec(" chat = 7 ") == {"chat": 7.0}
    assert parse_ttl_spec("chat=oops,=5,note=-1,") == {}  # malformed entries are skipped, not fatal
    assert parse_ttl_spec("") == {}


def test_forget_expired_by_kind() -> None:
    mem = _mem()
    now = time.time()
    stale_chat = mem.remember("just chatting about the weather", kind="chat",
                              created_at=now - 40 * 86_400)
    fresh_chat = mem.remember("more chat from yesterday", kind="chat", created_at=now - 86_400)
    old_fact = mem.remember("The launch date is September 14.", kind="fact",
                            created_at=now - 400 * 86_400)
    dropped = mem.forget_expired({"chat": 30}, now=now)
    assert dropped == [stale_chat.id]
    remaining = {r.id for r in mem.store.all()}
    assert fresh_chat.id in remaining and old_fact.id in remaining  # other kinds untouched


def test_forget_expired_protections() -> None:
    mem = _mem()
    now = time.time()
    confirmed = mem.remember("User said: always use tabs.", kind="chat",
                             provenance="user_confirmation", created_at=now - 100 * 86_400)
    old = mem.remember("The launch date is September 14.", kind="fact",
                       created_at=now - 100 * 86_400)
    new = mem.remember("The launch date is September 20.", kind="fact",
                       created_at=now - 90 * 86_400)
    rec = mem.store.get(old.id)
    rec.superseded_by = new.id  # a revision chain: old -> new
    mem.store.put(rec)
    dropped = mem.forget_expired({"chat": 30, "fact": 30}, now=now)
    assert dropped == []  # user-confirmed never expires silently; chain members are audit material
    assert {confirmed.id, old.id, new.id} <= {r.id for r in mem.store.all()}
