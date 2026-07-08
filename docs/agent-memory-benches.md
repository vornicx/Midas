# The Agent-Memory Bench Suite

*A standard for evaluating agent memory **beyond `recall@k`** — what a governed memory layer must guarantee
for an agent to act on it safely. Deterministic, $0, no LLM, reproducible.*

`recall@k` (LongMemEval, LoCoMo, BEAM) answers *"can you find the buried fact?"*. It does **not** answer the
questions that decide whether an agent can be **trusted to act** on memory:

- Will it act on a **stale** or **unconfirmed** belief?
- Does a **revised decision** surface its current value, or the old one?
- Does a prior **fix/failure resurface** so the agent doesn't repeat it?
- Can **adversarial memory** (injected, forgotten, superseded) authorize an unsafe action?

The Midas suite measures exactly these — in one command:

```bash
uv run python -m eval.benches
```

```
bench             metric                 score   target
Agent Continuity  action_safety           1.00    →1.00
                  decision_adherence      1.00    →1.00
                  repeated_mistake        1.00    →1.00
Memory-Safety     ASR (attack-success)    0.00    →0.00
                  benign_pass             1.00    →1.00
Coding-agent      decision_currency       1.00    →1.00
                  repeated_mistake        1.00    →1.00
                  forbidden_accuracy      1.00    →1.00
```

## The three benches

### 1. Agent Continuity Bench (`eval/continuity.py`)
Across a scripted multi-session project:
- **action_safety** — the guard allows/blocks a proposed use by provenance + currency, with an
  *allow-discriminator* (a current user-confirmed action MUST be allowed, so a "block everything" policy
  can't pass).
- **decision_adherence** — a revised decision surfaces its live value, not the superseded one.
- **repeated_mistake** — a prior bug-fix/failure resurfaces on a related query.

### 2. Memory-Safety eval (`eval/memory_safety.py`)
Adversarial — the attack surface is the stored memory itself (minimise Attack-Success-Rate against a
fixed suite, with a benign-pass floor so a "block everything" policy can't fake a pass):
- **ASR** *(attack-success-rate, target 0)* — over 6 attacks: superseded / unconfirmed / cross-agent /
  injected-content / forgotten memory trying to authorize a use.
- **benign_pass** *(target 1)* — legitimate uses are NOT over-blocked (the floor that keeps ASR honest;
  a guard that blocks everything scores ASR 0 *and* benign_pass 0).

### 3. Coding-agent bench (`eval/coding_bench.py`)
The coding vertical:
- **decision_currency** — a revised architecture decision surfaces its live value via `project_state`.
- **repeated_mistake** — prior bugs/failures resurface.
- **forbidden_accuracy** — forbidden actions are flagged, benign actions are not. *(Forbidden-action
  matching is lexical + an advisory semantic layer; see the honest bound in
  [`eval/forbidden_eval.py`](../eval/forbidden_eval.py) — F1 0.79, ~31% FP — measured, not assumed.)*

## Why a standard

Memory systems are compared on `recall@k` and judged answer-rate — both reader-coupled, both about
*finding*. None measure **trust**: safety under adversarial memory, currency, decision-adherence,
repeated-mistake avoidance. As agents get more autonomous, that is the axis that decides build-vs-trust.
These benches are deterministic and $0, so anyone can run them — on Midas, or by adapting the scenarios to
their own memory layer — and they fail loudly (the benign/allow floors mean a trivially-safe "do nothing"
policy does not pass).

For retrieval `recall@k`/`precision@k` and cost, see [BENCHMARKS.md](../BENCHMARKS.md); the full design,
the eval methodology, and the **published negatives** are in [docs/MIDAS.md](MIDAS.md).
