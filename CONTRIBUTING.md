# Contributing to Midas

Thanks for considering a contribution. Midas is **eval-first**: the project's one durable asset is that
its reported numbers are true. A few principles keep it that way.

1. **Measure, don't claim.** Any change that affects retrieval or forgetting must show its effect on a
   reproducible metric — `recall@k` is deterministic (`python -m eval.runner …` / `python -m
   eval.retention …`). For reader-independence, also check `--dumb-reader` (`answer_dumb` should track
   `recall@k`). Quote numbers with the command, dataset, `n`, and caveats.
2. **Regression-check.** Run the suite (`python -m pytest -q`); for retrieval changes, confirm LoCoMo
   `recall@k` is unchanged. A win on a toy can break real data — that has happened here before.
3. **No LLM at ingest or query.** The wedge is local, cheap, auditable. New *no-LLM* mechanisms are very
   welcome; an LLM in the ingest/query path is not.
4. **Honest negatives are valued.** A measured "this didn't work" is a real contribution — the design
   doc (`docs/long-horizon-memory.md`) and [`docs/methodology.md`](docs/methodology.md) keep several on
   purpose (failure traces, `--value-rank-only` forgetting failures, conflicts-v1 residual staleness).

## A lighter bar for docs & examples

The measure-everything bar above is for the **retrieval / forgetting / guard core** — the code whose
numbers are the project's reputation. It is **not** meant to gate a typo fix, a clearer sentence, a
client setup guide, a new example, or a dataset mirror. Those are genuinely welcome and merge on a much
lighter touch: they just need to be **accurate** (don't assert a number the repo hasn't measured) and to
pass `python -m pytest -q` if they touch runnable code. Good first contributions:

- **Client guides** — one page for wiring Midas into a specific client, verified with `midas doctor`.
- **Examples** — a short, self-contained script showing one capability end to end.
- **Docs & repro polish** — fixing a stale command, clarifying methodology, mirroring a dataset so a
  benchmark reproduces from a clean checkout.

Look for [`good first issue`](https://github.com/vornicx/Midas/labels/good%20first%20issue), or just
open a PR — for docs/examples you don't need to open an issue first.

## Eval quick reference

All offline / deterministic (no API key unless `--judge`):

```bash
python -m eval.runner --dataset synthetic --dumb-reader
python -m eval.multiday --dataset conflicts --context-only --ab-supersede --midas-only
python -m eval.retention --dataset multiday --trace
```

Full methodology, verbatim MCP policy, and failure cases: [`docs/methodology.md`](docs/methodology.md).
Numbers and reproduce commands: [`BENCHMARKS.md`](BENCHMARKS.md).

## Dev setup

```bash
git clone https://github.com/vornicx/Midas && cd Midas
pip install -e ".[all,dev]"
python -m pytest -q
```

Open an issue first for anything non-trivial. Small, measured, well-tested PRs merge fastest.
