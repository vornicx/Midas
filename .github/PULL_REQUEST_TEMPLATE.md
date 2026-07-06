<!-- Thanks for contributing. Small, measured, well-tested PRs merge fastest. -->

## What & why

<!-- What does this change, and what problem does it solve? Link the issue if there is one. -->

## Checklist

<!-- Tick what applies; delete rows that don't. Docs/examples-only PRs can skip the eval rows —
     see CONTRIBUTING.md "A lighter bar for docs & examples". -->

- [ ] `python -m pytest -q` passes
- [ ] `ruff check` is clean on changed files
- [ ] **Retrieval/forgetting change:** shows its effect on a reproducible metric (`recall@k` /
      retention), with the command, dataset, `n`, and caveats — and LoCoMo `recall@k` is unchanged
- [ ] **No LLM added to the ingest/query path** (or: explained why the value needs one)
- [ ] Docs/CHANGELOG updated if the public surface changed

## Measurements (if this touches retrieval, forgetting, or the guard)

<!-- Paste the command and the numbers. A measured negative result is a real contribution too. -->
