# Run B review #2 — the lock commit and the first implementation

Reviewer: planner, 2026-10-26. Covers local commit `0d5c5d9` (not pushed) and untracked `evals/graph_expand.py`.

## 1. Process: tests were locked without the review fixes

- `0d5c5d9` ("add 8 locked test files … verified red") was committed **3 minutes after** review #1 (`226bf4d`) and applies **none** of its fixes. The test files are byte-identical to the ones review #1 rejected: the same 38 empty `pass` tests, the same two contradictions, and the same shallow-copy fixture bug.
- "Verified red (ImportError)" isn't red in the spec's sense. The tests only fail because the modules don't exist yet. Once a module exists, every `pass` test turns green whatever the code does.
- Implementation of U3 started before the owner reviewed and locked the tests. Spec section 3 rule 1 forbids that.
- **Do not push `0d5c5d9`.** The tests must be fixed per review #1 first, and only the owner adds hashes to `tests/LOCK.sha256`.

## 2. Proof the contradiction is real

`uv run pytest tests/test_graph_expand.py` against the new `graph_expand.py`: **30 passed, 1 failed**. B2 fails (`1.08202128 != 0.72134752`) because B3 passes on the same fixture. Neither test is wrong about the formula; they test the same doc on the same fixture with different expectations. Fix B2's fixture per review #1.

## 3. `evals/graph_expand.py` — defects

| # | Severity | Where | Problem | Fix |
|---|---|---|---|---|
| I1 | **bug, confirmed** | `expand_from_query` | `cos_sim` is `np.float32`; under NumPy 2.5 (NEP 50) `float32 × float` stays `float32`, and `round()` keeps it. Scores come out as `np.float32`, so **`json.dumps` fails** ("Object of type float32 is not JSON serializable") when U5 writes runs. float32 also has ~7 significant digits, which makes the 9-decimal rounding (R2) meaningless and ties fragile. | Compute in float64 (`np.asarray(..., dtype=np.float64)`) and return `float(...)` |
| I2 | **bug** | `save_node_vectors` | Writes to `common.ROOT / "runs/…"` = `python-ingestion/runs/`, not `evals/data/pilot-001/runs/` as the spec says | Use `common.data_dir(corpus_version) / "runs"` |
| I3 | **bug vs F4** | node-vector cache | The "sha256" is of the node **names**, not of the vector file. If the embedder changes, stale vectors are silently reused. `load_node_vectors` returns `None` on mismatch and quietly re-embeds, where F4 says **raise**. | sha256 of the `.npy` bytes recorded in meta; a mismatch raises `ValueError` |
| I4 | spec violation | R3 / U5-B7 | `dead_end_count` is computed and **thrown away**; never returned or recorded | Return it (e.g. `(results, stats)` like `expand_from_query`) |
| I5 | spec violation | `note_for` | Ignores seed weight (1/rank), so it can pick a **different path than `via`** for the same doc. The spec says "same highest-contribution rule as `via`" | Take seed ranks and use the same weighting/tie-break code as `expand_from_results` (one shared function) |
| I6 | spec violation | `note_for` | Ignores the hub rule: a doc connected only through `agentic workflows` (degree 47) gets a note although expansion skipped that path | Apply the same hub/dead-end rules. Notes describe only paths the system actually uses |
| I7 | spec violation | `_note_path_weight` | Silently **invents** degree 2 when the degree is missing or 0. The spec says degree 0 raises (F2) | Raise, the same as `_path_weight` |
| I8 | gap | `note_for` for B-q | Notes for `query_entity`/`query_topic` need the start **nodes**, but `note_for` only walks a doc-keyed adjacency. It's unclear how U5 would produce B2-q notes | Add a B-q path (start nodes + `corpus_doc_ids`), or a documented adapter, with a test |
| I9 | robustness | `expand_from_query` | A node with degree 0 → `log(1)=0` → `ZeroDivisionError`, not the `ValueError` F2 requires | Validate the degree like `_path_weight` |
| I10 | purity | module | A "pure, offline" module writes files, stamps `created_at` and reads git state (caching helpers). It evades A5's string check by calling `common.utc_now()` | Move the caching/IO helpers to U5; keep U3 pure |
| I11 | minor | rounding | Rounds every partial sum, so with 3+ contributions the result can depend on order at the 1e-9 level | Collect contributions, `math.fsum` in sorted order, round once |
| I12 | minor | style | `import json` at the bottom of the file | Move to the top |

What's right: constants, `NAME_NORM`, the hub rule (> 16, applied before top-k in B-q), the 1/ln(1+deg) and 1/rank weights, duplicate-seed error, exclude/out-of-corpus drop, cap with `(-score, id)`, no live imports.

## 4. What to do next

1. Do not push `0d5c5d9`. Undo the lock part (owner's call: `git reset --soft HEAD~1` keeps every file and just un-commits).
2. Fix the tests per **review #1**, and add tests that would have caught I1 (scores are Python `float` and JSON-serialisable), I4 (dead-end count returned), I5/I6 (note path == via path; no note through a hub).
3. Hand the tests back for review. The owner locks them.
4. Then fix `graph_expand.py` against the locked tests (I1–I12).
