# Run B review #4 — implementation loop (`95545dc..305eb07`, branch `ladder/u11-graph-expand`)

Reviewer: planner, 2026-10-26.
**Verdict: the 8 units are implemented and the suite is green (461 passed). Fix the items below before any data step.**

## 0. Process

- **The tests were never locked.** `tests/LOCK.sha256` has 0 Run B entries, and the local tag `graph-expand-base` is still at `0d5c5d9`. The owner locks after section 3 below.
- The executor changed tests 4 times during the loop (`9eb1591`, `c838ca3`, `e9fa05b`, `db91f8e`). The planner checked all four and **accepts them**: each fixes a real test bug, and none weakens a check.
  - `1.082021280` → `1.082021281`: a planner error. Rounding once gives …281.
  - C1-E5: scoped to messages that fit the budget. A message already over 256 tokens can't be brought under, by spec.
  - U2-S3: `monkeypatch.undo()`.
  - U6-A1: regex.
- **No data step ran.** The freeze files (10-23) and the runs (10-24) are the old, invalid ones. There are no sheets.

## 1. Must fix (code)

| # | Where | Problem | Fix |
|---|---|---|---|
| K1 | `graph_expand.expand_from_query` line 224 | **Degree-1 start nodes are skipped.** `if degree == 1: continue`. But in B-q a degree-1 node attached to one doc is a *direct hit*: the query matched a topic, and that doc is the only one on it. The dead-end rule (R7) applies to **B-r only**, where the seed is the doc itself. 583 of 738 nodes have degree 1, so B-q currently ignores most of the graph. Spec U3: score = Σ cos × 1/ln(1+deg), no exception. | Remove the skip. A degree-1 start node contributes `cos / ln 2` to its doc. Also remove `w_path == 0.0` dead code in `note_for`/`note_for_query` (it can't happen). |
| K2 | `graph_expand.expand_from_results` line 117 | `seeds[:k]` takes the first k **as given**, not the k best ranks. A shuffled seed list (A2 shuffles, but only 2 seeds) would use the wrong seeds. | `used = sorted(seeds, key=lambda s: s["rank"])[:k]`. |
| K3 | `run_graph_expand.main` | **The input check is hollow.** `expected` is hashed from the file itself just before the check, so it can never fail. Only `documents.jsonl` is checked. The graph-set half skips the sha check and the re-ranker pin, and writes a sha of in-memory bytes, not of the `.npy` file (I3 again). The code is duplicated. | (a) U2 meta records `file_sha256: {name: sha}` for each file it writes. (b) U5 reads expected hashes from that meta (and `documents.jsonl`'s from the corpus manifest), for **every** frozen input. (c) One code path for both sets: `build_from_frozen(data_dir, expected, queries=..., cosine=..., suffix="" or ".graphq")`. The node-vector file is saved once and its file sha is used in both metas. |
| K4 | `run_graph_expand.build_systems` | **The C1 parity check is never called.** `denorm_control.assert_parity` exists but nothing runs it. Spec U4: if parity fails, escalate. | Before ranking C1, re-embed the *unchanged* docs (links, and messages with no in-corpus link) and call `assert_parity` against the stored Qdrant vectors from the frozen cosine source. If that isn't possible offline, record the stored vectors in U2 (read-only) and compare against them. |
| K5 | `topup_b` | **No entry point.** Data step d can't run. | `python -m evals.topup_b build` writes both sheets (with `check_sheet_size`) and prints only the two row counts. `merge` writes `qrels_v3.csv`. |
| K6 | `score_graph_expand` | **No entry point and no decision wiring.** Nothing computes steps 1–3 per set and per start (r/q), the graph set never uses `graph_beats` with `ladder_stats.win_tie_loss`, the predictions file isn't loaded, and Appendix D lacks the 4 metrics × 8 systems table and the path split. | Needed only after grading, but write it now: `main()` → per set, per start: step 1 (S2→C0), step 2 (C0→B2-x), step 3 (C1→B2-x and C2→B2-x). Original set uses `beats_original`; graph set uses `graph_beats(n, wins, losses)`. Then `decide_branch` → `attach_predictions` → `render_appendix` with the tables. |

## 2. Tests to add (they must fail on the current code, then pass after the fix)

- **U3-B12:** a B-q start node with degree 1 → its doc is returned with score `cos / ln 2` at 1e-9, and `via.path == "query_entity"`.
- **U3-E12:** 11 seeds given in **shuffled** order → only the 10 best ranks are used (the rank-11 seed's neighbor is absent).
- **U5-F3:** a U2 meta `file_sha256` that doesn't match a frozen file → `main`/`build_from_frozen` raises, naming the file, for the graph set as well as the original set.
- **U5-F4:** `build_systems` calls `denorm_control.assert_parity` (monkeypatch it to raise → the build raises before C1 is ranked).
- **U6-B8:** `topup_b` CLI `build` on a tmp data dir writes 2 CSVs with exactly `common.ANNOTATION_COLUMNS` and prints only counts (no `candidate_id` or content in stdout).
- **U8-B11:** `score_graph_expand.decide_for_set` on synthetic comparisons gives the right branch for each of: original set (bootstrap + 0.02) and graph set (scoreboard), both starts.

## 3. Then

1. One commit for the tests (section 2), one commit per fixed module. No push.
2. Planner quick check.
3. The owner locks all Run B test files.
4. Data steps a–d from the loop prompt, then stop for blind grading.
