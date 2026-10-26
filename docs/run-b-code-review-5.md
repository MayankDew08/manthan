# Run B review #5 — review #4 fixes (`55391dd..b3b684b`)

Reviewer: planner, 2026-10-26.

## Goal: what you are after

Run B must be able to run **end to end on the real data** and produce a result that can be trusted. "Trusted" means one thing: **if something is missing, stale or wrong, the code stops with an error that names it. It never carries on quietly.** Every fix below turns a silent "carry on" into a loud "stop".

### Expected outcome of this round (done when ALL hold)

1. `evals/graph_expand_predictions.json` exists and is **byte-identical** to the JSON block in spec section 11.
2. The 6 new tests exist, and each **failed on the code before your fixes** (report each failure message).
3. `uv run pytest -q` is green, and no existing test file was changed (`git diff b3b684b -- tests/` shows only additions).
4. Each of these stops with a `ValueError` naming the problem, instead of continuing:
   - a missing predictions file;
   - a missing run file when building sheets;
   - a missing stored-vectors file;
   - any frozen file (including `documents.jsonl`) whose sha256 differs from the one U2 recorded.
5. The scoring output, for each set (original, graph) and each start (r, q), shows:
   - the branch: `GRAPH OUT`, `SHIP THE SHORTCUT(S)`, `GRAPH STAYS`, or `TOO FEW TO DECIDE` (graph set, n ≤ 4);
   - the owner's and the planner's prediction for every comparison they predicted, with `prediction_held` true or false (`abstain` stays abstain);
   - averages over answerable queries only.
6. Topic vectors are embedded fresh in each build, never read from an old file.
7. Nothing else happened: no freeze, no run build, no sheets, no push, no edit to `tests/LOCK.sha256`.

### What comes after (not your job this round, so you know where it leads)

The owner locks the tests and the predictions file. Then (Neo4j and Qdrant up; the code itself checks that no Drive listener is running, and none is today):
- **freeze** (read-only);
- **build** all 8 systems for both question sets;
- **build two blind grading sheets** and print only their row counts.

The owner grades the sheets, then **scoring** prints one branch per set and start, and Appendix D is written. The final answer Run B gives is that branch: *does the graph find relevant messages that normal search and the cheap shortcuts miss?*

## Verified

- The 6 review #4 tests were run against the **pre-fix code** (`305eb07`) in a temp worktree: all 6 fail there and pass now. No existing test was changed. Suite: 467 passed.
- K1 (degree-1 start nodes), K2 (sort seeds by rank), K3 (U2 meta records `file_sha256`; one code path for both sets), K4 (stored Qdrant vectors frozen by U2; parity runs before C1) and K5 (topup_b CLI prints counts only) are correct in substance.
- Still not done: **the tests are not locked** (`LOCK.sha256` has 0 Run B entries), and no data step has run (correct).

## Must fix

| # | Where | Problem | Fix |
|---|---|---|---|
| P1 | `score_graph_expand.main`, `attach_predictions` | **The predictions are never used.** `evals/graph_expand_predictions.json` doesn't exist on any branch. If it's missing, `main` silently uses `{}`. Even with the file, `attach_predictions` looks up `"ladder_s2_cos_bm25_rr->ctrl_c0_deeper_rr"`, but the spec block (section 11) uses short names (`"S2->C0"`), nested under `owner`/`planner` → `original_set`/`graph_set`. Every prediction would come out `None`. | (a) Copy the section 11 JSON block **verbatim** to `evals/graph_expand_predictions.json`. (b) U8 maps short names (S2, C0, C1, C2, B1-r, B2-r, B1-q, B2-q) to run ids, and reports **owner** and **planner** predictions separately, per set, plus `expected_branch` vs the branch actually taken. "abstain" stays abstain. (c) A missing file **raises**. |
| P2 | `decide_for_set`, graph set | n ≤ 4: `graph_beats` returns `None`, `bool(None)` is `False`, and the branch becomes **"GRAPH OUT"**. D16 says n ≤ 4 means **"too few to decide", with no branch taken**. | If n ≤ 4, the branch is `"TOO FEW TO DECIDE"` for that set (both starts). Never coerce `None` to `False`. |
| P3 | `score_graph_expand.main` metrics | nDCG@5, recall_pool and judged@5 are averaged over **all** queries (`len(qids)`). Spec U8 says average over `ladder_stats.answerable` only. Unanswerable queries silently pull every average down. | Average over the answerable query ids for each set (the same list `score_sets` uses). |
| P4 | `topup_b.cmd_build` | `_read_jsonl_if_exists` turns a **missing run file into an empty run**. The sheets would then be built from a partial pool, the owner would grade it, and the gap would never show. | Every one of the 8 run files (both sets), the graph-set cosine run and the neighbors file must exist. Otherwise raise, naming the file. |
| P5 | `run_graph_expand.build_from_frozen` | If `graph_node_vectors_v1.npy` **already exists, it's reused without any check**. A stale file from an earlier freeze (different nodes or count) would silently feed B-q. `documents.jsonl` is still checked against its own hash taken a moment earlier (the comment says "manifest", but no manifest hash exists). | `main` embeds the node names **once** and passes the vectors to both calls; nothing is read from disk. U2 also records `documents.jsonl`'s sha256 in its meta (`file_sha256`), and U5 checks against that. |
| P6 | `build_systems` / `_load_stored_vectors` | If the frozen vectors are missing, the **parity check is silently skipped** (`return {}` → `if stored_vectors:`). | `build_from_frozen` raises if `document_vectors_v1.*` is missing. In `build_systems`, `stored_vectors=None` is allowed only for unit fixtures. |

## Tests to add first (each must fail on the current code)

- **U8-A3:** `attach_predictions` with the **real section 11 file shape** gives the owner's `"improved"` for S2→C0 on the original set and the planner's `"no detectable difference"`. A missing predictions file raises.
- **U8-B12:** graph set with 4 answerable queries → branch `"TOO FEW TO DECIDE"` for both starts.
- **U8-B13:** 3 queries, 1 of them unanswerable → the reported nDCG@5 average uses 2 queries.
- **U6-F4:** `cmd_build` with one of the 16 run files missing → raises, naming it, and writes no sheet.
- **U5-F5:** a stale `graph_node_vectors_v1.npy` (wrong row count) on disk → not used: the vectors are freshly embedded and match `len(nodes)`.
- **U5-F6:** `build_from_frozen` with no `document_vectors_v1.npy` → raises.

## Then

1. Tests commit (each new test fails first), then fixes, one module per commit. Don't edit existing tests. No data step, no push.
2. Planner quick check.
3. The owner locks the **8 test files plus `evals/graph_expand_predictions.json`**, and pastes the lock output.
4. Data steps a–d, then stop for blind grading.
