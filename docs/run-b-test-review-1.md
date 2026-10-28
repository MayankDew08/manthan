# Run B test review #1 — fix before locking

Reviewer: planner, 2026-10-23. Against `docs/graph-expansion-run-b-spec.md` section 13.
**Verdict: do not lock.** Fix everything below, hand back, re-review.

## Rule broken across all files: empty tests

Spec section 3 rule 1: tests are handed back **red** and every row of the section 13 table is a real test. A test whose body is `pass` is green on any implementation, so locking it locks a hole. **Every `pass` test must be implemented** (synthetic fixtures, `tmp_path`, injected fakes or monkeypatching), or marked `artifacts`/`live`/`model` **with a real body** that reads the real files or services.

| File | Empty (`pass`) tests |
|---|---|
| test_graph_expand.py | B7, E11, F4, A4 |
| test_denorm_control.py | C1-E5, C1-F1, C1-A2 |
| test_freeze_neighbors.py | unmapped corpus doc (U2-F1), count drift (U2-F2), determinism (U2-A1) |
| test_run_graph_expand.py | **11 of 12** (everything except A1) |
| test_topup_b.py | B3, B4, B7, F1, F2, F3 |
| test_latency_expand.py | B2, S1 |
| test_score_graph_expand.py | B1, B2, B5, B6, B7, B8, B9, B10, A2 |

## Contradictions (no implementation can pass both)

1. **test_graph_expand.py B2 vs B3.** Both assert B's score on the same `ADJACENCY` fixture: B2 expects `1/ln 4`, B3 expects `1/ln 4 + 1/ln 4 / 2`. Fix: B2 uses its own fixture where only S1 reaches the target through a degree-3 entity.
2. **test_denorm_control.py C1-B1 vs C1-B2.** `msg1` has two links, but B1 asserts `texts["msg1"] == "Hello world Link one text"` and B2 asserts it also contains "Link two text". Fix: B1 uses a message with exactly one in-corpus link.

## Tests that don't test what they claim

| Test | Problem | Fix |
|---|---|---|
| U3 B4 | Compares two constants; never calls the module | Two fixtures, the same path from a rank-1 vs a rank-2 seed; assert module score ratio == 2 |
| U3 B5 | Tie-break branch never triggers (no equal scores in the fixture) | Add two neighbors with equal score; assert id order |
| U3 B8 | D14's point is a doc **already in S2's pool** getting a note; not tested | Put the connected doc in `exclude` and assert a note is still returned |
| U3 B9 | Only contains/entity wording; no topic, query_entity, query_topic; no 80-char truncation | Assert all 5 templates exactly; a message seed longer than 80 chars is cut to 80 |
| U3 E3 | Doesn't check the dead-end **count** (R3) | Assert the returned/recorded `dead_ends == 1` |
| U3 E5 | Doesn't check the excluded neighbor still gets a note | Add the `note_for` assertion |
| U3 A2 | Unseeded `random.shuffle`; seeds not shuffled; rounding (R2) not checked | `random.Random(0)`; shuffle seed order too; assert every score == `round(score, 9)` |
| U4 C1-E2 | `TOY_DOCS.copy()` is shallow, so it **mutates the shared fixture** and other tests depend on run order | `copy.deepcopy` |
| U4 C2-B1 | Only checks path names | Assert the same scores (1e-9) and order as U3 B1–B5 on the mirrored fixture |
| U4 C2-B2 | msg1 and msg2 also share topic `agents`, so they connect without `NAME_NORM` | Fixture where the **only** shared string differs by case/dash |
| U4 C2-B3 | `hub_topic` is on one doc only (degree 1), so the test passes without a hub rule | 18 docs carrying the same string (degree 17) → skipped; same with 17 docs (degree 16) → kept |
| U2 S3 | Only checks the function exists | Assert `freeze_neighbors` calls `assert_listener_stopped` before its first read (monkeypatch to record call order) |
| U2 degree consistency | `if key in node_degree` silently skips missing nodes | Missing node → fail |
| U1 E1 | "flagged not answerable" not asserted | Assert the flag |

## Interface questions (decide, don't guess)

- **Where do message→link relations come from?** U4 and U6 tests read `doc["links"]`, but `documents.jsonl` has no such field. Per spec they come from `graph_neighbors_v1.jsonl` (CONTAINS rows). Either pass the neighbors in, or have U5 attach `links` from the neighbors file before calling, and say which in the test docstring. Don't add the field to `documents.jsonl`.
- `expand_from_results(seeds, adjacency, exclude, corpus, ...)` takes `corpus` as the 4th argument; that's fine, it's needed for the out-of-corpus drop. Keep it, and use it the same way in `expand_from_query` and `payload_expand`.

## Spec correction (planner error)

Section 13.2 C1-E3 said "6 tokens (accounting for the separator)". Correct value: budget 10 − message 4 − separator 1 = **5**. The test's 5 is right; the spec is fixed in the same commit as this review.
