# Run B test review #3 — commit `b2cd55f`

Reviewer: planner, 2026-10-26. Covers the 8 Run B test files in `b2cd55f` against spec section 13 and reviews #1/#2.
**Verdict: nearly there. Fix the items below, then the owner locks.**

## What's right

- Zero `pass` bodies. `tests/LOCK.sha256` untouched. One commit, not pushed.
- 94 fail and 8 pass. Every failure is a missing module or the interface the tests ask for (`(rows, stats)`, `note_for_query`, `save_node_vectors(..., path)`), not a broken test. The 8 passing tests are U3 checks that don't depend on the interface (B10, F1, F2, F3, F5, A1, A3, A5), which is expected because `graph_expand.py` exists.
- Both review #1 contradictions are gone. Every review #1 weak-test row is fixed. I1/I3/I4/I5/I6 each have a test that fails on the old code.
- U2-B8 (`out_of_corpus_neighbors`) is a good test, and it **already catches a real bug** (see section 3).

## 1. Must fix

| # | Test | Problem | Fix |
|---|---|---|---|
| T1 | U3 B8 line 199 | Asserts `note_for("C", ...)` is **not None**, but C is reached only through topic `agents`, degree 17, which is a hub. I6 (line 452) asserts no note through a hub, so B8 and I6 contradict each other. | Assert `note_for("C", ...) is None` and add a comment that it's the hub case. A and B stay as they are. |
| T2 | U3 E3, I4, B8 fixture (`dead1`, `x1..x3`) | A dead end is modelled as an edge **to a neighbor** with degree 1. That can't happen: degree 1 means the seed is the only doc on the connector. The real `graph_neighbors_v1.jsonl` has **0 rows with degree 1**, while `graph_nodes_v1.jsonl` has **583 degree-1 nodes**. So the real pipeline could never count a dead end. | Rule R7 (below): dead-end edges have `neighbor: None`. Change the fixtures to `_edge(None, "entity", "llm", 1)`. Add **U5-B9**: a seed in a degree-1 node's `corpus_doc_ids` gives `dead_ends == 1` in the counters. |
| T3 | U5 B8 lines 294–298 | Hollow: `min(embedded_f01, key=len)` picks a plain filler (`"filler alpha"`) instead of f01, so an implementation that sends the full glued text to the embedder still passes. | Capture the embedder input **for f01 by id** (for example, the fake records `(index, text)` and you map it back through the order of the docs passed in). Assert it equals exactly `"filler alpha " + " ".join(["l1"] * 253)`: 256 words = 2 message + 1 separator + 253 link. Then assert f01's counters per R8. |
| T4 | U4 C1 tests, U6 B6 | D21 writes CONTAINS in **both directions**, but no fixture has a `link → message` row, so "links are unchanged" is never tested against the real data shape. An implementation that glues neighbor text onto links would pass. | Add the reverse rows (`_contains("link:a", "msg1")`, etc.) to the C1-B2 and U6-B6 fixtures. Assert the link's text or content is unchanged. |
| T5 | U5 B2–B6 | No test checks what actually separates B2 from B1: the re-ranker input `"[connected: <note>] " + text` (section 4). | **U5-B10**: from `FakeReranker.calls`, B2-r/B2-q docs that have a note get exactly `"[connected: " + note + "] " + text`. Docs without a note, and every B1/S2/C0/C2 doc, get the plain text. |
| T6 | U3 | The B-q score formula is never checked numerically: B7 tests membership only, and I1/A2 test only types and determinism. `via` for B-q isn't checked either. | **U3-B11**: two start nodes with known cosines and degrees. Assert doc score = Σ cos × 1/ln(1+deg) at 1e-9, and `via == {"path": "query_entity", "connector": ..., "seed": None}`. |

## 2. Should fix

| # | Test | Problem | Fix |
|---|---|---|---|
| S1 | U4 C1-A2 | With the fake tokenizer the model is never loaded, so the test can't fail for a network-using implementation. | Mark it `model`: real tokenizer and embedder, `HF_HUB_OFFLINE=1`, `socket.socket` patched to raise. |
| S2 | U5 F1 | Tests only the helper `_assert_file_sha`, not that the build calls it for every frozen file, before writing anything. | Run the build entry point with a tmp runs dir holding one altered frozen file. Assert `ValueError` naming the file, and that no run file is written. |
| S3 | U5 A2 | Pure unit test marked `artifacts`, so it's deselected by default. | Drop the marker. Optionally add a real artifact version later. |
| S4 | U2 B8 line 162 | Accepts either rows or distinct docs as the meta count. | Pick one, distinct outside docs, and say so in the meta key's docstring. |
| S5 | U8 B5 | Only n = 4 is tested for "too few", and the spec says n ≤ 4. | Add (3, 3, 0) and (0, 0, 0) → `None`. |
| S6 | U4 C1 | Two functions (`denorm_texts` → dict, `denorm_texts_with_tokenizer` → (texts, meta)) do one job, so the counters can be lost on the first path. | One function returning `(texts, meta)`. |

## 3. Data finding (not a test problem): the U2 freeze dropped out-of-corpus neighbors

Read-only check of the frozen files (graph structure only, no system outputs): **72 nodes have `degree` > the number of their `corpus_doc_ids`**, so out-of-corpus neighbors exist, yet the neighbors file has **0 rows with `in_corpus: false`**. U2-B8 will fail on the current files, which is correct. After U2 is fixed, re-freeze; the U5 runs built from the old freeze are invalid anyway.

## 4. New rules (approved by the owner 2026-10-26; added to spec 13.1 and 13.2)

- **R7 — dead ends.** U5 adds, for every seed, an edge `{"neighbor": None, "path": <entity|topic>, "connector": <name>, "degree": 1}` for each degree-1 node in `graph_nodes_v1.jsonl` whose `corpus_doc_ids` contains the seed. U3 counts these edges as dead ends and returns nothing for them.
- **R8 — C1 counters, per link.** A link that keeps some but not all of its words counts toward `links_trimmed`. A link that keeps none counts toward `links_dropped_no_room`. Each separator costs 1 token. U5-B8's f01: 1 trimmed (l1), 1 dropped (l2).

## 5. Housekeeping

- An untracked `worktrees/u2-freeze-graph/` (Run A branch `ladder/u2-freeze-graph`) sits inside the main tree. Move it under `Manthan.worktrees/` (`git worktree move`) or add `worktrees/` to `.gitignore`. Don't delete it without the owner's OK.

## 6. Next

1. The executor fixes T1–T6 and S1–S6 in **one more commit on `retrieval-ladder`**. No implementation code, no LOCK edits, no push.
2. Planner quick check (review #4 should be short).
3. The owner adds the 8 hashes to `tests/LOCK.sha256`. Then implementation, per review #2 section 4.
