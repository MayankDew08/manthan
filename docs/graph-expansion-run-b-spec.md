# Run B — graph as candidate generator: spec

Status: **planning complete; ready for execution once O4 and O5 (section 12) are done.** · Planned 2026-10-21, revised 2026-10-23 · Branch `ladder/u11-graph-expand` (from `retrieval-ladder` after O4 and O5 are committed; the owner tags that commit `graph-expand-base`)
Corpus `pilot-001` (188 docs: 148 messages, 40 links, frozen) · Queries `queries.csv` (22 answerable) + `queries_graph.csv` (8; planner-drafted from doc text, owner-adjusted, D18)

This spec is the contract. The executor builds exactly what is described here. It changes no constant after seeing a result and does not edit any file listed in `tests/LOCK.sha256`. Everything in section 11 is a **prediction**, not a result.

---

## 1. Question

Run A (Appendix C) put the graph into the **ranking** job, and it added nothing: s2→s4 = +0.004 nDCG@5, CI [−0.007, 0.020], +24 ms p50. Run B asks a different question:

> **Can the graph find relevant documents that cosine + BM25 never return, and is that win something only a graph can deliver?**

The graph can only win where relevance comes from a **connection** and not from the doc's text: two docs share an entity, or a thin message ("check this out 🔥") is relevant through the link it shared.

## 2. Decisions

### 2.1 Teaching session (2026-10-21)

| # | Decision | Why |
|---|---|---|
| D1 | **Headroom first.** Count relevant docs outside S2's top 30 before building anything. | If that count is zero, Run B can't show a lift by construction. |
| D2 | **Graph = candidate generator.** It proposes docs; the cross-encoder makes the final ranking. | Ranking was already tested in Run A and lost. |
| D3 | **Doc-to-doc paths:** CONTAINS (1 hop); doc→Entity←doc and doc→Topic←doc (2 hops through a connector). | `store.py:50-87`: Entity/Topic nodes are connectors, never documents. |
| D4 | **Weight paths by 1/ln(1+degree) of the connector, and skip "too popular" connectors** (limit: open item O1). | Hub nodes are the graph equivalent of stopwords (the same flaw that sank graph v0, which had no IDF). |
| D5 | **Two re-rank variants:** plain text (B1), and text with a connection note (B2). | Graph-found docs are relevant through a connection, which a text-only cross-encoder can't see. |
| D6 | **Two no-graph shortcuts:** C1 denormalized embeddings, C2 payload entity overlap. | The graph only earns a query-time role if it beats the same join done without a graph. |
| D7 | **Tag every expanded doc with its path (`via`).** Split the lift by path from a single run. | No separate run per path. |
| D8 | **Blind top-up grading.** Every unjudged doc in a system's top 5 is graded before scoring. Report judged@5. | Corrects pooling bias, which otherwise works against B. |
| D9 | **A second query set** `queries_graph.csv`, reported **separately**. | The original 22 queries weren't written to need the graph. |
| D10 | **Four metrics only:** nDCG@5 (decides), Recall@pool (did it *find* anything), judged@5 (is the comparison valid), latency (recorded). | The owner asked for impactful metrics only. |

### 2.2 Planning review with the owner (2026-10-23)

| # | Decision | Why |
|---|---|---|
| D11 | **New control C0, "just read more":** rrf(cosine, BM25) → re-rank, with as many candidates as the largest B pool for that query. B must beat C0, not just S2. | **12 of the 19** missed relevant docs sit in the top 60 when cosine and BM25 simply look deeper (section 2.3). Without C0, the graph would take credit for a bigger pool. |
| D12 | **Two starting points for the walk:** B-r starts from S2's top results; B-q starts from the graph nodes closest to the query. | S2 is weakest exactly on the headroom queries, so walking from its results may start in the wrong place. |
| D13 | **B-q matches by meaning:** embed every Entity/Topic name once (same embedder as the index) and take the **5 nodes** closest to the query. | Word matching repeats BM25 (and is how graph v0 failed). An LLM extractor is non-deterministic, so it's ruled out. |
| D14 | **Notes on every connected doc.** In B2, *every* doc in the pool that connects to a starting point gets its connection note, whether it came from search or from the graph. | Otherwise the note itself becomes a "graph doc" badge the re-ranker might favour. |
| D15 | **All 7 new runs are kept** (C0, B1-r, B2-r, B1-q, B2-q, C1, C2). | The owner accepts the extra grading for the full picture. |
| D16 | **Graph set: 8 queries, scoreboard rule.** B "beats" X when it has **0 losses** and at least the wins below, where n = answerable graph queries (win/loss = Δ nDCG@5 beyond ±0.01, `ladder_stats.win_tie_loss`). n=8 → 7 wins · n=7 → 7 · n=6 → 6 · n=5 → 5 · n≤4 → **"too few to decide"** (no branch is taken for that set). | With 8 queries the bootstrap CI almost never excludes 0, even for a real effect. Each threshold keeps the chance of passing by luck ≤ 5% (3.5%, 0.8%, 1.6%, 3.1%). Locked before results so the bar can't move. |
| D17 | **Latency: record everything, no limit yet.** Latency decides nothing in this run. | The owner wants the numbers before setting a bar. |
| D18 | **Rules for writing the graph questions:** see P2. **Disclosure:** the planner drafted the 8 questions (2026-10-23) from `documents.jsonl` text only, and the owner adjusted them. The planner had already seen connector degrees (section 2.3), so Appendix D must state this. The planner registers **no predictions for the graph set** (O2). | Guards against questions written to suit the graph, and keeps the remaining bias visible. |
| D19 | **Merge duplicate connector names** before building paths: `norm(name) = lowercase, strip, collapse runs of whitespace/'-'/'_' into one space`. Plurals stay separate (`llm` ≠ `llms`). The same `norm` applies to C2's payload strings. | 28 entity groups and 41 topic groups are the same concept under different spellings, which splits paths that should join. Cleaning is applied equally to graph and shortcut. |
| D20 | **`HUB_RULE`: skip a connector linked to more than 16 docs, counted after the merge.** | Set from the graph's shape only, at the natural gap after merging (topics jump 17 → 10). It removes exactly `agentic workflows` (47), `local llm` (21), `performance` (19) and `knowledge graph` (17); all entities stay (max 11). |
| D21 | **CONTAINS is walked in both directions** (message → link and link → message). | Search often finds the *link* (it has the text); the backward step reaches the thin message that shared it. |
| D22 | **The freeze is pinned to the planning-time graph:** 228 docs (156 messages, 72 links), 79 CONTAINS edges, 365 merged entities, 373 merged topics, 4 connectors with degree > 16. Any difference → stop and escalate. | The hub limit and every planning fact were set on this graph. A changed graph means re-planning. |
| D23 | **C1 is the shortcut you would ship:** cosine, BM25 and the re-ranker all see the glued message + link text. Only the embedding input is trimmed to 256 tokens (R5); BM25 and the re-ranker read the full glued text. | The graph must beat the real alternative, not a half-built one. |
| D24 | **Grader view split:** the original-set top-up shows doc text only (as Run A); the graph-set sheet shows message text plus the titles of links it shared. | New original-set grades must match how the 1,079 existing grades were made. Thin-message answers (e.g. a bare Granola URL) can't be judged without the link title, which is a fact of the doc, identical for every system, so blindness holds. |
| D25 | **Live Qdrant point id drift (2026-10-26).** One corpus doc, `message:346a7af1-…` (sent_at 2026-08-04T12:53:00), now lives in Qdrant under point id `70927317-…`. Its stored text equals the corpus `embedded_text`, and `uuid5("msg-{sent_at}-{embedded_text}")` = `70927317-…`. The old id is the uuid5 of the Neo4j text, which has one extra newline. Every Run A file and grade uses `346a7af1-…`. **Decision:** corpus `candidate_id`s never change. U2 builds an alias map `{corpus candidate_id → live Qdrant point id}`, allowed **only** when the live id equals `uuid5("msg-{sent_at}-{embedded_text}")` of that corpus doc, exactly (no fuzzy matching). U2 uses it for the payload/vector scroll and translates graph-set cosine hits back to corpus ids. Meta records `qdrant_id_aliases`. Any corpus doc not found by its own id or an exact alias still raises. | Re-keying would rewrite Run A's grades and runs. Dropping the doc would change the corpus (it holds 3 grades) and break 19/104. The text is identical, so the doc and its vector are the same; only the key moved. Someone wrote to Qdrant after Run A; this is recorded, not hidden. |

### 2.3 Facts measured before any B/C run (planner, from S2, cosine, BM25 and qrels_v2 only)

**Headroom (2026-10-21):** on the original set, **19 of 104** relevant docs (grade ≥ 2) sit outside S2's top 30.

| Query | Relevant | Outside S2@30 |
|---|---|---|
| q001 | 13 | 4 |
| q003 | 11 | 1 |
| q004 | 11 | 3 |
| q005 | 24 | 7 |
| q006 | 5 | 3 |
| q007 | 2 | 1 |
| the other 16 | — | 0 |

**Where those 19 sit with deeper search (2026-10-23):** rrf(cosine, BM25) with no 30 cut puts **12 of 19 within its top 60** and 7 beyond. That's why C0 exists (D11).

**Graph shape (2026-10-23, read-only count of the live Neo4j):**

| | Before merge | After merge (D19) |
|---|---|---|
| Docs | 228 (156 messages, 72 links) | same |
| CONTAINS edges | 79 | same |
| Entity nodes | 393; **333 link only 1 doc**; max degree 10 | 365; 299 link 1 doc; max 11 (`claude code`) |
| Topic nodes | 420; **338 link only 1 doc**; 4 hubs (47, 21, 19, 17) | 373; 284 link 1 doc; same 4 hubs, then 10 |

Most connectors connect a single doc, so they're dead ends. Only ~65 entities and ~90 topics can actually lead from one doc to another. The live graph is barely larger than the 188-doc corpus.

## 3. Rules for the executor

The same rules as `docs/retrieval-ladder-spec.md` section 3 apply, plus these:

1. **Tests first.** For each unit, the executor writes the test file exactly as its table in section 13 says (one test per row; ids in the test names, e.g. `test_U3_B2_entity_weight`) and hands it back **red**. The owner checks it row by row against the table and adds it to `tests/LOCK.sha256`. Only then is the unit implemented. Locked tests are never edited.
2. **No writes to Qdrant or Neo4j.** Live access only through `evals/live_guard.py` (`read_only_store`, `read_only_vector_store`, `qdrant_read_only_client`), wrapped in `fingerprint` → `assert_unchanged`. Never reference `QDRANT_API_KEY`.
3. Expansion, controls and scoring are **offline and deterministic**, computed from frozen files (U2). Only U2 and U7 touch live stores.
4. Closed corpus: neighbors outside the 188 `pilot-001` docs are **counted, never ranked**.
5. Never push. Touch only the files your card owns. The owner merges.
6. Run everything from `python-ingestion/` with `uv run`.

## 4. Systems

| run_id | Short | Pipeline | Pool |
|---|---|---|---|
| `ladder_s2_cos_bm25_rr` | S2 | Baseline: rrf(cosine, BM25) top 30 → re-rank. Original set: the existing file. Graph set: rebuilt with the same code (U5). | 30 |
| `ctrl_c0_deeper_rr` | C0 | rrf(cosine, BM25) top **N_q** → re-rank, where N_q = max(pool size of B-r, pool size of B-q) for query q | N_q (30–60) |
| `graph_b1r_rr` | B1-r | S2's 30 ∪ graph neighbors of S2's top `SEEDS_K` → re-rank on plain text | ≤ 60 |
| `graph_b2r_ctx_rr` | B2-r | Same pool as B1-r. Every pool doc with a path to a seed is re-ranked on `"[connected: <note>] " + text` | ≤ 60 |
| `graph_b1q_rr` | B1-q | S2's 30 ∪ docs attached to the `QUERY_NODES_K` nodes closest to the query → re-rank on plain text | ≤ 60 |
| `graph_b2q_ctx_rr` | B2-q | Same pool as B1-q. Every pool doc attached to a start node gets its note | ≤ 60 |
| `ctrl_c1_denorm_rr` | C1 | rrf(cosine over glued texts, BM25 over glued texts) top 30 → re-rank on glued text (D23) | 30 |
| `ctrl_c2_payload_rr` | C2 | S2's 30 ∪ payload entity-overlap neighbors of S2's top `SEEDS_K` → re-rank on plain text | ≤ 60 |

Constants (pre-registered, never tuned after a result):

| Constant | Value |
|---|---|
| `SEEDS_K` | 10 |
| `QUERY_NODES_K` | 5 |
| `NEIGHBOR_CAP` | 30 |
| `NAME_NORM` | lowercase, strip, `[\s\-_]+` → `" "` (D19) |
| `HUB_RULE` | skip connectors with merged degree > **16** (D20) |
| `RRF_K` | 60 |
| `DEPTH` | 30 |
| re-ranker | `rerank.MODEL_NAME` @ `rerank.MODEL_REVISION` |
| bootstrap | 10,000 resamples, seed 42 |
| win/tie threshold | ±0.01 |

## 5. Units

### P1 — owner: settle `topup.py`
Commit the blank-row filter in `evals/topup.py:cmd_merge` (it's currently uncommitted drift), add a test in `tests/test_ladder_topup.py`, and re-lock `tests/LOCK.sha256`. U6 depends on this.

### P2 — owner: finalize `queries_graph.csv` (repo root)
- **Final set (2026-10-23, revised at the owner's direction to centre on link content):** 4 `thin_message` (g001–g004), 4 `relational` (g005–g008), `origin=planner_draft`. Each question **describes a link by its content, not its name** (e.g. "the executable-specification tool", not "Quint"), and asks for the link **plus what connects to it** (who shared it, the discussion, related links). Those connected messages mostly don't share the link's words, which is where the graph should help. The earlier time-only question was dropped.
- **8 queries**, same columns as `queries.csv` (`query_id,query_text,intent,origin,split,notes`), ids `g001…g008`.
- **Types:** at least 3 with `intent=relational` (relevant through a connection) and at least 3 with `intent=thin_message` (the answer is a thin message, or is reached through one).
- **`notes` must name the expected answer doc(s)**, written before anything runs.
- **Allowed:** browsing `evals/data/pilot-001/documents.jsonl`, so every expected answer is inside the 188 docs. **Not allowed:** looking at any search, re-rank or graph output for these queries before the file is committed.
- Commit before U2 runs.

### U1 — `evals/headroom.py` (offline)
- `headroom(s2_rows, qrels, depth=30) -> {qid: {"relevant": n, "outside": n, "outside_ids": [...]}}`. Relevant means grade ≥ 2.
- CLI prints the table and the total. On the original set it must reproduce **19 / 104**.
- For the graph set it runs after U6 (when grades exist).

### U2 — `evals/freeze_neighbors.py` (live, read-only)
Freezes everything the offline units need:
1. `runs/graph_neighbors_v1.jsonl`: one row per (doc, neighbor, path), for every corpus doc:
   `{"candidate_id", "neighbor_id", "path": "contains|entity|topic", "connector": <name|null>, "connector_degree": <int|null>, "in_corpus": bool}`.
   - CONTAINS rows are written in **both directions** (D21): `message → link` and `link → message`, both with `path: "contains"`.
   - Connectors are merged by `NAME_NORM` first; `connector` is the normalized name.
   - The degree counts **distinct documents (Message + Link) attached to the merged connector in the full live graph**.
   - Map nodes to ids with `freeze_graph.candidate_id_for({"type", "sent_at", "text", "url"})`.
2. `runs/graph_nodes_v1.jsonl`: one row per **merged** Entity/Topic node: `{"node_type": "entity|topic", "name": <normalized>, "raw_names": [...], "degree", "corpus_doc_ids": [...]}`. B-q's starting points come from this file; its node vectors embed the normalized `name`.
3. `runs/payload_entities_v1.json`: `{candidate_id: {"entities": [...], "topics": [...]}}` from a read-only Qdrant scroll (the manifest has no entity fields).
4. `runs/vector_cosine_graphq_v1.jsonl`: cosine run for `queries_graph.csv`, all 188 ranks, via `run_vector.rank_all` with `read_only_vector_store()`.
5. `runs/graph_neighbors_v1.meta.json`: `{fingerprint_before, fingerprint_after, unmapped_nodes, out_of_corpus_neighbors, degree_histogram, graph_counts, created_at, git_sha}`. Before writing anything, it checks `graph_counts` against D22 and raises on any difference. Ends with `assert_unchanged`.

### U3 — `evals/graph_expand.py` (offline, pure)
**Result-start (B-r):** `expand_from_results(seeds, adjacency, exclude, k=SEEDS_K, cap=NEIGHBOR_CAP, hub_rule=HUB_RULE) -> list[{"candidate_id", "score", "via"}]`
- Seed weight `w(seed) = 1 / rank` within S2's top `k`.
- Path weight: `contains` = 1.0. `entity` / `topic` = `1 / ln(1 + degree)`. Connectors caught by `hub_rule` are skipped.
- Score = Σ over every (seed, path) reaching the doc of `w(seed) × w(path)`.
- Drop docs in `exclude` (S2's 30) or out of corpus. Keep the top `cap` by `(-score, candidate_id)`.
- `via` = the single highest-contributing path `{"path", "connector", "seed"}`. Ties go contains < entity < topic, then connector name.

**Query-start (B-q):** `expand_from_query(query_vec, node_vecs, nodes, exclude, k=QUERY_NODES_K, cap=NEIGHBOR_CAP, hub_rule=HUB_RULE)`
- Node vectors: embed every node `name` with `embedder.embed_batch`, in memory and cached to `runs/graph_node_vectors_v1.npy` with a sha256 recorded in the meta.
- Start nodes = the `k` nodes with the highest cosine to the query, ties by `(node_type, name)`, with hub-rule nodes removed **before** taking the top `k`.
- Doc score = Σ over the start nodes it's attached to of `cos(query, node) × 1/ln(1 + degree)`.
- Same exclude, cap and tie rules. `via = {"path": "query_entity|query_topic", "connector": name, "seed": null}`.

**Notes (both B2 variants):** `note_for(doc, starts, adjacency, docs_by_id) -> str | None`. It returns a note for **any** pool doc with a path to a starting point, whatever its source, and `None` otherwise.
- contains → `shared in the same message as "<seed label>"`
- entity → `both mention "<connector>", like "<seed label>"`
- topic → `both about "<connector>", like "<seed label>"`
- query_entity → `mentions "<connector>"`
- query_topic → `about "<connector>"`

The seed label is the link's `title`, or the first 80 chars of a message's `embedded_text`. When several paths apply, use the same highest-contribution rule as `via`.

### U4 — `evals/denorm_control.py` (offline)
- **C1:** `denorm_texts(docs, neighbors) -> {cid: text}`.
  - A message's text = its `embedded_text` + `" "` + the `embedded_text` of each in-corpus link it CONTAINS, sorted by link id.
  - **Token budget (agreed 2026-10-23):** the embedder (`all-MiniLM-L6-v2`) reads at most `max_seq_length = 256` tokens and silently truncates the rest; 30 of the 188 docs already exceed it. The message text comes first, untouched. The appended link text is **trimmed at a word boundary** to fit the remaining budget, measured with the embedder's own tokenizer, special tokens included. If the message alone uses the whole budget, nothing is appended. Meta records, per run: `links_trimmed`, `links_dropped_no_room`.
  - Offline: everything runs with `HF_HUB_OFFLINE=1`.
  - Links are unchanged.
  - Embed with `embedder.embed_batch` **in memory** and rank all 188 by cosine against `embedder.embed_text(query)`. No Qdrant write.
  - Parity check: re-embedding *unchanged* texts must match the stored vectors at ≥ `common.MIN_PARITY_COSINE` (reuse `parity_check`). Otherwise escalate.
- **C2:** `payload_expand(seeds, payload_entities, exclude, ...)`. Same seed weights, caps, hub rule and ties as B-r, but paths come from shared `entities`/`topics` strings in the payload, **normalized with the same `NAME_NORM`**, and degree = number of corpus docs carrying that normalized string. `via.path` ∈ `payload_entity|payload_topic`.

### U5 — `evals/run_graph_expand.py` (offline)
- For both query sets, builds the 8 systems in section 4.
- Reuses `fusion.rrf` (with `depth=N_q` for C0), `run_bm25.bm25_rows`, `run_ladder._rerank_candidates`, `rerank.Reranker`.
- Writes `runs/<run_id>.jsonl` (original set) and `runs/<run_id>.graphq.jsonl` (graph set). Rows are `{"run_id", "query_id", "candidate_id", "rank", "score", "via", "note"}`; `via`/`note` are `null` where they don't apply.
- Writes `.meta.json` in the `run_ladder.run_meta` format, plus every constant in section 4 and the node-vector sha256.
- Re-ranking never changes membership. Ranks are contiguous.

### U6 — `evals/topup_b.py` (offline; owner grades)
Reuses `topup.unjudged_pairs`, `topup.topup_rows` and `topup.merge_qrels`. It doesn't modify `topup.py`.
- `build`:
  - **Original set:** unjudged pairs in the top 5 of every new system → `annotation_topup_b.csv`.
  - **Graph set:** union of the top 30 of all 8 systems and cosine → `annotation_graphq.csv` (a full pool, because the set has no grades yet).
  - Both are shuffled with seed 42, with **no run, rank, score, `via` or `note` column**. Print the row counts.
  - **What the grader sees (D24):** the original-set sheet uses `pooling.display_content(doc)`, exactly as Run A. The graph-set sheet uses `display_content(doc)` plus, for a message, ` [shared link: "<title>"]` for each in-corpus link it CONTAINS, in candidate_id order. `content` is always the doc's **own** text, never C1's glued text or a B2 note.
- `merge`: `qrels_v2.csv` + both graded sheets → `qrels_v3.csv`.
- **H-grade (owner):** grade both sheets blind.

### U7 — `evals/latency_expand.py` (live, read-only)
Per query (both sets), `WARMUP = 1`, `REPEATS = 5`, nearest-rank p50/p95 as in `latency_bench.summarize`:
- the live Cypher expansion for B-r's seeds through `read_only_store()`;
- B-q's start-node lookup (query vs node vectors) + the live Cypher fetch of attached docs;
- re-rank cost at each system's pool size (30 vs N_q vs ≤ 60), on CPU;
- C1's extra query-time cost, recorded as 0 ms (the denormalization is index-time).

Writes `runs/latency_expand_v1.json` with fingerprints and ends with `assert_unchanged`. **Recorded and reported only; no decision uses it (D17).**

### U8 — `evals/score_graph_expand.py` (offline)
- Calls `topup.assert_fully_judged` on every system's top 5 against `qrels_v3.csv` **first**.
- Averages over `ladder_stats.answerable` only. **The two query sets are always reported separately, never pooled.**
- Per system and set: `ndcg@5`, `recall_pool`, `judged@5`, latency p50/p95.
- Comparisons, in this order:

  | # | Comparison | Role |
  |---|---|---|
  | 1 | S2 → C0 | decides step 1 |
  | 2 | S2 → C1 | descriptive |
  | 3 | S2 → C2 | descriptive |
  | 4 | C0 → B2-r | decides step 2 |
  | 5 | C1 → B2-r | decides step 3 |
  | 6 | C2 → B2-r | decides step 3 |
  | 7 | C0 → B2-q | decides step 2 |
  | 8 | C1 → B2-q | decides step 3 |
  | 9 | C2 → B2-q | decides step 3 |
  | 10 | B1-r → B2-r | side check (do notes help?) |
  | 11 | B1-q → B2-q | side check |

- "Beats" (section 6) on the original set = `ladder_stats.verdict == "improved"` **and** mean Δ ≥ +0.02. On the graph set = the scoreboard rule (D16).
- **Split by path:** for each B system, the share of top-5 grade ≥ 2 hits by `via.path`.
- Applies section 6 per set and per starting point, and prints the branch taken.
- Writes `graph_expand_metrics.json` and `graph_expand_report.md`. With `--append-report`, it appends `## Appendix D — Graph as candidate generator` to `metrics/manthan_report.md`, keeping the existing bytes unchanged, and raises if Appendix D is already there.

## 6. Decision flow (pre-registered)

Applied **separately** to each query set, and to each starting point (r, q):

```
Step 1  Does C0 beat S2?
        yes → "deeper pool" is a free win for the default search (recorded whatever happens next)

Step 2  Does B2-x beat C0?
        no  → GRAPH OUT for start x (the graph adds nothing over reading more)

Step 3  Does B2-x beat C1 AND C2?
        no  → SHIP THE SHORTCUT(S) that beat S2; GRAPH WAITS
        yes → GRAPH STAYS as an expansion stage, using start x

Side    B1-x → B2-x: do the notes help? Reported, never decides.
```

How to read the outcome:
- Original set → affects the **default** retriever. Graph set → affects whether a **relational-query route** exists.
- If both starting points reach GRAPH STAYS, report both. Choosing between them is a follow-up, not part of this run.
- **Lifts don't add up:** combinations (e.g. C1 + B2) are untested and are not shipped on this evidence.
- Latency is reported next to every branch but doesn't change it (D17).

## 7. Execution order

```
P1, P2 (owner)
  └─► U1 (offline; checks 19/104)
  └─► U2 (live freeze) ──► U3 ─┬─► U5 ──► U6 build ──► H-grade ──► U6 merge ──► U1 (graph set) ──► U8
                         U4 ───┘                                                        U7 (live) ──┘
```

U3, U4 and U1 can run in parallel worktrees. U2 and U7 run on the owner's machine with Neo4j and Qdrant up. They check that no Drive listener is running (`assert_listener_stopped`) and refuse to start if one is; the owner does nothing extra.

## 8. Gates

| Gate | Check |
|---|---|
| G0 | P1 committed; `uv run pytest -q` green, including `test_ladder_lock.py` |
| G1 | `headroom.py` reproduces 19/104 on the original set |
| G2 | U2 meta: fingerprints equal, `unmapped_nodes == 0` for corpus docs |
| G3 | U3/U4 locked tests green (designed with the owner, section 13) |
| G4 | U5 runs: contiguous ranks; C0 pool size == N_q per query; notes present exactly on connected docs |
| G5 | `assert_fully_judged` passes on `qrels_v3.csv` |
| G6 | U7 fingerprints equal |
| G7 | Appendix D appended: 4 metrics × 8 systems × 2 sets, path split, the branch taken per set and start |

## 9. Acceptance

G0–G7 green. `uv run pytest -q` green. `git diff graph-expand-base -- python-ingestion/tests` shows only the new Run B test files and the LOCK update. Appendix D appended, with Appendix C's bytes unchanged.

## 10. Escalate: stop, change nothing, report to the owner

- Any live fingerprint mismatch, a write that isn't rejected, or the listener is running.
- `unmapped_nodes > 0` for corpus docs (Neo4j text ≠ Qdrant text, e.g. `synthesized_text`). Do not add fuzzy matching.
- The C1 parity check fails.
- `annotation_topup_b.csv` > 300 rows, or `annotation_graphq.csv` > 800 rows.
- `headroom.py` doesn't give 19/104 on the original set.
- Any temptation to change a constant in section 4 after seeing a result.

## 11. Predictions

Registered 2026-10-23, before any C0/B/C system existed. The owner answered first; the planner registered afterwards without revising. The planner registers the **original set only** (D18). The executor copies the block verbatim to `evals/graph_expand_predictions.json`, and the owner adds it to `tests/LOCK.sha256` **before U5 runs**.

```json
{
  "_notes": "Pre-registered 2026-10-23 before any Run B system existed. Facts known at registration: headroom 19/104; 12 of 19 misses within rrf top 60; graph shape (section 2.3). Owner answered first; planner registered afterwards without revising. Planner abstains on the graph set (drafted its questions, D18). Predictions, not results.",
  "registered_at": "2026-10-23",
  "owner": {
    "original_set": {
      "S2->C0": {"expected_verdict": "improved", "reason": "12 of 19 missed relevant docs sit in the deeper RRF list; the re-ranker will promote some of them."},
      "expected_branch": "GRAPH OUT"
    },
    "graph_set": {
      "B1-q->B2-q and B1-r->B2-r": {"expected_verdict": "improved", "reason": "Notes let the re-ranker read why a doc connects."},
      "expected_branch": "GRAPH OUT"
    }
  },
  "planner": {
    "original_set": {
      "S2->C0": {"expected_verdict": "no detectable difference", "reason": "Misses cluster in q001/q005, which already have 13 and 24 relevant docs; their top 5 is likely already relevant, so extra relevant candidates barely move nDCG@5."},
      "S2->C1": {"expected_verdict": "no detectable difference", "reason": "Few thin-message needs in the original set."},
      "S2->C2": {"expected_verdict": "no detectable difference", "reason": "Few connection-shaped needs; same re-ranker blind spot."},
      "C0->B2-r": {"expected_verdict": "no detectable difference", "reason": "A deeper pool covers most of what result-start expansion adds."},
      "C1->B2-r": {"expected_verdict": "no detectable difference", "reason": "Both near S2."},
      "C2->B2-r": {"expected_verdict": "no detectable difference", "reason": "Both near S2."},
      "C0->B2-q": {"expected_verdict": "no detectable difference", "reason": "Query-start nodes on concept queries are broad; the re-ranker filters the noise, but nothing new reaches the top 5."},
      "C1->B2-q": {"expected_verdict": "no detectable difference", "reason": "Both near S2."},
      "C2->B2-q": {"expected_verdict": "no detectable difference", "reason": "Both near S2."},
      "B1-r->B2-r": {"expected_verdict": "no detectable difference", "reason": "Few connected docs reach the top 5 on this set."},
      "B1-q->B2-q": {"expected_verdict": "no detectable difference", "reason": "Same."},
      "expected_branch": "GRAPH OUT (fails step 2); step 1 not improved"
    },
    "graph_set": "abstain (D18)"
  },
  "disagreement_to_watch": "S2->C0 on the original set: owner reads 'relevant docs get found'; planner reads 'nDCG@5 only sees the top 5, which may already be full'."
}
```

## 12. Open items

| # | Item | Status |
|---|---|---|
| O1 | `HUB_RULE`: the "too popular" limit | **Closed 2026-10-23:** merge duplicates (D19), skip merged degree > 16 (D20) |
| O2 | Predictions | **Closed 2026-10-23** (section 11) |
| O3 | Test design | **Closed 2026-10-23** (sections 13.1–13.8) |
| O4 | P1: commit the `topup.py` blank-row patch with a test, re-lock | **Owner, before handoff** |
| O5 | P2: owner edits `queries_graph.csv` (incl. the g008 keep/replace choice), then commits | **Owner, before handoff** |

## 13. Tests

Designed with the owner, unit by unit. For each: behaviour, boundaries, failure cases, and anti-cheat checks. Locked in `tests/LOCK.sha256` before implementation.

### 13.1 U3 `graph_expand` — `tests/test_graph_expand.py` (agreed 2026-10-23)

**Fixture: a toy graph built in the test file, with no frozen data.** Seeds S1 (rank 1), S2 (rank 2):
`S1 -contains-> A` · `S1 -entity "rust" (deg 3)-> B` · `S2 -entity "rust" (deg 3)-> B` · `S2 -topic "agents" (deg 17)-> C` · `S1 -entity "llm" (deg 1)` (dead end). Further rows are added per test as needed.

**Rules agreed for the implementation:**
- R1: duplicate seeds raise `ValueError`.
- R2: scores are rounded to **9 decimals** before sorting and are returned rounded.
- R3: dead-end connectors (degree 1) are skipped and **counted**.
- R4: B-q with fewer than `QUERY_NODES_K` usable nodes uses what's left and **records the count**.
- Both counts go into the U5 run meta, per query.
- R7 (approved 2026-10-26, review #3): a dead end is an adjacency edge `{"neighbor": None, "path": "entity|topic", "connector": <name>, "degree": 1}`. U5 adds one per seed for each degree-1 node in `graph_nodes_v1.jsonl` whose `corpus_doc_ids` contains the seed. U3 counts these edges as dead ends and returns nothing for them.

| # | Kind | Test | Expected |
|---|---|---|---|
| B1 | behaviour | contains hop from seed rank 1 | A score = 1.0 |
| B2 | behaviour | entity of degree 3 from seed rank 1 | weight = 1/ln(4) = 0.721347520 |
| B3 | behaviour | B reached from S1 and S2 | score = 0.721347520 + 0.721347520/2 = 1.082021280 |
| B4 | behaviour | same path from seed rank 2 vs rank 1 | exactly half |
| B5 | behaviour | output order | sorted by (−score, candidate_id) |
| B6 | behaviour | `via` | strongest single path; ties contains < entity < topic, then connector name |
| B7 | behaviour | B-q start nodes | top `QUERY_NODES_K` by cosine, ties by (node_type, name); hubs removed **before** top-k |
| B8 | behaviour | `note_for` | returns a note for a connected doc already in S2's pool; `None` for an unconnected doc |
| B9 | behaviour | note wording | exact strings from U3; label = link `title`, or the first 80 chars of a message's `embedded_text` |
| B10 | behaviour | `NAME_NORM` | `"AI-Development"`, `"ai development"`, `"AI_development"` → same; `"LLM"` ≠ `"LLMs"` |
| E1 | boundary | connector degree 16 | kept |
| E2 | boundary | connector degree 17 | skipped |
| E3 | boundary | degree-1 connector | yields nothing; dead-end count = 1 |
| E4 | boundary | a seed reachable from another seed | never returned as a neighbor |
| E5 | boundary | neighbor already in `exclude` | not in the expansion output; still gets a note (B8) |
| E6 | boundary | neighbor out of corpus | dropped |
| E7 | boundary | 31 candidates, cap 30, tie at the cut | 30 returned; the tie is broken by candidate_id |
| E8 | boundary | fewer than `SEEDS_K` seeds | uses all of them |
| E9 | boundary | 11 seeds given | the 11th is never used |
| E10 | boundary | no seeds / no neighbors | `[]`, no error |
| E11 | boundary | B-q with 3 usable nodes | uses 3; recorded count = 3 |
| F1 | failure | unknown path type in adjacency | `ValueError` naming the row |
| F2 | failure | entity/topic degree 0 | `ValueError` |
| F3 | failure | query vector dim ≠ node vector dim | `ValueError` |
| F4 | failure | node-vector sha256 ≠ recorded sha256 | `ValueError` |
| F5 | failure | duplicate seed ids | `ValueError` |
| A1 | anti-cheat | static check of `evals/graph_expand.py` | no import of `neo4j`, `qdrant_client`, `evals.live_guard` |
| A2 | anti-cheat | shuffle adjacency rows and node order | identical output |
| A3 | anti-cheat | module constants | `SEEDS_K == 10`, `QUERY_NODES_K == 5`, `NEIGHBOR_CAP == 30`, `HUB_MAX_DEGREE == 16` |
| A4 | anti-cheat | numeric checks | B2/B3 compared to hand-computed values at 1e-9, not inequalities |
| A5 | anti-cheat | static check | no `random`, `time`, `datetime` or `os.environ` use in the module |

### 13.2 U4 `denorm_control` — `tests/test_denorm_control.py` (agreed 2026-10-23)

**Rules agreed:**
- R5: C1 glues message first, then link text trimmed at a word boundary to the 256-token budget.
- R6: a corpus doc missing from `payload_entities_v1.json` raises.
- R8 (approved 2026-10-26, review #3): C1 counters are per link. A link that keeps some but not all of its words counts toward `links_trimmed`. A link that keeps none counts toward `links_dropped_no_room`. Each separator costs 1 token.

Tokenizer-dependent tests use the real embedder tokenizer under `--model`; the rest use an injected fake tokenizer (1 word = 1 token).

| # | Kind | Test | Expected |
|---|---|---|---|
| C1-B1 | behaviour | message with 1 in-corpus link | `msg + " " + link_text` |
| C1-B2 | behaviour | message with 2 links | links appended in candidate_id order |
| C1-B3 | behaviour | message with no link; any link doc | unchanged |
| C1-E1 | boundary | contained link outside the corpus | ignored |
| C1-E2 | boundary | same link contained twice | appended once |
| C1-E3 | boundary | fake budget 10, message 4 words, link 20 words | link trimmed to 5 tokens (10 − 4 message − 1 separator), cut on a word boundary; `links_trimmed == 1` |
| C1-E4 | boundary | message alone ≥ budget | message unchanged; `links_dropped_no_room == 1` |
| C1-E5 | boundary (`--model`) | real tokenizer: every output text | ≤ 256 tokens incl. special tokens |
| C1-F1 | failure | parity: re-embedded unchanged text vs stored vector < `MIN_PARITY_COSINE` | raises |
| C1-A1 | anti-cheat | static + runtime | no `QdrantClient`, no `upsert`/`create_collection`/`delete` call in the module |
| C1-A2 | anti-cheat | run under `HF_HUB_OFFLINE=1` | succeeds with no network |
| C2-B1 | behaviour | same toy graph as 13.1, expressed as payload strings | same scores, order and caps as U3 B1–B5 (paths renamed `payload_entity`/`payload_topic`) |
| C2-B2 | behaviour | names `"AI-Development"` on one doc, `"ai development"` on another | they connect (shared `NAME_NORM`) |
| C2-B3 | behaviour | payload string carried by 17 corpus docs | skipped (hub rule) |
| C2-E1 | boundary | doc with empty `entities` and `topics` | no paths, no error |
| C2-F1 | failure | corpus doc absent from the payload file | `ValueError` naming the doc |
| C2-A1 | anti-cheat | constants | imported from `graph_expand`, not redefined |

### 13.3 U2 `freeze_neighbors` — `tests/test_freeze_neighbors.py` (agreed 2026-10-23)

Live tests are marked `live`, run only with `--live`, and **fail, never skip**. Artifact tests are marked `artifacts`.

| # | Kind | Test | Expected |
|---|---|---|---|
| U2-B1 | artifacts | neighbors file ids | every `candidate_id` ∈ corpus; every `neighbor_id` ∈ corpus or `in_corpus == false` |
| U2-B2 | artifacts | symmetry | for entity/topic/contains: row (A, B, path, connector) exists ⇔ row (B, A, path, connector) exists |
| U2-B3 | artifacts | degree consistency | `connector_degree` in the neighbors file == `degree` in the nodes file for the same (type, name) |
| U2-B4 | artifacts | names merged | `NAME_NORM(name) == name` for every connector and node |
| U2-B5 | artifacts | payload completeness | the payload file has exactly the 188 corpus ids |
| U2-B6 | artifacts | graph-set cosine run | 8 queries × 188 rows, ranks 1..188 contiguous per query |
| U2-B7 | artifacts | pinned counts | meta `graph_counts` == D22 exactly |
| U2-S1 | live | fingerprints | `fingerprint_before == fingerprint_after` |
| U2-S2 | live/static | read-only access | the module uses only `live_guard` accessors; the string `QDRANT_API_KEY` is absent from the module |
| U2-S3 | live | listener | `assert_listener_stopped` is called before any read; with a fake listener process it raises |
| U2-F1 | live | unmapped corpus doc | a corpus doc not found in Neo4j → raises (no fuzzy matching) |
| U2-F2 | unit | count drift | given counts ≠ D22 → raises before any file is written |
| U2-A1 | live | determinism | two consecutive freezes → byte-identical output files (meta compared without `created_at`) |

### 13.4 U1 `headroom` — `tests/test_headroom.py` (agreed 2026-10-23)

| # | Kind | Test | Expected |
|---|---|---|---|
| U1-B1 | artifacts | real S2 + qrels_v2 | total 19 / 104; per-query table exactly as section 2.3 |
| U1-B2 | unit | toy qrels with grades 1, 2, 3 | only grades ≥ 2 count as relevant |
| U1-E1 | unit | query with no grade ≥ 2 | `relevant 0, outside 0`, flagged not answerable |
| U1-E2 | unit | `depth=60` vs default 30 on a toy run | the outside count changes accordingly |

### 13.5 U5 `run_graph_expand` — `tests/test_run_graph_expand.py` (agreed 2026-10-23)

Unit tests use an injected fake re-ranker (score = −len(text), for example); artifact tests check the real output files.

| # | Kind | Test | Expected |
|---|---|---|---|
| U5-B1 | artifacts | rebuild S2 for the original set with U5's code path | **byte-identical** rows to the existing `ladder_s2_cos_bm25_rr.jsonl` (run_id, query_id, candidate_id, rank, score) |
| U5-B2 | unit + artifacts | C0 pool size | == max(\|B-r pool\|, \|B-q pool\|) for every query |
| U5-B3 | unit + artifacts | B1 vs B2 membership | identical candidate sets per query, for both r and q |
| U5-B4 | unit + artifacts | notes | `note != null` ⇔ `graph_expand.note_for(...)` returns a note; never present in B1/S2/C0/C1/C2 rows |
| U5-B5 | unit + artifacts | via | `via != null` ⇔ the doc came from graph/payload expansion |
| U5-B6 | unit + artifacts | ranks and membership | the re-ranked set == the pool set; ranks 1..N contiguous |
| U5-B7 | artifacts | meta | contains every section 4 constant, the node-vector sha256, and per-query `dead_ends`, `bq_nodes_used`, `links_trimmed`, `links_dropped_no_room` |
| U5-B8 | unit | C1 texts | BM25 and the re-ranker receive the full glued text; the embedder receives the trimmed text (D23) |
| U5-F1 | unit | input sha mismatch (any frozen file vs its meta) | raises before building |
| U5-F2 | unit | re-ranker model/revision ≠ pinned | raises |
| U5-A1 | static | imports | no `neo4j`, `qdrant_client`, `evals.live_guard` |
| U5-A2 | artifacts | determinism | building twice → identical run files |

### 13.6 U6 `topup_b` — `tests/test_topup_b.py` (agreed 2026-10-23)

| # | Kind | Test | Expected |
|---|---|---|---|
| U6-B1 | unit | sheet columns | exactly `common.ANNOTATION_COLUMNS`; no column or cell contains a run_id, rank, score, `via`, `note`, or a system short name |
| U6-B2 | unit | content source | `content` == `display_content(doc)` (original sheet) or D24 enrichment (graph sheet); never C1 glued text, never a note |
| U6-B3 | unit | original sheet | only (query, doc) pairs in the top 5 of C0, B1-r, B2-r, B1-q, B2-q, C1, C2 that are absent from `qrels_v2` |
| U6-B4 | unit | graph sheet | union of the top 30 of all 8 systems + `vector_cosine_graphq_v1`, deduplicated |
| U6-B5 | unit | shuffle | `random.Random(42)`; the same input gives the same order |
| U6-B6 | unit | D24 enrichment | a message containing 2 in-corpus links shows both titles in candidate_id order; links and link-less messages unchanged |
| U6-B7 | artifacts | merge | `qrels_v2.csv` byte-identical before/after; `qrels_v3` = v2 ∪ both graded sheets |
| U6-F1 | unit | blank grade in either sheet | merge raises, naming the row |
| U6-F2 | unit | pair already graded appears again | merge raises (`... already ...`) |
| U6-F3 | unit | sheet size | original > 300 or graph > 800 rows → raises with the count (escalate) |
| U6-A1 | static | `topup.py` | unchanged by this unit; functions imported, not copied |

### 13.7 U7 `latency_expand` — `tests/test_latency_expand.py` (agreed 2026-10-23)

| # | Kind | Test | Expected |
|---|---|---|---|
| U7-B1 | unit | statistics | uses `latency_bench.time_call` / `summarize`; the warm-up sample is discarded |
| U7-B2 | artifacts | components | B-r walk, B-q node lookup + fetch, re-rank at 30 / N_q / B pool size; C1 extra recorded as 0 ms |
| U7-S1 | live | safety | listener stopped; fingerprints before == after |
| U7-A1 | static | separation | `score_graph_expand`'s decision function takes no latency input (D17) |

### 13.8 U8 `score_graph_expand` — `tests/test_score_graph_expand.py` (agreed 2026-10-23)

| # | Kind | Test | Expected |
|---|---|---|---|
| U8-B1 | unit | fully judged | `topup.UnjudgedError` raised before any metric when any system's top 5 has an unjudged pair |
| U8-B2 | unit | set separation | original and graph sets are never averaged together; each result is keyed by set |
| U8-B3 | unit | comparisons | exactly the 11 of U8, in order, per set |
| U8-B4 | unit | original-set "beats" | true iff `verdict == "improved"` and mean Δ ≥ 0.02. Boundary: Δ = 0.0200 → true; Δ = 0.0199 → false |
| U8-B5 | unit | scoreboard (D16) | parametrized over n = 8, 7, 6, 5, 4 at the threshold and one below; any loss → false; n ≤ 4 → "too few to decide" |
| U8-B6 | unit | decision branches | synthetic comparison results drive each branch: GRAPH OUT (fails step 2), SHIP SHORTCUT (beats C0, not C1 or not C2), GRAPH STAYS (beats C0, C1 and C2); step 1 recorded independently of the others |
| U8-B7 | unit | path split | shares across `via.path` values sum to 1.0 of graph-sourced grade ≥ 2 top-5 hits (or report 0 hits) |
| U8-B8 | unit | metrics | `recall_pool` = grade ≥ 2 docs anywhere in the ranked list / all grade ≥ 2 docs; `judged@5` = judged top-5 rows / 5 |
| U8-B9 | unit | appendix | starts with `## Appendix D — Graph as candidate generator`; existing report bytes unchanged; raises if Appendix D already present |
| U8-B10 | unit | disclosure | the appendix contains the D18 disclosure sentence and the D24 grader-view note |
| U8-A1 | static | latency isolation | the decision function's signature has no latency parameter (D17) |
| U8-A2 | unit | predictions | each comparison reports `prediction` and `prediction_held` from the locked predictions file |
