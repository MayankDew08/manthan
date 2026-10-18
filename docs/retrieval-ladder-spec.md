# Retrieval ladder: spec

Status: **ready for execution** · Planned 2026-10-18 · Branch `retrieval-ladder` (from `review-rag-eval`)
Corpus `pilot-001` (188 documents, frozen) · Queries `queries.csv` (25) · Grades `annotation.csv` (1,079, complete)

This spec is the contract. The tests in `python-ingestion/tests/test_ladder_*.py` are the executable
form of it and are **locked**. Where this text and a locked test disagree, the test wins and the
executor escalates (section 17); it never edits the test.

---

## 1. Questions this experiment answers

1. How good is current retrieval (cosine over Qdrant)?
2. What does each component add, **given the ones before it**: BM25, then a cross-encoder
   re-ranker, then the Neo4j graph?

Every result below is computed later by the executor. Every expectation in this document and in
`evals/ladder_predictions.json` is a **prediction**, not a result.

## 2. Decisions (owner interview, 2026-10-18)

| # | Decision |
|---|---|
| Q1 | **Freeze first.** Quality metrics are computed only from frozen files under `evals/data/pilot-001/`. |
| Q2 | **Closed corpus.** Only the 188 `pilot-001` documents compete. Graph results outside it are counted, never ranked. |
| Q4 | **Five-step ladder**, each step adds one component (table in section 4). |
| Q3/Q4 | Graph results are ordered by number of distinct query terms matched, ties by `candidate_id`, then fused by RRF. |
| Q5 | **Top-up grading.** Every ungraded (query, document) pair in any step's top 5 is graded blind by the owner. Scoring **fails hard** until none remain. |
| Q6 | Depth 30 per source, RRF k = 60, top 30 to the re-ranker. Candidate Success@30 is a lower bound where ranks 6–30 hold unjudged documents. |
| Q7 | `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision `c5ee24cb16019beea0893ab7796b1df96625c6b8`, on CPU, deterministic, reading untruncated `embedded_text`. |
| Q8 | Averages over **answerable** queries only (any grade ≥ 2; expected 22: q001–q021 + q025). No-answer queries are reported as an abstention observation. |
| Q9 | nDCG@5 is primary. Paired bootstrap (10,000 resamples, seed 42). The verdict is assigned by a function. Predictions are registered before any run. |
| Q10/Q11 | Latency is measured **live**. Read-only is **enforced**: a Qdrant read-only API key, Neo4j READ access mode, a fingerprint taken before and after, and the Drive listener checked as stopped. |
| Q12 | Production `search._terms` and `search._graph_candidates` are imported, never retyped. |
| Q13 | Planner-written failing tests, locked by `tests/LOCK.sha256` plus anti-tamper tests. |
| Q14 | Two waves: parallel offline units with a parallel live track, then a sequential integration (section 15). |
| Q15 | `qrels.csv` is never overwritten. The top-up produces `qrels_v2.csv`, and **all** steps are scored on it. The report gets "Appendix C", appended. |

### 2.1 Changes made while writing the spec (owner: please note)

- **Run configuration goes in a sidecar `runs/<run_id>.meta.json`**, not a header line inside the
  `.jsonl`. `common.read_jsonl` and every existing loader expect each line to be a row; a header
  line would break them. The content is the same as agreed in Q15.
- **Comparison s2 → s4 was added** to the pre-registered list. It is the graph's effect once a
  re-ranker is present, which is exactly "B vs the same stack without the graph".
- **New unit U0.** Qdrant gets an API key in H0, and today no client sends one: `VectorStore`,
  `app.py`, `main.py`, `pipeline.py`, `search.py` and 5 scripts would all fail with 401. U0 makes
  every client read `QDRANT_API_KEY` and must be merged **before** H0 restarts Qdrant.

### 2.2 Threat model for the live stores

The protection targets an **accidental** write by a careless executor (a fixture calling
`ensure_collection`, a "clean-up" step, a test that upserts). It does not defend against a
malicious executor: Neo4j Community has one all-powerful user, so its protection is READ access
mode enforced by the server, plus the fingerprint. Qdrant's protection is a real server-side
read-only key. The full Qdrant key (`QDRANT_API_KEY`) is never referenced by harness code (a test
enforces this).

## 3. Rules for every executor

1. **Implement only; never edit a locked file.** The locked files are listed in
   `python-ingestion/tests/LOCK.sha256`; `tests/test_ladder_lock.py` fails on any change.
   Implementation code may not contain `pytest`, `xfail`, `importorskip`, `sys.modules[` or
   `unittest.mock` (even in comments or docstrings). No new `conftest.py`, `pytest.ini`,
   `setup.cfg`, `sitecustomize.py`, or pytest `addopts`.
2. **Touch only the files your task card owns.** Check before handing back:
   `git diff --name-only ladder-base...HEAD` must list only the files the card owns.
3. **Red → green.** First run your card's test command and confirm the recorded red failure.
   Implement. You are done when the card's command is green **and** these are green:
   `uv run pytest tests/test_ladder_lock.py -q` and
   `uv run pytest -q --ignore-glob='tests/test_ladder_*.py'` (the 170 pre-existing tests).
   Other units' ladder test files stay red until those units merge; that is expected. After the
   **last** merge, plain `uv run pytest -q` (everything, no ignores) must be green.
4. **Live, model and artifact tests are deselected by default.** They run only with `--live`,
   `--model` or `--artifacts`, and when selected they **fail, never skip**. Never "fix" a live
   failure by mocking, retrying or catching; stop and escalate (section 17).
5. **Determinism.** No wall-clock time, randomness or environment value may influence a ranking or
   score. Randomness is only `random.Random(seed)` with the seed this spec names.
6. **No network** except the local Qdrant and Neo4j in live units. Model weights come from the
   local Hugging Face cache (`HF_HUB_OFFLINE=1`).
7. **Never push, never commit outside your branch, never touch `evals/data/` except the outputs
   your card names.** The owner merges.
8. Run every command from `python-ingestion/` with `uv run`.

## 4. The ladder

| Step | `run_id` | Pipeline | `score_scale` |
|---|---|---|---|
| 0 | `ladder_s0_cosine` | frozen `vector_cosine_v1`, unchanged (all 188 ranks) | `cosine` |
| 1 | `ladder_s1_cos_bm25` | `rrf([cosine, bm25])`, top 30 | `rrf` |
| 2 | `ladder_s2_cos_bm25_rr` | step 1's 30 candidates → cross-encoder | `cross_encoder` |
| 3 | `ladder_s3_cos_bm25_graph` | `rrf([cosine, bm25, graph])`, top 30 | `rrf` |
| 4 | `ladder_s4_cos_bm25_graph_rr` | step 3's 30 candidates → cross-encoder | `cross_encoder` |

Pre-registered comparisons, in this order:
`s0→s1, s1→s2, s1→s3, s3→s4, s2→s4, s0→s4`.

**Deferred** (listed so they are not forgotten): BM25 on its own; the re-ranker on cosine alone;
the production-faithful graph merge "B0" (0.5 × best score); graph v1 (whole-word + IDF matching);
a stronger re-ranker (`BAAI/bge-reranker-v2-m3`, already cached); growing to 50 queries.

## 5. Data formats (shared by every unit; do not invent others)

All paths are relative to `python-ingestion/evals/data/pilot-001/`.

**Run file** `runs/<run_id>.jsonl`: one JSON object per line:
`{"run_id", "query_id", "candidate_id", "rank", "score"}`, plus `"matched_terms"` in the graph
run. Ranks start at 1 and are contiguous per query. `candidate_id` is `message:<uuid>` or
`link:<uuid>` and must exist in `documents.jsonl`. The existing `vector_cosine_v1.jsonl` also has
`latency_ms`; it is read as is.

**Run meta** `runs/<run_id>.meta.json`:
`{"run_id", "score_scale", "rrf_k", "depth", "rerank_model", "rerank_revision", "inputs":
{name: {"path", "sha256"}}, "created_at", "git_sha", "git_dirty"}`. The rerank fields are `null`
for steps that do not re-rank. The graph meta (section 6) has its own fields.

**Qrels v2** `qrels_v2.csv`: columns `query_id,candidate_id,relevance,answer_bearing,reason`. It is
built from **`annotation.csv`** (the owner's complete 1,079-row sheet) plus `annotation_topup.csv`.
`qrels.csv` is the historic 5-query file and is never modified or deleted.

### 5.1 U0 — every Qdrant client sends `QDRANT_API_KEY`

`VectorStore(api_key=None)` uses `os.environ["QDRANT_API_KEY"]` when it is set and non-blank, and
otherwise `None`. An explicit `api_key` argument always wins. Every direct `QdrantClient(...)` call
in `evals/check_corpus.py`, `evals/export_corpus.py`, `evals/parity_check.py`,
`scripts/migrate_remove_sender.py` and `scripts/restore_qdrant_from_neo4j.py` passes
`api_key=os.environ.get("QDRANT_API_KEY") or None`. No other behaviour changes.

### 5.2 U1 — read-only live harness (`evals/live_guard.py`)

- `READ_ONLY_KEY_ENV = "QDRANT_READ_ONLY_API_KEY"`; `class LiveGuardError(RuntimeError)`.
- `qdrant_read_only_client(env=None) -> QdrantClient`: uses `env` or `os.environ`. It raises
  `LiveGuardError` if the read-only key is missing or blank, and never falls back to
  `QDRANT_API_KEY`. Host and port come from `QDRANT_HOST` / `QDRANT_PORT`.
- `read_only_vector_store(env=None) -> VectorStore`: same key, with collection from
  `QDRANT_COLLECTION`.
- `class ReadOnlyDriver(inner)`: `.inner`; `session(**kw)` always sets
  `default_access_mode=neo4j.READ_ACCESS`, overriding the caller and passing other kwargs through;
  `close()`.
- `class ReadOnlyStore(driver)`: `.driver` is a `ReadOnlyDriver`; `close()`. It duck-types
  `KnowledgeStore` for `search._graph_candidates`.
- `read_only_store(env=None) -> ReadOnlyStore`, from `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`.
- `neo4j_read(driver, cypher, **params) -> list[dict]` via `session.execute_read(...)` only.
- `fingerprint(qdrant_client, collection, neo4j_driver) -> dict` with:
  - `qdrant_points`: `client.count(collection_name=..., exact=True).count`;
  - `qdrant_ids_sha256`: sha256 of the sorted string ids joined by `\n`, scrolling the whole
    collection 256 at a time;
  - `neo4j_nodes`: from `MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n`;
  - `neo4j_rels`: from `MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n`.

  Both Neo4j dicts are sorted by key.
- `assert_unchanged(before, after)` raises `LiveGuardError` naming every differing key.
- `listener_pids(proc_cmdlines: dict[int, str]) -> list[int]` returns the pids whose command line
  contains `drive_event_listener`. `assert_listener_stopped(proc_cmdlines=None)` reads `/proc` when
  given `None`.
- The live modules (`live_guard`, `freeze_graph`, `latency_bench`) must not contain any
  write-capable call; the list is in `test_ladder_live_guard.py::FORBIDDEN_CALLS`.

## 6. U2 — freeze the graph (`evals/freeze_graph.py`)

- `from search import _terms, _graph_candidates` (the very same objects). `RUN_ID =
  "graph_lexical_v0"`.
- `source_sha256()`: sha256 of `inspect.getsource(_terms) + inspect.getsource(_graph_candidates)`.
- `candidate_id_for(result)`:
  - messages: `"message:" + uuid5(NAMESPACE_URL, f"msg-{sent_at}-{text}")`;
  - links: `"link:" + uuid5(NAMESPACE_URL, f"link-{url}")`.

  This is the `vector_store.py` formula; the text is **not** stripped.
- `matched_terms(result, terms)`: the distinct terms, in `terms` order, that occur as
  case-insensitive substrings of `text`, `title`, `entities` and `topics` (the same `CONTAINS`
  semantics as the Cypher).
- `graph_rows_for_query(query_id, terms, raw_results, corpus_ids) -> (rows, stats)`:
  - deduplicates by `candidate_id` (several Cypher branches return the same node);
  - keeps only in-corpus documents, ordered by `(-len(matched_terms), candidate_id)`, with
    `score = float(len(matched_terms))`;
  - applies **no depth cut**;
  - returns `stats = {query_id, terms, raw_results, distinct, in_corpus, out_of_corpus}`.
- `cosine_mismatches(live_rows, frozen_rows, depth=30, tol=1e-4) -> list[str]`. For each frozen
  query and each live row at rank r ≤ depth, the document's frozen score and the frozen score at
  rank r must both be within `tol` of the live score. This tolerates swaps inside score ties and
  catches real reorders. A query missing from the live rows is a mismatch.
- **CLI** `uv run python -m evals.freeze_graph` (live):
  1. assert the listener is stopped, then fingerprint;
  2. for each query, call
     `_graph_candidates(read_only_store, _terms(text), None, None, 1.0)`, timed;
  3. write `runs/graph_lexical_v0.jsonl` and `runs/graph_lexical_v0.meta.json` =
     `{run_id, source_sha256, per_query: {qid: stats}, latency_ms: {qid: ms}, fingerprint_before,
     fingerprint_after, created_at, git_sha, git_dirty}`;
  4. fingerprint again and `assert_unchanged`.

  `topics` and `min_quality` are `None`, the production defaults.

## 7. U3 — BM25 run (`evals/run_bm25.py`)

`RUN_ID = "bm25_v1"`. `bm25_rows(queries, docs)` uses the in-repo `evals.bm25.BM25` over
`embedded_text`. It ranks every document with score > 0, sorted by `(-score, candidate_id)`. CLI
`uv run python -m evals.run_bm25` writes `runs/bm25_v1.jsonl` and its meta file.

## 8. U4 — fusion (`evals/fusion.py`)

`RRF_K = 60`, `DEPTH = 30`.
- `ranking(rows, query_id) -> list[str]` returns ids in rank order.
- `rrf(rankings, k=RRF_K, depth=DEPTH) -> list[(candidate_id, score)]`:
  - each input list is cut to `depth`;
  - a repeat inside one list counts once, at its best rank;
  - score = Σ 1/(k + rank);
  - the output is sorted by `(-score, candidate_id)` and cut to `depth`.

## 9. U5 — re-ranker (`evals/rerank.py`)

- Constants: `MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"`, `MODEL_REVISION =
  "c5ee24cb16019beea0893ab7796b1df96625c6b8"`, `MAX_LENGTH = 512`, `SCORE_DECIMALS = 6`.
- `load_model()` returns `CrossEncoder(MODEL_NAME, revision=MODEL_REVISION, device="cpu",
  max_length=MAX_LENGTH, local_files_only=True)` with `.model.eval()`.
- `Reranker(score_pairs=None)`: `score_pairs(list[(query, text)]) -> sequence[float]`. It defaults
  to the loaded model's `predict`, loaded lazily.
- `rerank(query, [(candidate_id, text)])`:
  - calls `score_pairs` **once**, in input order;
  - rounds each score to 6 decimals as a plain `float`;
  - sorts by `(-score, candidate_id)`;
  - returns `[]` without calling the model when given no candidates.

## 10. U6 — top-up gate (`evals/topup.py`)

- `TOPUP_K = 5`, `QRELS_V2 = "qrels_v2.csv"`.
- `class UnjudgedError(RuntimeError)` with `.pairs`; its message includes the count.
- `unjudged_pairs(runs: {run_id: rows}, qrels, k=5)` returns the sorted, unique (query, candidate)
  pairs with rank ≤ k that have no qrels entry. `assert_fully_judged(runs, qrels, k=5)` raises
  `UnjudgedError` when any exist.
- `topup_rows(pairs, queries_by_id, docs_by_id, seed=42)`:
  - output columns are exactly `common.ANNOTATION_COLUMNS`, with blank `relevance`/`reason`;
  - `content = pooling.display_content(doc)`;
  - the order is the sorted unique pairs shuffled by `random.Random(seed)`;
  - no run, rank or score information appears anywhere.
- `merge_qrels(annotation_rows, topup_rows) -> qrels rows`:
  - every row must carry a grade in `0`–`3`, otherwise `ValueError` naming the row;
  - a top-up pair that is already graded raises `ValueError("... already ...")`.
- CLI:
  - `uv run python -m evals.topup build` checks all five ladder runs against `annotation.csv` and
    writes `annotation_topup.csv`, printing the count;
  - `uv run python -m evals.topup merge` writes `qrels_v2.csv`.

## 11. U7 — statistics (`evals/ladder_stats.py`) and predictions

- `TIE_EPS = 0.01`, `N_RESAMPLES = 10000`, `SEED = 42`.
- `answerable(qrels, query_ids)` returns the sorted query ids with any grade ≥ 2.
- `per_query(run_rows, qrels, query_ids, k=5, depth=30) -> {qid: {ndcg, success, mrr,
  candidate_success}}`. It uses `evals.metrics` (unjudged = grade 0; nDCG's ideal ordering is built
  from all judged grades) and orders rows by rank.
- `win_tie_loss(base, new)`: Δ = new − base. Δ ≥ 0.01 is a win, −Δ ≥ 0.01 a loss, anything else a
  tie. It returns `{wins, ties, losses, improved: [qids], regressed: [qids]}`. Different query
  sets raise `ValueError`.
- `paired_bootstrap(base, new, n_resamples, seed)`:
  1. take the deltas in sorted-qid order and seed `rng = random.Random(seed)`;
  2. for each of `n_resamples` resamples, draw `n` indices with `rng.randrange(n)`, in sequence,
     and take their mean;
  3. sort the means;
  4. `ci_low = means[int(0.025·R)]`, `ci_high = means[int(0.975·R) − 1]`.

  It returns `{mean_delta, ci_low, ci_high, n}`.
- `verdict(ci_low, ci_high, n)`: `"improved"` if `ci_low > 0`; `"regressed"` if `ci_high < 0`;
  otherwise `"no detectable difference at n={n}"`.
- `unjudged_at(run_rows, qrels, query_ids, depth)` counts the top-depth rows with no qrels entry.
- `abstention(run_rows, no_answer_ids, answerable_ids)` returns
  `{no_answer_top1: {qid: score}, answerable_top1: {...}, median_no_answer, median_answerable}`.
- **Predictions**: `evals/ladder_predictions.json` was written by the planner on 2026-10-18 before
  any step existed. It is locked.

## 12. U8 — latency (`evals/latency_bench.py`)

- `WARMUP = 1`, `REPEATS = 5`. `COMPONENTS = ("cosine_qdrant", "graph_neo4j", "bm25", "rrf",
  "rerank_cpu", "rerank_gpu")`.
- `time_call(fn, warmup, repeats, clock=time.perf_counter) -> list[ms]` discards the warm-up.
- `summarize(samples)` uses nearest-rank percentiles, `s[ceil(q·n) − 1]`, and returns
  `{n, p50, p95}`. An empty input raises `ValueError`.
- `component_summary(per_query)` applies `summarize` to the per-query medians.
- `machine_info()` returns `{platform, processor, cpu_count, gpu}`, with `gpu` `null` when CUDA is
  absent.
- **CLI** `uv run python -m evals.latency_bench` (live): listener check and fingerprint first, then
  each component per query:
  - cosine is `read_only_vector_store().search(text, 30)`;
  - graph is `_graph_candidates` through `read_only_store()`;
  - bm25, rrf and rerank (30 pairs) are timed in-process;
  - `rerank_gpu` is timed only if CUDA is available.

  It writes `runs/latency_v1.json = {machine, components: {name: {per_query, summary}},
  fingerprint_before, fingerprint_after}` and finishes with `assert_unchanged`. **Latency never
  affects a verdict.**

## 13. U9 — ladder assembly (`evals/run_ladder.py`)

- `STEPS` and `SCORE_SCALE` exactly as in section 4.
- `build_ladder(queries, docs_by_id, cosine_rows, bm25_rows, graph_rows, reranker) -> {step:
  rows}` follows section 4:
  - s0 is the cosine rows with the run id replaced;
  - re-ranking reads `docs_by_id[cid]["embedded_text"]` in fused order;
  - re-ranking never changes membership;
  - ranks are contiguous.
- `run_meta(step, inputs: {name: path})` is as in section 5.
- **CLI** `uv run python -m evals.run_ladder` (offline) loads `vector_cosine_v1`, `bm25_v1` and
  `graph_lexical_v0` plus the real `Reranker()`, then writes the five run files and their meta.

## 14. U10 — scoring and report (`evals/score_ladder.py`)

- `COMPARISONS` as in section 4.
- `score(runs, qrels, queries, predictions) -> dict`:
  - calls `topup.assert_fully_judged` **first**;
  - averages over `answerable` only;
  - per step: `{mean: {ndcg, success, mrr, candidate_success}, unjudged_at_30,
    candidate_success_is_lower_bound}`;
  - per comparison: `{base, new, win_tie_loss, bootstrap, verdict, prediction | None,
    prediction_held | None}`;
  - `abstention` only for s0, s2 and s4.

  The result must be JSON-serialisable.
- `render_appendix(result, latency) -> str`:
  - starts with `## Appendix C — Retrieval ladder`;
  - states `n=<n_answerable>`;
  - writes every verdict string verbatim;
  - marks Candidate Success@30 "lower bound" where applicable;
  - labels end-to-end latency "estimate across machines";
  - includes per-query win/tie/loss with the query ids;
  - includes predicted vs actual for each comparison.
- `append_appendix(report_path, text)` keeps the existing bytes unchanged, then appends `\n\n` and
  the appendix. It raises `ValueError` if `## Appendix C` is already present.
- **CLI** `uv run python -m evals.score_ladder`:
  - loads the five runs, `qrels_v2.csv`, the queries, the predictions and `runs/latency_v1.json`;
  - writes `ladder_metrics.json` (adding `"qrels_version": "qrels_v2.csv"`) and `ladder_report.md`;
  - then, only with `--append-report`, appends to `python-ingestion/metrics/manthan_report.md`.

## 15. Execution plan

```
WAVE 1 (parallel)
  offline track, one worktree per unit:   U3  U4  U5  U6  U7
  live track, on the owner's machine:     U0 ──merge──► H0 ──► U1 ──► U2
                                                              └────► U8   (needs U1; may run after U5 merges)
WAVE 2 (sequential)
  U9 (needs U2,U3,U4,U5) ──► U6 build ──► H1 owner grades ──► U6 merge ──► U10 (needs U7,U8)
```

Before starting, the owner tags the base: `git tag ladder-base` on the commit containing this spec.
After each merge, run `uv run pytest -q --ignore-glob='tests/test_ladder_*.py'`, the lock test, and the merged units' own test files.

Gates, in order (each is a test command that must be green):

| Gate | Command |
|---|---|
| G0 U0 merged | `uv run pytest tests/test_ladder_qdrant_api_key.py -q` |
| G1 harness proven | `uv run pytest --live -m live tests/test_ladder_live_guard.py -q` |
| G2 frozen inputs valid | `uv run pytest --live -m live tests/test_ladder_freeze_graph.py -q` and `uv run pytest --artifacts -m artifacts tests/test_ladder_freeze_graph.py tests/test_ladder_bm25.py -q` |
| G3 model pinned | `HF_HUB_OFFLINE=1 uv run pytest --model -m model tests/test_ladder_rerank.py -q` |
| G4 runs built | `uv run pytest --artifacts -m artifacts tests/test_ladder_assembly.py -q` |
| G5 fully judged | `uv run pytest --artifacts -m artifacts tests/test_ladder_topup.py -q` |
| G6 latency recorded | `uv run pytest --artifacts -m artifacts tests/test_ladder_latency.py -q` |
| G7 scored | `uv run pytest --artifacts -m artifacts tests/test_ladder_score_report.py -q` |

## 16. Acceptance

The experiment is complete when G0–G7 are green, the full offline suite is green, the lock test is
green, `git diff ladder-base -- python-ingestion/tests` is empty, and Appendix C is appended with
the original report bytes unchanged.

## 17. Escalate: stop, change nothing, report to the owner

- Any `--live` test fails: the read-only key is rejected or missing, an auto-commit write is *not*
  rejected, the fingerprints differ, the listener is running, or the collection drifted from
  `pilot-001`.
- `test_live_every_corpus_document_is_reachable_from_neo4j` fails. Neo4j text differs from Qdrant
  text (for example `synthesized_text`); do not add fuzzy matching.
- `test_live_cosine_ranking_matches_the_frozen_run` fails. The frozen Run A no longer reflects the
  store.
- A locked test looks wrong or contradicts this spec.
- The top-up sheet has more than 300 rows (the planning estimate was 50–150).
- Any temptation to change a constant (k, depth, seed, model, ε, resamples) after seeing a result.
