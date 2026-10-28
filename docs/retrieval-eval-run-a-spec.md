# Spec: Golden Dataset Pilot and Run A (Vector-Only Cosine Baseline)

Status: ready for execution
Builds on: `docs/golden-dataset-evaluation-plan.md` (metrics, grade scale, failure taxonomy, Runs A–D). This spec makes that plan concrete for the first 5 queries and fixes the data problems that would otherwise corrupt the baseline.

Decisions already made by the owner (do not re-litigate):

1. **Labelling uses pooled human judgments**, not LLM pre-labels. Pool = cosine top-30 ∪ BM25 top-20 ∪ keyword hits derived from the query `notes`.
2. **A message and the link it shares are graded independently** in Run A. Deduplication only matters when the hybrid `search()` is evaluated (Run B/D).
3. **Run A stays pure.** Long-document fixes (chunking) are measured as a separate variant, `vector_cosine_chunked_v1`, never folded into the baseline.
4. **Live Drive files are the source of truth** (2026-10-09). Trashed exports are not re-imported; the superseded Sep 9 Workflows export is trashed rather than imported.
5. **Reposts are allowed** (2026-10-09). The same text at the same timestamp is a true duplicate and fails the gate. The same text posted at a different time is a genuine repost: allowed and reported.

---

## Phase 0 — Data hygiene (blocking)

The live corpus is not fit to label. Findings from 2026-10-09:

| Problem | Evidence | Cause |
| --- | --- | --- |
| Branch is missing the sender removal | `review-rag-eval` is 5 commits behind `main`, 0 ahead. `f5c4661` / merges `d5a3fe4`, `81a40f7` are only on `main` | Work branch was cut before the merge |
| Sender written again after migration | Qdrant `manthan-fresh`: 46 points carry a `sender` key; Neo4j: 47 `Message` nodes with `sender`, 2 `Person` nodes | Drive listener and imports ran this branch's (pre-migration) code |
| **Duplicate vectors** | 193 points, 148 unique contents; 43 contents stored twice (one sender-based ID `msg-{sender}-{sent_at}`, one content-based ID `msg-{sent_at}-{text}`) | Both ID schemes wrote into the same collection |
| **Phone numbers inside message text** | 3 messages whose text is `+NN NNNNN NNNNN joined using a group link.` / `… was added` | Bug in `f5c4661`, `parser.py`: the header-without-`": "` branch changed from `continue` to `text = rest`, so WhatsApp system events became messages that start with the member's phone number |
| Stale manifest with PII | Root `documents.jsonl` (2026-09-26, pre-migration): 107 phone numbers in `sender`, 5 in `text`. Gitignored and never committed | Exported before the migration |

Phone-regex hits on arXiv IDs and tweet status IDs inside URLs are false positives; scans must strip URLs first.

Found during the 2026-10-09 rebuild (both fixed, with regression tests):

| Problem | Cause | Fix |
| --- | --- | --- |
| A re-import recorded 148 bogus "low quality" grades in 12 s | The graph thread came from `source_id + revision`; after revisions were reset it resumed an unfinished run with empty texts. The guard compared counts only | Batch path (and so the thread) includes a content digest; the guard compares grade texts |
| Trashed exports would be imported | Neither the change feed nor `list_files` checked `trashed`, and trashing a file is itself a change event | Change feed requests and rejects `trashed`; listing adds `trashed = false` |

### P0.1 Merge `main` into `review-rag-eval`

- Resolve conflicts so the result is sender-free (main wins) **and** keeps this branch's staged improvements (e.g. 24-hour timestamp support in `parser.py`: `TIME_FORMATS` with `%H:%M`, optional AM/PM in the header regex).
- Expect conflicts in `parser.py`, `main.py`, `incremental_import.py`, `query_worker.py`.
- Done when: `git rev-list --count HEAD..main` = 0, the full suite passes (`uv run python tests/run_tests.py`, `uv run pytest -q`), and `grep -rn "sender" python-ingestion/{parser,store,vector_store,search}.py` shows no sender identity/storage code.

### P0.2 Fix the system-event leak in `parser.py`

- A header line whose remainder has no `": "` is a WhatsApp system event (joined/added/left/changed subject/encryption notice). Restore `continue` for that branch.
- Add tests: `"+91 98765 43210 joined using a group link."`, `"+1 555 0100 was added"`, and `"Messages and calls are end-to-end encrypted…"` produce **no** messages. A normal `"Name: hello"` line still yields text `"hello"`, and multi-line continuation still works.

### P0.3 Rebuild the stores — **requires owner approval before `--apply`**

1. Stop writers (API, workers, Drive listener).
2. `uv run python scripts/migrate_remove_sender.py` (dry run) → save the audit output.
3. **Stop and show the audit to the owner.** Run `--apply` only after an explicit yes. It deletes the Qdrant collection, `Person` nodes and `processed_messages`, after a SQLite backup.
4. Re-import every Drive export (full import, not incremental).
5. Verify: 0 points with a `sender` key, 0 `Person` nodes, 0 duplicate contents, 0 phone numbers in text (URLs stripped), and `drive_watch_state` unchanged.

### P0.4 Delete the stale root `documents.jsonl`

It is superseded by the Phase 1 export. Confirm it is untracked (`git log --all -- documents.jsonl` is empty), then remove it.

---

## Phase 1 — Run A pipeline

All code lives in `python-ingestion/evals/`. All data lives in `python-ingestion/evals/data/<corpus_version>/`, which **must be gitignored** because it contains chat text. Replace the dummy `collect_retrievals.py`, `evaluate_retrieval.py` and `prepare_annotation_csv.py`, plus `pooled_demo.jsonl` / `annotation_demo.csv`. They return placeholder hits, label only 0/1, and treat pooled order as ranking.

Settings come from `python-ingestion/.env`. The live collection is **`QDRANT_COLLECTION=manthan-fresh`**; never hard-code `manthan`.

### Stage 1 — `export_corpus.py`

Scroll every point in Qdrant (payloads, plus vectors for Stage 2) and write:

`documents.jsonl`, one line per point:

```json
{"candidate_id": "message:<qdrant_point_id>", "candidate_type": "message",
 "qdrant_point_id": "…", "sent_at": "…", "title": null, "url": null,
 "embedded_text": "<exact string that was embedded>",
 "token_count": 312, "truncated": true,
 "content_hash": "sha256:…", "corpus_version": "pilot-001"}
```

- `embedded_text` must match what `vector_store.py` embeds. Messages: payload `text`. Links: `" ".join(p for p in (summary, what_it_is, problem_solved) if p)`.
- `token_count` uses the embedding model's tokenizer. `truncated = token_count > model.max_seq_length` (256 for `all-MiniLM-L6-v2`).
- No `sender` field.
- **PII gate:** strip URLs, then fail the export if any `embedded_text` matches `\+?\d[\d\s().-]{8,}\d`.

`meta.json`: corpus_version, collection, point_count, embed model name and `max_seq_length`, git SHA, created_at.

### Stage 2 — `parity_check.py`

Re-embed every `embedded_text` with the configured model and compare it with the stored Qdrant vector. **Fail if any cosine < 0.9999.** Also fail on true duplicates (same `content_hash` and same `sent_at`/URL); report reposts (same text, different time) without failing. This proves the manifest is the index.

### Stage 3 — `run_vector.py`

- For each query, call `VectorStore.search(query, limit=<point_count>)` directly, ranking the **whole** corpus. Never call hybrid `search()`.
- Write `runs/vector_cosine_v1.jsonl`, one row per (query, candidate): `run_id, query_id, candidate_id, rank, score, latency_ms`. Raw scores, no threshold.

### Stage 4 — `build_pool.py`

- Per query, take the union of: cosine top-30 from Stage 3, BM25 top-20 over `embedded_text` (`rank_bm25` or equivalent, lowercase plus a simple tokenizer), and documents containing keywords that a human extracted from the query's `notes` column. Put those keywords in a reviewed file, `pool_keywords.json`, not an automatic parse.
- Write `annotation.csv` with columns `query_id, query_text, candidate_id, candidate_type, content, relevance, reason`.
- **Blind:** no score, rank or source columns. Rows are shuffled within each query with a fixed seed (42).
- Write `pool_provenance.jsonl` separately, recording which source contributed each candidate. Annotators never see it.

### Stage 5 — Human labels → `qrels.csv`

The owner labels using the 0–3 scale from the plan doc. Columns: `query_id, candidate_id, relevance, answer_bearing, reason`. Unjudged documents count as 0. Any qrels `candidate_id` not in `documents.jsonl` is an error.

### Stage 6 — `evaluate.py` → `metrics.json` + `report.md`

Per-query table (the primary output at 5 queries):

| query_id | rank of best grade-3 | rank of first grade≥2 | nDCG@5 | Success@5 | MRR@5 | Recall@5 | failure class | distractors above first relevant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Aggregates: Candidate Success@30, Success@5, MRR@5, graded nDCG@5 (primary), Recall@5. Use the definitions in the plan doc; grade ≥ 2 counts as relevant.

Cosine-specific diagnostics:

- **Separability:** ROC-AUC of the cosine score as a relevant-vs-irrelevant classifier, per query and pooled.
- **Threshold sweep:** at thresholds 0.20, 0.25, …, 0.60, report pooled precision, recall and Success@5. Mark the current product value of 0.25.
- **Score scale:** top-1 score per query, to show that absolute scores don't compare across queries.

Failure classes are the plan doc's list plus:

- **Long-document miss:** a relevant document has `truncated=true` and ranks below 5.
- **Distractor** tag (not a class): non-relevant documents ranked above the first relevant one. Self-introduction messages are the expected main case.

---

## Pre-registered predictions

**These are hypotheses, not results.** They were written on 2026-10-10 from a look at the top cosine results on the rebuilt, restored corpus (`pilot-001`: 188 documents) and *before any labelling*. The real numbers come only from the human-labelled run. The report must compare outcomes against this table (`evals/predictions.json` holds the same content).

| Query | Predicted outcome | Reason |
| --- | --- | --- |
| q001 | Success, mediocre nDCG | Agent Orchestrator link at #1 (0.70); self-introductions take #3-#5 |
| q002 | Success | qdrant link #1, ParadeDB link #3, pgvector message #4 |
| q003 | Success | Rerank-claim message #1 (0.71), RAG survey link #2 |
| q004 | Success, distractor-heavy | ClawMetry author's message #1 (0.62); intros follow |
| q005 | **Failure: long-document miss** | Best answer is a 1,132-token "Local Model Setup" message at cosine #38 (0.30), truncated at 256 tokens; pooled only via BM25/keywords |

The pilot set is skewed toward easy, keyword-anchored queries. Do not read a 4/5 Success@5 as a property of the retriever.

---

## Acceptance tests

| # | Test | Pass condition |
| --- | --- | --- |
| 1 | Hygiene | 0 `sender` keys, 0 `Person` nodes, 0 true duplicates (same hash and time), 0 phone matches (URLs stripped); reposts reported |
| 2 | Corpus integrity | `len(documents.jsonl) == Qdrant point count`; every qrels id exists |
| 3 | Vector parity | min cosine(re-embedded, stored) ≥ 0.9999 |
| 4 | Determinism | Two Stage 3 runs produce identical `(query_id, candidate_id, rank)` |
| 5 | Metric unit tests | Hand-computed fixtures in `tests/test_eval_metrics.py`: graded nDCG@5 with grades [3,0,2,0,1] and the ideal ordering; MRR@5 = 0 when the first relevant result is at rank 6; Recall@5 with 3 relevant documents, 2 retrieved = 0.667; Success@5 treats grade 1 as not relevant |
| 6 | Blindness | `annotation.csv` has exactly the 7 specified columns, and row order differs from rank order |
| 7 | Pool coverage | For q005, the long "Local Model Setup" message is in the pool, via BM25 or keywords, even though its cosine rank is above 30 |
| 8 | Privacy | `git check-ignore python-ingestion/evals/data/x` succeeds; no file under `evals/` that `git status` shows contains chat text |

Wire tests 1–4 as `evals/check_corpus.py`, which exits non-zero on failure. Tests 5–6 are pytest.

---

## Follow-on variant (after Run A is labelled)

`vector_cosine_chunked_v1`: split documents into ≤256-token windows with 32-token overlap, embed each chunk, and score each document by its best-scoring chunk. Evaluate on the **same qrels**. This tests whether q005 is truncation (it should be fixed) or dilution (it shouldn't be). Write chunks to a separate collection; never to `manthan-fresh`.

## Scaling

5 queries → 20 → 50, then freeze 20 as the untouched test split. New queries must include, by quota:

- paraphrases with no shared keywords
- answers found in long documents
- person lookups by attribute
- questions about when something happened
- 3–5 questions whose answer is not in the corpus (tests whether any threshold can say "nothing found")

Write queries **before** looking at their results.
