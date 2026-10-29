# Manthan MCP server — v1 spec

Status: design agreed with the owner on 2026-10-28 (lesson and interview, Q1–Q13). This document is for the executor. The planner reviews; the owner locks tests.

## 0. Why this exists

Run A showed S2 is the best retrieval: cosine + BM25 fused with RRF, then a cross-encoder re-rank. Run B showed the graph does not improve search (GRAPH OUT ×4). The live `/search` (`search.py`) still runs cosine plus a graph keyword pass, with no BM25 and no re-ranker.

v1 does three things:
1. upgrades `/search` to S2 and proves it matches the measured system;
2. exposes Manthan to LLMs through a local MCP server;
3. uses the graph only for exact topic lookups.

## 1. Decisions (do not change without the planner)

| ID | Decision |
|---|---|
| M1 | The MCP server is a **thin stdio adapter over the FastAPI app**. It holds no DB clients and loads no models. It only makes HTTP calls to `MANTHAN_API_URL` (default `http://localhost:8000`). |
| M2 | **S2 lives in one place.** A new `retrieval.py` imports `evals.fusion`, `evals.rerank`, `evals.bm25` and `evals.common.embedded_text`. Copy nothing, so the eval and the product cannot drift apart. |
| M3 | Tools: `search`, `get_item`, `list_topics`, `browse_topic`. There are no person tools: senders are discarded at parse time (`parser.py:57-59`). |
| M4 | `search` returns at most 5 results. Each snippet is cut to 300 characters. `get_item(id)` returns the full text. |
| M5 | Confidence comes from the **top-1 re-rank score**: `strong` if ≥ 2.0, otherwise `weak`. `none` only when there are zero results. The 2.0 is fit on the same 25 graded queries, so it lives in one constant, `CONFIDENCE_STRONG_MIN`. Results are **never hidden** because of a low score: real answers have scored as low as −7.6 (q009). |
| M6 | `kind` filter (`any` / `introduction`) is applied **before** building the pool, in both the Qdrant query and the BM25 candidate set. It is never applied after the pool. Phase 3. |
| M7 | The MCP is **read-only**. It calls only `POST /search` and the new GET endpoints, never the ingest or link endpoints. |
| M8 | Item id = the Qdrant point id. Messages use `uuid5(NAMESPACE_URL, f"msg-{sent_at}-{text}")` and links use `uuid5(NAMESPACE_URL, f"link-{url}")`, as in `vector_store.py:51,76`. Neo4j results compute the same id. **Never hash enriched or embedded text into an id** (this protects v1.1/C1). |
| M9 | C1 (link text glued onto messages) is **not** in v1. See §8. |

## 2. Phase 1: `/search` becomes S2

### New file: `python-ingestion/retrieval.py`

```
search_s2(query, *, top_k=5, kind="any", topics=None, min_quality=None,
          vs, reranker) -> {"results": [...], "confidence": str, "top_score": float|None}
```

**Corpus cache.** Scroll every point in the collection once, building `{point_id: (embedded_text, payload)}`. Use `evals.common.embedded_text(payload)`, the same text the evals scored. Rebuild the BM25 index when `client.count()` differs from the cached size. This check costs one cheap call per request, and the corpus is about 200 docs.

**Pipeline.** The constants come from `evals.fusion` and are not re-declared:
1. Cosine: `vs.search(query, limit=DEPTH)`, then ids.
2. BM25: score all cached docs and take the top `DEPTH`.
3. Fuse the two lists with `fusion.rrf([cos, bm25], k=RRF_K, depth=DEPTH)`.
4. Re-rank: `rerank.Reranker().rerank(query, [(id, embedded_text)...])`.
5. Confidence: label from the top-1 score per M5. Return the first `top_k`.

**Result shape.** Keep every key that `search.py` results carry today (type, text, url, title, sent_at, score…), so `query_worker.format_results` keeps working. Add `id`. The re-ranker is loaded **once** at app startup in `lifespan`, next to `app.state.vs`.

**`topics` / `min_quality`.** These are existing request fields. Apply them as **pre-filters** (M6 mechanism). When they are `None`, nothing is filtered, so parity is unaffected.

### `app.py` changes

- `POST /search` calls `search_s2`. The response gains `confidence` and `top_score`.
- The `top_k` default stays as it is today for the bot. The MCP passes 5.
- `search.py` stays in the repo, unused by `/search`. Its removal is a separate cleanup commit later.

### Parity gate (Q9). Pass rule fixed before running.

New script `evals/parity_s2_live.py`:
1. Run `evals.parity_check` first.
2. Call `retrieval.search_s2` (in-process, `top_k=30`) for the 22 queries in `queries.csv` (exclude g* graph queries).
3. Compare with the frozen run `evals/data/pilot-001/runs/ladder_s2_cos_bm25_rr.jsonl`:
   - **If `parity_check` passed:** top-5 ids are **identical and in the same order** for all 22 queries.
     - Map live ids to corpus `candidate_id` with the D25 alias map (`freeze_neighbors.build_alias_map`).
   - **If `parity_check` failed (corpus grew):** nDCG@5 on `qrels_v3.csv`, over graded docs only, is within ±0.02 of **0.7517**. Report which branch ran.
4. Exit non-zero on failure. Phase 2 does not start until this passes.

## 3. Phase 2: read endpoints and the MCP server

### New GET endpoints in `app.py` (read-only Cypher / Qdrant retrieve)

| Endpoint | Returns |
|---|---|
| `GET /items/{id}` | The full record of one point: type, full text, sent_at, url, title, link summary, topics. 404 with a clear message if unknown. |
| `GET /topics` | `[{name, count}]`, sorted by count descending. Count = Messages + Links `ABOUT` the topic. |
| `GET /topics/{name}?limit=50` | Every message and link `ABOUT` the topic (id per M8, date, 300-char snippet, url, title), newest first, plus `total` and `neighbors`: the top 5 topics co-occurring on the same items, with counts. Unknown topic gives 404 and the 5 closest names (case-insensitive substring match). |

Reuse the Cypher style of `_query_messages` (`app.py:299`).

### New file: `python-ingestion/mcp_server.py`

- **SDK.** Pin the `mcp` package in `pyproject.toml`. Use the API of the pinned version: `from mcp.server.fastmcp import FastMCP` (1.x) or `from mcp.server import MCPServer` (2.x). Check which one is installed.
- **HTTP.** Use `httpx`, timeout 10 s.
- **stdout is the protocol channel. Never `print`.** Log with `logging` to stderr. Also append one JSON line per tool call to `logs/mcp_calls.jsonl`, containing: ts, tool, args, result_count, confidence, error.
- **API errors.** Return text, not an exception trace. Unreachable API: `"Manthan API is not reachable at <url>. Ask the user to start it (docker compose up + the FastAPI app)."` A 404 passes the endpoint's message through.

**Server instructions** (verbatim):
```
Manthan holds the user's saved knowledge from their tech WhatsApp groups. When you
use a result, cite it by its date and link (or a short quote). Never present
something as coming from the user's groups unless a tool returned it.
```

**Tool descriptions** (verbatim, agreed in Q6/Q7):

`search(query: str, kind: Literal["any","introduction"]="any")`
```
Searches the user's saved knowledge from their tech WhatsApp groups: messages
and shared links (articles, repos, videos) where members discussed tools,
libraries, errors, and trends, with their experiences and opinions.
Call it when the user is choosing a tool or approach, hits an error or crash,
or asks what their groups said or shared about something; skip it for general
knowledge or routine coding steps that don't need the community's view.
Pass a natural-language question (e.g. "problems people had with Qdrant
filters"), not single keywords. Set kind="introduction" only when the user
wants people to connect with: it searches members' self-introductions.
Returns up to 5 results, each with an id, date, short snippet, link title/URL
if any, plus an overall confidence: strong, weak, or none. On weak, check the
snippets really answer the question before using them.
Call get_item(id) for the full text; if confidence is weak or none, rephrase
once or tell the user their groups haven't discussed it. Never invent results.
```
In v1, before Phase 3, `kind` is **not exposed**. Add it in Phase 3 together with this description line.

`get_item(id: str)`: "Returns the full text of one search or browse result by its id, including the shared link's title and summary. Use it only for results you need in full."

`list_topics()`: "Lists the topics in the user's saved group knowledge with how many items each has. Use it to see what the groups talk about, or to find the exact topic name for browse_topic."

`browse_topic(topic: str)`: "Returns every saved item tagged with one exact topic, plus related topics. Use it when the user wants a complete list or overview of a topic (e.g. 'everything shared about RAG'); use search for a specific question. Summarize the count and offer the list; don't dump all items unprompted."

### Wiring

- Document in the README how to add the server to Claude Code (`claude mcp add manthan -- uv run --directory <path>/python-ingestion python mcp_server.py`) and to Claude Desktop (`mcpServers` JSON).
- Manual check: `uv run mcp dev mcp_server.py` (MCP Inspector). Call every tool once and paste the outputs into the PR.

## 4. Phase 3: persist `category`, add `kind`

**Write path.**
- The grader already produces `category` (`grader.py`), but nothing stores it.
- Add it to the enrich record (`enrich.py:~247`).
- Write `m.category` in `store.py`.
- Add `category` to the Qdrant payload in `vector_store.upsert_message`.
- Links get `category = "resource"`, as in `ingest.py:78`.

**Backfill (owner-run, writes to live stores).**
- Script: `scripts/backfill_category.py --dry-run|--apply`.
- Source: `graded_messages.json`, matched on `(sent_at, original_text)` to the M8 id.
- Qdrant: use **`set_payload` only**. No re-embedding, no id change.
- Neo4j: `SET m.category`.
- Print matched/unmatched counts.
- Must be idempotent.
- After `--apply`, `evals.parity_check` must still pass, since vectors are unchanged.

**Payload index.** Create a keyword payload index on `category` in Qdrant.

**Retrieval.**
- `kind="introduction"` → Qdrant `Filter(must=[category == "introduction"])` on the cosine query.
- BM25 runs only over cached docs whose payload has that category, then RRF and rerank as usual (M6).
- Expose `kind` on `/search` and on the MCP `search` tool.

## 5. Tests (executor writes; owner reviews and locks into `tests/LOCK.sha256`)

`tests/test_retrieval_s2.py`
- With a fake vector store, fake BM25 docs and an injected re-ranker scorer:
  - order equals `rrf` followed by `rerank` on hand-computed inputs;
  - `top_k` is respected.
- Confidence:
  - a top score of 2.0 gives `strong`;
  - 1.999 gives `weak`;
  - empty results give `none`.
- The pre-filter excludes non-matching docs from **both** cosine and BM25 before fusion. Test case: a filtered-out doc ranked #1 must never appear.
- `topics=None, min_quality=None, kind="any"` give byte-identical results to an unfiltered call.

`tests/test_mcp_server.py`
- Uses a fake HTTP transport (`httpx.MockTransport`) and calls the tool functions directly:
  - the snippet is ≤ 300 characters;
  - at most 5 results;
  - an unreachable API returns the instructional message, not a raise;
  - a 404 message is passed through;
  - each call appends one JSON line to the call log;
  - nothing is written to stdout (capture with `capsys`).
- The tool descriptions contain "Never invent results", and the server instructions contain "cite".

`tests/test_topics_endpoints.py`
- `/topics/{name}` unknown → 404 with suggestions.
- Ids follow M8.

`tests/test_backfill_category.py`
- Dry run writes nothing.
- Apply is idempotent.
- Unmatched rows are reported.
- Only `set_payload` is called, never `upsert`.

## 6. Phase 4: behaviour test (owner-run, Q13)

**The sheet.** `evals/mcp_behaviour.csv` has the columns `id, question, expect_tool, expect_args, expect_behaviour`. The planner drafts the rows and the owner edits them. They cover:
- 4 questions that should trigger search (from `queries.csv` intents: recommendation, concept, temporal);
- 2 no-answer questions (q022–q025 style), where the expected behaviour is "says not discussed; no invented items";
- 2 general-knowledge questions, where the expected behaviour is no Manthan call;
- 2 introduction questions ("who could I talk to about X"), where `kind=introduction` is expected;
- 1 "what is RAG"-type question, where `kind=introduction` must **not** be used;
- 1 "everything shared about X", where `browse_topic` is expected;
- 1 "what do my groups talk about", where `list_topics` is expected.

**How to run it.**
- Use a fresh Claude Code session for each run.
- Ask the question as written, with **no** "cite" or "use the tool" hints.
- Run each question 3 times.
- Grade from `logs/mcp_calls.jsonl` plus the final answer.

**Pass rule.**
- Each row needs at least 2 of 3 runs correct.
- **Zero** invented items across all no-answer runs. One invented item fails v1.

**If a row fails,** fix the description or instructions, not the test, then re-run the whole sheet.

## 7. Order, branch, and rules

**Order and gates.**
1. Phase 1: the parity gate must pass.
2. Phase 2: the Inspector check must pass.
3. Phase 3: the backfill must run with `parity_check` still green.
4. Phase 4: the behaviour sheet must pass.

**Branch and commits.**
- Branch `mcp/v1`.
- One commit per phase, plus test commits before implementation where practical.

**Rules.**
- Never bypass the pre-commit hook. Never push.
- No writes to Qdrant or Neo4j except the Phase 3 backfill, which the owner runs.
- Never print or log `QDRANT_API_KEY`.
- The executor does not edit `tests/LOCK.sha256`; the owner locks it.

## 8. v1.1: C1 (recorded here, specced later)

- **Approach.** The message embedding text = message + its links' titles and summaries. It goes in a **new collection**; the old one stays untouched, and the switch is one config value.
- **Re-embed trigger.** Re-embed a message when its link summary arrives (scraper or `/links/paste`).
- **Ids.** Ids stay per M8, hashed from raw text only.
- **Switch rule** (fixed now, 2026-10-28):
  - original 22 queries: **not proven worse** (CI upper bound ≥ 0);
  - graph set g001–g008: **proven better** (CI excludes 0).
  - Run B's offline C1 passed both. If live C1 fails, treat it as a pipeline bug, not a reason to drop C1.
