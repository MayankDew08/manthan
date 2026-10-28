# Golden Dataset and Retrieval Evaluation Plan

## Goal

Build one reusable, human-labelled golden dataset and use it to compare four controlled retrieval systems:

| Run | Candidate retrieval | Final ranking |
| --- | --- | --- |
| A | Vector database only | Cosine similarity |
| B | Vector and graph retrieval | Explicit fusion rule |
| C | Vector database only | Reranker |
| D | Vector and graph retrieval | Reranker |

The immediate focus is **Run A: vector-only cosine similarity**. Graph retrieval should have zero influence on this baseline.

The same queries, documents, and human relevance labels will later be reused for all four runs. This makes the comparisons fair and prevents the benchmark from changing between experiments.

## Current System Status

The current production search is not vector-only:

- Qdrant uses cosine similarity.
- The default embedding model is `all-MiniLM-L6-v2` with normalized vectors.
- Messages are vectorized only when their quality is at least 4.
- Links are vectorized when extracted content is available.
- The production `search()` function always mixes Qdrant and Neo4j results.
- Graph results currently receive a constant score equal to `0.5 * best_semantic_score`.
- A final minimum score threshold of `0.25` is applied.
- The search API does not currently return Qdrant point IDs.
- The existing evaluation scripts produce dummy results and are not yet connected to Qdrant.

The most recent stored metrics report lists:

- 11 Neo4j messages
- 12 Neo4j links
- 16 Qdrant points

These values are from a previous report, not a live database inspection. The Docker services were stopped during inspection, so the current Qdrant point count could not be verified.

### Why setting graph weight to zero is insufficient

The existing hybrid `search()` function still queries Neo4j for graph candidates, related links, link attribution, and deduplication. Therefore, setting a graph score to zero does not produce a clean vector-only experiment.

Run A should call `VectorStore.search()` directly. Neo4j may be used later for displaying metadata, but it must not affect candidate selection or ranking.

## What a 50-Query Golden Dataset Means

A golden label describes a relationship between a query and a candidate document:

```text
Query + candidate document + human relevance judgment
```

For example:

```text
Query:
"What tool was recommended for managing coding agents?"

Candidate A:
A message that names and recommends the tool
Relevance: 3

Candidate B:
A general article about AI agents
Relevance: 1

Candidate C:
An unrelated vector database message
Relevance: 0
```

The target is **50 queries**, not 50 total labelled rows.

If the corpus contains 23 eligible messages and links, exhaustive annotation requires:

```text
50 queries * 23 documents = 1,150 relevance judgments
```

Because the current corpus is small, exhaustive annotation is valuable. It provides true ground truth and makes it possible to detect documents that vector search completely misses.

Start with a smoke test:

1. Write 5 queries.
2. Label every eligible document for those queries.
3. Verify the collector and metric calculations.
4. Expand to all 50 queries.

## Evaluation Corpus

The frozen evaluation corpus should contain every record that could reasonably answer a query:

- Searchable messages
- Searchable links
- Records that might later be found through graph retrieval

Do not create the golden dataset only from the current top-five search results. Doing so hides relevant documents that cosine retrieval misses.

Each frozen corpus export should have a unique version, for example:

```text
pilot-001
```

All systems in a comparison must use the same corpus version.

## Required Files

Keep human truth separate from machine-generated rankings.

### 1. `queries.csv`

This file contains the 50 questions.

```csv
query_id,query_text,intent,origin,split,notes
q001,What tool was recommended for managing coding agents?,recommendation,real,test,
q002,What is Qdrant used for?,concept,real,development,
q003,Who had experience with vector indexes?,person_lookup,real,test,
```

Fields:

- `query_id`: permanent query identifier, such as `q001`
- `query_text`: exact text sent to the retrieval system
- `intent`: recommendation, concept, person lookup, link lookup, and so on
- `origin`: real, known-item, difficult, or unanswerable
- `split`: `development` or `test`
- `notes`: optional private explanation

For the 50-query pilot, use:

- 30 development queries
- 20 locked test queries

Use development queries while improving the system. Do not tune graph retrieval, thresholds, fusion, or the reranker against the locked test queries.

### 2. `documents.jsonl`

This file is the frozen corpus manifest.

Message example:

```json
{
  "candidate_id": "message:3db65f45-...",
  "candidate_type": "message",
  "qdrant_point_id": "3db65f45-...",
  "sender": "Alice",
  "sent_at": "2026-08-14T10:31:05",
  "text": "The full searchable message text",
  "url": null,
  "corpus_version": "pilot-001",
  "content_hash": "sha256:..."
}
```

Link example:

```json
{
  "candidate_id": "link:9da1782d-...",
  "candidate_type": "link",
  "qdrant_point_id": "9da1782d-...",
  "sender": null,
  "sent_at": null,
  "title": "Qdrant documentation",
  "text": "Summary and searchable link content",
  "url": "https://example.com/qdrant",
  "corpus_version": "pilot-001",
  "content_hash": "sha256:..."
}
```

### 3. `qrels.csv`

Qrels are the human relevance judgments and form the actual golden dataset.

```csv
query_id,candidate_id,relevance,answer_bearing,reason
q001,message:3db65f45-...,3,1,Directly names and recommends the tool
q001,link:9da1782d-...,2,1,Useful supporting source
q001,message:7aa502ab-...,1,0,Discusses agents but does not answer the question
q001,message:15fab332-...,0,0,Unrelated vector database discussion
```

Use this relevance scale:

| Grade | Meaning |
| ---: | --- |
| 3 | Direct, complete, or best answer |
| 2 | Useful answer, but incomplete |
| 1 | Related context, but does not answer |
| 0 | Irrelevant |

For binary success metrics, grades 2 and 3 are relevant.

### 4. `runs.jsonl`

This file contains machine-generated rankings. It is regenerated for every system configuration and is not manually labelled.

```json
{
  "run_id": "vector_cosine_v1",
  "query_id": "q001",
  "candidate_id": "message:3db65f45-...",
  "rank": 1,
  "score": 0.4321,
  "latency_ms": 18.4
}
```

Suggested run names:

```text
vector_cosine_v1
vector_graph_v1
vector_reranker_v1
vector_graph_reranker_v1
```

## Candidate and Message IDs

Use the general name `candidate_id` because search candidates can be either messages or links.

Qdrant already assigns a deterministic point ID to every vector.

Messages currently use:

```python
uuid.uuid5(
    uuid.NAMESPACE_URL,
    f"msg-{sender}-{sent_at}",
)
```

Links currently use:

```python
uuid.uuid5(
    uuid.NAMESPACE_URL,
    f"link-{url}",
)
```

Every Qdrant search result contains this value as `hit.id`. The current production result formatter discards it, but the evaluation collector should preserve it:

```python
hits = vector_store.search(query, limit=candidate_depth)

for rank, hit in enumerate(hits, start=1):
    payload = hit.payload or {}
    candidate_id = f"{payload['type']}:{hit.id}"

    result = {
        "candidate_id": candidate_id,
        "candidate_type": payload["type"],
        "rank": rank,
        "score": float(hit.score),
        "text": payload.get("text") or payload.get("summary") or "",
    }
```

Example IDs:

```text
message:3db65f45-0a67-5d5f-b130-090cc2404d6a
link:9da1782d-f192-5027-82d3-c7f86b83e44a
```

### Current ID limitation

The current Qdrant message UUID contains only the sender and timestamp. It does not include:

- Source or chat ID
- Original message text
- Duplicate occurrence
- Telegram message ID

Two messages from the same sender at the same timestamp could therefore collide.

For the first frozen pilot, the existing Qdrant point ID is acceptable because it maps directly to the indexed vector.

The durable solution is to generate one canonical `document_id` during ingestion and store it in:

- The Qdrant point or payload
- The Neo4j node
- Search API results
- Evaluation artifacts

A stronger identity can be derived from:

```text
source_id + timestamp + sender + normalized original text + occurrence
```

The incremental import subsystem already uses this concept. Do not use Neo4j internal node IDs as golden identifiers because those IDs are storage-specific and may change.

## Selecting the 50 Queries

Write the queries before inspecting search results. Otherwise, the benchmark can become biased toward questions the current system already handles well.

Use this initial mix:

| Query group | Count |
| --- | ---: |
| Natural questions you would genuinely ask | 20 |
| Known message or link lookup | 10 |
| Conceptual or paraphrased questions | 10 |
| Known difficult questions | 5 |
| Deliberately unanswerable questions | 5 |
| **Total** | **50** |

Example queries:

- What was the recommended alternative to Pinecone?
- Who mentioned experience building vector indexes?
- What was the reason for using Qdrant?
- Find the report about the state of AI.
- Which tool helps manage multiple coding agents?

Avoid copying the exact wording of the stored message. Use the wording a real person would remember and type later.

## Labelling Instructions

Create a temporary annotation sheet with these columns:

```csv
query_id,query_text,candidate_id,candidate_type,content,relevance,reason
```

Do not show the annotator:

- Cosine score
- Current rank
- Retrieval source
- Graph score
- Reranker score
- System or run name

For every query-candidate pair, ask:

> If this result appeared in response to the question, how useful would it be for answering the question?

Then assign relevance 0, 1, 2, or 3.

When uncertain between two grades, record the reason. After the first ten queries, review the labels for consistency before continuing.

If possible, have another person independently label 20% to 25% of the pairs. Resolve important disagreements, especially when one label is 0 or 1 and the other is 2 or 3.

## Run A: Vector-Only Cosine Baseline

For the clean baseline:

1. Load the frozen queries.
2. Call `VectorStore.search()` directly.
3. Do not call the hybrid `search()` function.
4. Retrieve every point if the corpus still has only 16 points.
5. Record the Qdrant point ID, candidate type, cosine score, rank, payload, and latency.
6. Save the results as `runs.jsonl` with `run_id=vector_cosine_v1`.
7. Join the results with the frozen qrels.
8. Calculate metrics at five results.

### Evaluate the score threshold separately

Produce two baseline reports:

1. Raw cosine ranking without the `0.25` threshold.
2. Current product behavior with `score >= 0.25`.

This separates poor ranking from threshold-related failures.

## Evaluation Metrics

### Candidate Success@N

The percentage of queries whose candidate pool contains at least one grade-2-or-3 result:

```text
Candidate Success@N =
queries with a relevant candidate in top N
/
total queries
```

If this value is low, candidate retrieval is weak. A reranker cannot repair a query when no relevant document is present in its candidate pool.

### Success@5

The percentage of queries whose displayed top five contains at least one grade-2-or-3 result:

```text
Success@5 =
queries with a relevant result in top 5
/
total queries
```

### MRR@5

MRR measures how early the first useful result appears:

- Relevant at rank 1: `1.0`
- Relevant at rank 2: `0.5`
- Relevant at rank 3: `0.333`
- No relevant result in the first five: `0`

### nDCG@5

nDCG evaluates the complete graded ordering:

- Grade 3 should rank ahead of grade 2.
- Grade 2 should rank ahead of grade 1.
- The ideal ordering approaches `1.0`.
- Poor ordering approaches `0`.

Use graded `nDCG@5` as the primary metric.

### Recall@5

When the corpus is labelled exhaustively:

```text
Recall@5 =
relevant documents retrieved in top 5
/
all relevant documents in the corpus
```

## Failure Classification

Assign each query one diagnostic status:

| Status | Meaning | Likely next action |
| --- | --- | --- |
| No corpus answer | No relevant document exists | Improve ingestion or data coverage |
| Not vectorized | A relevant record exists but is absent from Qdrant | Fix indexing rules |
| Cosine retrieval miss | A relevant vector exists but is outside candidate depth | Improve retrieval or embeddings |
| Ranking miss | A relevant candidate exists but is below the top five | Reranker opportunity |
| Threshold miss | A relevant result was removed by the `0.25` cutoff | Tune the threshold |
| Success | A relevant result appears in the top five | Protect against regression |

This breakdown shows which technology can fix each failure. Aggregate metrics alone cannot provide that diagnosis.

## The Four Controlled Experiments

### Run A: Vector and cosine similarity

```text
Qdrant candidates -> cosine ordering
```

This is the baseline.

### Run B: Vector and graph retrieval

```text
Qdrant candidates + graph candidates -> declared fusion rule
```

Report two separate effects:

1. Did graph retrieval introduce relevant candidates that vector retrieval missed?
2. Did the fusion rule place those candidates usefully in the top five?

The current constant-score graph implementation should be recorded as `graph_lexical_v0`, not presented as mature relationship-aware graph retrieval.

### Run C: Vector and reranker

```text
The same Qdrant candidates as Run A -> reranker ordering
```

Run C must receive the exact same candidate IDs as Run A. Candidate Success@N must therefore be identical. Any top-five improvement can be attributed to the reranker.

### Run D: Vector, graph, and reranker

```text
Union of vector and graph candidates -> reranker ordering
```

This measures whether graph retrieval introduces useful candidates and whether the reranker orders the combined pool effectively.

## Comparing Runs Fairly

For every comparison, freeze:

- Query set
- Test split
- Corpus version
- Embedding model
- Filters
- Candidate depth
- Final result count
- Relevance judgments

The only intended differences should be the candidate source or ranking method named by the run.

For Run A versus Run C, the candidate IDs must be identical. This is the clean causal test of reranking.

For graph runs, report both candidate recall and final ranking quality. Otherwise, graph candidate improvements and fusion-score changes become mixed together.

## Recommended Report Format

The first trustworthy baseline report should contain:

```text
Corpus version: pilot-001
Queries: 50
Documents: <count>
Vectorized documents: <count>

Candidate Success@N: <value>
Success@5: <value>
MRR@5: <value>
nDCG@5: <value>
Recall@5: <value>

Failure breakdown:
- No corpus answer: <count>
- Not vectorized: <count>
- Cosine retrieval miss: <count>
- Ranking miss: <count>
- Threshold miss: <count>
- Success: <count>
```

When systems are compared, report:

| Metric | Baseline | New system | Absolute change |
| --- | ---: | ---: | ---: |
| Candidate Success@N | | | |
| Success@5 | | | |
| MRR@5 | | | |
| nDCG@5 | | | |
| Recall@5 | | | |
| p50 latency | | | |
| p95 latency | | | |

Also report:

- Number of queries improved
- Number unchanged
- Number regressed
- Baseline ranking failures rescued
- Existing baseline successes lost

With 50 queries, treat the results as a pilot. Confidence intervals may still be wide. Expand the test set after the workflow is stable.

## First Implementation Milestone

Complete these steps in order:

1. Start Qdrant and verify the live point count.
2. Export a frozen document manifest with stable candidate IDs.
3. Write five natural queries before inspecting their results.
4. Label every eligible document for those five queries.
5. Replace the dummy collector with a real vector-only collector.
6. Produce the first `vector_cosine_v1` run.
7. Calculate Success@5, MRR@5, nDCG@5, and Recall@5.
8. Classify every failed query.
9. Correct any collection, identity, or metric problems.
10. Expand from five to 50 queries.
11. Freeze 20 queries as the untouched test set.

Once this baseline is frozen, graph retrieval and reranking improvements can be measured against stable human judgments instead of subjective inspection.
