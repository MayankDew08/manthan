# Manthan metrics report

_generated 2026-08-18 09:44 UTC_

## 1. Pipeline funnel

- **Source:** Private chat export (redacted) — **146** messages parsed
- **Kept → grading:** 131
- **Dropped (heuristic):** 15 (**drop 10.3%**)

| drop reason | count | share |
|---|---|---|
| media_only_no_caption | 15 | 100% |

## 2. Grading

**Quality distribution (1–5):**
- q1: **5** `#####`
- q2: **19** `###################`
- q3: **16** `################`
- q4: **66** `##################################################################`
- q5: **25** `#########################`
- confidence: mean **0.893** (min 0.65, max 1.00)
- avg topics/message: **2.9**

**Categories:**
- introduction: 52
- resource: 26
- insight: 18
- discussion: 13
- experience: 7
- social: 5
- meta: 3
- question: 3
- noise: 3
- validation: 1

## 3. Enrichment

- records: **49** (49 with links)
- total links: **57** (avg 1.16/record)
- avg entities/record: **1.7**
- avg topics/record: **2.4**
- records with link_intent: **49/49**
- link previews: {'blocked': 19, 'scraped': 36, 'ask_user': 2}

## 4. Scraped links

- links: **35**
- raw_text chars: mean **9181** (min 133, max 46694)
- `summary` present in **35/35**
- `what_it_is` present in **35/35**
- `problem_solved` present in **35/35**
- `how_useful` present in **35/35**
- avg topics=3.0, avg entities=4.3

## 5. Blocked / ask-user

- blocked: **19** {'not_found': 4, 'failed': 15}
- ask-user: **2** {'auth_required': 2}

## 6. Tokens & latency

**Measured (from this run, via llama.cpp `usage` + client timing):**
- LLM calls: **122**
- prompt tokens: **241019**
- completion tokens: **23694**
- total tokens: **264713**
- LLM wall time: **772.837s** (avg 6.335s/call, max 26.845s)

**Pipeline stats:**
- scraped: 27
- blocked: 15
- ask_user: 2
- skipped_existing: 1
- no_links: 53

**Estimated (char-based, `(len+3)//4`, from stored files):**
- system prompts: **4351**
- grading input: **14501**
- enrichment input: **3359**
- summarization input: **80344**

## 7. Storage

**Neo4j nodes:**
- Topic: 43
- Entity: 42
- Link: 12
- Message: 11
- Person: 6
- relationships: 116
**Qdrant:**
- points: 16
- vectors: 384

---

# Appendix A — Final sender-free re-ingest (2026-10-03)

_Sender-free pipeline (`remove-sender`): message identity, Neo4j/Qdrant schemas,
API, workers and Go bridge carry no sender; `Person` nodes and `SENT` edges removed.
Sections 1–7 above (2026-08-18 run) are preserved unchanged._

## A.1 Files ingested

Two Drive ZIPs sharing one display name (distinct file IDs); four other tracked
files skipped as unchanged (revision match, zero Gemma calls):

| File | Parsed | Kept | Dropped | Low quality | Stored |
|---|---|---|---|---|---|
| export A (358 msgs) | 358 | 340 | 18 (`media_only_no_caption`) | 233 | 107 |
| export B (316 msgs) | 316 | 301 | 15 (`media_only_no_caption`) | 203 | 98 |
| **Run total** | **674** | **641** | **33** | **436** | **205** |

Final-file quality distribution (301 grades): q1 **120** / q2 **66** / q3 **17** / q4 **70** / q5 **28**.

## A.2 Grading batches and retries

- Pass 1: 68 + 61 batches, max 5/batch, 25717-token budget; **58 batches resumed
  from checkpoint (0 LLM calls)**.
- One batch returned 10 grade objects for 5 messages and was retried with an
  explicit count reminder before succeeding (`grade_batch` retry, max 3 attempts).
- Pass 2 re-graded quality-3 messages one at a time as in the original run.

## A.3 Tokens (process-cumulative, both files)

- Pass 1: 217 LLM calls, 357323 in / 30022 out (387345 total)
- Pass 1+2: 282 LLM calls, 406971 in / 34949 out (441920 total)
- Run total incl. summaries: **287 LLM calls, 444366 total tokens**

## A.4 Cumulative store state after this run

- SQLite: **1017** processed rows (51 dropped / 658 low-quality / 308 stored),
  **6/6** sources revisioned; page tokens and watch rows preserved.
- **Neo4j:** 145 Messages, 68 Links, **0 Persons, 0 SENT edges, 0 sender props**.
- **Qdrant** (`manthan-fresh`, 384-dim cosine): **144 points**, status green.

---

# Appendix B — Clean rebuild and re-ingest (2026-10-09)

_Sections 1–7 and Appendix A above are preserved unchanged. This appendix records
the Phase 0 rebuild from `docs/retrieval-eval-run-a-spec.md`. It contains ingestion
and store metrics only: **no retrieval-evaluation numbers exist yet**. Run A metrics
come after the corpus is frozen and human-labelled._

## B.1 Why a rebuild

The working branch had not merged the sender removal, so imports re-wrote `sender`
and stored every message twice under two point-ID schemes. Separately, the parser
turned WhatsApp system events ("+NN … joined using a group link") into messages,
leaking members' phone numbers into message text. Both were fixed before rebuilding
(parser skips system events; the importer's temp-file serializer emits a
placeholder author so its own output still round-trips).

| Store | Before rebuild | Action |
|---|---|---|
| Qdrant `manthan-fresh` | 193 points, 148 unique contents, 46 with `sender`, 3 phone-number texts | Snapshot, then collection deleted |
| Neo4j | 193 Messages / 149 unique, 47 with `sender`, 2 Persons, 3 phone-number events | Persons + `sender` removed; 3 events and 44 duplicate nodes deleted → 146 |
| SQLite | 1,032 processed rows, 10 sources | Backed up; processed rows wiped, revisions reset |

## B.2 Files ingested

| File | Parsed | Dropped (heuristic) | Low quality | Stored | Note |
|---|---:|---:|---:|---:|---|
| Workflows export, Oct 7 (upload 2) | 164 | 16 | 42 | 106 | Valid run |
| Workflows export, Oct 7 (upload 1) | 164 | 16 | 148 | 0 | **Invalid**, see B.4 |
| Test group exports (3 uploads) | 15 | 0 | 6 | 9 | Same 5 messages each |
| **Run total** | **343** | **32** | **196** | **115** | |

Not ingested: the **Sep 9 Workflows export** (still live in Drive). The change
listener only imports files that changed, and this file did not. It holds
10 messages absent from the Oct 7 export (145 shared, 19 newer only in Oct 7).

Valid-run grade distribution (148 kept): pass 1 q1 **4** / q2 **22** / q3 **31** /
q4 **60** / q5 **31**; after pass 2 q3 **16** / q4 **75** (15 promoted), so
**106** at quality ≥ 4.

## B.3 Tokens and latency (process-cumulative, `last_run.json`)

- **120 LLM calls**, 196,969 in / 18,411 out (**215,380 total**)
- LLM time 469.7 s; mean 3.91 s per call, max 12.8 s
- Enrichment: 1 link scraped, 2 queued for manual paste, 45 already known, 65 messages without links

## B.4 Defect found: stale graph resume

Upload 1 of the Oct 7 export was "graded" in 12 s: all 148 messages got quality 1
with the reason "The message is empty". The identical upload 2 graded normally
(106 stored). Cause: the LangGraph thread ID is derived from `source_id + revision`.
An attempt this morning on pre-merge code left that thread unfinished with empty
texts. After the rebuild reset revisions, the re-import mapped to the same thread
and **resumed the stale state instead of grading**. The importer's guard compares
counts only (148 = 148), so it passed.

Impact: upload 1's 148 SQLite rows are wrong. Qdrant and Neo4j are unaffected,
because the identical content was graded and stored correctly by upload 2.

## B.5 Store state after this run

- **SQLite:** 343 processed rows (32 dropped / 196 low-quality / 115 stored); 5 of 12 sources revisioned; page tokens and watch rows preserved.
- **Neo4j:** 156 Messages, 72 Links, **0 Persons, 0 `sender` props, 0 duplicate nodes, 0 phone-number events, 0 empty texts**. Neo4j still holds messages from earlier (now trashed) exports, because the migration never deletes Message nodes, so its coverage is wider than Qdrant's.
- **Qdrant** (`manthan-fresh`, 384-dim cosine): **148 points** (118 messages, 30 links), **0 `sender` keys, 0 PII-gate matches**; 1 repeated text (the same message posted a minute apart).

## B.6 Post-run corrections (same day)

- **Stale-resume and trashed-file defects fixed** (regression tests added). The
  batch path now carries a content digest, and the importer rejects grades whose
  texts do not match its batch. Drive listing and change handling skip trashed files.
- **Bogus rows removed.** Upload 1's 164 rows (148 bogus "low quality" + 16
  dropped) and its source row were deleted, after a SQLite backup. Qdrant and
  Neo4j were untouched.
- **SQLite now:** 179 processed rows (16 dropped / 48 low-quality / 115 stored).
  B.5's totals describe the run as it happened.
- **Pending owner actions in Drive:** trash the Sep 9 export (decision: live Oct 7
  export is the source of truth) and the duplicate Oct 7 upload 1.
