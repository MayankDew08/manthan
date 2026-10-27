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


## Appendix C — Retrieval ladder

n=22 answerable queries.

### Per-step metrics (answerable queries)

| Step | nDCG@5 | Success@5 | MRR@5 | Candidate Success@30 (lower bound) |
|------|--------|-----------|-------|-----------------------------------|
| ladder_s0_cosine | 0.5643 | 0.8636 | 0.7159 | 1.0000 |
| ladder_s1_cos_bm25 | 0.6765 | 0.9545 | 0.8220 | 1.0000 (lower bound) |
| ladder_s2_cos_bm25_rr | 0.7685 | 1.0000 | 0.9167 | 1.0000 (lower bound) |
| ladder_s3_cos_bm25_graph | 0.7020 | 0.9545 | 0.8523 | 1.0000 (lower bound) |
| ladder_s4_cos_bm25_graph_rr | 0.7724 | 1.0000 | 0.9129 | 1.0000 (lower bound) |

### Pairwise comparisons

**ladder_s0_cosine → ladder_s1_cos_bm25**
- Verdict: improved
- Win/tie/loss: 12 / 4 / 6 (improved: ['q004', 'q005', 'q006', 'q007', 'q008', 'q009', 'q011', 'q012', 'q015', 'q017', 'q020', 'q021']; regressed: ['q001', 'q002', 'q010', 'q016', 'q018', 'q025'])
- Bootstrap (n=22): mean Δ=0.1122, CI=[0.0073, 0.2160]
- Prediction: expected "no detectable difference" (BM25 should help exact-name queries (q006 GitLost, q013 Anima, q014-q017 people and companies), but cosine likely already finds several of them, and RRF can also demote a strong cosine #1 that BM25 ranks low. Expect a small positive mean delta whose CI crosses 0 at n=22, wins concentrated in person_lookup/temporal, and Candidate Success@30 >= step 0.) — ✗

**ladder_s1_cos_bm25 → ladder_s2_cos_bm25_rr**
- Verdict: improved
- Win/tie/loss: 12 / 7 / 3 (improved: ['q001', 'q002', 'q004', 'q006', 'q008', 'q009', 'q010', 'q011', 'q013', 'q015', 'q020', 'q025']; regressed: ['q003', 'q012', 'q021'])
- Bootstrap (n=22): mean Δ=0.0920, CI=[0.0282, 0.1565]
- Prediction: expected "improved" (The cross-encoder reads query and document together, up to 512 tokens (26 of the 30 truncated documents fully), and should demote the self-introduction distractors that sit in cosine's top 5 on non-person queries. Risk: informal WhatsApp text is far from MS MARCO, so the gain could be smaller than on web passages.) — ✓

**ladder_s1_cos_bm25 → ladder_s3_cos_bm25_graph**
- Verdict: no detectable difference at n=22
- Win/tie/loss: 8 / 7 / 7 (improved: ['q001', 'q002', 'q004', 'q007', 'q010', 'q011', 'q018', 'q020']; regressed: ['q003', 'q005', 'q006', 'q009', 'q012', 'q021', 'q025'])
- Bootstrap (n=22): mean Δ=0.0256, CI=[-0.0363, 0.0882]
- Prediction: expected "no detectable difference" (graph_lexical_v0 is substring keyword matching with no IDF, so it repeats BM25's signal, and does it worse (generic terms like 'group', 'when', 'first', 'llm' match a large share of messages). Expect at most 1-2 queries to change, Candidate Success@30 equal or +1, raw graph counts above 100 for queries with generic words, and a non-zero out-of-corpus count.) — ✓

**ladder_s3_cos_bm25_graph → ladder_s4_cos_bm25_graph_rr**
- Verdict: improved
- Win/tie/loss: 10 / 6 / 6 (improved: ['q001', 'q004', 'q005', 'q006', 'q008', 'q009', 'q010', 'q013', 'q015', 'q025']; regressed: ['q002', 'q003', 'q007', 'q012', 'q018', 'q021'])
- Bootstrap (n=22): mean Δ=0.0704, CI=[0.0001, 0.1441]
- Prediction: expected "improved" (Same mechanism as s1->s2: re-ordering 30 candidates with a cross-encoder.) — ✓

**ladder_s2_cos_bm25_rr → ladder_s4_cos_bm25_graph_rr**
- Verdict: no detectable difference at n=22
- Win/tie/loss: 1 / 19 / 2 (improved: ['q001']; regressed: ['q005', 'q007'])
- Bootstrap (n=22): mean Δ=0.0039, CI=[-0.0068, 0.0202]
- Prediction: expected "no detectable difference" (This is the graph's effect once a re-ranker is present. The graph should add few relevant candidates that cosine+BM25 missed, and the re-ranker should push any graph noise it adds out of the top 5.) — ✓

**ladder_s0_cosine → ladder_s4_cos_bm25_graph_rr**
- Verdict: improved
- Win/tie/loss: 14 / 3 / 5 (improved: ['q001', 'q004', 'q005', 'q006', 'q007', 'q008', 'q009', 'q011', 'q012', 'q013', 'q015', 'q017', 'q020', 'q021']; regressed: ['q002', 'q003', 'q016', 'q018', 'q025'])
- Bootstrap (n=22): mean Δ=0.2082, CI=[0.0938, 0.3344]
- Prediction: expected "improved" (End to end, driven mainly by the re-ranker, not by BM25 or the graph.) — ✓

### Abstention (no-answer queries)

**ladder_s0_cosine**
- No-answer top-1 scores: {'q022': 0.3345571, 'q023': 0.5514544, 'q024': 0.5420687}
- Answerable top-1 scores: {'q001': 0.7000346, 'q002': 0.5093109, 'q003': 0.7103596, 'q004': 0.62046635, 'q005': 0.4320637, 'q006': 0.5195841, 'q007': 0.47300118, 'q008': 0.50330126, 'q009': 0.36281174, 'q010': 0.5073825, 'q011': 0.50927067, 'q012': 0.57647854, 'q013': 0.5465317, 'q014': 0.6848414, 'q015': 0.42414474, 'q016': 0.7518954, 'q017': 0.6363015, 'q018': 0.475621, 'q019': 0.7355914, 'q020': 0.46584958, 'q021': 0.4892819, 'q025': 0.38170046}
- Median no-answer: 0.5421
- Median answerable: 0.5093

**ladder_s2_cos_bm25_rr**
- No-answer top-1 scores: {'q022': -9.282022, 'q023': -1.302, 'q024': -5.894752}
- Answerable top-1 scores: {'q001': 7.062949, 'q002': -3.154519, 'q003': 4.428553, 'q004': 5.228491, 'q005': 4.995416, 'q006': -4.447292, 'q007': -4.482089, 'q008': -0.102569, 'q009': -7.611075, 'q010': -0.658157, 'q011': 3.054396, 'q012': 0.934486, 'q013': 7.686624, 'q014': 7.979046, 'q015': 3.743757, 'q016': 9.07021, 'q017': 7.18703, 'q018': 2.16608, 'q019': 3.233656, 'q020': 4.190549, 'q021': -2.751807, 'q025': -8.997515}
- Median no-answer: -5.8948
- Median answerable: 3.1440

**ladder_s4_cos_bm25_graph_rr**
- No-answer top-1 scores: {'q022': -9.282022, 'q023': -1.302, 'q024': -5.894752}
- Answerable top-1 scores: {'q001': 7.062949, 'q002': -3.154518, 'q003': 4.428553, 'q004': 5.228491, 'q005': 4.995416, 'q006': -4.447292, 'q007': -4.482089, 'q008': -0.102569, 'q009': -7.611075, 'q010': -0.658157, 'q011': 3.054396, 'q012': 0.934486, 'q013': 7.686624, 'q014': 7.979046, 'q015': 3.743757, 'q016': 9.07021, 'q017': 7.18703, 'q018': 2.166078, 'q019': 3.233656, 'q020': 4.190549, 'q021': -2.751807, 'q025': -8.997515}
- Median no-answer: -5.8948
- Median answerable: 3.1440

### End-to-end latency (estimate across machines)

- Machine: x86_64, 16 CPUs, GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- cosine_qdrant: n=25, p50=8.1ms, p95=10.1ms
- graph_neo4j: n=25, p50=24.0ms, p95=60.4ms
- bm25: n=25, p50=0.1ms, p95=0.1ms
- rrf: n=25, p50=0.0ms, p95=0.0ms
- rerank_cpu: n=25, p50=52.8ms, p95=70.0ms
- rerank_gpu: n=25, p50=6.0ms, p95=14.4ms

## Appendix D — Graph as candidate generator

Disclosure (D18): the planner drafted the 8 questions (2026-10-23) from documents.jsonl text only, and the owner adjusted them. The planner had already seen connector degrees (section 2.3) before drafting.

grader view (D24): the original-set top-up sheet shows doc text only, as Run A. The graph-set sheet shows message text plus the titles of its in-corpus shared links, in candidate_id order.

{
  "sets": {
    "graph": {
      "comparisons": [
        {
          "base": "ladder_s2_cos_bm25_rr",
          "ci_high": 0.016210505188746574,
          "ci_low": 0.0,
          "mean_delta": 0.005403501729582191,
          "n": 8,
          "new": "ctrl_c0_deeper_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "no detectable difference at n=8"
        },
        {
          "base": "ladder_s2_cos_bm25_rr",
          "ci_high": 0.2750275597477409,
          "ci_low": 0.028849971431620824,
          "mean_delta": 0.15269355186461558,
          "n": 8,
          "new": "ctrl_c1_denorm_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "improved"
        },
        {
          "base": "ladder_s2_cos_bm25_rr",
          "ci_high": 0.0,
          "ci_low": 0.0,
          "mean_delta": 0.0,
          "n": 8,
          "new": "ctrl_c2_payload_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "no detectable difference at n=8"
        },
        {
          "base": "ctrl_c0_deeper_rr",
          "ci_high": 0.0749121859030277,
          "ci_low": -0.166580901731833,
          "mean_delta": -0.060527123408201725,
          "n": 8,
          "new": "graph_b2r_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "no detectable difference at n=8"
        },
        {
          "base": "ctrl_c1_denorm_rr",
          "ci_high": -0.09161987847648576,
          "ci_low": -0.3314044290196765,
          "mean_delta": -0.20781717354323512,
          "n": 8,
          "new": "graph_b2r_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "regressed"
        },
        {
          "base": "ctrl_c2_payload_rr",
          "ci_high": 0.08062908862830664,
          "ci_low": -0.16521416094842944,
          "mean_delta": -0.05512362167861953,
          "n": 8,
          "new": "graph_b2r_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "no detectable difference at n=8"
        },
        {
          "base": "ctrl_c0_deeper_rr",
          "ci_high": 0.018723226184554284,
          "ci_low": -0.016210505188746574,
          "mean_delta": 0.0008375736652692367,
          "n": 8,
          "new": "graph_b2q_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "no detectable difference at n=8"
        },
        {
          "base": "ctrl_c1_denorm_rr",
          "ci_high": -0.0249712965634815,
          "ci_low": -0.26878648435288943,
          "mean_delta": -0.14645247646976417,
          "n": 8,
          "new": "graph_b2q_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "regressed"
        },
        {
          "base": "ctrl_c2_payload_rr",
          "ci_high": 0.018723226184554284,
          "ci_low": 0.0,
          "mean_delta": 0.006241075394851428,
          "n": 8,
          "new": "graph_b2q_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "no detectable difference at n=8"
        },
        {
          "base": "graph_b1r_rr",
          "ci_high": 0.0749121859030277,
          "ci_low": -0.166580901731833,
          "mean_delta": -0.060527123408201725,
          "n": 8,
          "new": "graph_b2r_ctx_rr",
          "owner_prediction": "improved",
          "owner_prediction_held": false,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "no detectable difference at n=8"
        },
        {
          "base": "graph_b1q_rr",
          "ci_high": 0.018723226184554284,
          "ci_low": 0.0,
          "mean_delta": 0.006241075394851428,
          "n": 8,
          "new": "graph_b2q_ctx_rr",
          "owner_prediction": "improved",
          "owner_prediction_held": false,
          "planner_prediction": null,
          "planner_prediction_held": null,
          "verdict": "no detectable difference at n=8"
        }
      ],
      "decisions": {
        "q": {
          "branch": "GRAPH OUT",
          "step1_deeper_pool": false,
          "step2_graph_over_c0": false,
          "step3_graph_over_shortcuts": false
        },
        "r": {
          "branch": "GRAPH OUT",
          "step1_deeper_pool": false,
          "step2_graph_over_c0": false,
          "step3_graph_over_shortcuts": false
        }
      },
      "metrics": {
        "ctrl_c0_deeper_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.6892438994529767,
          "recall_pool": 0.8556818181818182
        },
        "ctrl_c1_denorm_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.8365339495880101,
          "recall_pool": 0.9772727272727273
        },
        "ctrl_c2_payload_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.6838403977233944,
          "recall_pool": 0.7401515151515151
        },
        "graph_b1q_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.6838403977233944,
          "recall_pool": 0.7818181818181819
        },
        "graph_b1r_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.6892438994529767,
          "recall_pool": 0.9772727272727273
        },
        "graph_b2q_ctx_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.690081473118246,
          "recall_pool": 0.7818181818181819
        },
        "graph_b2r_ctx_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.6287167760447749,
          "recall_pool": 0.9772727272727273
        },
        "ladder_s2_cos_bm25_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.6838403977233944,
          "recall_pool": 0.7401515151515151
        }
      },
      "path_split": {
        "q": {
          "hits": 0,
          "shares": {}
        },
        "r": {
          "hits": 0,
          "shares": {}
        }
      }
    },
    "original": {
      "comparisons": [
        {
          "base": "ladder_s2_cos_bm25_rr",
          "ci_high": 0.025358738614439876,
          "ci_low": -0.005654354185717808,
          "mean_delta": 0.007636417285628141,
          "n": 22,
          "new": "ctrl_c0_deeper_rr",
          "owner_prediction": "improved",
          "owner_prediction_held": false,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "no detectable difference at n=22"
        },
        {
          "base": "ladder_s2_cos_bm25_rr",
          "ci_high": 0.07700503098647746,
          "ci_low": -0.0058334899679854495,
          "mean_delta": 0.03487196675021214,
          "n": 22,
          "new": "ctrl_c1_denorm_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "no detectable difference at n=22"
        },
        {
          "base": "ladder_s2_cos_bm25_rr",
          "ci_high": 0.0,
          "ci_low": 0.0,
          "mean_delta": 0.0,
          "n": 22,
          "new": "ctrl_c2_payload_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "no detectable difference at n=22"
        },
        {
          "base": "ctrl_c0_deeper_rr",
          "ci_high": -0.0033181599992767544,
          "ci_low": -0.17420303442183704,
          "mean_delta": -0.08639776803053834,
          "n": 22,
          "new": "graph_b2r_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "regressed"
        },
        {
          "base": "ctrl_c1_denorm_rr",
          "ci_high": -0.022134897145858482,
          "ci_low": -0.20810177495713605,
          "mean_delta": -0.11363331749512233,
          "n": 22,
          "new": "graph_b2r_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "regressed"
        },
        {
          "base": "ctrl_c2_payload_rr",
          "ci_high": 0.004693035435385707,
          "ci_low": -0.16643185699478033,
          "mean_delta": -0.07876135074491018,
          "n": 22,
          "new": "graph_b2r_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "no detectable difference at n=22"
        },
        {
          "base": "ctrl_c0_deeper_rr",
          "ci_high": 0.003017255426662249,
          "ci_low": -0.06094659624513776,
          "mean_delta": -0.02633307697195566,
          "n": 22,
          "new": "graph_b2q_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "no detectable difference at n=22"
        },
        {
          "base": "ctrl_c1_denorm_rr",
          "ci_high": -0.022677161305139023,
          "ci_low": -0.08922224456193145,
          "mean_delta": -0.05356862643653966,
          "n": 22,
          "new": "graph_b2q_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "regressed"
        },
        {
          "base": "ctrl_c2_payload_rr",
          "ci_high": 0.010631039939369797,
          "ci_low": -0.05463288229681051,
          "mean_delta": -0.018696659686327513,
          "n": 22,
          "new": "graph_b2q_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "no detectable difference at n=22"
        },
        {
          "base": "graph_b1r_rr",
          "ci_high": -0.002857385802283235,
          "ci_low": -0.17451888954429826,
          "mean_delta": -0.08625296780974491,
          "n": 22,
          "new": "graph_b2r_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "regressed"
        },
        {
          "base": "graph_b1q_rr",
          "ci_high": 0.010654402714060523,
          "ci_low": -0.05372100605001185,
          "mean_delta": -0.018345377523472196,
          "n": 22,
          "new": "graph_b2q_ctx_rr",
          "owner_prediction": null,
          "owner_prediction_held": null,
          "planner_prediction": "no detectable difference",
          "planner_prediction_held": false,
          "verdict": "no detectable difference at n=22"
        }
      ],
      "decisions": {
        "q": {
          "branch": "GRAPH OUT",
          "step1_deeper_pool": false,
          "step2_graph_over_c0": false,
          "step3_graph_over_shortcuts": false
        },
        "r": {
          "branch": "GRAPH OUT",
          "step1_deeper_pool": false,
          "step2_graph_over_c0": false,
          "step3_graph_over_shortcuts": false
        }
      },
      "metrics": {
        "ctrl_c0_deeper_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.7593152156688621,
          "recall_pool": 0.9057675657675657
        },
        "ctrl_c1_denorm_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.786550765133446,
          "recall_pool": 0.891163382072473
        },
        "ctrl_c2_payload_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.7516787983832338,
          "recall_pool": 0.8609116640934823
        },
        "graph_b1q_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.7513275162203786,
          "recall_pool": 0.8726726303999031
        },
        "graph_b1r_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.7591704154480685,
          "recall_pool": 0.9259959737232464
        },
        "graph_b2q_ctx_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.7329821386969063,
          "recall_pool": 0.8726726303999031
        },
        "graph_b2r_ctx_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.6729174476383236,
          "recall_pool": 0.9259959737232464
        },
        "ladder_s2_cos_bm25_rr": {
          "judged_at_5": 1.0,
          "ndcg_at_5": 0.7516787983832338,
          "recall_pool": 0.8609116640934823
        }
      },
      "path_split": {
        "q": {
          "hits": 0,
          "shares": {}
        },
        "r": {
          "hits": 0,
          "shares": {}
        }
      }
    }
  }
}
