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
