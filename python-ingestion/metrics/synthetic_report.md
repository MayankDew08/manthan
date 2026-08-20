# Manthan metrics report

_generated 2026-08-17 09:28 UTC_

## 1. Pipeline funnel

- **Source:** Synthetic test fixture — **44** messages parsed
- **Kept → grading:** 30
- **Dropped (heuristic):** 14 (**drop 31.8%**)

| drop reason | count | share |
|---|---|---|
| short_acknowledgment | 9 | 64% |
| no_alphanumeric | 3 | 21% |
| media_only_no_caption | 1 | 7% |
| empty | 1 | 7% |

## 2. Grading

**Quality distribution (1–5):**
- q1: **5** `#####`
- q2: **8** `########`
- q3: **1** `#`
- q4: **10** `##########`
- q5: **6** `######`
- confidence: mean **0.947** (min 0.85, max 1.00)
- avg topics/message: **2.7**

**Categories:**
- resource: 11
- social: 5
- question: 4
- experience: 4
- validation: 2
- introduction: 2
- insight: 2

## 3. Enrichment

- records: **8** (8 with links)
- total links: **9** (avg 1.12/record)
- avg entities/record: **1.1**
- avg topics/record: **1.2**
- records with link_intent: **8/8**
- link previews: {'blocked': 3, 'scraped': 6}

## 4. Scraped links

- links: **6**
- raw_text chars: mean **7085** (min 381, max 17585)
- `summary` present in **6/6**
- `what_it_is` present in **6/6**
- `problem_solved` present in **6/6**
- `how_useful` present in **6/6**
- avg topics=3.0, avg entities=4.3

## 5. Blocked / ask-user

- blocked: **3** {'not_found': 3}
- ask-user: none

## 6. Tokens & latency

**Measured:** none stored yet — run the pipeline on actual data to populate `metrics/last_run.json`.

**Estimated (char-based, `(len+3)//4`, from stored files):**
- system prompts: **4351**
- grading input: **1336**
- enrichment input: **215**
- summarization input: **10632**

## 7. Storage

**Neo4j nodes:**
- (storage query error: ServiceUnavailable)
**Qdrant:**
- error: ResponseHandlingException
