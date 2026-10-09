"""Build the blind annotation pool (cosine top-N ∪ BM25 top-N ∪ keyword hits)."""

import csv
import random
import re
from pathlib import Path

from evals.bm25 import BM25
from evals.common import ANNOTATION_COLUMNS

SHUFFLE_SEED = 42
COSINE_TOP = 30
BM25_TOP = 20


def keyword_pattern(keyword: str) -> re.Pattern:
    """Whole-token, case-insensitive match (so 'moe' does not hit 'Samoed')."""
    return re.compile(r"(?<!\w)" + re.escape(keyword.strip()) + r"(?!\w)", re.IGNORECASE)


def display_content(doc: dict) -> str:
    """What the annotator reads. Links get their title/URL; no scores or ranks."""
    if doc["candidate_type"] == "link":
        head = "\n".join(p for p in (doc.get("title"), doc.get("url")) if p)
        return (head + "\n" + doc["embedded_text"]).strip()
    return doc["embedded_text"]


def select_pool(query_text: str, docs: list[dict], cosine_ids: list[str],
                bm25: BM25, keywords: list[str],
                cosine_top: int = COSINE_TOP, bm25_top: int = BM25_TOP) -> dict[str, dict]:
    """Return {candidate_id: provenance} for the union of the three sources."""
    pool: dict[str, dict] = {}

    def add(cid: str, source: str, **extra):
        entry = pool.setdefault(cid, {"sources": [], "cosine_rank": None, "bm25_rank": None})
        if source not in entry["sources"]:
            entry["sources"].append(source)
        entry.update({k: v for k, v in extra.items() if v is not None})

    for rank, cid in enumerate(cosine_ids[:cosine_top], start=1):
        add(cid, f"cosine_top{cosine_top}", cosine_rank=rank)
    for rank, (idx, _score) in enumerate(bm25.top(query_text, bm25_top), start=1):
        add(docs[idx]["candidate_id"], f"bm25_top{bm25_top}", bm25_rank=rank)
    for kw in keywords:
        pat = keyword_pattern(kw)
        for d in docs:
            if pat.search(d["embedded_text"]):
                add(d["candidate_id"], f"keyword:{kw}")
    return pool


def shuffle_pool(candidate_ids: list[str], cosine_order: list[str],
                 rng: random.Random) -> list[str]:
    """Deterministic shuffle that never lands on the cosine order (blindness)."""
    ids = sorted(candidate_ids)
    reference = [c for c in cosine_order if c in set(ids)] + \
                [c for c in ids if c not in set(cosine_order)]
    for _ in range(20):
        rng.shuffle(ids)
        if len(ids) < 2 or ids != reference:
            break
    return ids


def annotation_rows(query: dict, ordered_ids: list[str],
                    docs_by_id: dict[str, dict]) -> list[dict]:
    return [{
        "query_id": query["query_id"],
        "query_text": query["query_text"],
        "candidate_id": cid,
        "candidate_type": docs_by_id[cid]["candidate_type"],
        "content": display_content(docs_by_id[cid]),
        "relevance": "",
        "reason": "",
    } for cid in ordered_ids]


def write_annotation_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def verify_annotation(path: Path, cosine_order_by_query: dict[str, list[str]]) -> list[str]:
    """Acceptance test 6: exact columns, and row order differs from rank order."""
    problems = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ANNOTATION_COLUMNS:
            problems.append(f"columns {reader.fieldnames} != {ANNOTATION_COLUMNS}")
        by_query: dict[str, list[str]] = {}
        for row in reader:
            by_query.setdefault(row["query_id"], []).append(row["candidate_id"])
    for qid, ids in by_query.items():
        reference = [c for c in cosine_order_by_query.get(qid, []) if c in set(ids)]
        if len(ids) > 1 and ids == reference:
            problems.append(f"{qid}: annotation rows are in cosine rank order")
    return problems


def verify_must_pool(must_pool: dict[str, list[str]], docs: list[dict],
                     pools: dict[str, dict[str, dict]]) -> list[str]:
    """Acceptance test 7: every document containing a marker must be pooled.

    This guards the reason BM25 and keywords exist: a relevant document that
    cosine ranks below the cutoff (e.g. a long message) must still reach the
    annotator. A marker with no matching document, or a matching document
    missing from the pool, is a failure.
    """
    problems = []
    for qid, markers in must_pool.items():
        pool = pools.get(qid, {})
        for marker in markers:
            pat = re.compile(re.escape(marker), re.IGNORECASE)
            matches = [d for d in docs if pat.search(d["embedded_text"])]
            if not matches:
                problems.append(f"{qid}: no document contains marker {marker!r}")
                continue
            for d in matches:
                if d["candidate_id"] not in pool:
                    problems.append(f"{qid}: {d['candidate_id']} ({marker!r}) is not in the pool")
    return problems
