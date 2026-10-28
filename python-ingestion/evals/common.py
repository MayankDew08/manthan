"""Shared helpers for the retrieval evaluation pipeline.

Kept free of heavy imports (no model or Qdrant client at import time) so the
pure logic can be unit-tested quickly.
"""

import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # python-ingestion/
REPO_ROOT = ROOT.parent
EVALS_DIR = ROOT / "evals"

DEFAULT_CORPUS_VERSION = "pilot-001"
DEFAULT_RUN_ID = "vector_cosine_v1"
DEFAULT_QUERIES = REPO_ROOT / "queries.csv"
KEYWORDS_FILE = EVALS_DIR / "pool_keywords.json"

# Minimum acceptable cosine between a re-embedded manifest text and the stored
# Qdrant vector. Anything lower means the manifest is not the index.
MIN_PARITY_COSINE = 0.9999

ANNOTATION_COLUMNS = ["query_id", "query_text", "candidate_id", "candidate_type",
                      "content", "relevance", "reason"]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Spec PII gate: run on text with URLs removed (arXiv ids and tweet status ids
# inside URLs are otherwise false positives).
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")


def load_env() -> None:
    """Load python-ingestion/.env so evals use the same stores as the app."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


def qdrant_settings() -> dict:
    return {
        "host": os.environ.get("QDRANT_HOST", "localhost"),
        "port": int(os.environ.get("QDRANT_PORT", "6333")),
        "collection": os.environ.get("QDRANT_COLLECTION", "manthan"),
    }


def data_dir(corpus_version: str, create: bool = True) -> Path:
    path = EVALS_DIR / "data" / corpus_version
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# ---- text helpers -----------------------------------------------------------

def strip_urls(text: str) -> str:
    return URL_RE.sub(" ", text or "")


def find_phone_numbers(text: str) -> list[str]:
    """Phone-number-like substrings of ``text`` after URLs are removed."""
    return [m.group().strip() for m in PHONE_RE.finditer(strip_urls(text))]


def mask_digits(value: str) -> str:
    """Hide all but the last two digits so reports never echo a number."""
    digits = [i for i, ch in enumerate(value) if ch.isdigit()]
    keep = set(digits[-2:])
    return "".join("#" if ch.isdigit() and i not in keep else ch
                   for i, ch in enumerate(value))


def embedded_text(payload: dict) -> str:
    """Rebuild the exact string vector_store.py embedded for this payload."""
    if payload.get("type") == "link":
        parts = (payload.get("summary") or "", payload.get("what_it_is") or "",
                 payload.get("problem_solved") or "")
        text = " ".join(p for p in parts if p)
    else:
        text = payload.get("text") or ""
    return text.strip()  # embedder.embed_text strips before encoding


def content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def looks_like_intro(text: str) -> bool:
    """Heuristic: a self-introduction message (the expected top distractor)."""
    head = (text or "").strip().lower()[:200]
    return bool(re.match(
        r"^(hi|hello|hey|namaste|good (morning|evening)|greetings)\b", head
    ) and re.search(r"\b(i['’]?m|i am|my name|here)\b", head)) or "introduce myself" in head


# ---- file helpers -----------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queries(path: Path = DEFAULT_QUERIES) -> list[dict]:
    """Read queries.csv; blank rows are skipped. Requires query_id + query_text."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("query_id") or "").strip()]
    for r in rows:
        if not (r.get("query_text") or "").strip():
            raise ValueError(f"query {r['query_id']} has no query_text")
    return rows


def load_documents(corpus_version: str) -> list[dict]:
    return read_jsonl(data_dir(corpus_version, create=False) / "documents.jsonl")


def load_run(corpus_version: str, run_id: str) -> list[dict]:
    return read_jsonl(data_dir(corpus_version, create=False) / "runs" / f"{run_id}.jsonl")


def load_keywords(path: Path = KEYWORDS_FILE) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"keywords": data.get("keywords", {}), "must_pool": data.get("must_pool", {})}


def load_qrels(path: Path, valid_ids: set[str] | None = None) -> dict[str, dict[str, dict]]:
    """Return {query_id: {candidate_id: {"relevance": int, "answer_bearing": int}}}.

    Unjudged documents are simply absent (callers treat them as grade 0).
    Raises on a grade outside 0-3 or a candidate_id missing from the corpus.
    """
    qrels: dict[str, dict[str, dict]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            qid, cid = row["query_id"].strip(), row["candidate_id"].strip()
            raw = (row.get("relevance") or "").strip()
            if raw == "":
                continue  # blank grade = not judged yet
            try:
                grade = int(raw)
            except ValueError:
                raise ValueError(f"{path.name} line {n}: relevance {raw!r} is not an integer")
            if grade not in (0, 1, 2, 3):
                raise ValueError(f"{path.name} line {n}: relevance {grade} not in 0-3")
            if valid_ids is not None and cid not in valid_ids:
                raise ValueError(f"{path.name} line {n}: unknown candidate_id {cid}")
            ab = (row.get("answer_bearing") or "").strip()
            qrels.setdefault(qid, {})[cid] = {
                "relevance": grade, "answer_bearing": int(ab) if ab in ("0", "1") else 0}
    return qrels


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"git_sha": run("rev-parse", "HEAD") or None,
            "git_dirty": bool(run("status", "--porcelain"))}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
