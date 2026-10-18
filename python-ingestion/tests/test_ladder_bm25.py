"""U3: BM25 run file over the frozen corpus (spec section 7).

LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common, run_bm25  # noqa: E402
from evals.bm25 import BM25  # noqa: E402


def doc(cid, text):
    return {"candidate_id": cid, "candidate_type": cid.split(":")[0], "embedded_text": text}


DOCS = [
    doc("message:c", "qdrant is a vector database for rag"),
    doc("message:a", "qdrant qdrant qdrant self hosted"),
    doc("link:b", "pgvector extends postgres"),
    doc("message:d", "nothing relevant at all"),
]
QUERIES = [{"query_id": "q1", "query_text": "qdrant vector database"},
           {"query_id": "q2", "query_text": "postgres"}]


def test_run_id():
    assert run_bm25.RUN_ID == "bm25_v1"


def test_row_schema_and_ranks():
    rows = run_bm25.bm25_rows(QUERIES, DOCS)
    assert set(rows[0]) == {"run_id", "query_id", "candidate_id", "rank", "score"}
    q1 = [r for r in rows if r["query_id"] == "q1"]
    assert [r["rank"] for r in q1] == list(range(1, len(q1) + 1))
    assert all(r["run_id"] == "bm25_v1" for r in rows)


def test_scores_are_the_in_repo_bm25_over_embedded_text():
    rows = run_bm25.bm25_rows(QUERIES, DOCS)
    expected = BM25([d["embedded_text"] for d in DOCS]).scores("qdrant vector database")
    by_id = {r["candidate_id"]: r["score"] for r in rows if r["query_id"] == "q1"}
    for d, s in zip(DOCS, expected):
        if s > 0:
            assert by_id[d["candidate_id"]] == pytest.approx(s)


def test_zero_score_documents_are_not_ranked():
    rows = run_bm25.bm25_rows(QUERIES, DOCS)
    assert "message:d" not in {r["candidate_id"] for r in rows}
    assert [r["candidate_id"] for r in rows if r["query_id"] == "q2"] == ["link:b"]


def test_sorted_by_score_desc():
    rows = [r for r in run_bm25.bm25_rows(QUERIES, DOCS) if r["query_id"] == "q1"]
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_ties_break_by_candidate_id_not_corpus_order():
    docs = [doc("message:z", "same words"), doc("message:m", "same words")]
    rows = run_bm25.bm25_rows([{"query_id": "q", "query_text": "same words"}], docs)
    assert [r["candidate_id"] for r in rows] == ["message:m", "message:z"]


def test_deterministic():
    assert run_bm25.bm25_rows(QUERIES, DOCS) == run_bm25.bm25_rows(QUERIES, DOCS)


@pytest.mark.artifacts
def test_artifact_bm25_run_covers_every_query():
    rows = common.load_run(common.DEFAULT_CORPUS_VERSION, "bm25_v1")
    corpus = {d["candidate_id"] for d in common.load_documents(common.DEFAULT_CORPUS_VERSION)}
    assert {r["query_id"] for r in rows} == {q["query_id"] for q in common.load_queries()}
    assert {r["candidate_id"] for r in rows} <= corpus
