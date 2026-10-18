"""U9: assemble the five ladder steps from frozen inputs (spec section 13).

LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common, fusion, run_ladder  # noqa: E402

S0, S1, S2, S3, S4 = run_ladder.STEPS


def rows(run_id, qid, ids):
    return [{"run_id": run_id, "query_id": qid, "candidate_id": c, "rank": i + 1,
             "score": 1.0 / (i + 1)} for i, c in enumerate(ids)]


class LengthReranker:
    """Fake cross-encoder: longer text wins. Records every call."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates):
        self.calls.append((query, list(candidates)))
        scored = [(cid, float(len(text))) for cid, text in candidates]
        return sorted(scored, key=lambda x: (-x[1], x[0]))


IDS = [f"message:{i:03d}" for i in range(50)]
DOCS = {cid: {"candidate_id": cid, "embedded_text": "x" * (i + 1)} for i, cid in enumerate(IDS)}
QUERIES = [{"query_id": "q1", "query_text": "first"}, {"query_id": "q2", "query_text": "second"}]
COSINE = rows("vector_cosine_v1", "q1", IDS[:40]) + rows("vector_cosine_v1", "q2", IDS[10:50])
BM25 = rows("bm25_v1", "q1", IDS[20:45]) + rows("bm25_v1", "q2", IDS[:5])
GRAPH = rows("graph_lexical_v0", "q1", IDS[45:50] + IDS[:3])  # q2: graph found nothing


@pytest.fixture
def built():
    rr = LengthReranker()
    out = run_ladder.build_ladder(QUERIES, DOCS, COSINE, BM25, GRAPH, rr)
    return out, rr


def by_query(step_rows, qid):
    return [r["candidate_id"] for r in sorted(step_rows, key=lambda r: r["rank"])
            if r["query_id"] == qid]


def test_step_names_and_scales():
    assert run_ladder.STEPS == ("ladder_s0_cosine", "ladder_s1_cos_bm25", "ladder_s2_cos_bm25_rr",
                                "ladder_s3_cos_bm25_graph", "ladder_s4_cos_bm25_graph_rr")
    assert run_ladder.SCORE_SCALE == {S0: "cosine", S1: "rrf", S2: "cross_encoder",
                                      S3: "rrf", S4: "cross_encoder"}


def test_all_five_steps_are_built_with_their_run_ids(built):
    out, _ = built
    assert set(out) == set(run_ladder.STEPS)
    for step, step_rows in out.items():
        assert step_rows and all(r["run_id"] == step for r in step_rows)
        assert set(step_rows[0]) == {"run_id", "query_id", "candidate_id", "rank", "score"}


def test_step0_is_the_frozen_cosine_ranking_unchanged(built):
    out, _ = built
    assert by_query(out[S0], "q1") == IDS[:40]
    assert [r["score"] for r in out[S0] if r["query_id"] == "q1"][:2] == [1.0, 0.5]


def test_step1_is_rrf_of_cosine_and_bm25(built):
    out, _ = built
    expected = fusion.rrf([fusion.ranking(COSINE, "q1"), fusion.ranking(BM25, "q1")])
    assert by_query(out[S1], "q1") == [c for c, _ in expected]
    assert len(by_query(out[S1], "q1")) == 30


def test_step3_adds_graph_as_a_third_rrf_input(built):
    out, _ = built
    expected = fusion.rrf([fusion.ranking(COSINE, "q1"), fusion.ranking(BM25, "q1"),
                           fusion.ranking(GRAPH, "q1")])
    assert by_query(out[S3], "q1") == [c for c, _ in expected]


def test_query_without_graph_results_still_fuses(built):
    out, _ = built
    assert by_query(out[S3], "q2") == by_query(out[S1], "q2")


def test_reranking_never_changes_candidate_membership(built):
    out, _ = built
    for qid in ("q1", "q2"):
        assert sorted(by_query(out[S2], qid)) == sorted(by_query(out[S1], qid))
        assert sorted(by_query(out[S4], qid)) == sorted(by_query(out[S3], qid))


def test_reranker_reads_embedded_text_of_the_fused_top30_in_fused_order(built):
    out, rr = built
    s1_q1 = by_query(out[S1], "q1")
    call = next(c for c in rr.calls if c[0] == "first" and [x for x, _ in c[1]] == s1_q1)
    assert call[1] == [(cid, DOCS[cid]["embedded_text"]) for cid in s1_q1]


def test_reranked_order_and_scores_come_from_the_reranker(built):
    out, _ = built
    q1 = [r for r in sorted(out[S2], key=lambda r: r["rank"]) if r["query_id"] == "q1"]
    scores = [r["score"] for r in q1]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == float(len(DOCS[q1[0]["candidate_id"]]["embedded_text"]))


def test_ranks_are_contiguous_per_query(built):
    out, _ = built
    for step_rows in out.values():
        for qid in ("q1", "q2"):
            ranks = sorted(r["rank"] for r in step_rows if r["query_id"] == qid)
            assert ranks == list(range(1, len(ranks) + 1))


def test_run_meta_records_configuration(tmp_path):
    f = tmp_path / "in.jsonl"
    f.write_text("{}\n", encoding="utf-8")
    meta = run_ladder.run_meta(S4, {"cosine": f})
    assert meta["run_id"] == S4 and meta["score_scale"] == "cross_encoder"
    assert meta["rrf_k"] == 60 and meta["depth"] == 30
    assert meta["rerank_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert meta["rerank_revision"] == "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    assert meta["inputs"]["cosine"]["sha256"] == common.content_hash("{}\n").split(":")[1]
    assert {"created_at", "git_sha", "git_dirty"} <= set(meta)


def test_run_meta_for_non_rerank_step_has_no_model(tmp_path):
    meta = run_ladder.run_meta(S1, {})
    assert meta["rerank_model"] is None and meta["rerank_revision"] is None


@pytest.mark.artifacts
def test_artifact_all_ladder_runs_exist_with_meta():
    runs = common.data_dir(common.DEFAULT_CORPUS_VERSION, create=False) / "runs"
    qids = {q["query_id"] for q in common.load_queries()}
    for step in run_ladder.STEPS:
        step_rows = common.read_jsonl(runs / f"{step}.jsonl")
        meta = json.loads((runs / f"{step}.meta.json").read_text(encoding="utf-8"))
        assert {r["query_id"] for r in step_rows} == qids
        assert meta["run_id"] == step
