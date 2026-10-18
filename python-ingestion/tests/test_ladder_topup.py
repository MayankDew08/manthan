"""U6: top-up grading sheet, hard-fail gate, qrels_v2 merge (spec section 10).

LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common, topup  # noqa: E402


def r(q, cid, rank, run_id="s1"):
    return {"run_id": run_id, "query_id": q, "candidate_id": cid, "rank": rank, "score": 1.0 / rank}


QRELS = {"q1": {"message:a": {"relevance": 3, "answer_bearing": 0},
                "message:b": {"relevance": 0, "answer_bearing": 0}}}
RUNS = {
    "s0": [r("q1", "message:a", 1, "s0"), r("q1", "message:b", 2, "s0")],
    "s1": [r("q1", "message:c", 1), r("q1", "message:a", 2), r("q1", "message:x", 6)],
    "s2": [r("q1", "message:c", 1, "s2"), r("q2", "link:d", 3, "s2")],
}
DOCS = {cid: {"candidate_id": cid, "candidate_type": cid.split(":")[0],
              "embedded_text": f"text of {cid}", "title": "None", "url": "None"}
        for cid in ("message:a", "message:b", "message:c", "message:x", "link:d")}
QUERIES = {"q1": {"query_id": "q1", "query_text": "first query"},
           "q2": {"query_id": "q2", "query_text": "second query"}}


class TestUnjudgedPairs:
    def test_k_is_five(self):
        assert topup.TOPUP_K == 5

    def test_collects_unjudged_top_k_pairs_across_runs_deduped_and_sorted(self):
        assert topup.unjudged_pairs(RUNS, QRELS) == [("q1", "message:c"), ("q2", "link:d")]

    def test_ranks_below_k_are_ignored(self):
        assert ("q1", "message:x") not in topup.unjudged_pairs(RUNS, QRELS)

    def test_fully_judged_is_empty(self):
        assert topup.unjudged_pairs({"s0": RUNS["s0"]}, QRELS) == []


class TestGate:
    def test_gate_passes_when_everything_is_judged(self):
        topup.assert_fully_judged({"s0": RUNS["s0"]}, QRELS)

    def test_gate_fails_hard_and_lists_the_pairs(self):
        with pytest.raises(topup.UnjudgedError) as exc:
            topup.assert_fully_judged(RUNS, QRELS)
        assert exc.value.pairs == [("q1", "message:c"), ("q2", "link:d")]
        assert "2" in str(exc.value)

    def test_unjudged_error_is_not_a_skip(self):
        assert issubclass(topup.UnjudgedError, RuntimeError)


class TestTopupSheet:
    def rows(self, seed=42):
        pairs = [("q1", "message:c"), ("q2", "link:d"), ("q1", "message:x")]
        return topup.topup_rows(pairs, QUERIES, DOCS, seed=seed)

    def test_columns_are_the_annotation_columns(self):
        assert all(list(row) == common.ANNOTATION_COLUMNS for row in self.rows())

    def test_grades_are_blank_and_no_run_information_leaks(self):
        for row in self.rows():
            assert row["relevance"] == "" and row["reason"] == ""
            assert not ({"run_id", "rank", "score", "source"} & set(row))

    def test_content_and_query_text_are_filled(self):
        row = next(x for x in self.rows() if x["candidate_id"] == "link:d")
        assert row["query_text"] == "second query"
        assert "text of link:d" in row["content"]
        assert row["candidate_type"] == "link"

    def test_shuffle_is_deterministic_per_seed(self):
        assert self.rows(42) == self.rows(42)

    def test_shuffle_does_not_depend_on_input_order(self):
        pairs = [("q1", "message:c"), ("q2", "link:d"), ("q1", "message:x")]
        a = topup.topup_rows(pairs, QUERIES, DOCS, seed=42)
        b = topup.topup_rows(list(reversed(pairs)), QUERIES, DOCS, seed=42)
        assert a == b

    def test_every_pair_appears_once(self):
        got = sorted((x["query_id"], x["candidate_id"]) for x in self.rows())
        assert got == [("q1", "message:c"), ("q1", "message:x"), ("q2", "link:d")]


def ann(q, cid, rel, reason="why"):
    return {"query_id": q, "query_text": "t", "candidate_id": cid, "candidate_type": "message",
            "content": "c", "relevance": rel, "reason": reason}


class TestMergeQrels:
    def test_union_of_annotation_and_topup(self):
        out = topup.merge_qrels([ann("q1", "message:a", "3")], [ann("q1", "message:c", "1")])
        assert [(x["query_id"], x["candidate_id"], x["relevance"]) for x in out] == [
            ("q1", "message:a", "3"), ("q1", "message:c", "1")]

    def test_output_columns_are_the_qrels_columns(self):
        out = topup.merge_qrels([ann("q1", "message:a", "3")], [])
        assert list(out[0]) == ["query_id", "candidate_id", "relevance", "answer_bearing", "reason"]

    def test_blank_topup_grade_fails(self):
        with pytest.raises(ValueError, match="message:c"):
            topup.merge_qrels([ann("q1", "message:a", "3")], [ann("q1", "message:c", "")])

    def test_blank_annotation_grade_fails(self):
        with pytest.raises(ValueError):
            topup.merge_qrels([ann("q1", "message:a", " ")], [])

    def test_grade_outside_scale_fails(self):
        with pytest.raises(ValueError):
            topup.merge_qrels([], [ann("q1", "message:c", "4")])

    def test_topup_pair_already_graded_fails(self):
        with pytest.raises(ValueError, match="already"):
            topup.merge_qrels([ann("q1", "message:a", "3")], [ann("q1", "message:a", "0")])

    def test_merged_file_round_trips_through_load_qrels(self, tmp_path):
        out = topup.merge_qrels([ann("q1", "message:a", "3")], [ann("q2", "link:d", "2")])
        path = tmp_path / topup.QRELS_V2
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        q = common.load_qrels(path)
        assert q["q1"]["message:a"]["relevance"] == 3 and q["q2"]["link:d"]["relevance"] == 2

    def test_qrels_v2_name(self):
        assert topup.QRELS_V2 == "qrels_v2.csv"


@pytest.mark.artifacts
def test_artifact_qrels_v2_judges_every_top5_pair_of_every_ladder_step():
    from evals import run_ladder
    data = common.data_dir(common.DEFAULT_CORPUS_VERSION, create=False)
    corpus = {d["candidate_id"] for d in common.load_documents(common.DEFAULT_CORPUS_VERSION)}
    qrels = common.load_qrels(data / topup.QRELS_V2, valid_ids=corpus)
    runs = {s: common.load_run(common.DEFAULT_CORPUS_VERSION, s) for s in run_ladder.STEPS}
    topup.assert_fully_judged(runs, qrels)
    assert (data / "qrels.csv").exists(), "qrels.csv (v1) must never be removed"
