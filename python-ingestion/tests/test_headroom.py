"""U1 headroom: count relevant docs outside S2's top N (spec section 13.4).

Run B, `retrieval-ladder`. The module under test is imported *inside* each test
so the file collects on its own and every test fails with ``ImportError`` until
``evals/headroom.py`` exists (spec section 3 rule 1).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common  # noqa: E402


def _headroom():
    """Lazy import: red means the module is missing, not that the file is broken."""
    from evals import headroom
    return headroom


def _crow(qid, cid, rank, score=1.0):
    return {"query_id": qid, "candidate_id": cid, "rank": rank, "score": score}


def _qrels(pairs):
    """Build the common.load_qrels shape: {qid: {cid: {"relevance", ...}}}."""
    out = {}
    for qid, cid, grade in pairs:
        out.setdefault(qid, {})[cid] = {"relevance": grade, "answer_bearing": 0}
    return out


# Section 2.3: relevant (grade >= 2) and outside S2's top 30, per query.
SECTION_2_3 = {
    "q001": (13, 4),
    "q003": (11, 1),
    "q004": (11, 3),
    "q005": (24, 7),
    "q006": (5, 3),
    "q007": (2, 1),
}


class TestHeadroomArtifacts:
    @pytest.mark.artifacts
    def test_U1_B1_original_set_reproduces_19_of_104(self):
        """Real S2 + qrels_v2: total 19/104 and the section 2.3 per-query table."""
        h = _headroom()
        corpus = common.load_documents(common.DEFAULT_CORPUS_VERSION)
        s2_rows = common.load_run(common.DEFAULT_CORPUS_VERSION, "ladder_s2_cos_bm25_rr")
        qrels = common.load_qrels(
            common.data_dir(common.DEFAULT_CORPUS_VERSION) / "qrels_v2.csv",
            valid_ids={d["candidate_id"] for d in corpus},
        )

        result = h.headroom(s2_rows, qrels, depth=30)

        assert sum(v["relevant"] for v in result.values()) == 104
        assert sum(v["outside"] for v in result.values()) == 19

        for qid, (relevant, outside) in SECTION_2_3.items():
            assert result[qid]["relevant"] == relevant, qid
            assert result[qid]["outside"] == outside, qid
            assert len(result[qid]["outside_ids"]) == outside, qid

        for qid, entry in result.items():
            if qid not in SECTION_2_3:
                assert entry["outside"] == 0, qid
                assert entry["outside_ids"] == [], qid


class TestHeadroomUnit:
    def test_U1_B2_only_grades_ge_2_count_as_relevant(self):
        """Grades 1, 2, 3: only 2 and 3 are relevant."""
        h = _headroom()
        rows = [
            _crow("q1", "d1", rank=1),
            _crow("q1", "d2", rank=2),
            _crow("q1", "d3", rank=31),
        ]
        qrels = _qrels([("q1", "d1", 1), ("q1", "d2", 2), ("q1", "d3", 3)])

        result = h.headroom(rows, qrels, depth=30)["q1"]

        assert result["relevant"] == 2
        assert result["outside"] == 1
        assert result["outside_ids"] == ["d3"]
        assert result["answerable"] is True

    def test_U1_E1_query_with_no_grade_ge_2_is_flagged_not_answerable(self):
        """A query whose best grade is 1 is answerable=False, not a quiet zero."""
        h = _headroom()
        rows = [_crow("q1", "d1", rank=1)]
        qrels = _qrels([("q1", "d1", 1), ("q1", "d2", 0)])

        entry = h.headroom(rows, qrels, depth=30)["q1"]

        assert entry["relevant"] == 0
        assert entry["outside"] == 0
        assert entry["outside_ids"] == []
        assert entry["answerable"] is False

    def test_U1_E2_depth_60_changes_the_outside_count(self):
        """A doc at rank 31 is outside at depth 30 and inside at depth 60."""
        h = _headroom()
        rows = [_crow("q1", "d1", rank=1), _crow("q1", "d2", rank=31)]
        qrels = _qrels([("q1", "d1", 3), ("q1", "d2", 3)])

        at_30 = h.headroom(rows, qrels, depth=30)["q1"]
        at_60 = h.headroom(rows, qrels, depth=60)["q1"]

        assert (at_30["relevant"], at_30["outside"]) == (2, 1)
        assert (at_60["relevant"], at_60["outside"]) == (2, 0)
