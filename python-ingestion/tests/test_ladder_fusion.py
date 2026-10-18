"""U4: Reciprocal Rank Fusion (spec section 8).

LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import fusion  # noqa: E402


def test_constants_are_pre_registered():
    assert fusion.RRF_K == 60
    assert fusion.DEPTH == 30


def test_single_list_keeps_its_order_and_rrf_scores():
    out = fusion.rrf([["a", "b", "c"]])
    assert [c for c, _ in out] == ["a", "b", "c"]
    assert out[0][1] == pytest.approx(1 / 61)
    assert out[2][1] == pytest.approx(1 / 63)


def test_scores_sum_across_lists():
    out = dict(fusion.rrf([["a", "b"], ["b", "a"]]))
    assert out["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_document_in_both_lists_beats_document_in_one():
    out = fusion.rrf([["x", "shared"], ["shared", "y"]])
    assert out[0][0] == "shared"


def test_ties_break_by_candidate_id():
    out = fusion.rrf([["b", "a"], ["a", "b"]])  # identical scores
    assert [c for c, _ in out] == ["a", "b"]


def test_each_input_is_cut_to_depth_before_fusing():
    long = [f"d{i:02d}" for i in range(40)]
    out = dict(fusion.rrf([long, ["d35"]], depth=30))
    assert out["d35"] == pytest.approx(1 / 61)  # its rank 36 in `long` was cut


def test_output_is_cut_to_depth():
    a = [f"a{i:02d}" for i in range(30)]
    b = [f"b{i:02d}" for i in range(30)]
    assert len(fusion.rrf([a, b])) == 30


def test_custom_k():
    out = fusion.rrf([["a"]], k=10)
    assert out[0][1] == pytest.approx(1 / 11)


def test_empty_lists():
    assert fusion.rrf([[], []]) == []
    assert [c for c, _ in fusion.rrf([[], ["a"]])] == ["a"]


def test_duplicate_within_one_list_counts_once_at_its_best_rank():
    out = dict(fusion.rrf([["a", "b", "a"]]))
    assert out["a"] == pytest.approx(1 / 61)


def test_ranking_extracts_one_query_in_rank_order():
    rows = [{"query_id": "q1", "candidate_id": "b", "rank": 2},
            {"query_id": "q2", "candidate_id": "z", "rank": 1},
            {"query_id": "q1", "candidate_id": "a", "rank": 1}]
    assert fusion.ranking(rows, "q1") == ["a", "b"]
    assert fusion.ranking(rows, "q3") == []
