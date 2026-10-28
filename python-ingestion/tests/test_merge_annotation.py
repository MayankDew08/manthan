"""Grades already entered must survive a pool regeneration; losses must be reported, never silent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.merge_annotation import merge_preserving_grades


def row(qid, cid, relevance="", reason="", content="c", ctype="message"):
    return {"query_id": qid, "query_text": "t", "candidate_id": cid, "candidate_type": ctype,
            "content": content, "relevance": relevance, "reason": reason}


def test_existing_grade_carries_over_onto_the_matching_new_row():
    old = [row("q001", "message:1", "3", "direct answer")]
    new = [row("q001", "message:1")]
    merged, orphaned = merge_preserving_grades(old, new)
    assert merged[0]["relevance"] == "3" and merged[0]["reason"] == "direct answer"
    assert orphaned == []


def test_new_query_rows_with_no_old_counterpart_stay_blank():
    old = [row("q001", "message:1", "3")]
    new = [row("q001", "message:1"), row("q006", "message:2")]
    merged, _ = merge_preserving_grades(old, new)
    assert merged[1]["relevance"] == ""


def test_ungraded_old_rows_are_never_treated_as_grades():
    old = [row("q001", "message:1", "")]  # blank: never graded
    new = [row("q001", "message:1")]
    merged, orphaned = merge_preserving_grades(old, new)
    assert merged[0]["relevance"] == "" and orphaned == []


def test_grade_whose_candidate_left_the_pool_is_reported_not_dropped_silently():
    old = [row("q001", "message:1", "2", "ok"), row("q001", "message:99", "3", "gone")]
    new = [row("q001", "message:1")]  # message:99 no longer pooled for q001
    merged, orphaned = merge_preserving_grades(old, new)
    assert [r["candidate_id"] for r in orphaned] == ["message:99"]
    assert orphaned[0]["relevance"] == "3"


def test_output_order_follows_the_new_pool_not_the_old_sheet():
    old = [row("q001", "message:2", "1"), row("q001", "message:1", "3")]
    new = [row("q001", "message:1"), row("q001", "message:2")]  # pool's own shuffle order
    merged, _ = merge_preserving_grades(old, new)
    assert [r["candidate_id"] for r in merged] == ["message:1", "message:2"]
    assert [r["relevance"] for r in merged] == ["3", "1"]


def test_content_and_type_come_from_the_new_pool_not_the_old_sheet():
    old = [row("q001", "message:1", "3", content="stale content", ctype="message")]
    new = [row("q001", "message:1", content="fresh content", ctype="link")]
    merged, _ = merge_preserving_grades(old, new)
    assert merged[0]["content"] == "fresh content" and merged[0]["candidate_type"] == "link"
    assert merged[0]["relevance"] == "3"
