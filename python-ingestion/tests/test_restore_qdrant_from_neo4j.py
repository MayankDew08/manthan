"""Selection rule for restoring Qdrant entries from Neo4j (scripts/restore_qdrant_from_neo4j.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from restore_qdrant_from_neo4j import select_restorable


def msg(sent_at, text, quality):
    return {"sent_at": sent_at, "text": text, "quality": quality,
            "topics": ["t"], "entities": [], "trusted": False}


def link(url, summary="s", status=None, what="", problem=""):
    return {"url": url, "title": "T", "summary": summary, "what_it_is": what,
            "problem_solved": problem, "status": status, "topics": [], "entities": []}


def restore(neo_msgs=(), neo_links=(), q_msgs=(), q_urls=(), live=()):
    return select_restorable(list(neo_msgs), list(neo_links), set(q_msgs), set(q_urls), set(live))


def test_old_high_quality_message_missing_from_qdrant_and_live_file_is_restored():
    m, _ = restore([msg("t1", "qdrant rocks", 5)])
    assert [x["text"] for x in m] == ["qdrant rocks"]


def test_low_quality_messages_are_never_restored():
    m, _ = restore([msg("t1", "meh", 2), msg("t2", "ok", 3), msg("t3", "none", None)])
    assert m == []


def test_message_already_in_qdrant_is_skipped():
    m, _ = restore([msg("t1", "a", 4)], q_msgs=[("t1", "a")])
    assert m == []


def test_message_present_in_a_live_file_is_skipped_because_the_live_grade_wins():
    m, _ = restore([msg("t1", "a", 4)], live=[("t1", "a")])
    assert m == []


def test_keys_ignore_surrounding_whitespace():
    m, _ = restore([msg("t1", "  a \n", 4)], q_msgs=[("t1", "a")])
    assert m == []


def test_raw_text_is_preserved_so_point_ids_match_the_original_import():
    m, _ = restore([msg("t1", "  a \n", 4)])
    assert m[0]["text"] == "  a \n"


def test_link_with_content_missing_from_qdrant_is_restored():
    _, links = restore(neo_links=[link("u1", summary="useful")])
    assert [x["url"] for x in links] == ["u1"]


def test_link_with_only_what_it_is_or_problem_solved_counts_as_content():
    _, links = restore(neo_links=[link("u1", summary="", what="a tool"),
                                  link("u2", summary="", problem="a problem")])
    assert {x["url"] for x in links} == {"u1", "u2"}


def test_link_without_any_content_is_skipped():
    _, links = restore(neo_links=[link("u1", summary="")])
    assert links == []


def test_blocked_or_pending_links_are_skipped():
    _, links = restore(neo_links=[link("u1", status="blocked"), link("u2", status="pending_paste"),
                                  link("u3", status="skipped")])
    assert links == []


def test_link_already_in_qdrant_is_skipped():
    _, links = restore(neo_links=[link("u1")], q_urls=["u1"])
    assert links == []
