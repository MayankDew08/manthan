"""GET /items/{id}, GET /topics, GET /topics/{name} (spec section 3).

Offline: a fake Neo4j session dispatches ``run(query, **params)`` on a
substring of each query's own text (same style as the FakeTx in
tests/test_ladder_live_guard.py), so the real Cypher in app.py is exercised
without a real driver. One ``@pytest.mark.live`` test (deselected unless
--live) hits the real, read-only Neo4j.
"""

import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402


def _message_id(sent_at, text):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"msg-{sent_at}-{text}"))


def _link_id(url):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"link-{url}"))


class FakeSession:
    """``answers`` is ``[(marker, rows)]``; ``run`` returns the rows for the
    first marker that is a substring of the query text."""

    def __init__(self, answers):
        self._answers = answers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        for marker, rows in self._answers:
            if marker in query:
                return SimpleNamespace(data=lambda rows=rows: rows)
        raise AssertionError(f"FakeSession: no marker matched query:\n{query}")


def _use_store(answers):
    app.app.state.store = SimpleNamespace(
        driver=SimpleNamespace(session=lambda: FakeSession(answers)))


# ---- GET /items/{id} -------------------------------------------------------

def test_get_item_finds_a_message_by_its_m8_id():
    sent_at, text = "2026-01-01T00:00:00", "hello world"
    item_id = _message_id(sent_at, text)
    _use_store([
        ("OPTIONAL MATCH (m)-[:CONTAINS]->(l:Link)", [
            {"sent_at": sent_at, "text": text, "topics": ["python"],
             "links": [{"url": "", "title": None}]},
        ]),
        ("OPTIONAL MATCH (l)-[:ABOUT]->(t:Topic)", []),
    ])
    out = app.get_item(item_id)
    assert out["id"] == item_id
    assert out["type"] == "message"
    assert out["text"] == text
    assert out["topics"] == ["python"]
    assert out["related_links"] == []  # the one candidate link had no url


def test_get_item_finds_a_link_by_its_m8_id():
    url = "https://example.com/post"
    item_id = _link_id(url)
    _use_store([
        ("OPTIONAL MATCH (m)-[:CONTAINS]->(l:Link)", []),
        ("OPTIONAL MATCH (l)-[:ABOUT]->(t:Topic)", [
            {"url": url, "title": "A Post", "summary": "a full summary",
             "sent_at": "2026-01-02T00:00:00", "topics": ["rag"]},
        ]),
    ])
    out = app.get_item(item_id)
    assert out["id"] == item_id
    assert out["type"] == "link"
    assert out["url"] == url
    assert out["title"] == "A Post"
    assert out["text"] == "a full summary"  # full text, no truncation (M4)
    assert out["topics"] == ["rag"]


def test_get_item_unknown_id_is_404_with_a_clear_message():
    _use_store([
        ("OPTIONAL MATCH (m)-[:CONTAINS]->(l:Link)", []),
        ("OPTIONAL MATCH (l)-[:ABOUT]->(t:Topic)", []),
    ])
    with pytest.raises(app.HTTPException) as exc_info:
        app.get_item("not-a-real-id")
    assert exc_info.value.status_code == 404
    assert "not-a-real-id" in exc_info.value.detail


# ---- GET /topics ------------------------------------------------------------

def test_list_topics_counts_messages_plus_links_sorted_descending():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "python", "n": 2}, {"name": "rag", "n": 1},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", [
            {"name": "rag", "n": 3},
        ]),
    ])
    out = app.list_topics()
    assert out["topics"] == [{"name": "rag", "count": 4}, {"name": "python", "count": 2}]


# ---- GET /topics/{name} -----------------------------------------------------

def test_browse_topic_returns_items_newest_first_plus_total_and_neighbors():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(:Topic {name: $name})", [
            {"sent_at": "2026-01-01T00:00:00", "text": "older message"},
            {"sent_at": "2026-03-01T00:00:00", "text": "newer message"},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(:Topic {name: $name})", [
            {"url": "u", "title": "T", "summary": "s", "sent_at": "2026-02-01T00:00:00"},
        ]),
        ("<-[:ABOUT]-(m:Message)-[:ABOUT]->(other:Topic)", [{"name": "rag", "n": 2}]),
        ("<-[:ABOUT]-(l:Link)-[:ABOUT]->(other:Topic)", [{"name": "rag", "n": 1}]),
    ])
    out = app.browse_topic("python", limit=50)
    assert out["total"] == 3
    assert [it["sent_at"] for it in out["items"]] == [
        "2026-03-01T00:00:00", "2026-02-01T00:00:00", "2026-01-01T00:00:00"]
    assert out["neighbors"] == [{"name": "rag", "count": 3}]


def test_browse_topic_snippet_is_cut_to_300_characters():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(:Topic {name: $name})", [
            {"sent_at": "2026-01-01T00:00:00", "text": "x" * 1000},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(:Topic {name: $name})", []),
        ("<-[:ABOUT]-(m:Message)-[:ABOUT]->(other:Topic)", []),
        ("<-[:ABOUT]-(l:Link)-[:ABOUT]->(other:Topic)", []),
    ])
    out = app.browse_topic("python", limit=50)
    assert len(out["items"][0]["text"]) == 300


def test_browse_topic_ids_follow_m8():
    sent_at, text = "2026-01-01T00:00:00", "a tracked message"
    url = "https://example.com"
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(:Topic {name: $name})", [
            {"sent_at": sent_at, "text": text},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(:Topic {name: $name})", [
            {"url": url, "title": "T", "summary": "s", "sent_at": sent_at},
        ]),
        ("<-[:ABOUT]-(m:Message)-[:ABOUT]->(other:Topic)", []),
        ("<-[:ABOUT]-(l:Link)-[:ABOUT]->(other:Topic)", []),
    ])
    out = app.browse_topic("python", limit=50)
    by_type = {it["type"]: it for it in out["items"]}
    assert by_type["message"]["id"] == _message_id(sent_at, text)
    assert by_type["link"]["id"] == _link_id(url)


def test_browse_topic_unknown_gives_404_with_suggestions():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(:Topic {name: $name})", []),
        ("MATCH (l:Link)-[:ABOUT]->(:Topic {name: $name})", []),
        ("MATCH (t:Topic) RETURN t.name AS name", [
            {"name": "python"}, {"name": "pythonic-code"}, {"name": "rag"},
        ]),
    ])
    with pytest.raises(app.HTTPException) as exc_info:
        app.browse_topic("python-tips")
    assert exc_info.value.status_code == 404
    assert "python-tips" in exc_info.value.detail
    assert "python" in exc_info.value.detail


def test_browse_topic_unknown_with_no_close_name_still_404s_cleanly():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(:Topic {name: $name})", []),
        ("MATCH (l:Link)-[:ABOUT]->(:Topic {name: $name})", []),
        ("MATCH (t:Topic) RETURN t.name AS name", [{"name": "zzz"}]),
    ])
    with pytest.raises(app.HTTPException) as exc_info:
        app.browse_topic("totally-unrelated")
    assert exc_info.value.status_code == 404


# ---- live (deselected unless --live) ---------------------------------------

@pytest.mark.live
def test_live_browse_topic_items_resolve_via_get_item():
    """A real topic's browse_topic items must each resolve back through
    get_item -- i.e. the M8 ids computed from raw Neo4j rows actually match."""
    from evals import common, live_guard

    common.load_env()
    original_store = app.app.state.store
    store = live_guard.read_only_store()
    app.app.state.store = store
    try:
        topics = app.list_topics()["topics"]
        assert topics, "fixture assumption: at least one topic exists"
        topic_name = topics[0]["name"]

        browsed = app.browse_topic(topic_name, limit=50)
        assert browsed["items"], "fixture assumption: this topic has an item"

        for item in browsed["items"][:3]:
            fetched = app.get_item(item["id"])
            assert fetched["id"] == item["id"]
            assert fetched["type"] == item["type"]
    finally:
        app.app.state.store = original_store
        store.close()
