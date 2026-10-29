"""GET /items/{id}, GET /topics, GET /topics/{name} (spec section 3, plus the
section-3 review fixes: topic spelling variants and slash topics).

Offline: a fake Neo4j session dispatches ``run(query, **params)`` on a
substring of each query's own text (same style as the FakeTx in
tests/test_ladder_live_guard.py), and a fake Qdrant client answers
``retrieve``. Two ``@pytest.mark.live`` tests (deselected unless --live) hit
the real, read-only Neo4j.
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
    """``answers`` is ``[(marker, rows)]``; ``run`` returns the rows for
    whichever marker is both a substring of the query text and the longest
    such match -- so a broader query's marker (e.g. no ``WHERE``) never
    shadows a more specific one that happens to start the same way."""

    def __init__(self, answers):
        self._answers = answers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        candidates = [(marker, rows) for marker, rows in self._answers if marker in query]
        if not candidates:
            raise AssertionError(f"FakeSession: no marker matched query:\n{query}")
        _marker, rows = max(candidates, key=lambda mr: len(mr[0]))
        return SimpleNamespace(data=lambda rows=rows: rows,
                               single=lambda rows=rows: (rows[0] if rows else None))


def _use_store(answers):
    app.app.state.store = SimpleNamespace(
        driver=SimpleNamespace(session=lambda: FakeSession(answers)))


class FakeVS:
    """Enough of VectorStore for get_item's ``vs.client.retrieve`` lookup."""

    def __init__(self, points_by_id):
        self._points_by_id = points_by_id  # {id: payload}
        self.collection = "manthan"
        self.client = SimpleNamespace(retrieve=self._retrieve)

    def _retrieve(self, collection_name, ids, with_payload=True, **kw):
        return [SimpleNamespace(id=i, payload=self._points_by_id[i])
                for i in ids if i in self._points_by_id]


def _use_vs(points_by_id):
    app.app.state.vs = FakeVS(points_by_id)


# ---- topic_key --------------------------------------------------------------

def test_topic_key_folds_the_spec_examples():
    assert app.topic_key("Agentic Workflow") == app.topic_key("agentic-workflows")
    assert app.topic_key("Local LLMs") == app.topic_key("local-llm")
    assert app.topic_key("AI/ML") == "ai ml"


def test_topic_key_does_not_merge_different_words():
    assert app.topic_key("Agentic AI") != app.topic_key("agentic-workflows")


def test_topic_key_leaves_short_words_and_double_s_alone():
    assert app.topic_key("rag") == "rag"       # len 3: never stripped
    assert app.topic_key("glass") == "glass"   # ends in "ss": never stripped


# ---- GET /items/{id} ---------------------------------------------------------

def test_get_item_finds_a_message_via_qdrant_retrieve():
    sent_at, text = "2026-01-01T00:00:00", "hello world"
    item_id = _message_id(sent_at, text)
    _use_vs({item_id: {"type": "message", "sent_at": sent_at, "text": text}})
    _use_store([
        ("collect(DISTINCT {url: l.url, title: l.title})", [
            {"topics": ["python"], "links": [{"url": "https://x", "title": "X"}]},
        ]),
    ])
    out = app.get_item(item_id)
    assert out["id"] == item_id
    assert out["type"] == "message"
    assert out["text"] == text
    assert out["topics"] == ["python"]
    assert out["related_links"] == [{"url": "https://x", "title": "X"}]


def test_get_item_finds_a_link_via_qdrant_retrieve():
    url = "https://example.com/post"
    item_id = _link_id(url)
    _use_vs({item_id: {"type": "link", "url": url, "title": "A Post",
                       "summary": "a full summary"}})
    _use_store([
        ("min(m.sent_at) AS earliest_sent_at", [
            {"topics": ["rag"], "earliest_sent_at": "2026-01-02T00:00:00"},
        ]),
    ])
    out = app.get_item(item_id)
    assert out["id"] == item_id
    assert out["type"] == "link"
    assert out["url"] == url
    assert out["title"] == "A Post"
    assert out["text"] == "a full summary"  # full text, no truncation (M4)
    assert out["topics"] == ["rag"]
    assert out["sent_at"] == "2026-01-02T00:00:00"


def test_get_item_unknown_id_is_404_without_touching_neo4j():
    """The id IS the Qdrant point id (M8): an empty retrieve must 404
    immediately, never fall back to scanning Neo4j."""
    _use_vs({})
    _use_store([])  # any session.run() call here raises AssertionError
    with pytest.raises(app.HTTPException) as exc_info:
        app.get_item(str(uuid.uuid4()))
    assert exc_info.value.status_code == 404


def test_get_item_malformed_id_is_404_without_calling_qdrant():
    """Not a valid UUID -> straight to 404. The id IS the Qdrant point id
    (M8), so a non-UUID can never match a real point; Qdrant is never
    called (no 500 from a malformed retrieve)."""
    def _boom(*a, **kw):
        raise AssertionError("must not call Qdrant for a malformed id")

    app.app.state.vs = SimpleNamespace(collection="manthan",
                                       client=SimpleNamespace(retrieve=_boom))
    with pytest.raises(app.HTTPException) as exc_info:
        app.get_item("not-a-uuid")
    assert exc_info.value.status_code == 404
    assert "not-a-uuid" in exc_info.value.detail


def test_get_item_well_formed_but_unknown_id_is_404():
    """A syntactically valid UUID Qdrant simply doesn't have."""
    _use_vs({})
    unknown_id = str(uuid.uuid4())
    with pytest.raises(app.HTTPException) as exc_info:
        app.get_item(unknown_id)
    assert exc_info.value.status_code == 404
    assert unknown_id in exc_info.value.detail


# ---- GET /topics: spelling-variant grouping -------------------------------------

def test_list_topics_groups_variants_dedupes_items_and_picks_canonical():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "agentic-workflows", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"name": "Agentic Workflow", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"name": "Agentic Workflow", "sent_at": "2026-01-02T00:00:00", "text": "m2"},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", []),
    ])
    out = app.list_topics()
    assert out["topics"] == [{
        "name": "Agentic Workflow",  # tagged on 2 items vs. agentic-workflows' 1
        "count": 2,                  # m1 (tagged by both names) + m2, deduped
        "variants": ["Agentic Workflow", "agentic-workflows"],
    }]


def test_list_topics_canonical_name_tie_break_is_alphabetical():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "Rag", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"name": "rag", "sent_at": "2026-01-02T00:00:00", "text": "m2"},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", []),
    ])
    out = app.list_topics()
    assert out["topics"][0]["name"] == "Rag"  # 1 item each (tie); "Rag" < "rag"


# ---- GET /topics/{name:path}: union, dedupe, neighbors on keys -----------------

def test_browse_topic_unions_variants_and_dedupes_by_id():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "agentic-workflows", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"name": "Agentic Workflow", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"name": "Agentic Workflow", "sent_at": "2026-01-02T00:00:00", "text": "m2"},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", []),
        ("WHERE t.name IN $names RETURN DISTINCT m.sent_at", [
            {"sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"sent_at": "2026-01-02T00:00:00", "text": "m2"},
        ]),
        ("WHERE t.name IN $names RETURN DISTINCT l.url", []),
    ])
    out = app.browse_topic("agentic workflow", limit=50)
    assert out["total"] == 2
    ids = {it["id"] for it in out["items"]}
    assert ids == {_message_id("2026-01-01T00:00:00", "m1"),
                   _message_id("2026-01-02T00:00:00", "m2")}
    assert [it["sent_at"] for it in out["items"]] == [
        "2026-01-02T00:00:00", "2026-01-01T00:00:00"]  # newest first


def test_browse_topic_snippet_is_cut_with_ellipsis_and_truncated_flag():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "python", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", []),
        ("WHERE t.name IN $names RETURN DISTINCT m.sent_at", [
            {"sent_at": "2026-01-01T00:00:00", "text": "x" * 1000},
        ]),
        ("WHERE t.name IN $names RETURN DISTINCT l.url", []),
    ])
    out = app.browse_topic("python", limit=50)
    item = out["items"][0]
    assert len(item["text"]) == 300
    assert item["text"].endswith("…")
    assert item["truncated"] is True


def test_browse_topic_short_snippet_is_not_truncated():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "python", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", []),
        ("WHERE t.name IN $names RETURN DISTINCT m.sent_at", [
            {"sent_at": "2026-01-01T00:00:00", "text": "short"},
        ]),
        ("WHERE t.name IN $names RETURN DISTINCT l.url", []),
    ])
    out = app.browse_topic("python", limit=50)
    item = out["items"][0]
    assert item["text"] == "short"
    assert item["truncated"] is False


def test_browse_topic_ids_follow_m8():
    sent_at, text = "2026-01-01T00:00:00", "a tracked message"
    url = "https://example.com"
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "python", "sent_at": sent_at, "text": text},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", [
            {"name": "python", "url": url},
        ]),
        ("WHERE t.name IN $names RETURN DISTINCT m.sent_at", [
            {"sent_at": sent_at, "text": text},
        ]),
        ("WHERE t.name IN $names RETURN DISTINCT l.url", [
            {"url": url, "title": "T", "summary": "s", "sent_at": sent_at},
        ]),
    ])
    out = app.browse_topic("python", limit=50)
    by_type = {it["type"]: it for it in out["items"]}
    assert by_type["message"]["id"] == _message_id(sent_at, text)
    assert by_type["link"]["id"] == _link_id(url)


def test_browse_topic_neighbors_are_computed_on_keys_and_exclude_self():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "python", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"name": "Python", "sent_at": "2026-01-02T00:00:00", "text": "m2"},
            {"name": "rag", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"name": "rag", "sent_at": "2026-01-02T00:00:00", "text": "m2"},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", []),
        ("WHERE t.name IN $names RETURN DISTINCT m.sent_at", [
            {"sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"sent_at": "2026-01-02T00:00:00", "text": "m2"},
        ]),
        ("WHERE t.name IN $names RETURN DISTINCT l.url", []),
    ])
    out = app.browse_topic("python", limit=50)
    assert out["neighbors"] == [{"name": "rag", "count": 2}]


def test_browse_topic_unknown_gives_404_with_suggestions_compared_on_keys():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "python-tips", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
            {"name": "rag", "sent_at": "2026-01-02T00:00:00", "text": "m2"},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", []),
    ])
    with pytest.raises(app.HTTPException) as exc_info:
        app.browse_topic("python", limit=50)
    assert exc_info.value.status_code == 404
    assert "python" in exc_info.value.detail
    assert "python-tips" in exc_info.value.detail


def test_browse_topic_unknown_with_no_close_name_still_404s_cleanly():
    _use_store([
        ("MATCH (m:Message)-[:ABOUT]->(t:Topic)", [
            {"name": "zzz", "sent_at": "2026-01-01T00:00:00", "text": "m1"},
        ]),
        ("MATCH (l:Link)-[:ABOUT]->(t:Topic)", []),
    ])
    with pytest.raises(app.HTTPException) as exc_info:
        app.browse_topic("totally-unrelated", limit=50)
    assert exc_info.value.status_code == 404


# ---- SearchRequest.kind (spec section 4, item 3) ---------------------------------

def test_search_request_kind_defaults_to_any():
    assert app.SearchRequest(query="q").kind == "any"


def test_search_request_kind_accepts_introduction():
    assert app.SearchRequest(query="q", kind="introduction").kind == "introduction"


def test_search_request_kind_rejects_anything_else():
    with pytest.raises(Exception):  # pydantic ValidationError
        app.SearchRequest(query="q", kind="not-a-real-kind")


def test_do_search_passes_kind_through_to_search_s2():
    seen = {}
    original = app.retrieval.search_s2

    def fake_search_s2(query, **kwargs):
        seen.update(kwargs)
        return {"results": [], "confidence": "none", "top_score": None}

    app.retrieval.search_s2 = fake_search_s2
    app.app.state.vs = object()
    app.app.state.reranker = object()
    app.app.state.corpus = object()
    app.app.state.store = object()
    try:
        app.do_search(app.SearchRequest(query="q", kind="introduction"))
    finally:
        app.retrieval.search_s2 = original
    assert seen["kind"] == "introduction"


# ---- live (deselected unless --live) ---------------------------------------

@pytest.mark.live
def test_live_browse_topic_items_resolve_via_get_item():
    """A real topic's browse_topic items must each resolve back through
    get_item -- i.e. the M8 ids computed from raw Neo4j rows actually match,
    and get_item's Qdrant retrieve finds the same point."""
    from evals import common, live_guard

    common.load_env()
    original_store, original_vs = app.app.state.store, app.app.state.vs
    store = live_guard.read_only_store()
    vs = live_guard.read_only_vector_store()
    app.app.state.store = store
    app.app.state.vs = vs
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
        app.app.state.vs = original_vs
        store.close()
        vs.close()


@pytest.mark.live
def test_live_browse_topic_folds_a_known_spelling_variant():
    """"agentic workflow" (singular, no punctuation) must fold to the same
    key as the real "agentic-workflows" topic and return at least as many
    items as that one spelling alone used to (a regression floor, not an
    exact count: the live corpus can only grow)."""
    from evals import common, live_guard

    common.load_env()
    original_store = app.app.state.store
    store = live_guard.read_only_store()
    app.app.state.store = store
    try:
        out = app.browse_topic("agentic workflow", limit=500)
        assert out["total"] >= 47
    finally:
        app.app.state.store = original_store
        store.close()
