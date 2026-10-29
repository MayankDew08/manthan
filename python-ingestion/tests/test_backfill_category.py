"""scripts/backfill_category.py (docs/mcp-server-spec.md section 4, tests
per section 5): dry run writes nothing, apply is idempotent, unmatched
rows are reported, only set_payload is called, never upsert.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import backfill_category as bc  # noqa: E402


# ---- fakes -------------------------------------------------------------------

class FakeSession:
    def __init__(self, driver):
        self._driver = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        return self._driver._run(query, **params)


class FakeDriver:
    """An in-memory Message/Link graph. Recognizes exactly the four
    queries backfill_category.py issues: full scans of each label, and a
    single-node SET for each (never a MERGE/CREATE)."""

    def __init__(self, messages=(), links=()):
        self.messages = [dict(m) for m in messages]  # {"sent_at","text","category"}
        self.links = [dict(lk) for lk in links]       # {"url","category"}
        self.write_calls = []

    def session(self):
        return FakeSession(self)

    def _run(self, query, **params):
        if "SET m.category" in query:
            self.write_calls.append(("message", dict(params)))
            for m in self.messages:
                if m["sent_at"] == params["sent_at"] and m["text"] == params["text"]:
                    m["category"] = params["category"]
            return SimpleNamespace(data=lambda: [])
        if "SET l.category" in query:
            self.write_calls.append(("link", dict(params)))
            for lk in self.links:
                if lk["url"] == params["url"]:
                    lk["category"] = params["category"]
            return SimpleNamespace(data=lambda: [])
        if "MATCH (m:Message)" in query:
            return SimpleNamespace(data=lambda: [dict(m) for m in self.messages])
        if "MATCH (l:Link)" in query:
            return SimpleNamespace(data=lambda: [dict(lk) for lk in self.links])
        raise AssertionError(f"unexpected query: {query}")


class FakeQdrantClient:
    def __init__(self):
        self.set_payload_calls = []
        self.index_calls = []

    def set_payload(self, collection_name, payload, points):
        self.set_payload_calls.append((collection_name, dict(payload), list(points)))

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.index_calls.append((collection_name, field_name, field_schema))

    def upsert(self, *args, **kwargs):
        raise AssertionError("backfill must never call upsert")

    def upload_points(self, *args, **kwargs):
        raise AssertionError("backfill must never call upload_points")


def _grade(text, category, quality=4, confidence=0.9, reason="", topics=()):
    return {"original_text": text, "category": category, "quality": quality,
            "confidence": confidence, "reason": reason, "topics": list(topics)}


def _message(sent_at, text, category=None):
    return {"sent_at": sent_at, "text": text, "category": category}


def _link(url, category=None):
    return {"url": url, "category": category}


# ---- align_grades_to_messages ----------------------------------------------

def test_align_pairs_a_grade_to_the_message_with_the_same_text():
    grades = [_grade("hello", "discussion")]
    messages = [_message("t1", "hello")]
    matched, unmatched = bc.align_grades_to_messages(grades, messages)
    assert unmatched == []
    assert matched == [{"sent_at": "t1", "text": "hello",
                        "old_category": None, "new_category": "discussion"}]


def test_align_reports_a_grade_with_no_matching_message_as_unmatched():
    grades = [_grade("not live anywhere", "discussion")]
    matched, unmatched = bc.align_grades_to_messages(grades, [])
    assert matched == []
    assert unmatched == grades


def test_align_pairs_repeated_text_by_chronological_order():
    """Two reposts of the same text: the earlier grade (file order) pairs
    with the earlier live message (sent_at order), matching the kind of
    positional alignment enrich._load_existing_grades uses."""
    grades = [_grade("repost", "discussion"), _grade("repost", "resource")]
    messages = [_message("t2", "repost"), _message("t1", "repost")]  # out of order
    matched, unmatched = bc.align_grades_to_messages(grades, messages)
    assert unmatched == []
    by_sent_at = {row["sent_at"]: row["new_category"] for row in matched}
    assert by_sent_at == {"t1": "discussion", "t2": "resource"}


def test_align_never_fuzzy_matches_on_a_count_mismatch():
    """Repeated text, but the counts don't agree 1:1 -- every grade for
    that text must be reported unmatched, not guessed."""
    grades = [_grade("repost", "discussion"), _grade("repost", "resource")]
    messages = [_message("t1", "repost")]  # only one live message, two grades
    matched, unmatched = bc.align_grades_to_messages(grades, messages)
    assert matched == []
    assert unmatched == grades


# ---- plan --------------------------------------------------------------------

def test_plan_splits_matched_rows_into_already_set_and_to_update():
    driver = FakeDriver(
        messages=[_message("t1", "a", category=None),
                 _message("t2", "b", category="resource")],
        links=[_link("u1", category=None), _link("u2", category="resource")],
    )
    grades = [_grade("a", "discussion"), _grade("b", "resource")]
    result = bc.plan(grades, driver)

    assert len(result["to_update"]) == 1
    assert result["to_update"][0]["new_category"] == "discussion"
    assert len(result["already_set"]) == 1
    assert result["unmatched"] == []
    assert len(result["links_to_update"]) == 1
    assert result["links_to_update"][0]["url"] == "u1"
    assert len(result["links_already_set"]) == 1


def test_plan_reports_unmatched_rows():
    driver = FakeDriver(messages=[_message("t1", "a")])
    grades = [_grade("a", "discussion"), _grade("ghost", "noise")]
    result = bc.plan(grades, driver)
    assert [row["original_text"] for row in result["unmatched"]] == ["ghost"]


# ---- apply: set_payload only, Neo4j SET only, idempotent -----------------------

def test_apply_messages_sets_category_on_an_existing_node_only():
    driver = FakeDriver(messages=[_message("t1", "a", category=None)])
    to_update = [{"sent_at": "t1", "text": "a", "old_category": None,
                 "new_category": "discussion"}]
    bc.apply_messages(driver, to_update)
    assert driver.messages[0]["category"] == "discussion"
    assert driver.write_calls == [("message", {"sent_at": "t1", "text": "a",
                                               "category": "discussion"})]


def test_apply_links_neo4j_sets_resource_category():
    driver = FakeDriver(links=[_link("u1", category=None)])
    bc.apply_links_neo4j(driver, [_link("u1")])
    assert driver.links[0]["category"] == "resource"


def test_apply_qdrant_only_calls_set_payload_never_upsert():
    client = FakeQdrantClient()
    to_update = [{"sent_at": "t1", "text": "a", "new_category": "discussion"}]
    links_to_update = [_link("u1")]
    bc.apply_qdrant(client, "manthan", to_update, links_to_update)

    assert len(client.set_payload_calls) == 2  # grouped by category value
    categories_set = {payload["category"] for _coll, payload, _ids in client.set_payload_calls}
    assert categories_set == {"discussion", "resource"}
    for _coll, _payload, ids in client.set_payload_calls:
        assert all(isinstance(i, str) for i in ids)  # M8 uuid5 strings, not re-embedded


def test_apply_qdrant_groups_same_category_into_one_call():
    client = FakeQdrantClient()
    to_update = [
        {"sent_at": "t1", "text": "a", "new_category": "discussion"},
        {"sent_at": "t2", "text": "b", "new_category": "discussion"},
    ]
    bc.apply_qdrant(client, "manthan", to_update, [])
    assert len(client.set_payload_calls) == 1
    _coll, payload, ids = client.set_payload_calls[0]
    assert payload == {"category": "discussion"}
    assert len(ids) == 2


def test_ensure_category_index_creates_a_keyword_index():
    client = FakeQdrantClient()
    bc.ensure_category_index(client, "manthan")
    assert len(client.index_calls) == 1
    collection, field_name, _schema = client.index_calls[0]
    assert collection == "manthan"
    assert field_name == "category"


def test_dry_run_writes_nothing():
    driver = FakeDriver(messages=[_message("t1", "a", category=None)])
    grades = [_grade("a", "discussion")]
    result = bc.plan(grades, driver)  # plan() alone is what --dry-run runs
    assert result["to_update"]  # there IS something that would change
    assert driver.write_calls == []  # but nothing was written


def test_apply_is_idempotent():
    driver = FakeDriver(messages=[_message("t1", "a", category=None)])
    links = [_link("u1", category=None)]
    driver.links = links
    grades = [_grade("a", "discussion")]
    client = FakeQdrantClient()

    # first apply: one message and one link actually change.
    result = bc.plan(grades, driver)
    bc.apply_messages(driver, result["to_update"])
    bc.apply_links_neo4j(driver, result["links_to_update"])
    bc.apply_qdrant(client, "manthan", result["to_update"], result["links_to_update"])
    first_write_count = len(driver.write_calls)
    first_set_payload_count = len(client.set_payload_calls)
    assert first_write_count == 2  # one message, one link
    assert first_set_payload_count > 0

    # second apply against the now-updated state: nothing left to change.
    result2 = bc.plan(grades, driver)
    assert result2["to_update"] == []
    assert result2["links_to_update"] == []
    bc.apply_messages(driver, result2["to_update"])
    bc.apply_links_neo4j(driver, result2["links_to_update"])
    bc.apply_qdrant(client, "manthan", result2["to_update"], result2["links_to_update"])
    assert len(driver.write_calls) == first_write_count  # no new writes
    assert len(client.set_payload_calls) == first_set_payload_count
