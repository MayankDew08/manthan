"""scripts/backfill_category.py (docs/mcp-server-spec.md section 4, tests
per section 5, plus the independent-stores fix): dry run writes nothing,
apply is idempotent, unmatched rows are reported, only set_payload is
called (never upsert), and Neo4j/Qdrant are planned independently so a
store that's still behind -- for any reason, including a missing Qdrant
point or a prior partial apply -- is never silently skipped.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    """Mimics real Qdrant closely enough to matter here: ``retrieve`` is
    lenient (an id simply absent from the result if it isn't a live
    point), but ``set_payload`` is strict -- it raises, like Qdrant's own
    PointIdError, if ANY given id doesn't exist. This is exactly the
    failure mode the independent-planning fix exists to avoid triggering.
    """

    def __init__(self, points=None):
        self.points = dict(points or {})  # {id: payload}
        self.set_payload_calls = []
        self.index_calls = []

    def retrieve(self, collection_name, ids, with_payload=True):
        return [SimpleNamespace(id=i, payload=self.points[i]) for i in ids if i in self.points]

    def set_payload(self, collection_name, payload, points):
        missing = [i for i in points if i not in self.points]
        if missing:
            raise RuntimeError(f"PointIdError: missing point id(s) {missing}")
        self.set_payload_calls.append((collection_name, dict(payload), list(points)))
        for i in points:
            self.points[i] = {**self.points[i], **payload}

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


# ---- align_grades_to_messages (unaffected by the independent-stores fix) ------

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


# ---- plan: Neo4j and Qdrant are independent ------------------------------------

def test_plan_qdrant_still_flags_an_update_when_neo4j_already_matches():
    """The bug this fix addresses: a prior run set Neo4j's category but
    never reached Qdrant (or Qdrant still had a stale value). plan must
    label Qdrant as needing an update regardless of Neo4j's own state."""
    sent_at, text = "t1", "a"
    mid = bc.message_id(sent_at, text)
    driver = FakeDriver(messages=[_message(sent_at, text, category="discussion")])
    client = FakeQdrantClient(points={mid: {"category": "noise"}})
    grades = [_grade(text, "discussion")]

    result = bc.plan(grades, driver, client, "manthan")

    assert result["neo4j_to_update"] == []            # Neo4j already matches
    assert len(result["qdrant_msg_to_update"]) == 1    # Qdrant does not
    assert result["qdrant_msg_to_update"][0]["new_category"] == "discussion"


def test_plan_reports_a_missing_qdrant_message_point_separately():
    sent_at, text = "t1", "a"
    driver = FakeDriver(messages=[_message(sent_at, text, category=None)])
    client = FakeQdrantClient(points={})  # never vectorized (e.g. below quality)
    grades = [_grade(text, "discussion")]

    result = bc.plan(grades, driver, client, "manthan")

    assert len(result["neo4j_to_update"]) == 1    # Neo4j SET is still fine
    assert result["qdrant_msg_to_update"] == []
    assert result["qdrant_msg_already_set"] == []
    assert len(result["qdrant_msg_not_in_qdrant"]) == 1


def test_plan_reports_a_missing_qdrant_link_point_separately():
    driver = FakeDriver(links=[_link("u1", category=None)])
    client = FakeQdrantClient(points={})  # e.g. a link with no scraped content
    result = bc.plan([], driver, client, "manthan")

    assert len(result["links_neo4j_to_update"]) == 1
    assert result["qdrant_link_to_update"] == []
    assert len(result["qdrant_link_not_in_qdrant"]) == 1


def test_plan_qdrant_already_set_when_payload_matches():
    sent_at, text = "t1", "a"
    mid = bc.message_id(sent_at, text)
    driver = FakeDriver(messages=[_message(sent_at, text, category="discussion")])
    client = FakeQdrantClient(points={mid: {"category": "discussion"}})
    grades = [_grade(text, "discussion")]

    result = bc.plan(grades, driver, client, "manthan")
    assert result["neo4j_to_update"] == []
    assert result["qdrant_msg_to_update"] == []
    assert len(result["qdrant_msg_already_set"]) == 1


def test_plan_reports_unmatched_rows():
    driver = FakeDriver(messages=[_message("t1", "a")])
    client = FakeQdrantClient(points={})
    grades = [_grade("a", "discussion"), _grade("ghost", "noise")]
    result = bc.plan(grades, driver, client, "manthan")
    assert [row["original_text"] for row in result["unmatched"]] == ["ghost"]


# ---- apply: set_payload only, never on an id Qdrant doesn't have ---------------

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


def test_fake_qdrant_set_payload_raises_on_a_missing_id_like_real_qdrant():
    """Documents the exact failure this fix avoids triggering: Qdrant's
    real set_payload raises if even one id in the batch doesn't exist."""
    client = FakeQdrantClient(points={})
    with pytest.raises(RuntimeError):
        client.set_payload(collection_name="manthan", payload={"category": "x"},
                           points=["not-a-real-id"])


def test_apply_with_a_missing_qdrant_id_succeeds_because_plan_excluded_it():
    """plan() must never hand apply_qdrant an id it hasn't confirmed
    exists -- so apply succeeds even though one of the two candidate
    messages was never vectorized into Qdrant at all."""
    present = ("t1", "a")
    missing = ("t2", "b")
    present_id = bc.message_id(*present)
    driver = FakeDriver(messages=[_message(*present, category=None),
                                  _message(*missing, category=None)])
    client = FakeQdrantClient(points={present_id: {"category": "noise"}})
    grades = [_grade(present[1], "discussion"), _grade(missing[1], "resource")]

    result = bc.plan(grades, driver, client, "manthan")
    assert len(result["qdrant_msg_to_update"]) == 1
    assert len(result["qdrant_msg_not_in_qdrant"]) == 1

    bc.apply_qdrant(client, "manthan", result["qdrant_msg_to_update"], [])  # must not raise
    assert client.points[present_id]["category"] == "discussion"


def test_apply_qdrant_groups_same_category_into_one_call():
    client = FakeQdrantClient(points={
        bc.message_id("t1", "a"): {"category": None},
        bc.message_id("t2", "b"): {"category": None},
    })
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


# ---- dry run, partial-failure recovery, full idempotency ---------------------

def test_dry_run_writes_nothing():
    mid = bc.message_id("t1", "a")
    driver = FakeDriver(messages=[_message("t1", "a", category=None)])
    client = FakeQdrantClient(points={mid: {"category": None}})
    grades = [_grade("a", "discussion")]

    result = bc.plan(grades, driver, client, "manthan")  # plan() alone is what --dry-run runs
    assert result["qdrant_msg_to_update"]  # there IS something that would change
    assert driver.write_calls == []
    assert client.set_payload_calls == []


def test_plan_after_a_partial_failure_still_labels_qdrant_for_the_retry():
    """Simulates the exact crash scenario reported live: a prior run's
    Neo4j SET succeeded, but Qdrant's set_payload never ran (it crashed
    on an unrelated missing id before/during this one, or this one just
    hadn't been reached). A re-run's plan must still catch Qdrant up."""
    sent_at, text = "t1", "a"
    mid = bc.message_id(sent_at, text)
    driver = FakeDriver(messages=[_message(sent_at, text, category="discussion")])
    client = FakeQdrantClient(points={mid: {"category": None}})
    grades = [_grade(text, "discussion")]

    result = bc.plan(grades, driver, client, "manthan")
    assert result["neo4j_to_update"] == []
    assert len(result["qdrant_msg_to_update"]) == 1

    bc.apply_qdrant(client, "manthan", result["qdrant_msg_to_update"], [])
    assert client.points[mid]["category"] == "discussion"


def test_full_apply_then_rerun_makes_zero_write_calls():
    sent_at, text = "t1", "a"
    mid = bc.message_id(sent_at, text)
    lid = bc.link_id("u1")
    driver = FakeDriver(messages=[_message(sent_at, text, category=None)],
                        links=[_link("u1", category=None)])
    client = FakeQdrantClient(points={mid: {"category": None}, lid: {"category": None}})
    grades = [_grade(text, "discussion")]

    result = bc.plan(grades, driver, client, "manthan")
    bc.apply_messages(driver, result["neo4j_to_update"])
    bc.apply_links_neo4j(driver, result["links_neo4j_to_update"])
    bc.apply_qdrant(client, "manthan", result["qdrant_msg_to_update"],
                    result["qdrant_link_to_update"])

    first_neo4j_writes = len(driver.write_calls)
    first_qdrant_writes = len(client.set_payload_calls)
    assert first_neo4j_writes == 2  # one message, one link
    assert first_qdrant_writes > 0

    result2 = bc.plan(grades, driver, client, "manthan")
    assert result2["neo4j_to_update"] == []
    assert result2["links_neo4j_to_update"] == []
    assert result2["qdrant_msg_to_update"] == []
    assert result2["qdrant_link_to_update"] == []

    bc.apply_messages(driver, result2["neo4j_to_update"])
    bc.apply_links_neo4j(driver, result2["links_neo4j_to_update"])
    bc.apply_qdrant(client, "manthan", result2["qdrant_msg_to_update"],
                    result2["qdrant_link_to_update"])

    assert len(driver.write_calls) == first_neo4j_writes
    assert len(client.set_payload_calls) == first_qdrant_writes
