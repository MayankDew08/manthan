"""store.py's Message category write (spec section 4, item 3 fix): an
empty/missing category must never erase an already-set one on re-ingest.

Offline: a fake Neo4j tx records the Cypher and parameters add_message
sends. Neo4j's own MERGE/CASE evaluation isn't reimplemented here (that
would just be testing our own re-implementation of Neo4j's semantics,
not Neo4j's); instead this checks the two things store.py is actually
responsible for -- the query text carries the CASE guard, and the
category parameter it sends is "" (never None) when the message has
none, which the CASE in the query then knows to treat as "leave it
alone" on ON MATCH.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import store as store_module  # noqa: E402


class FakeTx:
    def __init__(self):
        self.queries = []  # [(cypher, params)]

    def run(self, cypher, **params):
        self.queries.append((cypher, params))
        return SimpleNamespace(data=lambda: [], single=lambda: None)


class FakeSession:
    def __init__(self, tx):
        self.tx = tx

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_write(self, fn, *args):
        return fn(self.tx, *args)


class FakeDriver:
    def __init__(self):
        self.tx = FakeTx()

    def session(self):
        return FakeSession(self.tx)


def _store_with_fake_driver() -> store_module.KnowledgeStore:
    """Bypasses __init__ (which opens a real neo4j.GraphDatabase.driver)."""
    store = store_module.KnowledgeStore.__new__(store_module.KnowledgeStore)
    store.driver = FakeDriver()
    return store


def test_message_merge_query_guards_an_empty_category_on_match():
    """The guard must live in the ON MATCH branch of the Cypher Neo4j
    actually runs -- not something only true of our test double."""
    source = Path(store_module.__file__).read_text(encoding="utf-8")
    assert "m.category = CASE WHEN $category = '' THEN m.category ELSE $category END" in source


def test_add_message_sends_empty_string_not_none_when_no_category():
    """The CASE guard matches on $category = '', so our code must send
    "" (never None/absent) when the message carries no category."""
    store = _store_with_fake_driver()
    store.add_message({"sent_at": "t1", "original_text": "hello"})  # no "category" key
    _query, params = store.driver.tx.queries[0]
    assert params["category"] == ""


def test_add_message_sends_a_real_category_through_unchanged():
    store = _store_with_fake_driver()
    store.add_message({"sent_at": "t1", "original_text": "hello", "category": "discussion"})
    _query, params = store.driver.tx.queries[0]
    assert params["category"] == "discussion"
