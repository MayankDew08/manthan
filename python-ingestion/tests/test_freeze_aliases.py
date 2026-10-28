"""U2 freeze_neighbors: Qdrant id aliases (spec D25).

A corpus doc's candidate_id never changes. If its own qdrant_point_id isn't
found live, it may still be found by the EXACT uuid5 of its own (sent_at,
embedded_text) -- the id its current text would hash to -- never by any
other fuzzy match. Found this way, it is still keyed by its corpus
candidate_id everywhere downstream.
"""

import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _freeze():
    from evals import freeze_neighbors
    return freeze_neighbors


def _own_id(sent_at, text):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"msg-{sent_at}-{text}"))


def _doc(cid, point_id, sent_at="2026-08-04T12:53:00", text="hello", kind="message"):
    return {"candidate_id": cid, "candidate_type": kind, "qdrant_point_id": point_id,
            "embedded_text": text, "sent_at": sent_at}


class _Point:
    def __init__(self, id_, payload=None, vector=None):
        self.id = id_
        self.payload = payload or {}
        self.vector = vector


class _FakeScrollClient:
    """One batch of points, then stop (offset=None)."""

    def __init__(self, points):
        self.points = points
        self.calls = 0

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        self.calls += 1
        return self.points, None


class _Hit:
    def __init__(self, id_, score, payload=None):
        self.id = id_
        self.score = score
        self.payload = payload or {}


class _FakeVectorStore:
    def __init__(self, hits_by_query):
        self.hits_by_query = hits_by_query

    def search(self, query_text, limit):
        return self.hits_by_query[query_text]


class TestFreezeAliasesPayload:
    def test_live_id_matching_the_d25_uuid5_is_keyed_by_the_corpus_id(self):
        fn = _freeze()
        sent_at, text = "2026-08-04T12:53:00", "same text, moved key"
        alias_id = _own_id(sent_at, text)  # what the live store now uses
        doc = _doc("message:OLDKEY", "oldkey-not-live", sent_at=sent_at, text=text)
        corpus_docs = [doc]

        client = _FakeScrollClient([_Point(alias_id, payload={"entities": ["x"], "topics": []},
                                           vector=[1.0, 0.0])])

        payload, vectors, aliases = fn.build_payload_entities(client, "c", corpus_docs)

        assert payload == {"message:OLDKEY": {"entities": ["x"], "topics": []}}
        assert vectors == {"message:OLDKEY": [1.0, 0.0]}
        assert aliases == {"message:OLDKEY": alias_id}

    def test_doc_matching_neither_its_own_id_nor_the_exact_alias_raises(self):
        fn = _freeze()
        doc = _doc("message:LOST", "lost-point-id", sent_at="2026-01-01T00:00:00", text="gone")
        client = _FakeScrollClient([_Point("some-other-unrelated-id")])

        with pytest.raises(ValueError, match="message:LOST"):
            fn.build_payload_entities(client, "c", [doc])

    def test_no_fuzzy_matching_a_near_miss_alias_still_raises(self):
        """The alias must be the EXACT uuid5 of (sent_at, embedded_text) --
        a close-but-different text must not match."""
        fn = _freeze()
        sent_at = "2026-08-04T12:53:00"
        doc = _doc("message:ALMOST", "almost-point-id", sent_at=sent_at, text="hello world")
        near_miss_id = _own_id(sent_at, "hello world ")  # one trailing space difference

        client = _FakeScrollClient([_Point(near_miss_id)])

        with pytest.raises(ValueError, match="message:ALMOST"):
            fn.build_payload_entities(client, "c", [doc])


class TestFreezeAliasesGraphqCosine:
    def test_graph_cosine_hit_on_the_live_alias_id_is_written_under_the_corpus_id(self):
        fn = _freeze()
        sent_at, text = "2026-08-04T12:53:00", "same text, moved key"
        alias_id = _own_id(sent_at, text)
        doc = _doc("message:OLDKEY", "oldkey-not-live", sent_at=sent_at, text=text)
        alias_map = fn.build_alias_map([doc])

        vs = _FakeVectorStore({
            "g001": [_Hit(alias_id, 0.9, payload={"type": "message"})],
        })

        rows = fn.build_graphq_cosine_rows(
            [{"query_id": "g001", "query_text": "g001"}], vs, {"message:OLDKEY"},
            "vector_cosine_graphq_v1", alias_map=alias_map)

        assert len(rows) == 1
        assert rows[0]["candidate_id"] == "message:OLDKEY"
        assert rows[0]["rank"] == 1


class TestFreezeAliasesMap:
    def test_build_alias_map_keys_by_live_id_values_by_corpus_id(self):
        fn = _freeze()
        sent_at, text = "2026-08-04T12:53:00", "abc"
        alias_id = _own_id(sent_at, text)
        doc = _doc("message:X", "x-point", sent_at=sent_at, text=text)
        link = {"candidate_id": "link:Y", "candidate_type": "link",
                "qdrant_point_id": "y-point", "title": "t", "url": "http://x"}

        alias_map = fn.build_alias_map([doc, link])

        assert alias_map == {alias_id: "message:X"}  # links are never aliased this way
