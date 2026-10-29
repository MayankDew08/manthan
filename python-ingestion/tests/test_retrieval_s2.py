"""S2 retrieval (docs/mcp-server-spec.md section 2, tests per section 5).

Offline only: a fake vector store, fake BM25-ranked documents, a fake Neo4j
store, and an injected re-ranker scorer. No real Qdrant, no real Neo4j, no
real model weights.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import retrieval  # noqa: E402
from evals.rerank import Reranker  # noqa: E402


# ---- fakes -------------------------------------------------------------------

class FakeQdrantClient:
    """Enough of the qdrant_client surface for Corpus.refresh: count and a
    paged scroll. ``fail_scroll_at`` raises on the Nth scroll call, to test
    that an interrupted refresh leaves the old snapshot untouched.
    """

    def __init__(self, docs):
        self.docs = docs  # [(point_id, payload)], in corpus order
        self.scroll_calls = 0
        self.fail_scroll_at: Optional[int] = None

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(self.docs))

    def scroll(self, collection_name, limit=256, offset=None, with_payload=False,
               with_vectors=False):
        self.scroll_calls += 1
        if self.fail_scroll_at == self.scroll_calls:
            raise RuntimeError("boom: simulated Qdrant scroll failure")
        start = offset or 0
        batch = [SimpleNamespace(id=pid, payload=payload)
                 for pid, payload in self.docs[start:start + limit]]
        nxt = start + limit if start + limit < len(self.docs) else None
        return batch, nxt


def _matches_filter(payload, query_filter):
    """Minimal interpreter for the Filter shapes retrieval._qdrant_filter builds."""
    if query_filter is None:
        return True
    for condition in query_filter.must:
        if hasattr(condition, "should"):  # nested Filter(should=[...]) for min_quality
            if not any(_matches_condition(payload, c) for c in condition.should):
                return False
        elif not _matches_condition(payload, condition):
            return False
    return True


def _matches_condition(payload, condition):
    key, match = condition.key, condition.match
    if match is not None:
        if hasattr(match, "any"):
            return bool(set(payload.get(key) or []).intersection(match.any))
        if hasattr(match, "value"):
            return payload.get(key) == match.value
    range_ = condition.range
    if range_ is not None:
        return (payload.get(key) or 0) >= range_.gte
    return True


class FakeVectorStore:
    """Cosine ranking fixed per test; ``search`` applies ``query_filter`` itself,
    same contract as ``vector_store.VectorStore.search`` (item 4: retrieval.py
    never touches ``vs.client`` or ``vs._embed`` for the cosine query).
    """

    def __init__(self, docs, cosine_order):
        self.client = FakeQdrantClient(docs)
        self.collection = "manthan"
        self._cosine_order = cosine_order  # [point_id, ...] rank order
        self._payloads = dict(docs)

    def search(self, query, limit, query_filter=None):
        ordered = [pid for pid in self._cosine_order
                   if _matches_filter(self._payloads.get(pid, {}), query_filter)]
        return [SimpleNamespace(id=pid, score=1.0) for pid in ordered[:limit]]


class FakeStore:
    """Enough of ``store.KnowledgeStore`` for ``_graph_enrich``: two
    independent read-only queries over an in-memory CONTAINS edge list.

    ``query_log`` records which query actually ran (``"message_links"`` /
    ``"link_attribution"``), so tests can assert the empty-input side is
    skipped rather than issued as a no-op round trip.
    """

    def __init__(self, contains=()):
        self._contains = list(contains)  # [{"sent_at", "text", "url", "title"}]
        self.query_log: list = []
        self.driver = SimpleNamespace(session=lambda: _FakeSession(self))

    @property
    def query_count(self) -> int:
        return len(self.query_log)

    def run(self, messages=None, urls=None):
        if messages is not None:
            self.query_log.append("message_links")
            return [
                dict(c) for c in self._contains
                if any(c["sent_at"] == m["sent_at"] and c["text"] == m["text"]
                       for m in messages)
            ]
        if urls is not None:
            self.query_log.append("link_attribution")
            return [
                {"link_url": url, "earliest_sent_at": min(
                    (c["sent_at"] for c in self._contains if c["url"] == url), default=None)}
                for url in urls
            ]
        raise AssertionError("FakeStore.run needs messages= or urls=")


class _FakeSession:
    def __init__(self, store):
        self._store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        rows = self._store.run(**params)
        return SimpleNamespace(data=lambda: rows)


def _store(contains=()):
    return FakeStore(contains)


def _reranker(scores):
    """A Reranker whose score comes from the candidate id, not the text, so
    distinct candidates may share identical embedded text in a test.

    ``Reranker.rerank`` only ever hands its scorer (query, text) pairs, with
    no id -- so this wraps ``rerank`` itself rather than ``score_pairs``.
    """
    r = Reranker()

    def rerank(query, candidates):
        text_to_id = {text: cid for cid, text in candidates}
        pairs = [(query, text) for _cid, text in candidates]
        raw = [scores[text_to_id[text]] for _q, text in pairs]
        scored = [(cid, round(float(s), 6)) for (cid, _t), s in zip(candidates, raw)]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    r.rerank = rerank  # type: ignore[method-assign]
    return r


def _message(point_id, text, topics=None, quality=None, sent_at="2026-01-01T00:00:00"):
    return (point_id, {"type": "message", "text": text, "sent_at": sent_at,
                        "topics": topics or [], "quality": quality})


def _link(point_id, summary, topics=None, url="u"):
    return (point_id, {"type": "link", "summary": summary, "title": "T", "url": url,
                        "topics": topics or []})


# ---- ordering: rrf then rerank ------------------------------------------------

def test_order_equals_rrf_then_rerank_on_hand_computed_inputs():
    docs = [_message("a", "alpha"), _message("b", "beta"), _message("c", "gamma")]
    # cosine: a, b, c  |  bm25 (via corpus.bm25, patched below): b, a, c
    vs = FakeVectorStore(docs, cosine_order=["a", "b", "c"])
    corpus = retrieval.Corpus()
    corpus.refresh(vs)
    corpus.bm25 = SimpleNamespace(top=lambda query, n: [(1, 9.0), (0, 5.0), (2, 1.0)])
    # fused order by hand: RRF(["a","b","c"], ["b","a","c"]) -> a and b tie at
    # rank1+rank2, broken by id ("a" < "b"), then c last.
    expected_fused = [cid for cid, _ in retrieval.rrf([["a", "b", "c"], ["b", "a", "c"]])]
    assert expected_fused == ["a", "b", "c"]

    reranker = _reranker({"a": 1.0, "b": 5.0, "c": 0.5})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker, corpus=corpus, store=_store())
    assert [r["id"] for r in out["results"]] == ["b", "a", "c"]


def test_top_k_is_respected():
    docs = [_message(f"m{i}", f"text {i}") for i in range(10)]
    vs = FakeVectorStore(docs, cosine_order=[f"m{i}" for i in range(10)])
    reranker = _reranker({f"m{i}": float(i) for i in range(10)})
    out = retrieval.search_s2("q", top_k=3, vs=vs, reranker=reranker, store=_store())
    assert len(out["results"]) == 3


# ---- confidence ----------------------------------------------------------------

def test_confidence_strong_at_top_score_2_0():
    docs = [_message("a", "alpha")]
    vs = FakeVectorStore(docs, cosine_order=["a"])
    reranker = _reranker({"a": 2.0})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker, store=_store())
    assert out["top_score"] == 2.0
    assert out["confidence"] == "strong"


def test_confidence_weak_just_under_threshold():
    docs = [_message("a", "alpha")]
    vs = FakeVectorStore(docs, cosine_order=["a"])
    reranker = _reranker({"a": 1.999})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker, store=_store())
    assert out["confidence"] == "weak"


def test_confidence_none_when_empty():
    docs: list = []
    vs = FakeVectorStore(docs, cosine_order=[])
    reranker = _reranker({})
    store = _store()
    out = retrieval.search_s2("q", vs=vs, reranker=reranker, store=store)
    assert out["results"] == []
    assert out["top_score"] is None
    assert out["confidence"] == "none"
    assert store.query_count == 0  # nothing to enrich: no Neo4j round trip


# ---- pre-pool filtering (M6) -----------------------------------------------------

def test_filtered_out_top_ranked_doc_never_appears():
    docs = [
        _message("top", "the best match", topics=["other"]),
        _message("second", "a decent match", topics=["python"]),
    ]
    # "top" ranks first on both cosine and BM25, but is filtered out by topic.
    vs = FakeVectorStore(docs, cosine_order=["top", "second"])
    corpus = retrieval.Corpus()
    corpus.refresh(vs)
    reranker = _reranker({"top": 10.0, "second": 1.0})

    out = retrieval.search_s2("q", vs=vs, reranker=reranker, corpus=corpus,
                              topics=["python"], store=_store())
    ids = [r["id"] for r in out["results"]]
    assert "top" not in ids
    assert ids == ["second"]


def test_min_quality_filter_excludes_low_quality_messages_but_not_links():
    docs = [
        _message("lowq", "low quality", quality=1),
        _message("highq", "high quality", quality=9),
        _link("alink", "a link summary"),
    ]
    vs = FakeVectorStore(docs, cosine_order=["lowq", "highq", "alink"])
    corpus = retrieval.Corpus()
    corpus.refresh(vs)
    reranker = _reranker({"lowq": 5.0, "highq": 3.0, "alink": 1.0})

    out = retrieval.search_s2("q", vs=vs, reranker=reranker, corpus=corpus,
                              min_quality=5, store=_store())
    ids = {r["id"] for r in out["results"]}
    assert "lowq" not in ids
    assert {"highq", "alink"} <= ids


def test_unfiltered_call_is_byte_identical_to_explicit_none_any():
    docs = [_message("a", "alpha", topics=["x"]), _message("b", "beta", topics=["y"])]
    vs = FakeVectorStore(docs, cosine_order=["a", "b"])
    corpus = retrieval.Corpus()
    corpus.refresh(vs)
    reranker = _reranker({"a": 1.0, "b": 2.0})

    implicit = retrieval.search_s2("q", vs=vs, reranker=reranker, corpus=corpus, store=_store())
    explicit = retrieval.search_s2("q", vs=vs, reranker=reranker, corpus=corpus,
                                   topics=None, min_quality=None, kind="any", store=_store())
    assert implicit == explicit


# ---- result shape ---------------------------------------------------------------

def test_result_keeps_every_key_search_py_carries_and_adds_id():
    docs = [_message("m1", "a message with text", topics=["t"])]
    vs = FakeVectorStore(docs, cosine_order=["m1"])
    reranker = _reranker({"m1": 3.0})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker, store=_store())
    result = out["results"][0]
    for key in ("type", "sent_at", "text", "topics", "entities", "score", "id",
                "related_links"):
        assert key in result


def test_full_text_is_not_truncated():
    """/search returns full text again (item 2); SNIPPET_CHARS is reserved
    for Phase 2's MCP layer, not applied here."""
    long_text = "x" * 1000
    docs = [_message("m1", long_text)]
    vs = FakeVectorStore(docs, cosine_order=["m1"])
    reranker = _reranker({"m1": 1.0})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker, store=_store())
    assert len(out["results"][0]["text"]) == 1000


def test_snippet_chars_constant_is_still_exported_for_phase_2():
    assert retrieval.SNIPPET_CHARS == 300


# ---- graph enrichment (item 1): related_links / link sent_at -------------------
#
# Two independent queries, not one combined pipeline (regression: a single
# query built from two CALL subqueries drops ALL rows, including the
# message side, whenever the url side's input is empty -- see the mixed,
# messages-only and links-only cases below).

def test_graph_enrichment_mixed_batch_runs_both_queries():
    docs = [_message("m1", "message one"), _link("l1", "link summary", url="u")]
    vs = FakeVectorStore(docs, cosine_order=["m1", "l1"])
    reranker = _reranker({"m1": 3.0, "l1": 2.0})
    store = _store(contains=[
        {"sent_at": "2025-12-31T00:00:00", "text": "an earlier message",
         "url": "u", "title": "T"},
        {"sent_at": "2026-02-01T00:00:00", "text": "a later message",
         "url": "u", "title": "T"},
        {"sent_at": "2026-01-01T00:00:00", "text": "message one",
         "url": "other-url", "title": "Other"},
    ])

    out = retrieval.search_s2("q", vs=vs, reranker=reranker, store=store)

    by_id = {r["id"]: r for r in out["results"]}
    assert by_id["m1"]["related_links"] == [{"url": "other-url", "title": "Other"}]
    assert by_id["l1"]["sent_at"] == "2025-12-31T00:00:00"  # earliest of two edges to "u"
    assert store.query_log == ["message_links", "link_attribution"]


def test_graph_enrichment_messages_only_batch_still_fills_related_links():
    """Regression: no link results in the top-k must not blank out the
    message side too (the bug: an empty $urls made the whole combined query
    return zero rows, silently dropping every message's related_links)."""
    docs = [_message("m1", "message one")]
    vs = FakeVectorStore(docs, cosine_order=["m1"])
    reranker = _reranker({"m1": 3.0})
    store = _store(contains=[
        {"sent_at": "2026-01-01T00:00:00", "text": "message one",
         "url": "other-url", "title": "Other"},
    ])

    out = retrieval.search_s2("q", vs=vs, reranker=reranker, store=store)

    assert out["results"][0]["related_links"] == [{"url": "other-url", "title": "Other"}]
    assert store.query_log == ["message_links"]  # link_attribution skipped: no urls


def test_graph_enrichment_links_only_batch_still_fills_sent_at():
    """Regression: no message results in the top-k must not blank out the
    link side (empty $messages making the combined query return zero rows)."""
    docs = [_link("l1", "link summary", url="u")]
    vs = FakeVectorStore(docs, cosine_order=["l1"])
    reranker = _reranker({"l1": 1.0})
    store = _store(contains=[
        {"sent_at": "2025-12-31T00:00:00", "text": "an earlier message",
         "url": "u", "title": "T"},
    ])

    out = retrieval.search_s2("q", vs=vs, reranker=reranker, store=store)

    assert out["results"][0]["sent_at"] == "2025-12-31T00:00:00"
    assert store.query_log == ["link_attribution"]  # message_links skipped: no messages


def test_graph_enrichment_leaves_message_without_links_empty():
    docs = [_message("m1", "lonely message")]
    vs = FakeVectorStore(docs, cosine_order=["m1"])
    reranker = _reranker({"m1": 1.0})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker, store=_store())
    assert out["results"][0]["related_links"] == []


# ---- graph enrichment, live (deselected unless --live) --------------------------

@pytest.mark.live
def test_live_graph_enrich_fills_related_links_for_messages_only_batch():
    """Same regression as the offline messages-only test, against the real,
    read-only Neo4j: a messages-only batch (no link results in the top-k)
    must still come back with at least one non-empty related_links.
    """
    from evals import common, live_guard

    common.load_env()
    store = live_guard.read_only_store()
    try:
        rows = live_guard.neo4j_read(
            store.driver,
            "MATCH (m:Message)-[:CONTAINS]->(:Link) "
            "RETURN m.sent_at AS sent_at, m.text AS text LIMIT 1")
        assert rows, "fixture assumption: at least one Message CONTAINS a Link"
        sent_at, text = rows[0]["sent_at"], rows[0]["text"]

        results = [{"type": "message", "sent_at": sent_at, "text": text,
                    "related_links": []}]
        retrieval._graph_enrich(store, results)

        assert results[0]["related_links"], "messages-only batch lost its related_links"
    finally:
        store.close()


# ---- Corpus: refresh-on-size-change, atomic snapshot swap (item 3) ---------------

def test_corpus_rebuilds_only_when_collection_size_changes():
    docs = [_message("a", "alpha")]
    vs = FakeVectorStore(docs, cosine_order=["a"])
    corpus = retrieval.Corpus()
    corpus.refresh(vs)
    first_bm25 = corpus.bm25
    corpus.refresh(vs)  # same size: no rebuild
    assert corpus.bm25 is first_bm25

    vs.client.docs.append(_link("l1", "a new link"))
    corpus.refresh(vs)  # size grew: rebuild
    assert corpus.bm25 is not first_bm25
    assert "l1" in corpus.ids


def test_corpus_refresh_leaves_old_snapshot_untouched_if_interrupted():
    """A crash mid-refresh (e.g. Qdrant drops a scroll page) must never leave
    ``ids``/``payloads``/``bm25`` partially updated -- the old snapshot stays
    whole until the single assignment that replaces it succeeds."""
    docs = [_message("a", "alpha")]
    vs = FakeVectorStore(docs, cosine_order=["a"])
    corpus = retrieval.Corpus()
    corpus.refresh(vs)
    old_ids, old_bm25, old_payloads = corpus.ids, corpus.bm25, corpus.payloads

    vs.client.docs.append(_link("l1", "a new link"))  # size changes: refresh must rebuild
    vs.client.fail_scroll_at = vs.client.scroll_calls + 1  # fail the very next scroll call

    with pytest.raises(RuntimeError):
        corpus.refresh(vs)

    assert corpus.ids == old_ids
    assert corpus.bm25 is old_bm25
    assert corpus.payloads == old_payloads
