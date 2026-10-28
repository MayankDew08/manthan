"""S2 retrieval (docs/mcp-server-spec.md section 2, tests per section 5).

Offline only: a fake vector store, fake BM25-ranked documents, and an
injected re-ranker scorer. No real Qdrant, no real model weights.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import retrieval  # noqa: E402
from evals.rerank import Reranker  # noqa: E402


# ---- fakes -------------------------------------------------------------------

class FakeQdrantClient:
    """Enough of the qdrant_client surface for Corpus.refresh and a filtered
    cosine query: count, scroll (paged), and a filtered search.
    """

    def __init__(self, docs):
        self.docs = docs  # [(point_id, payload)], in corpus order
        self.search_calls = []

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(self.docs))

    def scroll(self, collection_name, limit=256, offset=None, with_payload=False,
               with_vectors=False):
        start = offset or 0
        batch = [SimpleNamespace(id=pid, payload=payload)
                 for pid, payload in self.docs[start:start + limit]]
        nxt = start + limit if start + limit < len(self.docs) else None
        return batch, nxt

    def search(self, collection_name, query_vector, limit, query_filter=None):
        """Only called when a filter is active: apply it, then keep corpus order."""
        self.search_calls.append(query_filter)
        kept = [(pid, payload) for pid, payload in self.docs
                if _matches_filter(payload, query_filter)]
        return [SimpleNamespace(id=pid, score=1.0) for pid, _ in kept[:limit]]


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
    """Cosine ranking fixed per test, independent of the fake Qdrant client."""

    def __init__(self, docs, cosine_order):
        self.client = FakeQdrantClient(docs)
        self.collection = "manthan"
        self._cosine_order = cosine_order  # [point_id, ...] rank order

    def search(self, query, limit):
        return [SimpleNamespace(id=pid, score=1.0) for pid in self._cosine_order[:limit]]

    def _embed(self, query):
        return [0.0]


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


def _message(point_id, text, topics=None, quality=None):
    return (point_id, {"type": "message", "text": text, "sent_at": "2026-01-01T00:00:00",
                        "topics": topics or [], "quality": quality})


def _link(point_id, summary, topics=None):
    return (point_id, {"type": "link", "summary": summary, "title": "T", "url": "u",
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
    out = retrieval.search_s2("q", vs=vs, reranker=reranker, corpus=corpus)
    assert [r["id"] for r in out["results"]] == ["b", "a", "c"]


def test_top_k_is_respected():
    docs = [_message(f"m{i}", f"text {i}") for i in range(10)]
    vs = FakeVectorStore(docs, cosine_order=[f"m{i}" for i in range(10)])
    reranker = _reranker({f"m{i}": float(i) for i in range(10)})
    out = retrieval.search_s2("q", top_k=3, vs=vs, reranker=reranker)
    assert len(out["results"]) == 3


# ---- confidence ----------------------------------------------------------------

def test_confidence_strong_at_top_score_2_0():
    docs = [_message("a", "alpha")]
    vs = FakeVectorStore(docs, cosine_order=["a"])
    reranker = _reranker({"a": 2.0})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker)
    assert out["top_score"] == 2.0
    assert out["confidence"] == "strong"


def test_confidence_weak_just_under_threshold():
    docs = [_message("a", "alpha")]
    vs = FakeVectorStore(docs, cosine_order=["a"])
    reranker = _reranker({"a": 1.999})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker)
    assert out["confidence"] == "weak"


def test_confidence_none_when_empty():
    docs: list = []
    vs = FakeVectorStore(docs, cosine_order=[])
    reranker = _reranker({})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker)
    assert out["results"] == []
    assert out["top_score"] is None
    assert out["confidence"] == "none"


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
                              topics=["python"])
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
                              min_quality=5)
    ids = {r["id"] for r in out["results"]}
    assert "lowq" not in ids
    assert {"highq", "alink"} <= ids


def test_unfiltered_call_is_byte_identical_to_explicit_none_any():
    docs = [_message("a", "alpha", topics=["x"]), _message("b", "beta", topics=["y"])]
    vs = FakeVectorStore(docs, cosine_order=["a", "b"])
    corpus = retrieval.Corpus()
    corpus.refresh(vs)
    reranker = _reranker({"a": 1.0, "b": 2.0})

    implicit = retrieval.search_s2("q", vs=vs, reranker=reranker, corpus=corpus)
    explicit = retrieval.search_s2("q", vs=vs, reranker=reranker, corpus=corpus,
                                   topics=None, min_quality=None, kind="any")
    assert implicit == explicit


# ---- result shape ---------------------------------------------------------------

def test_result_keeps_every_key_search_py_carries_and_adds_id():
    docs = [_message("m1", "a message with text", topics=["t"])]
    vs = FakeVectorStore(docs, cosine_order=["m1"])
    reranker = _reranker({"m1": 3.0})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker)
    result = out["results"][0]
    for key in ("type", "sent_at", "text", "topics", "entities", "score", "id"):
        assert key in result


def test_snippet_is_cut_to_300_characters():
    long_text = "x" * 1000
    docs = [_message("m1", long_text)]
    vs = FakeVectorStore(docs, cosine_order=["m1"])
    reranker = _reranker({"m1": 1.0})
    out = retrieval.search_s2("q", vs=vs, reranker=reranker)
    assert len(out["results"][0]["text"]) == 300


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
