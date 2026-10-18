"""U2: freeze the production graph retrieval to a run file (spec section 6).

LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import hashlib
import inspect
import json
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import search  # noqa: E402
from evals import common, freeze_graph  # noqa: E402


def msg(sent_at, text, topics=(), entities=()):
    return search._message_result({"sent_at": sent_at, "text": text, "topics": list(topics),
                                   "entities": list(entities)}, 1.0, "graph")


def link(url, title="", summary="", topics=()):
    return search._link_result({"url": url, "title": title, "summary": summary,
                                "topics": list(topics), "entities": []}, 1.0, "graph")


def mid(sent_at, text):
    return "message:" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"msg-{sent_at}-{text}"))


def lid(url):
    return "link:" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"link-{url}"))


# ---- production code is imported, never retyped ---------------------------------

class TestProductionFidelity:
    def test_terms_is_the_production_function(self):
        assert freeze_graph._terms is search._terms

    def test_graph_candidates_is_the_production_function(self):
        assert freeze_graph._graph_candidates is search._graph_candidates

    def test_source_hash_covers_both_production_functions(self):
        expected = hashlib.sha256(
            (inspect.getsource(search._terms) + inspect.getsource(search._graph_candidates))
            .encode("utf-8")).hexdigest()
        assert freeze_graph.source_sha256() == expected

    def test_run_id(self):
        assert freeze_graph.RUN_ID == "graph_lexical_v0"


# ---- candidate ids use the vector_store.py point-id formula -----------------------

class TestCandidateIds:
    def test_message_id(self):
        r = msg("2026-07-21T21:04:00", "Anima is out")
        assert freeze_graph.candidate_id_for(r) == mid("2026-07-21T21:04:00", "Anima is out")

    def test_link_id(self):
        r = link("https://github.com/x/anima")
        assert freeze_graph.candidate_id_for(r) == lid("https://github.com/x/anima")

    def test_message_text_is_not_stripped_before_hashing(self):
        r = msg("t", "  padded \n")
        assert freeze_graph.candidate_id_for(r) == mid("t", "  padded \n")


# ---- matched terms and ordering ------------------------------------------------

class TestMatchedTerms:
    def test_distinct_terms_in_query_order(self):
        r = msg("t", "Anima: an LLM interpretability tool. anima again")
        terms = ["llm", "interpretability", "anima", "group"]
        assert freeze_graph.matched_terms(r, terms) == ["llm", "interpretability", "anima"]

    def test_substring_semantics_match_production_contains(self):
        r = msg("t", "OpenAI released tools")
        assert freeze_graph.matched_terms(r, ["open", "tool"]) == ["open", "tool"]

    def test_case_insensitive(self):
        assert freeze_graph.matched_terms(msg("t", "QDRANT rocks"), ["qdrant"]) == ["qdrant"]

    def test_topics_and_entities_count(self):
        r = msg("t", "nothing here", topics=["rag"], entities=["Qdrant"])
        assert freeze_graph.matched_terms(r, ["rag", "qdrant"]) == ["rag", "qdrant"]

    def test_link_title_and_summary_count(self):
        r = link("https://u", title="Anima repo", summary="interpretability")
        assert freeze_graph.matched_terms(r, ["anima", "interpretability"]) == [
            "anima", "interpretability"]


class TestGraphRows:
    TERMS = ["anima", "interpretability", "group"]

    def corpus(self, *ids):
        return set(ids)

    def test_ordered_by_matched_term_count_then_candidate_id(self):
        a = msg("t1", "group chat")                                # 1 term
        b = msg("t2", "Anima interpretability")                    # 2 terms
        c = msg("t3", "anima in the group with interpretability")  # 3 terms
        d = msg("t4", "another group note")                        # 1 term
        ids = [freeze_graph.candidate_id_for(x) for x in (a, b, c, d)]
        rows, _ = freeze_graph.graph_rows_for_query("q018", self.TERMS, [a, b, c, d], set(ids))
        got = [r["candidate_id"] for r in rows]
        assert got[:2] == [ids[2], ids[1]]
        assert got[2:] == sorted([ids[0], ids[3]])
        assert [r["rank"] for r in rows] == [1, 2, 3, 4]
        assert [r["score"] for r in rows] == [3.0, 2.0, 1.0, 1.0]

    def test_row_schema(self):
        a = msg("t1", "anima")
        rows, _ = freeze_graph.graph_rows_for_query(
            "q018", self.TERMS, [a], {freeze_graph.candidate_id_for(a)})
        assert set(rows[0]) == {"run_id", "query_id", "candidate_id", "rank", "score",
                                "matched_terms"}
        assert rows[0]["run_id"] == "graph_lexical_v0" and rows[0]["query_id"] == "q018"
        assert rows[0]["matched_terms"] == ["anima"]

    def test_duplicates_from_several_cypher_arms_collapse(self):
        a = msg("t1", "anima")
        same = msg("t1", "anima")  # e.g. matched by text AND by entity
        rows, stats = freeze_graph.graph_rows_for_query(
            "q018", self.TERMS, [a, same], {freeze_graph.candidate_id_for(a)})
        assert len(rows) == 1
        assert stats["raw_results"] == 2 and stats["distinct"] == 1

    def test_out_of_corpus_results_are_counted_never_ranked(self):
        inside = msg("t1", "anima")
        outside = msg("t9", "anima interpretability group")  # quality-3, not in Qdrant
        rows, stats = freeze_graph.graph_rows_for_query(
            "q018", self.TERMS, [inside, outside], {freeze_graph.candidate_id_for(inside)})
        assert [r["candidate_id"] for r in rows] == [freeze_graph.candidate_id_for(inside)]
        assert stats == {"query_id": "q018", "terms": self.TERMS, "raw_results": 2,
                         "distinct": 2, "in_corpus": 1, "out_of_corpus": 1}

    def test_no_depth_cut_at_freeze_time(self):
        results = [msg(f"t{i:03d}", "group") for i in range(45)]
        ids = {freeze_graph.candidate_id_for(r) for r in results}
        rows, _ = freeze_graph.graph_rows_for_query("q1", self.TERMS, results, ids)
        assert len(rows) == 45  # the 30-cut happens in fusion, so stats can show rank 40

    def test_empty(self):
        rows, stats = freeze_graph.graph_rows_for_query("q1", self.TERMS, [], set())
        assert rows == [] and stats["raw_results"] == 0


# ---- live cosine consistency ------------------------------------------------------

def crow(q, cid, rank, score):
    return {"query_id": q, "candidate_id": cid, "rank": rank, "score": score}


class TestCosineMismatches:
    def test_identical_is_clean(self):
        rows = [crow("q1", "a", 1, 0.9), crow("q1", "b", 2, 0.8)]
        assert freeze_graph.cosine_mismatches(rows, rows) == []

    def test_swap_inside_a_score_tie_is_tolerated(self):
        frozen = [crow("q1", "a", 1, 0.80000), crow("q1", "b", 2, 0.80000)]
        live = [crow("q1", "b", 1, 0.80001), crow("q1", "a", 2, 0.80000)]
        assert freeze_graph.cosine_mismatches(live, frozen) == []

    def test_real_reorder_is_reported(self):
        frozen = [crow("q1", "a", 1, 0.9), crow("q1", "b", 2, 0.5)]
        live = [crow("q1", "b", 1, 0.9), crow("q1", "a", 2, 0.5)]
        problems = freeze_graph.cosine_mismatches(live, frozen)
        assert problems and "q1" in problems[0]

    def test_only_top_depth_is_compared(self):
        frozen = [crow("q1", "a", 1, 0.9), crow("q1", "b", 2, 0.5), crow("q1", "c", 3, 0.1)]
        live = [crow("q1", "a", 1, 0.9), crow("q1", "b", 2, 0.5), crow("q1", "z", 3, 0.4)]
        assert freeze_graph.cosine_mismatches(live, frozen, depth=2) == []

    def test_missing_query_is_reported(self):
        frozen = [crow("q1", "a", 1, 0.9), crow("q2", "a", 1, 0.9)]
        live = [crow("q1", "a", 1, 0.9)]
        assert any("q2" in p for p in freeze_graph.cosine_mismatches(live, frozen))


# ---- live --------------------------------------------------------------------

@pytest.mark.live
def test_live_every_corpus_document_is_reachable_from_neo4j():
    """If this fails, ids computed from Neo4j text differ from Qdrant ids: STOP, escalate."""
    from evals import live_guard
    common.load_env()
    store = live_guard.read_only_store()
    try:
        msgs = live_guard.neo4j_read(store.driver,
                                     "MATCH (m:Message) RETURN m.sent_at AS sent_at, m.text AS text")
        links = live_guard.neo4j_read(store.driver, "MATCH (l:Link) RETURN l.url AS url")
    finally:
        store.close()
    reachable = {mid(m["sent_at"] or "", m["text"] or "") for m in msgs}
    reachable |= {lid(l["url"]) for l in links if l["url"]}
    corpus = {d["candidate_id"] for d in common.load_documents(common.DEFAULT_CORPUS_VERSION)}
    missing = sorted(corpus - reachable)
    assert missing == [], f"{len(missing)} corpus documents unreachable from Neo4j"


@pytest.mark.live
def test_live_cosine_ranking_matches_the_frozen_run():
    from evals import live_guard
    common.load_env()
    vs = live_guard.read_only_vector_store()
    try:
        live_rows = []
        for q in common.load_queries():
            for rank, hit in enumerate(vs.search(q["query_text"], limit=30), start=1):
                live_rows.append(crow(q["query_id"], f"{(hit.payload or {}).get('type')}:{hit.id}",
                                      rank, float(hit.score)))
    finally:
        vs.close()
    frozen = common.load_run(common.DEFAULT_CORPUS_VERSION, "vector_cosine_v1")
    assert freeze_graph.cosine_mismatches(live_rows, frozen, depth=30) == []


@pytest.mark.artifacts
def test_artifact_frozen_graph_run_is_complete():
    data = common.data_dir(common.DEFAULT_CORPUS_VERSION, create=False) / "runs"
    rows = common.read_jsonl(data / "graph_lexical_v0.jsonl")
    meta = json.loads((data / "graph_lexical_v0.meta.json").read_text(encoding="utf-8"))
    corpus = {d["candidate_id"] for d in common.load_documents(common.DEFAULT_CORPUS_VERSION)}
    qids = {q["query_id"] for q in common.load_queries()}
    assert {r["candidate_id"] for r in rows} <= corpus
    assert set(meta["per_query"]) == qids
    assert meta["source_sha256"] == freeze_graph.source_sha256()
    assert meta["fingerprint_before"] == meta["fingerprint_after"]
    for qid, stats in meta["per_query"].items():
        ranks = [r["rank"] for r in rows if r["query_id"] == qid]
        assert ranks == list(range(1, stats["in_corpus"] + 1))
