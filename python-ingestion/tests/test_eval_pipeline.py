"""Pure-logic tests for the eval pipeline (no Qdrant, Neo4j or model needed)."""

import csv
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import common, evaluate, pooling
from evals.bm25 import BM25
from evals.export_corpus import pii_findings, points_to_documents


def doc(n: int, text: str, ctype: str = "message", truncated: bool = False, **kw) -> dict:
    return {"candidate_id": f"{ctype}:{n:04d}", "candidate_type": ctype,
            "qdrant_point_id": f"{n:04d}", "embedded_text": text, "truncated": truncated,
            "title": kw.get("title"), "url": kw.get("url")}


# ---- PII gate ----------------------------------------------------------------

class TestPiiGate:
    def test_flags_a_real_phone_number(self):
        assert common.find_phone_numbers("call +91 98765 43210 now")

    def test_flags_a_system_event_prefix(self):
        assert common.find_phone_numbers("+1 555 0100 was added")

    def test_ignores_numbers_inside_urls(self):
        text = ("https://arxiv.org/html/2312.10997v1 and "
                "https://x.com/trq212/status/2088280188362867185")
        assert common.find_phone_numbers(text) == []

    def test_ordinary_prose_is_clean(self):
        assert common.find_phone_numbers("35b MoE at around 90 tokens per second") == []

    def test_mask_keeps_only_the_last_two_digits(self):
        assert common.mask_digits("+91 98765 43210") == "+## ##### ###10"

    def test_findings_never_echo_the_number(self):
        found = pii_findings([doc(1, "ping +91 98765 43210")])
        assert found and "98765" not in found[0]


# ---- manifest ----------------------------------------------------------------

class TestManifest:
    def points(self):
        return [
            SimpleNamespace(id="b", payload={"type": "message", "text": "  hello world  ",
                                             "sent_at": "2026-01-01T00:00:00"}),
            SimpleNamespace(id="a", payload={"type": "link", "summary": "S", "what_it_is": "W",
                                             "problem_solved": "", "title": "T", "url": "http://x"}),
        ]

    def test_embedded_text_matches_vector_store_rules(self):
        docs = {d["candidate_type"]: d for d in
                points_to_documents(self.points(), lambda t: len(t.split()), 256, "v")}
        assert docs["message"]["embedded_text"] == "hello world"
        assert docs["link"]["embedded_text"] == "S W"

    def test_truncation_flag_uses_the_model_limit(self):
        docs = points_to_documents(self.points(), lambda t: 300 if t == "S W" else 3, 256, "v")
        flags = {d["candidate_type"]: d["truncated"] for d in docs}
        assert flags == {"message": False, "link": True}

    def test_order_is_deterministic_and_ids_are_typed(self):
        docs = points_to_documents(self.points(), len, 256, "v")
        assert [d["candidate_id"] for d in docs] == ["link:a", "message:b"]

    def test_no_sender_field_is_ever_exported(self):
        pts = self.points()
        pts[0].payload["sender"] = "someone"
        docs = points_to_documents(pts, len, 256, "v")
        assert all("sender" not in d for d in docs)

    def test_unknown_point_type_is_rejected(self):
        with pytest.raises(ValueError):
            points_to_documents([SimpleNamespace(id="z", payload={"type": "x"})], len, 256, "v")

    def test_content_hash_differs_for_different_text(self):
        assert common.content_hash("a") != common.content_hash("b")


# ---- duplicates vs reposts (owner decision 2026-10-09) -----------------------

class TestDuplicates:
    def mk(self, n, text, sent_at):
        d = doc(n, text)
        d["sent_at"], d["content_hash"] = sent_at, common.content_hash(text)
        return d

    def test_same_text_same_time_is_a_true_duplicate(self):
        from evals.parity_check import duplicate_documents
        docs = [self.mk(1, "thanks", "2026-05-30T22:36:00"), self.mk(2, "thanks", "2026-05-30T22:36:00")]
        assert duplicate_documents(docs) == [common.content_hash("thanks")]

    def test_same_text_different_time_is_an_allowed_repost(self):
        from evals.parity_check import duplicate_documents, reposts
        docs = [self.mk(1, "thanks", "2026-05-30T22:36:00"), self.mk(2, "thanks", "2026-05-30T22:37:00")]
        assert duplicate_documents(docs) == []
        assert reposts(docs) == [common.content_hash("thanks")]

    def test_unique_texts_are_neither(self):
        from evals.parity_check import duplicate_documents, reposts
        docs = [self.mk(1, "a", "t1"), self.mk(2, "b", "t1")]
        assert duplicate_documents(docs) == [] and reposts(docs) == []


# ---- intro heuristic ---------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Hi everyone! I'm Anup, a Software Engineer with around 4 years", True),
    ("Hello, I am Avneesh. I work as Software Dev/ML engineer", True),
    ("Retrieval quality beats model size in RAG systems", False),
    ("Key insight from todays reading: hybrid search wins", False),
])
def test_looks_like_intro(text, expected):
    assert common.looks_like_intro(text) is expected


# ---- pooling, blindness, coverage -------------------------------------------

def corpus(n=60):
    docs = [doc(i, f"filler message number {i} about nothing") for i in range(n)]
    docs[45]["embedded_text"] = "Local Model Setup: started with a 35b MoE then llama.cpp"
    docs[50]["embedded_text"] = "Qdrant is a vector database"
    return docs


class TestPooling:
    def setup_method(self):
        self.docs = corpus()
        self.ids = [d["candidate_id"] for d in self.docs]
        self.bm25 = BM25([d["embedded_text"] for d in self.docs])

    def test_pool_is_the_union_with_provenance(self):
        pool = pooling.select_pool("local model moe", self.docs, self.ids, self.bm25, ["qdrant"])
        assert self.ids[0] in pool and pool[self.ids[0]]["cosine_rank"] == 1
        assert "bm25_top20" in pool[self.ids[45]]["sources"]
        assert "keyword:qdrant" in pool[self.ids[50]]["sources"]

    def test_document_below_cosine_cutoff_is_rescued_by_bm25(self):
        pool = pooling.select_pool("local model moe", self.docs, self.ids, self.bm25, [])
        assert self.ids[45] in pool and pool[self.ids[45]]["cosine_rank"] is None

    def test_keywords_match_whole_tokens_only(self):
        docs = [doc(1, "Samoed dogs"), doc(2, "a MoE model")]
        pool = pooling.select_pool("zzz", docs, [], BM25([d["embedded_text"] for d in docs]),
                                   ["moe"])
        assert list(pool) == ["message:0002"]

    def test_shuffle_is_seeded_and_never_equals_cosine_order(self):
        ids = self.ids[:30]
        a = pooling.shuffle_pool(ids, ids, random.Random(42))
        b = pooling.shuffle_pool(ids, ids, random.Random(42))
        assert a == b and a != ids and sorted(a) == sorted(ids)

    def test_annotation_rows_are_blind(self):
        q = {"query_id": "q1", "query_text": "text"}
        link = doc(7, "summary", "link", title="Title", url="http://u")
        rows = pooling.annotation_rows(q, [link["candidate_id"]], {link["candidate_id"]: link})
        assert list(rows[0]) == common.ANNOTATION_COLUMNS
        assert rows[0]["content"].startswith("Title\nhttp://u")
        assert not {"score", "rank", "source", "run"} & set(rows[0])

    def _write(self, path, ordered_ids, columns=None):
        pooling.write_annotation_csv(path, [
            {"query_id": "q1", "query_text": "t", "candidate_id": c, "candidate_type": "message",
             "content": "x", "relevance": "", "reason": ""} for c in ordered_ids])
        if columns:
            text = path.read_text().replace("reason", columns)
            path.write_text(text)

    def test_blindness_passes_for_a_shuffled_sheet(self, tmp_path):
        p = tmp_path / "annotation.csv"
        self._write(p, ["c3", "c1", "c2"])
        assert pooling.verify_annotation(p, {"q1": ["c1", "c2", "c3"]}) == []

    def test_blindness_fails_when_rows_follow_rank_order(self, tmp_path):
        p = tmp_path / "annotation.csv"
        self._write(p, ["c1", "c2", "c3"])
        assert pooling.verify_annotation(p, {"q1": ["c1", "c2", "c3"]})

    def test_blindness_fails_on_extra_or_renamed_columns(self, tmp_path):
        p = tmp_path / "annotation.csv"
        self._write(p, ["c3", "c1", "c2"], columns="score")
        assert pooling.verify_annotation(p, {"q1": ["c1", "c2", "c3"]})

    def test_must_pool_passes_when_marker_document_is_pooled(self):
        pool = {"q5": {self.ids[45]: {}}}
        assert pooling.verify_must_pool({"q5": ["Local Model Setup"]}, self.docs, pool) == []

    def test_must_pool_fails_when_marker_document_is_missing_from_pool(self):
        assert pooling.verify_must_pool({"q5": ["Local Model Setup"]}, self.docs, {"q5": {}})

    def test_must_pool_fails_when_no_document_has_the_marker(self):
        assert pooling.verify_must_pool({"q5": ["nonexistent phrase"]}, self.docs, {"q5": {}})


# ---- qrels -------------------------------------------------------------------

class TestQrels:
    def write(self, tmp_path, rows):
        p = tmp_path / "qrels.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "candidate_id", "relevance", "answer_bearing", "reason"])
            w.writerows(rows)
        return p

    def test_blank_grade_means_not_judged(self, tmp_path):
        p = self.write(tmp_path, [["q1", "c1", "", "", ""], ["q1", "c2", "2", "1", "ok"]])
        assert common.load_qrels(p, {"c1", "c2"}) == {"q1": {"c2": {"relevance": 2, "answer_bearing": 1}}}

    def test_grade_outside_scale_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            common.load_qrels(self.write(tmp_path, [["q1", "c1", "4", "", ""]]), {"c1"})

    def test_unknown_candidate_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            common.load_qrels(self.write(tmp_path, [["q1", "zz", "1", "", ""]]), {"c1"})

    def test_annotation_sheet_converts_to_qrels(self, tmp_path):
        ann = tmp_path / "annotation.csv"
        pooling.write_annotation_csv(ann, [
            {"query_id": "q1", "query_text": "t", "candidate_id": "c1", "candidate_type": "message",
             "content": "x", "relevance": "3", "reason": "direct"},
            {"query_id": "q1", "query_text": "t", "candidate_id": "c2", "candidate_type": "message",
             "content": "y", "relevance": "", "reason": ""}])
        out = tmp_path / "qrels.csv"
        assert evaluate.annotation_to_qrels(ann, out) == 1
        assert common.load_qrels(out, {"c1", "c2"})["q1"]["c1"]["relevance"] == 3


# ---- end-to-end evaluate() on a tiny synthetic run ---------------------------

def test_evaluate_end_to_end():
    docs = {f"message:{i}": doc(i, f"text {i}", truncated=(i == 9)) for i in range(10)}
    queries = [{"query_id": "q1", "intent": "a"}, {"query_id": "q2", "intent": "b"},
               {"query_id": "q3", "intent": "c"}]

    def run(qid, order):
        return [{"query_id": qid, "candidate_id": f"message:{i}", "rank": r,
                 "score": 0.9 - 0.08 * r} for r, i in enumerate(order, start=1)]

    rows = run("q1", range(10)) + run("q2", range(10)) + run("q3", range(10))
    qrels = {"q1": {"message:0": {"relevance": 3, "answer_bearing": 1}},   # rank 1
             "q2": {"message:9": {"relevance": 3, "answer_bearing": 1}}}   # rank 10, truncated
    result = evaluate.evaluate(queries, docs, rows, qrels,
                               {"q1": {"expected": "success"}, "q2": {"expected": "success"}})
    agg, pq = result["aggregate"], result["per_query"]
    assert agg["queries_unlabelled"] == ["q3"] and agg["queries_evaluated"] == 2
    assert pq["q1"]["failure_class"] == "success" and pq["q1"]["ndcg_at_5"] == pytest.approx(1.0)
    assert pq["q2"]["failure_class"] == "long_document_miss" and pq["q2"]["success_at_5"] is False
    assert agg["success_at_5"] == 0.5 and agg["candidate_success_at_30"] == 1.0
    assert result["predictions"]["q1"]["match"] is True
    assert result["predictions"]["q2"]["match"] is False
    report = evaluate.render_report(result, {"point_count": 10}, "r", "v")
    assert "Threshold sweep" in report and "← current" in report
