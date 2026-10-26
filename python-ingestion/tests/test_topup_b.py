"""U6 topup_b: blind top-up sheets and qrels_v3 (spec section 13.6).

Reuses ``topup.py``; it never edits it (U6-A1). Message-to-link relations come
from the frozen ``graph_neighbors_v1.jsonl`` CONTAINS rows, never from a
``doc["links"]`` field.
"""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common  # noqa: E402
from evals import pooling  # noqa: E402
from evals import topup  # noqa: E402


def _topup_b():
    from evals import topup_b
    return topup_b


def _row(qid, cid, rank, run_id="r", via=None):
    return {"run_id": run_id, "query_id": qid, "candidate_id": cid,
            "rank": rank, "score": 1.0, "via": via, "note": None}


def _qrels(pairs):
    out = {}
    for qid, cid, grade in pairs:
        out.setdefault(qid, {})[cid] = {"relevance": grade, "answer_bearing": 0}
    return out


def _doc(cid, kind="message", text="own text", title=None):
    doc = {"candidate_id": cid, "candidate_type": kind, "embedded_text": text}
    if kind == "link":
        doc["title"] = title or cid
        doc["url"] = f"http://example.test/{cid}"
    return doc


def _contains(msg_id, link_id, in_corpus=True):
    return {"candidate_id": msg_id, "neighbor_id": link_id, "path": "contains",
            "connector": None, "connector_degree": None, "in_corpus": in_corpus}


QUERIES = {"q1": {"query_id": "q1", "query_text": "alpha"}}

DOCS = {
    "msg1": _doc("msg1", text="message one"),
    "msg2": _doc("msg2", text="message two"),
    "link:a": _doc("link:a", kind="link", text="alpha body", title="Alpha"),
    "link:b": _doc("link:b", kind="link", text="beta body", title="Beta"),
}

def _neighbors(*rows):
    """Convert frozen CONTAINS rows to the adjacency shape U3/U4/U6 consume."""
    out = {}
    for row in rows:
        out.setdefault(row["candidate_id"], []).append({
            "neighbor": row["neighbor_id"], "path": row["path"],
            "connector": row["connector"],
            "degree": row["connector_degree"],
            "in_corpus": row["in_corpus"],
        })
    return out


# D21: CONTAINS is written in both directions, so the link -> message rows are here too.
NEIGHBORS = _neighbors(
    _contains("msg1", "link:b"), _contains("msg1", "link:a"),
    _contains("link:a", "msg1"), _contains("link:b", "msg1"))


class TestTopupBUnit:
    def test_U6_B1_sheet_columns_and_no_provenance(self):
        tb = _topup_b()
        rows = tb.build_sheet([("q1", "msg1"), ("q1", "link:a")],
                              QUERIES, DOCS, kind="original")

        assert rows
        assert list(rows[0].keys()) == common.ANNOTATION_COLUMNS
        for forbidden in ("rank", "score", "via", "note", "run_id"):
            assert forbidden not in common.ANNOTATION_COLUMNS

        blob = json.dumps(rows)
        for secret in ("graph_b1r_rr", "ctrl_c0_deeper_rr", "ladder_s2_cos_bm25_rr",
                       "B1-r", "B2-r", "B1-q", "B2-q", "S2", "C0", "C1", "C2"):
            assert secret not in blob, secret

    def test_U6_B2_content_is_the_docs_own_text_never_glued_or_a_note(self):
        tb = _topup_b()
        original = tb.build_sheet([("q1", "msg1")], QUERIES, DOCS, kind="original")
        assert original[0]["content"] == pooling.display_content(DOCS["msg1"])

        graph = tb.build_sheet([("q1", "msg1")], QUERIES, DOCS,
                               kind="graph", neighbors=NEIGHBORS)
        content = graph[0]["content"]
        assert content.startswith(pooling.display_content(DOCS["msg1"]))
        # D24 shows link *titles*, never link bodies (that would be glued text).
        assert "alpha body" not in content
        assert "beta body" not in content
        # Never a B2 note.
        assert "both mention" not in content
        assert "both about" not in content
        assert "shared in the same message" not in content

    def test_U6_B3_original_sheet_is_top5_of_the_seven_new_systems_only(self):
        tb = _topup_b()
        runs = {
            "ctrl_c0_deeper_rr": [_row("q1", f"c0-{i}", i, "ctrl_c0_deeper_rr")
                                  for i in range(1, 7)],
            "graph_b1r_rr": [_row("q1", f"b1r-{i}", i, "graph_b1r_rr")
                             for i in range(1, 7)],
            # S2 is not one of the seven new systems and must not contribute.
            "ladder_s2_cos_bm25_rr": [_row("q1", "s2-only", 1, "ladder_s2_cos_bm25_rr")],
        }
        qrels = _qrels([("q1", "c0-1", 2), ("q1", "b1r-1", 3)])

        pairs = tb.original_pairs(runs, qrels, k=5)

        got = {cid for qid, cid in pairs if qid == "q1"}
        assert got == {f"c0-{i}" for i in range(2, 6)} | {f"b1r-{i}" for i in range(2, 6)}
        assert "s2-only" not in got
        assert "c0-6" not in got
        assert "b1r-6" not in got
        assert pairs == sorted(set(pairs))

    def test_U6_B4_graph_sheet_is_top30_of_eight_systems_plus_cosine(self):
        tb = _topup_b()
        systems = list(tb.GRAPH_SYSTEMS)
        # Spec section 4 has 8 systems (S2, C0, B1-r, B2-r, B1-q, B2-q, C1, C2).
        assert len(systems) == 8
        assert "ladder_s2_cos_bm25_rr" in systems
        runs = {
            systems[0]: [_row("q1", f"d{i}", i, systems[0]) for i in range(1, 36)],
            systems[1]: [_row("q1", f"d{i}", i, systems[1]) for i in range(5, 40)],
        }
        for rid in systems[2:]:
            runs[rid] = []
        cosine = [_row("q1", f"e{i}", i, "vector_cosine_graphq_v1")
                  for i in range(1, 36)]

        pairs = tb.graph_pairs(runs, cosine, k=30)

        # Top 30 of a = d1..d30, of b = d5..d34 -> d1..d34; plus e1..e30.
        expected = {("q1", f"d{i}") for i in range(1, 35)} | \
                   {("q1", f"e{i}") for i in range(1, 31)}
        assert set(pairs) == expected
        assert len(pairs) == len(set(pairs))
        assert pairs == sorted(pairs)
        assert ("q1", "d35") not in pairs

    def test_U6_B5_shuffle_is_random_42_and_reproducible(self):
        tb = _topup_b()
        pairs = [("q1", "link:b"), ("q1", "link:a"), ("q1", "msg1"), ("q1", "msg2")]

        first = tb.build_sheet(pairs, QUERIES, DOCS, kind="original")
        second = tb.build_sheet(list(reversed(pairs)), QUERIES, DOCS, kind="original")

        expected = sorted(set(pairs))
        random.Random(42).shuffle(expected)
        order = [(r["query_id"], r["candidate_id"]) for r in first]
        assert order == expected
        assert order == [(r["query_id"], r["candidate_id"]) for r in second]

    def test_U6_B6_d24_shows_both_in_corpus_link_titles_in_candidate_id_order(self):
        tb = _topup_b()
        # A bogus ``links`` field must be ignored: relations come from CONTAINS rows.
        msg = dict(DOCS["msg1"])
        msg["links"] = ["link:zzz"]
        docs = {**DOCS, "msg1": msg}

        content = tb.graph_content(
            docs["msg1"], NEIGHBORS, docs)

        assert content == (pooling.display_content(docs["msg1"]) +
                           ' [shared link: "Alpha"] [shared link: "Beta"]')
        # Links and link-less messages are unchanged.
        assert tb.graph_content(docs["link:a"], NEIGHBORS, docs) == \
            pooling.display_content(docs["link:a"])
        assert tb.graph_content(docs["msg2"], NEIGHBORS, docs) == \
            pooling.display_content(docs["msg2"])

    @pytest.mark.artifacts
    def test_U6_B7_merge_is_v2_plus_both_sheets_and_leaves_v2_byte_identical(self):
        tb = _topup_b()
        v2_path = common.data_dir(common.DEFAULT_CORPUS_VERSION) / "qrels_v2.csv"
        before = v2_path.read_bytes()
        v2_rows = topup.read_csv(v2_path)

        template = dict(v2_rows[0])
        sheet_b = {**template, "query_id": "q-topup-b", "candidate_id": "doc-b",
                   "relevance": "2", "answer_bearing": "0", "reason": "B"}
        sheet_g = {**template, "query_id": "q-graph", "candidate_id": "doc-g",
                   "relevance": "3", "answer_bearing": "1", "reason": "G"}

        out = tb.merge_qrels_v3(v2_rows, [sheet_b], [sheet_g])

        assert v2_path.read_bytes() == before
        assert len(out) == len(v2_rows) + 2

        def key(row):
            return row["query_id"], row["candidate_id"]

        got = {key(r) for r in out}
        assert {(r["query_id"], r["candidate_id"]) for r in v2_rows} <= got
        assert ("q-topup-b", "doc-b") in got
        assert ("q-graph", "doc-g") in got
        assert got == {key(r) for r in v2_rows} | {
            ("q-topup-b", "doc-b"), ("q-graph", "doc-g")}

        # The real merged file, once the owner has graded both sheets.
        v3_path = common.data_dir(common.DEFAULT_CORPUS_VERSION) / "qrels_v3.csv"
        if not v3_path.exists():
            pytest.fail(f"missing merged qrels: {v3_path.name} (run the U6 merge)")
        v3_keys = {key(r) for r in topup.read_csv(v3_path)}
        v2_keys = {key(r) for r in topup.read_csv(v2_path)}
        assert v2_keys <= v3_keys
        assert v2_path.read_bytes() == before
        assert len(v3_keys) > len(v2_keys)

    @pytest.mark.parametrize("which", ["original", "graph"])
    def test_U6_F1_blank_grade_in_either_sheet_names_the_row(self, which):
        tb = _topup_b()
        blank = {"query_id": "q1", "candidate_id": "c1", "relevance": "",
                 "answer_bearing": "0", "reason": ""}
        with pytest.raises(ValueError, match=r"q1/c1"):
            tb.merge_qrels_v3([], [blank] if which == "original" else [],
                              [blank] if which == "graph" else [])

    def test_U6_F2_pair_already_graded_raises(self):
        tb = _topup_b()
        v2 = [{"query_id": "q1", "candidate_id": "c1", "relevance": "2",
               "answer_bearing": "0", "reason": "old"}]
        sheet = [{"query_id": "q1", "candidate_id": "c1", "relevance": "3",
                  "answer_bearing": "0", "reason": "new"}]

        with pytest.raises(ValueError, match="already"):
            tb.merge_qrels_v3(v2, sheet, [])
        with pytest.raises(ValueError, match="already"):
            tb.merge_qrels_v3([], sheet, sheet)

    def test_U6_F3_sheet_size_escalates_with_the_count(self):
        tb = _topup_b()
        tb.check_sheet_size("original", 300)
        tb.check_sheet_size("graph", 800)

        with pytest.raises(ValueError, match="301"):
            tb.check_sheet_size("original", 301)
        with pytest.raises(ValueError, match="801"):
            tb.check_sheet_size("graph", 801)

    def test_U6_A1_topup_py_is_imported_not_copied(self):
        tb = _topup_b()
        assert tb.topup is topup

        source = Path(tb.__file__).read_text(encoding="utf-8")
        assert "topup." in source
        assert "topup.topup_rows" in source or "topup.unjudged_pairs" in source
        assert "def topup_rows" not in source
        assert "def merge_qrels" not in source
