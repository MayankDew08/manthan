"""U3 graph_expand: offline graph expansion (spec section 13.1).

The toy graph lives in this file; no frozen data is read. Every score is a
hand-computed value checked at 1e-9. ``expand_from_results`` and
``expand_from_query`` return ``(rows, stats)`` so the counters required by
R3/R4 are returned, not discarded (review #2 I4).

Review #1 fixes: B2 and B3 use separate fixtures; B4 compares two real module
calls; B5 has a real tie; B8 covers a doc already in S2's pool; B9 checks all
five note templates and the 80-character cut; E3 and E5 assert the counts and
notes they describe; A2 shuffles seeds too and checks rounding (R2).
"""

import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _graph_expand():
    """Lazy import: a missing module is the red state, not a broken test file."""
    from evals import graph_expand
    return graph_expand


def _seed(cid, rank):
    return {"candidate_id": cid, "rank": rank}


def _edge(neighbor, path="contains", connector=None, degree=None):
    return {"neighbor": neighbor, "path": path, "connector": connector, "degree": degree}


def _doc(cid, kind="message", text="text", title=None):
    doc = {"candidate_id": cid, "candidate_type": kind, "embedded_text": text}
    if kind == "link":
        doc["title"] = title if title is not None else cid
        doc["url"] = f"http://example.test/{cid}"
    return doc


def _expand(seeds, adjacency, exclude=None, corpus=None, **kwargs):
    return _graph_expand().expand_from_results(
        seeds, adjacency, set(exclude or ()), set(corpus or ()), **kwargs)


def _expand_q(query_vec, node_vecs, nodes, exclude=None, corpus=None, **kwargs):
    return _graph_expand().expand_from_query(
        query_vec, node_vecs, nodes, set(exclude or ()), set(corpus or ()), **kwargs)


def _by_id(rows):
    return {row["candidate_id"]: row for row in rows}


# Section 13.1 fixture: S1 rank 1, S2 rank 2.
BASE_SEEDS = [_seed("S1", 1), _seed("S2", 2)]
BASE_ADJ = {
    "S1": [
        _edge("A", "contains"),
        _edge("B", "entity", "rust", 3),
        _edge(None, "entity", "llm", 1),     # dead end (R7): no neighbour
    ],
    "S2": [
        _edge("B", "entity", "rust", 3),
        _edge("C", "topic", "agents", 17),
    ],
}
CORPUS = {"S1", "S2", "A", "B", "C"}
DOCS = {
    "S1": _doc("S1", text="seed one"),
    "S2": _doc("S2", text="seed two"),
    "A": _doc("A"),
    "B": _doc("B"),
    "C": _doc("C"),
}

# Review #1: B2 and B3 contradicted each other on one fixture. B2 gets a
# one-seed graph; B3 keeps the two-seed graph above.
B2_SEEDS = [_seed("S1", 1)]
B2_ADJ = {"S1": [_edge("B", "entity", "rust", 3)]}

W_RUST = 1.0 / math.log(4.0)              # 1/ln(1+degree) for degree 3
W_RUST_R2 = W_RUST / 2.0                  # seed rank 2


def _node(name, node_type="entity", degree=2):
    return {"node_type": node_type, "name": name, "degree": degree,
            "corpus_doc_ids": [f"doc_{name}"]}


class TestGraphExpandBehaviour:
    def test_U3_B1_contains_hop_from_seed_rank_1(self):
        rows, _ = _expand(BASE_SEEDS, BASE_ADJ, set(), CORPUS)

        a = _by_id(rows)["A"]
        assert a["score"] == pytest.approx(1.0, abs=1e-9)
        assert a["via"] == {"path": "contains", "connector": None, "seed": "S1"}

    def test_U3_B2_entity_degree_3_from_seed_rank_1(self):
        rows, _ = _expand(B2_SEEDS, B2_ADJ, set(), CORPUS)

        assert [r["candidate_id"] for r in rows] == ["B"]
        assert rows[0]["score"] == pytest.approx(0.721347520, abs=1e-9)
        assert rows[0]["score"] == pytest.approx(1.0 / math.log(4.0), abs=1e-9)

    def test_U3_B3_b_reached_from_s1_and_s2_sums_seed_weights(self):
        rows, _ = _expand(BASE_SEEDS, BASE_ADJ, set(), CORPUS)

        b = _by_id(rows)["B"]
        # R2/I11: round once, after summing at full precision -- not per contribution.
        # 1/ln4 + (1/ln4)/2 at full precision rounds to .082021281, not .082021280
        # (rounding each term to 9 decimals first, then summing, gives .082021280).
        assert b["score"] == pytest.approx(1.0 / math.log(4.0) + 1.0 / math.log(4.0) / 2, abs=1e-9)
        assert b["score"] == pytest.approx(1.082021281, abs=1e-9)
        # The strongest single path is S1's.
        assert b["via"] == {"path": "entity", "connector": "rust", "seed": "S1"}

    def test_U3_B4_same_path_from_rank_2_is_exactly_half(self):
        rank1, _ = _expand([_seed("S1", 1)], B2_ADJ, set(), CORPUS)
        rank2, _ = _expand([_seed("S1", 2)], B2_ADJ, set(), CORPUS)

        assert rank2[0]["score"] * 2 == pytest.approx(rank1[0]["score"], abs=1e-9)
        assert rank2[0]["score"] == pytest.approx(W_RUST_R2, abs=1e-9)

    def test_U3_B5_output_is_sorted_by_minus_score_then_candidate_id(self):
        # X and Y tie exactly (both entity, degree 3, rank 1): id order decides.
        adj = {"S1": [
            _edge("A", "contains"),
            _edge("Y", "entity", "bb", 3),
            _edge("X", "entity", "aa", 3),
        ]}
        rows, _ = _expand([_seed("S1", 1)], adj, set(), CORPUS | {"X", "Y"})

        assert [r["candidate_id"] for r in rows] == ["A", "X", "Y"]
        assert rows[1]["score"] == rows[2]["score"]

    def test_U3_B6_via_is_the_strongest_single_path_with_the_agreed_ties(self):
        adj = {"S1": [
            _edge("D1", "entity", "bbb", 3),
            _edge("D1", "topic", "zzz", 3),   # same weight -> entity beats topic
            _edge("D2", "entity", "bbb", 3),
            _edge("D2", "entity", "aaa", 3),  # same path -> smaller connector name
        ]}
        rows, _ = _expand([_seed("S1", 1)], adj, set(), CORPUS | {"D1", "D2"})
        got = _by_id(rows)

        assert got["D1"]["via"]["path"] == "entity"
        assert got["D1"]["via"]["connector"] == "bbb"
        assert got["D2"]["via"]["path"] == "entity"
        assert got["D2"]["via"]["connector"] == "aaa"

    def test_U3_B7_bq_start_nodes_hubs_removed_before_top_k_and_typed_ties(self):
        def vec(cosine):
            return [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))]

        nodes = [
            _node("hub", "entity", 47),
            _node("a", "entity"),
            _node("b", "topic"),
            _node("c", "entity"),
            _node("d", "entity"),
            _node("e", "entity"),
            _node("f", "entity"),
            _node("g", "entity"),
        ]
        cosines = {"hub": 0.95, "a": 1.0, "b": 0.9, "c": 0.9,
                   "d": 0.8, "e": 0.7, "f": 0.6, "g": 0.5}
        vecs = np.array([vec(cosines[n["name"]]) for n in nodes], dtype=np.float32)
        corpus = {"doc_" + n["name"] for n in nodes}
        q = np.array([1.0, 0.0], dtype=np.float32)

        rows, stats = _expand_q(q, vecs, nodes, set(), corpus, k=2)
        # Hub removed first; c (entity 0.9) beats b (topic 0.9) on (node_type, name).
        assert stats["bq_nodes_used"] == 2
        assert {r["candidate_id"] for r in rows} == {"doc_a", "doc_c"}

        rows5, stats5 = _expand_q(q, vecs, nodes, set(), corpus, k=5)
        got = {r["candidate_id"] for r in rows5}
        assert stats5["bq_nodes_used"] == 5
        assert got == {"doc_a", "doc_c", "doc_b", "doc_d", "doc_e"}
        assert "doc_hub" not in got

    def test_U3_B8_connected_doc_already_in_s2_pool_still_gets_a_note(self):
        gx = _graph_expand()
        s2_pool = {"S1", "S2", "A", "B", "C"}   # S2's top 30 already holds these

        rows, _ = _expand(BASE_SEEDS, BASE_ADJ, s2_pool, CORPUS)
        assert rows == []                       # nothing new to expand

        assert gx.note_for("A", ["S1", "S2"], BASE_ADJ, DOCS) is not None
        assert gx.note_for("B", ["S1", "S2"], BASE_ADJ, DOCS) is not None
        # C is reached only through topic "agents" (degree 17): a hub, so no note (I6).
        assert gx.note_for("C", ["S1", "S2"], BASE_ADJ, DOCS) is None
        assert gx.note_for("Z", ["S1", "S2"], BASE_ADJ, DOCS) is None

    def test_U3_B11_query_start_score_and_via_by_hand(self):
        # Start nodes: rust (entity, degree 3, cos 0.8) and agents (topic, degree 2, cos 0.6).
        # Unit vectors, so cos(query=[1,0], node) is the first coordinate.
        # Hand values: D gets 0.8 x 1/ln4 + 0.6 x 1/ln3; E gets 0.6 x 1/ln3.
        nodes = [
            {"node_type": "entity", "name": "rust", "degree": 3, "corpus_doc_ids": ["D"]},
            {"node_type": "topic", "name": "agents", "degree": 2, "corpus_doc_ids": ["D", "E"]},
        ]
        node_vecs = np.array([[0.8, 0.6], [0.6, 0.8]], dtype=np.float64)
        q = np.array([1.0, 0.0], dtype=np.float64)

        rows, _ = _expand_q(q, node_vecs, nodes, set(), {"D", "E"}, k=2)
        got = _by_id(rows)

        expected_d = 0.8 / math.log(4.0) + 0.6 / math.log(3.0)
        assert got["D"]["score"] == pytest.approx(expected_d, abs=1e-9)
        assert got["E"]["score"] == pytest.approx(0.6 / math.log(3.0), abs=1e-9)
        # D's strongest single path is rust (0.5771 > 0.5461).
        assert got["D"]["via"] == {"path": "query_entity", "connector": "rust", "seed": None}
        assert got["E"]["via"] == {"path": "query_topic", "connector": "agents", "seed": None}

    def test_U3_B12_degree_1_start_node_is_a_direct_hit_not_a_dead_end(self):
        """Review #4 K1: R7's dead-end rule is for B-r (the seed IS the doc); a
        degree-1 B-q start node means the query matched a topic/entity that
        only one doc carries, which is a hit, not something to skip."""
        nodes = [{"node_type": "topic", "name": "niche", "degree": 1,
                  "corpus_doc_ids": ["D"]}]
        node_vecs = np.array([[1.0, 0.0]], dtype=np.float64)
        q = np.array([1.0, 0.0], dtype=np.float64)

        rows, stats = _expand_q(q, node_vecs, nodes, set(), {"D"}, k=1)

        assert [r["candidate_id"] for r in rows] == ["D"]
        assert rows[0]["score"] == pytest.approx(1.0 / math.log(2.0), abs=1e-9)
        assert rows[0]["via"] == {"path": "query_topic", "connector": "niche", "seed": None}
        assert stats["bq_nodes_used"] == 1

    def test_U3_B9_note_wording_and_labels(self):
        gx = _graph_expand()
        adj = {"S1": [
            _edge("A", "contains"),
            _edge("B", "entity", "rust", 3),
            _edge("C", "topic", "agents", 3),
        ]}
        docs = {
            "S1": _doc("S1", text="seed one"),
            "A": _doc("A"), "B": _doc("B"), "C": _doc("C"),
        }

        # B-r templates: contains, entity, topic.
        assert gx.note_for("A", ["S1"], adj, docs) == \
            'shared in the same message as "seed one"'
        assert gx.note_for("B", ["S1"], adj, docs) == \
            'both mention "rust", like "seed one"'
        assert gx.note_for("C", ["S1"], adj, docs) == \
            'both about "agents", like "seed one"'

        # A link seed is labelled by its title.
        link_docs = {**docs, "L": _doc("L", kind="link", title="Seed Link")}
        assert gx.note_for("A", ["L"], {"L": [_edge("A", "contains")]}, link_docs) == \
            'shared in the same message as "Seed Link"'

        # A message seed is labelled by its first 80 characters.
        long_docs = {**docs, "M": _doc("M", text="x" * 100)}
        assert gx.note_for("A", ["M"], {"M": [_edge("A", "contains")]}, long_docs) == \
            'shared in the same message as "' + "x" * 80 + '"'

        # B-q templates: query_entity and query_topic name the start node.
        nodes = [
            {"node_type": "entity", "name": "rust", "degree": 3, "corpus_doc_ids": ["B"]},
            {"node_type": "topic", "name": "agents", "degree": 3, "corpus_doc_ids": ["C"]},
        ]
        starts_e = [{"path": "query_entity", "connector": "rust", "score": 0.9}]
        starts_t = [{"path": "query_topic", "connector": "agents", "score": 0.8}]
        assert gx.note_for_query("B", starts_e, nodes) == 'mentions "rust"'
        assert gx.note_for_query("C", starts_t, nodes) == 'about "agents"'
        assert gx.note_for_query("Z", starts_e, nodes) is None

    def test_U3_B10_name_norm(self):
        gx = _graph_expand()

        assert gx.NAME_NORM("AI-Development") == gx.NAME_NORM("ai development")
        assert gx.NAME_NORM("AI_development") == gx.NAME_NORM("ai development")
        assert gx.NAME_NORM("LLM") != gx.NAME_NORM("LLMs")
        assert gx.NAME_NORM("  Claude   Code ") == "claude code"


class TestGraphExpandBoundaries:
    def test_U3_E1_degree_16_is_kept(self):
        adj = {"S1": [_edge("keep", "entity", "e16", 16)]}
        rows, _ = _expand([_seed("S1", 1)], adj, set(), {"S1", "keep"})

        assert [r["candidate_id"] for r in rows] == ["keep"]
        assert rows[0]["score"] == pytest.approx(1.0 / math.log(17.0), abs=1e-9)

    def test_U3_E2_degree_17_is_skipped(self):
        adj = {"S1": [_edge("skip", "entity", "e17", 17)]}
        rows, _ = _expand([_seed("S1", 1)], adj, set(), {"S1", "skip"})

        assert rows == []

    def test_U3_E3_degree_1_yields_nothing_and_counts_one_dead_end(self):
        rows, stats = _expand(BASE_SEEDS, BASE_ADJ, set(), CORPUS)

        assert set(_by_id(rows)) == {"A", "B"}
        assert stats["dead_ends"] == 1

    def test_U3_E4_a_seed_is_never_returned_as_a_neighbour(self):
        adj = {**BASE_ADJ, "S1": BASE_ADJ["S1"] + [_edge("S2", "contains")]}

        rows, _ = _expand(BASE_SEEDS, adj, set(), CORPUS)

        assert "S2" not in _by_id(rows)

    def test_U3_E5_excluded_neighbour_is_not_expanded_but_still_gets_a_note(self):
        gx = _graph_expand()

        rows, _ = _expand(BASE_SEEDS, BASE_ADJ, {"B"}, CORPUS)

        assert "B" not in _by_id(rows)
        assert gx.note_for("B", ["S1", "S2"], BASE_ADJ, DOCS) is not None

    def test_U3_E6_out_of_corpus_neighbour_is_dropped(self):
        adj = {**BASE_ADJ, "S1": BASE_ADJ["S1"] + [_edge("OUT", "contains")]}

        rows, _ = _expand(BASE_SEEDS, adj, set(), CORPUS)

        assert "OUT" not in _by_id(rows)

    def test_U3_E7_cap_30_and_the_tie_at_the_cut_breaks_on_candidate_id(self):
        docs = [f"n{i:02d}" for i in range(1, 32)]
        adj = {"S1": [_edge(doc, "contains") for doc in docs]}

        rows, _ = _expand([_seed("S1", 1)], adj, set(), {"S1", *docs}, cap=30)

        assert len(rows) == 30
        assert [r["candidate_id"] for r in rows] == docs[:30]
        assert "n31" not in _by_id(rows)

    def test_U3_E8_fewer_than_seeds_k_uses_all_of_them(self):
        rows, _ = _expand(BASE_SEEDS, BASE_ADJ, set(), CORPUS)

        assert "A" in _by_id(rows)          # from S1
        assert _by_id(rows)["B"]["score"] == pytest.approx(1.082021281, abs=1e-9)  # S1+S2

    def test_U3_E9_an_eleventh_seed_is_never_used(self):
        seeds = [_seed(f"S{i:02d}", i) for i in range(1, 12)]
        adj = {s["candidate_id"]: [_edge(f"doc{i:02d}", "contains")]
               for i, s in enumerate(seeds, start=1)}
        corpus = {s["candidate_id"] for s in seeds} | {
            f"doc{i:02d}" for i in range(1, 12)}

        rows, _ = _expand(seeds, adj, set(), corpus)

        got = {r["candidate_id"] for r in rows}
        assert got == {f"doc{i:02d}" for i in range(1, 11)}
        assert "doc11" not in got

    def test_U3_E12_eleven_seeds_shuffled_still_uses_the_best_ten_ranks(self):
        """Review #4 K2: seeds[:k] must take the k best RANKS, not the first k
        list positions -- a caller-shuffled seed list must not change the set
        of seeds used."""
        seeds_by_rank = [_seed(f"S{i:02d}", i) for i in range(1, 12)]
        shuffled = list(reversed(seeds_by_rank))  # rank 11 first positionally
        adj = {s["candidate_id"]: [_edge(f"doc{i:02d}", "contains")]
               for i, s in enumerate(seeds_by_rank, start=1)}
        corpus = {s["candidate_id"] for s in seeds_by_rank} | {
            f"doc{i:02d}" for i in range(1, 12)}

        rows, _ = _expand(shuffled, adj, set(), corpus)

        got = {r["candidate_id"] for r in rows}
        assert got == {f"doc{i:02d}" for i in range(1, 11)}
        assert "doc11" not in got

    def test_U3_E10_no_seeds_or_no_neighbours_is_empty_with_no_error(self):
        gx = _graph_expand()

        rows, stats = gx.expand_from_results([], {}, set(), CORPUS)
        assert (rows, stats) == ([], {"dead_ends": 0})

        rows, stats = gx.expand_from_results(BASE_SEEDS, {}, set(), CORPUS)
        assert (rows, stats) == ([], {"dead_ends": 0})

    def test_U3_E11_bq_with_three_usable_nodes_records_three(self):
        nodes = [_node(f"n{i}") | {"corpus_doc_ids": [f"d{i}"]} for i in range(1, 4)]
        vecs = np.array([[1.0, 0.0]] * 3, dtype=np.float32)

        rows, stats = _expand_q(np.array([1.0, 0.0], dtype=np.float32), vecs,
                                nodes, set(), {"d1", "d2", "d3"})

        assert stats["bq_nodes_used"] == 3
        assert {r["candidate_id"] for r in rows} == {"d1", "d2", "d3"}


class TestGraphExpandFailures:
    def test_U3_F1_unknown_path_type_names_the_row(self):
        adj = {"S1": [_edge("X", "friend", "x", 2)]}

        with pytest.raises(ValueError, match="friend"):
            _expand([_seed("S1", 1)], adj, set(), {"S1", "X"})

    def test_U3_F2_entity_or_topic_degree_0_raises(self):
        for path in ("entity", "topic"):
            adj = {"S1": [_edge("X", path, "x", 0)]}
            with pytest.raises(ValueError, match="degree"):
                _expand([_seed("S1", 1)], adj, set(), {"S1", "X"})

    def test_U3_F3_query_and_node_vector_dimensions_must_match(self):
        nodes = [_node("a") | {"corpus_doc_ids": ["d1"]}]
        with pytest.raises(ValueError, match="dimension"):
            _expand_q([1.0, 0.0], np.zeros((1, 3), dtype=np.float32),
                      nodes, set(), {"d1"})

    def test_U3_F4_node_vector_sha256_mismatch_raises(self, tmp_path):
        gx = _graph_expand()
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        path = tmp_path / "node_vectors.npy"

        sha = gx.save_node_vectors(vecs, ["a", "b"], path)
        assert isinstance(sha, str) and len(sha) == 64
        loaded = gx.load_node_vectors(path, sha)
        assert np.array_equal(loaded, vecs)

        with pytest.raises(ValueError, match="sha256"):
            gx.load_node_vectors(path, "0" * 64)

    def test_U3_F5_duplicate_seed_ids_raise(self):
        seeds = [_seed("S1", 1), _seed("S1", 2)]

        with pytest.raises(ValueError, match="duplicate"):
            _expand(seeds, BASE_ADJ, set(), CORPUS)


class TestGraphExpandReviewTwo:
    """Review #2 requirements: the defects I1, I3, I4, I5 and I6 must be caught."""

    def test_U3_I1_scores_are_python_float_and_json_serialisable(self):
        rows, _ = _expand(BASE_SEEDS, BASE_ADJ, set(), CORPUS)
        assert rows
        for row in rows:
            assert type(row["score"]) is float, row
        json.dumps(rows)

        nodes = [_node("a") | {"corpus_doc_ids": ["d1"]},
                 _node("b") | {"corpus_doc_ids": ["d2"]},
                 _node("c", "topic") | {"corpus_doc_ids": ["d3"]}]
        vecs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=np.float32)
        q_rows, _ = _expand_q(np.array([1.0, 0.0], dtype=np.float32), vecs, nodes,
                              set(), {"d1", "d2", "d3"})
        assert q_rows
        for row in q_rows:
            assert type(row["score"]) is float, row
        json.dumps(q_rows)

    def test_U3_I3_node_vector_file_tampered_on_disk_raises(self, tmp_path):
        gx = _graph_expand()
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        path = tmp_path / "node_vectors.npy"

        sha = gx.save_node_vectors(vecs, ["a", "b"], path)
        data = bytearray(path.read_bytes())
        data[-1] ^= 0x01                        # flip one bit of the vector data
        path.write_bytes(bytes(data))

        with pytest.raises(ValueError, match="sha256"):
            gx.load_node_vectors(path, sha)

    def test_U3_I4_dead_end_count_is_returned_not_discarded(self):
        # Dead ends are edges with neighbor None (R7), one per degree-1 connector.
        adj = {"S1": [
            _edge(None, "entity", "d1", 1),
            _edge(None, "entity", "d2", 1),
            _edge(None, "topic", "d3", 1),
            _edge("A", "contains"),
        ]}
        rows, stats = _expand([_seed("S1", 1)], adj, set(), {"S1", "A"})

        assert [r["candidate_id"] for r in rows] == ["A"]
        assert stats["dead_ends"] == 3

    def test_U3_I5_note_names_the_same_path_as_via(self):
        gx = _graph_expand()
        # X: entity "rust" from S1 (1 x 1/ln4 = 0.7213) beats contains from S2 (0.5 x 1.0).
        adj = {
            "S1": [_edge("X", "entity", "rust", 3)],
            "S2": [_edge("X", "contains")],
        }
        seeds = [_seed("S1", 1), _seed("S2", 2)]
        docs = {"S1": _doc("S1", text="seed one"), "S2": _doc("S2", text="seed two"),
                "X": _doc("X")}

        rows, _ = _expand(seeds, adj, set(), {"S1", "S2", "X"})

        assert _by_id(rows)["X"]["via"] == {"path": "entity", "connector": "rust", "seed": "S1"}
        assert gx.note_for("X", ["S1", "S2"], adj, docs) == \
            'both mention "rust", like "seed one"'

    def test_U3_I6_no_note_or_expansion_through_a_hub_connector(self):
        gx = _graph_expand()
        adj = {"S1": [_edge("B", "topic", "agentic workflows", 47)]}
        docs = {"S1": _doc("S1", text="seed one"), "B": _doc("B")}

        rows, _ = _expand([_seed("S1", 1)], adj, set(), {"S1", "B"})
        assert rows == []
        assert gx.note_for("B", ["S1"], adj, docs) is None

        # B-q: a hub node never gives a note, even if handed in as a start.
        nodes = [{"node_type": "topic", "name": "agentic workflows",
                  "degree": 47, "corpus_doc_ids": ["B"]}]
        q_rows, _ = _expand_q(np.array([1.0, 0.0], dtype=np.float32),
                              np.array([[1.0, 0.0]], dtype=np.float32), nodes,
                              set(), {"B"})
        assert q_rows == []
        assert gx.note_for_query(
            "B", [{"path": "query_topic", "connector": "agentic workflows", "score": 1.0}],
            nodes) is None


class TestGraphExpandAntiCheat:
    def test_U3_A1_no_live_store_imports(self):
        gx = _graph_expand()
        source = Path(gx.__file__).read_text(encoding="utf-8")

        assert "import neo4j" not in source
        assert "qdrant_client" not in source
        assert "live_guard" not in source

    def test_U3_A2_shuffling_rows_seeds_and_nodes_changes_nothing_and_scores_are_rounded(self):
        baseline, base_stats = _expand(BASE_SEEDS, BASE_ADJ, set(), CORPUS)

        rng = random.Random(0)
        shuffled = {key: list(value) for key, value in BASE_ADJ.items()}
        for rows in shuffled.values():
            rng.shuffle(rows)
        items = list(shuffled.items())
        rng.shuffle(items)
        seeds = list(BASE_SEEDS)
        rng.shuffle(seeds)

        got, got_stats = _expand(seeds, dict(items), set(), CORPUS)
        assert got == baseline
        assert got_stats == base_stats

        # R2: every returned score is already rounded to 9 decimals.
        for row in baseline:
            assert row["score"] == round(row["score"], 9), row

        nodes = [_node("a") | {"corpus_doc_ids": ["d1"]},
                 _node("b") | {"corpus_doc_ids": ["d2"]},
                 _node("c", "topic") | {"corpus_doc_ids": ["d3"]}]
        vecs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=np.float32)
        q = np.array([1.0, 0.0], dtype=np.float32)
        base_q, base_q_stats = _expand_q(q, vecs, nodes, set(), {"d1", "d2", "d3"})

        order = list(range(3))
        rng.shuffle(order)
        q_rows, q_stats = _expand_q(q, vecs[order], [nodes[i] for i in order],
                                    set(), {"d1", "d2", "d3"})
        assert (q_rows, q_stats) == (base_q, base_q_stats)
        for row in base_q:
            assert row["score"] == round(row["score"], 9), row

    def test_U3_A3_module_constants(self):
        gx = _graph_expand()
        assert gx.SEEDS_K == 10
        assert gx.QUERY_NODES_K == 5
        assert gx.NEIGHBOR_CAP == 30
        assert gx.HUB_MAX_DEGREE == 16

    def test_U3_A4_numeric_checks_are_exact_to_1e_9(self):
        b2, _ = _expand(B2_SEEDS, B2_ADJ, set(), CORPUS)
        b3, _ = _expand(BASE_SEEDS, BASE_ADJ, set(), CORPUS)

        assert abs(b2[0]["score"] - 0.721347520) < 1e-9
        assert abs(_by_id(b3)["B"]["score"] - 1.082021281) < 1e-9
        assert b2[0]["score"] == pytest.approx(1.0 / math.log(4.0), abs=1e-9)
        assert _by_id(b3)["B"]["score"] == pytest.approx(
            1.0 / math.log(4.0) + 1.0 / math.log(4.0) / 2.0, abs=1e-9)

    def test_U3_A5_no_random_time_datetime_or_environment(self):
        gx = _graph_expand()
        source = Path(gx.__file__).read_text(encoding="utf-8")

        for module in ("random", "time", "datetime"):
            assert not re.search(rf"^\s*(import|from)\s+{module}\b", source, re.M), module
        assert "os.environ" not in source
