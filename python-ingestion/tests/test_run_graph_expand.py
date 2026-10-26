"""U5 run_graph_expand: build the eight Run B systems (spec section 13.5).

Unit tests inject a fake re-ranker, fake embedder and fake tokenizer. Artifact
tests read the frozen run files. ``build_systems`` returns ``(runs, counters)``
so per-query dead ends, B-q node counts and C1 trim counts are not lost.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common  # noqa: E402
from evals import run_bm25  # noqa: E402


def _run_graph_expand():
    from evals import run_graph_expand
    return run_graph_expand


def _graph_expand():
    from evals import graph_expand
    return graph_expand


class WordTokenizer:
    def encode(self, text):
        return text.split()


class FakeReranker:
    """score = -len(text), deterministic and independent of any model."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates):
        candidates = list(candidates)
        self.calls.append((query, candidates))
        scored = [(cid, -float(len(text))) for cid, text in candidates]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored


def _msg(cid, text):
    return {"candidate_id": cid, "candidate_type": "message", "embedded_text": text}


def _link(cid, text):
    return {"candidate_id": cid, "candidate_type": "link",
            "title": cid, "url": f"http://example.test/{cid}", "embedded_text": text}


def _frozen_row(candidate_id, neighbor_id, path, connector=None, degree=None):
    """One graph_neighbors_v1.jsonl row (U2 schema)."""
    return {"candidate_id": candidate_id, "neighbor_id": neighbor_id, "path": path,
            "connector": connector, "connector_degree": degree, "in_corpus": True}


def _contains_row(candidate_id, neighbor_id):
    return _frozen_row(candidate_id, neighbor_id, "contains")


def _adjacency(rows):
    """Convert frozen rows to the U3 adjacency shape that build_systems consumes."""
    out = {}
    for row in rows:
        out.setdefault(row["candidate_id"], []).append({
            "neighbor": row["neighbor_id"], "path": row["path"],
            "connector": row["connector"], "degree": row["connector_degree"],
            "in_corpus": row["in_corpus"]})
    return out


FILLERS = [f"f{i:02d}" for i in range(1, 31)]
LONG_LINK = "l1 " * 300


def _fixture():
    docs = {cid: _msg(cid, "filler alpha") for cid in FILLERS}
    docs.update({
        "m1": _msg("m1", "m1 alpha"),
        "m2": _msg("m2", "m2 beta"),
        "m3": _msg("m3", "m3"),
        "m4": _msg("m4", "m4"),
        "l1": _link("l1", LONG_LINK),
        "l2": _link("l2", "l2 body"),
    })
    order = FILLERS + ["m1", "m2", "m3", "m4", "l1", "l2"]

    queries = [{"query_id": "q1", "query_text": "alpha"}]
    cosine = [{"run_id": "vector_cosine_v1", "query_id": "q1",
               "candidate_id": cid, "rank": rank, "score": 1.0}
              for rank, cid in enumerate(order, start=1)]
    bm25 = [{"run_id": "bm25_v1", "query_id": "q1",
             "candidate_id": cid, "rank": rank, "score": 1.0}
            for rank, cid in enumerate(order, start=1)]

    # Message-to-link relations come from frozen CONTAINS rows, never from doc["links"].
    frozen_rows = [
        _contains_row("f01", "l1"),
        _contains_row("f01", "l2"),
        _contains_row("f03", "m3"),
        _contains_row("f04", "m4"),
        _frozen_row("f01", "m1", "entity", "alpha", 2),
        _frozen_row("f02", "m2", "topic", "beta", 2),
    ]
    adjacency = _adjacency(frozen_rows)

    nodes = [
        {"node_type": "entity", "name": "alpha", "degree": 2,
         "corpus_doc_ids": ["f01", "m1"]},
        {"node_type": "topic", "name": "beta", "degree": 2,
         "corpus_doc_ids": ["f02", "m2"]},
    ]
    node_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    payload = {cid: {"entities": [], "topics": []} for cid in docs}
    payload["f01"] = {"entities": ["alpha"], "topics": []}
    payload["m1"] = {"entities": ["alpha"], "topics": []}
    payload["f02"] = {"entities": [], "topics": ["beta"]}
    payload["m2"] = {"entities": [], "topics": ["beta"]}

    return {
        "queries": queries,
        "docs": docs,
        "cosine": cosine,
        "bm25": bm25,
        "adjacency": adjacency,
        "nodes": nodes,
        "node_vecs": node_vecs,
        "payload": payload,
    }


def _build(fx, monkeypatch, reranker=None, *, embed_batch=None, tokenizer=None):
    """Inject the fakes the way U5 keeps its signature: patch the imported modules."""
    rge = _run_graph_expand()
    reranker = reranker or FakeReranker()
    monkeypatch.setattr(rge.embedder, "embed_text", lambda text: [1.0, 0.0])
    monkeypatch.setattr(rge.embedder, "embed_batch",
                        embed_batch or (lambda texts: [[1.0, 0.0] for _ in texts]))
    monkeypatch.setattr(rge.denorm_control, "_get_tokenizer",
                        lambda: tokenizer or WordTokenizer())
    return rge.build_systems(
        fx["queries"], fx["docs"], fx["cosine"], fx["bm25"],
        fx["adjacency"], fx["nodes"], fx["node_vecs"], fx["payload"],
        reranker,
    )


def _rows_for(runs, run_id, qid="q1"):
    return [r for r in runs[run_id] if r["query_id"] == qid]



class TestRunGraphExpandUnit:
    def test_U5_B2_c0_pool_is_max_of_br_and_bq(self, monkeypatch):
        fx = _fixture()
        runs, _ = _build(fx, monkeypatch)

        br = len(_rows_for(runs, "graph_b1r_rr"))
        bq = len(_rows_for(runs, "graph_b1q_rr"))
        c0 = len(_rows_for(runs, "ctrl_c0_deeper_rr"))

        assert c0 == max(br, bq)
        assert br > 30 and bq > 30  # both pools actually expanded

    def test_U5_B3_b1_and_b2_have_identical_membership(self, monkeypatch):
        fx = _fixture()
        runs, _ = _build(fx, monkeypatch)

        for plain, noted in (("graph_b1r_rr", "graph_b2r_ctx_rr"),
                             ("graph_b1q_rr", "graph_b2q_ctx_rr")):
            assert {r["candidate_id"] for r in _rows_for(runs, plain)} == \
                   {r["candidate_id"] for r in _rows_for(runs, noted)}, plain

    def test_U5_B4_notes_only_where_note_for_returns_one(self, monkeypatch):
        fx = _fixture()
        gx = _graph_expand()
        runs, _ = _build(fx, monkeypatch)

        seed_ids = [r["candidate_id"] for r in
                    sorted(_rows_for(runs, "ladder_s2_cos_bm25_rr"),
                           key=lambda r: r["rank"])[:10]]
        assert seed_ids == FILLERS[:10]

        for row in _rows_for(runs, "graph_b2r_ctx_rr"):
            expected = gx.note_for(row["candidate_id"], seed_ids,
                                   fx["adjacency"], fx["docs"])
            assert row["note"] == expected, row["candidate_id"]

        # B-q note for every doc attached to a start node.
        start_nodes = [
            {"path": "query_entity", "connector": "alpha", "score": 1.0},
            {"path": "query_topic", "connector": "beta", "score": 0.0},
        ]
        for row in _rows_for(runs, "graph_b2q_ctx_rr"):
            expected = gx.note_for_query(row["candidate_id"], start_nodes, fx["nodes"])
            assert row["note"] == expected, row["candidate_id"]

        for system in ("ladder_s2_cos_bm25_rr", "ctrl_c0_deeper_rr",
                       "ctrl_c1_denorm_rr", "ctrl_c2_payload_rr",
                       "graph_b1r_rr", "graph_b1q_rr"):
            assert all(r["note"] is None for r in runs[system]), system

    def test_U5_B5_via_is_set_exactly_for_expanded_docs(self, monkeypatch):
        fx = _fixture()
        runs, _ = _build(fx, monkeypatch)

        s2 = {r["candidate_id"] for r in _rows_for(runs, "ladder_s2_cos_bm25_rr")}
        for system in ("graph_b1r_rr", "graph_b2r_ctx_rr",
                       "graph_b1q_rr", "graph_b2q_ctx_rr",
                       "ctrl_c2_payload_rr"):
            for row in _rows_for(runs, system):
                if row["candidate_id"] in s2:
                    assert row["via"] is None, (system, row)
                else:
                    assert row["via"] is not None, (system, row)

        for system in ("ladder_s2_cos_bm25_rr", "ctrl_c0_deeper_rr",
                       "ctrl_c1_denorm_rr"):
            assert all(r["via"] is None for r in runs[system]), system

    def test_U5_B6_ranks_are_contiguous_and_sets_are_the_pools(self, monkeypatch):
        fx = _fixture()
        gx = _graph_expand()
        runs, _ = _build(fx, monkeypatch)

        for system in runs:
            ranks = [r["rank"] for r in _rows_for(runs, system)]
            assert ranks == list(range(1, len(ranks) + 1)), system
            cids = [r["candidate_id"] for r in _rows_for(runs, system)]
            assert len(cids) == len(set(cids)), system

        s2 = {r["candidate_id"] for r in _rows_for(runs, "ladder_s2_cos_bm25_rr")}
        seeds = [{"candidate_id": cid, "rank": rank}
                 for rank, cid in enumerate(FILLERS[:10], start=1)]
        expanded, _ = gx.expand_from_results(seeds, fx["adjacency"], s2, set(fx["docs"]))
        assert {r["candidate_id"] for r in _rows_for(runs, "graph_b1r_rr")} == \
               s2 | {r["candidate_id"] for r in expanded}

        q_expanded, _ = gx.expand_from_query(
            np.array([1.0, 0.0], dtype=np.float32), fx["node_vecs"], fx["nodes"],
            s2, set(fx["docs"]))
        assert {r["candidate_id"] for r in _rows_for(runs, "graph_b1q_rr")} == \
               s2 | {r["candidate_id"] for r in q_expanded}

    def test_U5_B8_reranker_gets_full_glued_text_embedder_gets_trimmed(self, monkeypatch):
        fx = _fixture()
        rge = _run_graph_expand()

        bm25_calls = []
        embedded = []

        def fake_bm25_rows(queries, docs):
            bm25_calls.append({d["candidate_id"]: d["embedded_text"] for d in docs})
            rows = []
            for q in queries:
                for rank, d in enumerate(docs, start=1):
                    rows.append({"run_id": "bm25_c1", "query_id": q["query_id"],
                                 "candidate_id": d["candidate_id"],
                                 "rank": rank, "score": 1.0})
            return rows

        def fake_embed_batch(texts):
            embedded.extend(texts)
            return [[1.0, 0.0] for _ in texts]

        monkeypatch.setattr(rge.run_bm25, "bm25_rows", fake_bm25_rows)
        reranker = FakeReranker()
        runs, counters = _build(fx, monkeypatch, reranker, embed_batch=fake_embed_batch)

        # BM25 and the re-ranker see the full glued text (message + long link).
        # C1's BM25 call is the one that sees glued text; check every call made.
        glued_calls = [c for c in bm25_calls if "f01" in c]
        assert glued_calls, "C1 BM25 was never called over the glued texts"
        full_f01 = glued_calls[-1]["f01"]
        assert full_f01.count("l1") > 100
        assert any(
            text.count("l1") > 100
            for _query, candidates in reranker.calls
            for _cid, text in candidates
        )

        # The embedder gets f01's trimmed text, identified by its content: l1 is
        # contained only by f01. Budget 256 = 2 message words + 1 separator + 253 link words.
        f01_inputs = [t for t in embedded if t.startswith("filler alpha l1")]
        assert f01_inputs == ["filler alpha " + " ".join(["l1"] * 253)]

        # R8: l1 keeps some of its words (trimmed); l2 keeps none (dropped). Only f01 has links.
        c1 = counters["ctrl_c1_denorm_rr"]["q1"]
        assert c1["links_trimmed"] == 1
        assert c1["links_dropped_no_room"] == 1

    def test_U5_B9_a_degree_1_connector_counts_one_dead_end(self, monkeypatch):
        # R7: the seed f01 is the only doc on this degree-1 node, so it is one dead end.
        fx = _fixture()
        fx["nodes"] = fx["nodes"] + [
            {"node_type": "entity", "name": "lonely", "degree": 1, "corpus_doc_ids": ["f01"]}]
        fx["node_vecs"] = np.vstack([fx["node_vecs"], [[0.5, 0.5]]]).astype(np.float32)

        _, counters = _build(fx, monkeypatch)

        assert counters["graph_b1r_rr"]["q1"]["dead_ends"] == 1

    def test_U5_B10_b2_reranker_input_is_the_connected_prefix_only_where_noted(self, monkeypatch):
        fx = _fixture()
        reranker = FakeReranker()
        runs, _ = _build(fx, monkeypatch, reranker)
        plain = {cid: doc["embedded_text"] for cid, doc in fx["docs"].items()}

        notes_r = {r["candidate_id"]: r["note"] for r in _rows_for(runs, "graph_b2r_ctx_rr")
                   if r["note"] is not None}
        notes_q = {r["candidate_id"]: r["note"] for r in _rows_for(runs, "graph_b2q_ctx_rr")
                   if r["note"] is not None}
        assert notes_r and notes_q, "both B2 systems must have noted docs"

        # One re-rank call per system (section 4 has 8). Only the two B2 calls carry
        # the "[connected: " prefix; C1's glued text is not a connection note.
        assert len(reranker.calls) == 8

        carrying = []
        for _query, candidates in reranker.calls:
            tagged = {(cid, text) for cid, text in candidates
                      if text.startswith("[connected: ")}
            if tagged:
                carrying.append(tagged)
        assert len(carrying) == 2

        def connected(cid, note):
            return (cid, "[connected: " + note + "] " + plain[cid])

        expected = [
            {connected(cid, note) for cid, note in notes_r.items()},
            {connected(cid, note) for cid, note in notes_q.items()},
        ]
        # Each B2 call tags exactly its noted docs, with the note and the plain text.
        assert sorted(map(sorted, carrying)) == sorted(map(sorted, expected))

    def test_U5_F1_build_checks_every_frozen_input_before_writing(self, tmp_path):
        """Interface: build_from_frozen(data_dir, expected_sha256) hashes each named
        input under data_dir, raises ValueError naming the first mismatch, and writes
        no run file under data_dir/runs."""
        rge = _run_graph_expand()
        data = tmp_path / "data"
        data.mkdir()
        names = ["documents.jsonl", "graph_neighbors_v1.jsonl",
                 "graph_nodes_v1.jsonl", "payload_entities_v1.json"]
        for name in names:
            (data / name).write_bytes(f"frozen {name}\n".encode("utf-8"))
        expected = {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                    for name in names}
        # One frozen input changes on disk after its sha was recorded.
        (data / "graph_neighbors_v1.jsonl").write_bytes(b"altered\n")

        with pytest.raises(ValueError, match="graph_neighbors_v1.jsonl"):
            rge.build_from_frozen(data, expected)

        runs = data / "runs"
        assert not runs.exists() or not any(runs.iterdir()), \
            "a sha mismatch must stop the build before any run file is written"

    def test_U5_F2_reranker_pin_is_enforced(self, monkeypatch):
        rge = _run_graph_expand()

        rge._assert_reranker_pin()
        monkeypatch.setattr(rge.rerank, "MODEL_REVISION", "deadbeef")
        with pytest.raises(ValueError, match="pinned|revision"):
            rge._assert_reranker_pin()

    def test_U5_A1_no_live_store_imports(self):
        rge = _run_graph_expand()
        source = Path(rge.__file__).read_text(encoding="utf-8")

        assert "import neo4j" not in source
        assert "qdrant_client" not in source
        assert "live_guard" not in source

    def test_U5_A2_building_twice_is_identical(self, monkeypatch):
        fx = _fixture()
        first, first_counters = _build(fx, monkeypatch)
        second, second_counters = _build(fx, monkeypatch)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
        assert first_counters == second_counters

    def test_U5_counters_carry_all_four_per_query_numbers(self, monkeypatch):
        fx = _fixture()
        _, counters = _build(fx, monkeypatch)

        entry = counters["graph_b1q_rr"]["q1"]
        assert set(entry) >= {"dead_ends", "bq_nodes_used",
                              "links_trimmed", "links_dropped_no_room"}
        assert entry["bq_nodes_used"] == 2


class TestRunGraphExpandArtifacts:
    @pytest.mark.artifacts
    @pytest.mark.model
    def test_U5_B1_rebuild_s2_is_byte_identical(self):
        rge = _run_graph_expand()
        from evals import rerank

        corpus = common.load_documents(common.DEFAULT_CORPUS_VERSION)
        docs_by_id = {d["candidate_id"]: d for d in corpus}
        queries = common.load_queries()
        cosine = common.load_run(common.DEFAULT_CORPUS_VERSION, "vector_cosine_v1")
        bm25 = run_bm25.bm25_rows(queries, corpus)

        rows = rge.build_s2(queries, docs_by_id, cosine, bm25, rerank.Reranker())
        existing = common.load_run(common.DEFAULT_CORPUS_VERSION,
                                   "ladder_s2_cos_bm25_rr")

        def five(row):
            return (row["run_id"], row["query_id"], row["candidate_id"],
                    row["rank"], row["score"])

        assert [five(r) for r in rows] == [five(r) for r in existing]

    @pytest.mark.artifacts
    def test_U5_B7_meta_has_constants_sha_and_per_query_counters(self):
        _run_graph_expand()
        runs_dir = common.data_dir(common.DEFAULT_CORPUS_VERSION, create=False) / "runs"
        constants = ("SEEDS_K", "QUERY_NODES_K", "NEIGHBOR_CAP", "NAME_NORM",
                     "HUB_RULE", "HUB_MAX_DEGREE", "RRF_K", "DEPTH",
                     "rerank_model", "rerank_revision", "bootstrap_resamples",
                     "bootstrap_seed", "win_tie_threshold")

        for run_id in ("graph_b1r_rr", "graph_b2r_ctx_rr", "graph_b1q_rr",
                       "graph_b2q_ctx_rr", "ctrl_c0_deeper_rr",
                       "ctrl_c1_denorm_rr", "ctrl_c2_payload_rr",
                       "ladder_s2_cos_bm25_rr"):
            for suffix in ("", ".graphq"):
                path = runs_dir / f"{run_id}{suffix}.meta.json"
                if not path.exists():
                    pytest.fail(f"missing run meta: {path.name}")
                meta = json.loads(path.read_text(encoding="utf-8"))
                for key in constants:
                    assert key in meta, (path.name, key)
                assert meta["node_vectors_sha256"], path.name
                assert meta["per_query"], path.name
                for qid, entry in meta["per_query"].items():
                    assert set(entry) >= {"dead_ends", "bq_nodes_used",
                                          "links_trimmed",
                                          "links_dropped_no_room"}, (path.name, qid)
                    assert all(isinstance(entry[k], int) for k in
                               ("dead_ends", "bq_nodes_used",
                                "links_trimmed", "links_dropped_no_room"))
