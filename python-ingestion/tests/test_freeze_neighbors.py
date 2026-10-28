"""U2 freeze_neighbors: freeze the graph for offline expansion (spec section 13.3).

Artifacts are the frozen files under ``evals/data/pilot-001/runs``. Live tests
only run with ``--live``; an unreachable store is a failure, never a skip.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common  # noqa: E402


def _freeze():
    from evals import freeze_neighbors
    return freeze_neighbors


def _data():
    return common.data_dir(common.DEFAULT_CORPUS_VERSION, create=False)


def _neighbors():
    return common.read_jsonl(_data() / "runs" / "graph_neighbors_v1.jsonl")


def _nodes():
    return common.read_jsonl(_data() / "runs" / "graph_nodes_v1.jsonl")


def _payload():
    return json.loads((_data() / "runs" / "payload_entities_v1.json").read_text(encoding="utf-8"))


def _meta():
    return json.loads((_data() / "runs" / "graph_neighbors_v1.meta.json").read_text(encoding="utf-8"))


def _corpus_ids():
    return {d["candidate_id"] for d in common.load_documents(common.DEFAULT_CORPUS_VERSION)}


D22 = {
    "docs_total": 228,
    "messages": 156,
    "links": 72,
    "contains_edges": 79,
    "merged_entities": 365,
    "merged_topics": 373,
    "connectors_gt_16": 4,
}


def _d22(fn):
    """The pinned D22 counts as the dict the freeze compares and writes."""
    return {
        "docs_total": fn.D22_DOCS_TOTAL,
        "messages": fn.D22_MESSAGES,
        "links": fn.D22_LINKS,
        "contains_edges": fn.D22_CONTAINS_EDGES,
        "merged_entities": fn.D22_MERGED_ENTITIES,
        "merged_topics": fn.D22_MERGED_TOPICS,
        "connectors_gt_16": fn.D22_CONNECTORS_GT_16,
    }


class TestFreezeNeighborsArtifacts:
    @pytest.mark.artifacts
    def test_U2_B1_neighbor_ids_are_valid(self):
        _freeze()
        corpus = _corpus_ids()
        for row in _neighbors():
            assert row["candidate_id"] in corpus, row
            if not row["in_corpus"]:
                assert row["neighbor_id"] not in corpus, row
            else:
                assert row["neighbor_id"] in corpus, row

    @pytest.mark.artifacts
    def test_U2_B2_symmetry_of_every_path(self):
        _freeze()
        rows = _neighbors()
        keys = {(r["candidate_id"], r["neighbor_id"], r["path"], r["connector"])
                for r in rows if r["in_corpus"]}
        for a, b, path, connector in keys:
            assert (b, a, path, connector) in keys, (b, a, path, connector)

    @pytest.mark.artifacts
    def test_U2_B3_degree_consistency_requires_every_node(self):
        _freeze()
        node_degree = {(n["node_type"], n["name"]): n["degree"] for n in _nodes()}
        assert node_degree, "nodes file is empty"

        for row in _neighbors():
            if row["path"] == "contains":
                continue
            key = (row["path"], row["connector"])
            assert key in node_degree, f"missing node for connector {key}"
            assert row["connector_degree"] == node_degree[key], row

    @pytest.mark.artifacts
    def test_U2_B4_names_are_merged(self):
        fn = _freeze()
        assert fn.NAME_NORM("AI-Development") == fn.NAME_NORM("ai development")
        assert fn.NAME_NORM("AI_development") == fn.NAME_NORM("ai development")
        assert fn.NAME_NORM("LLM") != fn.NAME_NORM("LLMs")

        names = {r["connector"] for r in _neighbors() if r["connector"] is not None}
        names |= {n["name"] for n in _nodes()}
        for name in names:
            assert fn.NAME_NORM(name) == name, name

    @pytest.mark.artifacts
    def test_U2_B5_payload_has_exactly_the_corpus(self):
        _freeze()
        assert set(_payload()) == _corpus_ids()
        assert len(_payload()) == 188

    @pytest.mark.artifacts
    def test_U2_B6_graph_set_cosine_run_is_complete(self):
        _freeze()
        rows = common.read_jsonl(_data() / "runs" / "vector_cosine_graphq_v1.jsonl")
        by_query = {}
        for row in rows:
            by_query.setdefault(row["query_id"], []).append(row["rank"])

        assert set(by_query) == {f"g{i:03d}" for i in range(1, 9)}
        for qid, ranks in by_query.items():
            assert len(ranks) == 188, qid
            assert sorted(ranks) == list(range(1, 189)), qid

    @pytest.mark.artifacts
    def test_U2_B7_meta_graph_counts_are_pinned_to_d22(self):
        fn = _freeze()
        assert _d22(fn) == D22
        assert _meta()["graph_counts"] == D22

    @pytest.mark.artifacts
    def test_U2_B8_out_of_corpus_neighbors_are_counted_not_dropped(self):
        """The live graph has 228 docs; the corpus has 188. Dropping the other 40 is a bug."""
        _freeze()
        meta = _meta()
        corpus = _corpus_ids()
        expected_outside = meta["graph_counts"]["docs_total"] - len(corpus)
        assert expected_outside == 40

        assert meta["out_of_corpus_neighbors"] > 0, (
            "out_of_corpus_neighbors is 0: out-of-corpus neighbours were dropped "
            f"(graph has {meta['graph_counts']['docs_total']} docs, corpus has "
            f"{len(corpus)})"
        )
        outside_rows = [r for r in _neighbors() if not r["in_corpus"]]
        assert outside_rows, "no in_corpus=false rows: out-of-corpus neighbours were dropped"
        distinct_outside = {r["neighbor_id"] for r in outside_rows}
        assert len(distinct_outside) <= expected_outside
        # Meta key contract (review #3 S4): out_of_corpus_neighbors counts DISTINCT
        # out-of-corpus docs, not neighbour rows.
        assert meta["out_of_corpus_neighbors"] == len(distinct_outside)


class TestFreezeNeighborsUnit:
    def test_U2_F2_count_drift_raises_before_writing(self):
        fn = _freeze()
        written = []

        # The pinned literal D22, not the module's own constants, is the reference.
        fn.run_freeze(dict(D22), lambda: written.append("wrote"))
        assert written == ["wrote"]

        with pytest.raises(ValueError, match="docs_total"):
            fn.run_freeze({**D22, "docs_total": 229},
                          lambda: written.append("bad"))
        assert written == ["wrote"], "count drift must raise before any file is written"

    def test_U2_F1_unmapped_corpus_doc_raises_without_fuzzy_matching(self):
        fn = _freeze()
        fn.assert_all_mapped({"a"}, {"a"})

        with pytest.raises(ValueError, match="message:abc-1"):
            fn.assert_all_mapped({"message:abc-1"}, {"message:abc"})


class TestFreezeNeighborsLive:
    @pytest.mark.live
    def test_U2_S1_fingerprint_before_equals_after(self):
        _freeze()
        from evals import live_guard

        common.load_env()
        meta = _meta()
        client = live_guard.qdrant_read_only_client()
        store = live_guard.read_only_store()
        try:
            now = live_guard.fingerprint(
                client, common.qdrant_settings()["collection"], store.driver)
        finally:
            client.close()
            store.close()

        assert meta["fingerprint_before"] == meta["fingerprint_after"] == now

    @pytest.mark.live
    def test_U2_S3_listener_checked_before_the_first_read(self, monkeypatch):
        fn = _freeze()
        from evals import live_guard

        calls = []

        def boom(*args, **kwargs):
            calls.append("read")
            raise RuntimeError("stop before the freeze body")

        monkeypatch.setattr(fn.live_guard, "assert_listener_stopped",
                            lambda: calls.append("listener"))
        for accessor in ("qdrant_read_only_client", "read_only_vector_store",
                         "read_only_store"):
            monkeypatch.setattr(fn.live_guard, accessor, boom)

        with pytest.raises(RuntimeError, match="stop before"):
            fn._run_cli()

        assert calls[0] == "listener"
        assert "read" in calls

        # Undo the stub above before exercising the real assert_listener_stopped
        # with a fake process list -- it takes a procs kwarg the stub doesn't.
        monkeypatch.undo()
        with pytest.raises(live_guard.LiveGuardError, match="listener"):
            live_guard.assert_listener_stopped(
                procs={1234: "python -m drive_event_listener"})

    def test_U2_S2_read_only_accessors_only(self):
        """Static check of the module source: no write-capable client, no key."""
        fn = _freeze()
        source = Path(fn.__file__).read_text(encoding="utf-8")

        assert "QDRANT_API_KEY" not in source
        assert "QdrantClient(" not in source
        assert "GraphDatabase" not in source
        assert "live_guard." in source

    @pytest.mark.live
    def test_U2_A1_two_freezes_are_byte_identical(self, tmp_path):
        fn = _freeze()
        first, second = tmp_path / "first", tmp_path / "second"

        fn._run_cli(out_dir=first)
        fn._run_cli(out_dir=second)

        first_runs = first / "runs"
        second_runs = second / "runs"
        names = sorted(p.name for p in first_runs.glob("*"))
        assert names == sorted(p.name for p in second_runs.glob("*"))
        assert names, "the freeze wrote no files"

        for name in names:
            a = (first_runs / name).read_bytes()
            b = (second_runs / name).read_bytes()
            if name.endswith(".meta.json"):
                ma, mb = json.loads(a), json.loads(b)
                ma.pop("created_at", None)
                mb.pop("created_at", None)
                assert ma == mb, name
            else:
                assert a == b, name
