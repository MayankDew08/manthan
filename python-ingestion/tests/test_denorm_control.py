"""U4 denorm_control: C1 glued texts and C2 payload expansion (section 13.2).

Message-to-link relations come from the frozen ``graph_neighbors_v1.jsonl``
CONTAINS rows the test builds by hand; a ``doc["links"]`` field is never used.
Token-dependent tests inject a fake tokenizer (1 word = 1 token); C1-E5 uses
the real embedder tokenizer under ``--model``.
"""

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common  # noqa: E402


def _denorm():
    from evals import denorm_control
    return denorm_control


def _graph_expand():
    from evals import graph_expand
    return graph_expand


def _doc(cid, kind="message", text="", title=None):
    doc = {"candidate_id": cid, "candidate_type": kind, "embedded_text": text}
    if kind == "link":
        doc["title"] = title if title is not None else cid
        doc["url"] = f"http://example.test/{cid}"
    return doc


def _contains(msg_id, link_id, in_corpus=True):
    """One frozen ``graph_neighbors_v1.jsonl`` CONTAINS row."""
    return {"candidate_id": msg_id, "neighbor_id": link_id, "path": "contains",
            "connector": None, "connector_degree": None, "in_corpus": in_corpus}


def _neighbors(*rows):
    """Convert frozen rows to the adjacency shape U3/U4 consume."""
    out = {}
    for row in rows:
        edge = {"neighbor": row["neighbor_id"], "path": row["path"],
                "connector": row["connector"],
                "degree": row["connector_degree"],
                "in_corpus": row["in_corpus"]}
        out.setdefault(row["candidate_id"], []).append(edge)
    return out


def _seed(cid, rank):
    return {"candidate_id": cid, "rank": rank}


class WordTokenizer:
    """Injected fake: one word is one token, no special tokens."""

    def encode(self, text):
        return text.split()


class TestC1Behaviour:
    def test_C1_B1_one_in_corpus_link_is_appended(self):
        dc = _denorm()
        docs = {
            "msg1": _doc("msg1", text="hello world"),
            "link:a": _doc("link:a", kind="link", text="alpha body", title="Alpha"),
        }
        neighbors = _neighbors(_contains("msg1", "link:a"))

        texts = dc.denorm_texts(docs, neighbors, tokenizer=WordTokenizer())

        assert texts["msg1"] == "hello world alpha body"
        assert texts["link:a"] == "alpha body"

    def test_C1_B2_two_links_are_appended_in_candidate_id_order(self):
        dc = _denorm()
        docs = {
            "msg1": _doc("msg1", text="hello"),
            "link:b": _doc("link:b", kind="link", text="beta body", title="Beta"),
            "link:a": _doc("link:a", kind="link", text="alpha body", title="Alpha"),
        }
        # Frozen rows arrive in graph order; the output must be candidate_id order.
        neighbors = _neighbors(_contains("msg1", "link:b"), _contains("msg1", "link:a"))

        texts = dc.denorm_texts(docs, neighbors, tokenizer=WordTokenizer())

        assert texts["msg1"] == "hello alpha body beta body"

    def test_C1_B3_message_without_a_link_and_link_docs_are_unchanged(self):
        dc = _denorm()
        docs = {
            "msg1": _doc("msg1", text="just a message"),
            "link:a": _doc("link:a", kind="link", text="alpha body", title="Alpha"),
        }

        texts = dc.denorm_texts(docs, {}, tokenizer=WordTokenizer())

        assert texts["msg1"] == "just a message"
        assert texts["link:a"] == "alpha body"


class TestC1Boundaries:
    def test_C1_E1_link_outside_the_corpus_is_ignored(self):
        dc = _denorm()
        docs = {"msg1": _doc("msg1", text="hello")}
        neighbors = _neighbors(_contains("msg1", "link:outside", in_corpus=False))

        texts = dc.denorm_texts(docs, neighbors, tokenizer=WordTokenizer())

        assert texts["msg1"] == "hello"

    def test_C1_E2_the_same_link_contained_twice_is_appended_once(self):
        dc = _denorm()
        docs = {
            "msg1": _doc("msg1", text="hello"),
            "link:a": _doc("link:a", kind="link", text="alpha body"),
        }
        neighbors = _neighbors(_contains("msg1", "link:a"), _contains("msg1", "link:a"))

        texts = dc.denorm_texts(docs, neighbors, tokenizer=WordTokenizer())

        assert texts["msg1"] == "hello alpha body"

    def test_C1_E3_link_is_trimmed_to_the_budget_on_a_word_boundary(self):
        dc = _denorm()
        docs = {
            "msg1": _doc("msg1", text="m1 m2 m3 m4"),
            "link:a": _doc("link:a", kind="link", text=" ".join(f"l{i}" for i in range(1, 21))),
        }
        neighbors = _neighbors(_contains("msg1", "link:a"))

        texts, meta = dc.denorm_texts_with_tokenizer(
            docs, neighbors, WordTokenizer(), max_tokens=10)

        assert texts["msg1"] == "m1 m2 m3 m4 l1 l2 l3 l4 l5"
        assert meta["links_trimmed"] == 1
        assert meta["links_dropped_no_room"] == 0

    def test_C1_E4_message_alone_at_the_budget_drops_the_link(self):
        dc = _denorm()
        message = " ".join(f"m{i}" for i in range(1, 11))
        docs = {
            "msg1": _doc("msg1", text=message),
            "link:a": _doc("link:a", kind="link", text="alpha body"),
        }
        neighbors = _neighbors(_contains("msg1", "link:a"))

        texts, meta = dc.denorm_texts_with_tokenizer(
            docs, neighbors, WordTokenizer(), max_tokens=10)

        assert texts["msg1"] == message
        assert meta["links_trimmed"] == 0
        assert meta["links_dropped_no_room"] == 1

    @pytest.mark.model
    def test_C1_E5_real_tokenizer_keeps_every_output_at_most_256_tokens(self):
        dc = _denorm()
        docs = {d["candidate_id"]: d for d in common.load_documents(common.DEFAULT_CORPUS_VERSION)}
        rows = common.read_jsonl(
            common.data_dir(common.DEFAULT_CORPUS_VERSION) / "runs" / "graph_neighbors_v1.jsonl")
        neighbors = _neighbors(*[r for r in rows if r["path"] == "contains"])

        texts = dc.denorm_texts(docs, neighbors)
        tokenizer = dc._get_tokenizer()

        for cid, text in texts.items():
            assert len(tokenizer.encode(text)) <= dc.MAX_SEQ_LENGTH, cid


class TestC1FailuresAndAntiCheat:
    def test_C1_F1_parity_below_min_cosine_raises(self):
        dc = _denorm()
        texts = {"msg1": "hello", "link:a": "alpha"}
        stored = {"msg1": [1.0, 0.0], "link:a": [0.0, 1.0]}

        # Re-embedding the unchanged texts reproduces the stored vectors: no raise.
        same = {"hello": [1.0, 0.0], "alpha": [0.0, 1.0]}
        dc.assert_parity(texts, stored, lambda batch: [same[t] for t in batch])

        # A re-embedder that returns the wrong direction is below the parity bar.
        wrong = [[0.0, 1.0]]
        with pytest.raises(ValueError, match="parity|cosine"):
            dc.assert_parity(texts, stored, lambda batch: wrong * len(batch))

    def test_C1_A1_static_and_no_write_calls(self, monkeypatch):
        dc = _denorm()
        source = Path(dc.__file__).read_text(encoding="utf-8")

        assert "QdrantClient" not in source
        assert ".upsert(" not in source
        assert ".create_collection(" not in source
        assert ".delete(" not in source

        import socket

        def no_network(*args, **kwargs):
            raise AssertionError("C1 must not touch the network")

        monkeypatch.setattr(socket, "socket", no_network)
        docs = {"msg1": _doc("msg1", text="hello")}
        assert dc.denorm_texts(docs, {}, tokenizer=WordTokenizer()) == {"msg1": "hello"}

    def test_C1_A2_runs_offline(self, monkeypatch):
        dc = _denorm()
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        import socket

        def no_network(*args, **kwargs):
            raise AssertionError("C1 must not open a socket")

        monkeypatch.setattr(socket, "socket", no_network)

        docs = {"msg1": _doc("msg1", text="hello")}
        assert dc.denorm_texts(docs, {}, tokenizer=WordTokenizer()) == {"msg1": "hello"}


class TestC2Behaviour:
    def test_C2_B1_payload_expansion_matches_the_toy_graph(self):
        """Mirror of U3 B2-B5 as payload strings.

        The contains hop (U3 B1) has no payload form, so it is not mirrored.
        Degree is the number of corpus docs carrying the string: "rust" is on
        S1, S2 and B (3); "agents" on S2 and C (2). Hand-computed:
        B = 1/ln4 + (1/ln4)/2 = 1.082021280; C = (1/2) x 1/ln3 = 0.455119613.
        """
        dc = _denorm()
        gx = _graph_expand()

        payload = {
            "S1": {"entities": ["rust"], "topics": []},
            "S2": {"entities": ["rust"], "topics": ["agents"]},
            "B": {"entities": ["rust"], "topics": []},
            "C": {"entities": [], "topics": ["agents"]},
        }
        corpus = {"S1", "S2", "B", "C"}
        seeds = [_seed("S1", 1), _seed("S2", 2)]

        # The same graph as U3 adjacency rows, built independently of the payload.
        adjacency = {
            "S1": [{"neighbor": "B", "path": "entity", "connector": "rust", "degree": 3}],
            "S2": [{"neighbor": "B", "path": "entity", "connector": "rust", "degree": 3},
                   {"neighbor": "C", "path": "topic", "connector": "agents", "degree": 2}],
        }
        graph_rows, _ = gx.expand_from_results(seeds, adjacency, set(), corpus)
        payload_rows = dc.payload_expand(seeds, payload, set(), corpus)

        assert [r["candidate_id"] for r in payload_rows] == \
               [r["candidate_id"] for r in graph_rows] == ["B", "C"]
        for got, want in zip(payload_rows, graph_rows):
            assert got["score"] == pytest.approx(want["score"], abs=1e-9)
        by_id = {r["candidate_id"]: r for r in payload_rows}
        assert by_id["B"]["score"] == pytest.approx(1.082021280, abs=1e-9)
        assert by_id["C"]["score"] == pytest.approx(0.5 / math.log(3.0), abs=1e-9)
        assert [r["via"]["path"] for r in payload_rows] == \
               ["payload_entity", "payload_topic"]
        assert by_id["B"]["via"]["seed"] == "S1"

        # Cap behaves like U3's.
        capped = dc.payload_expand(seeds, payload, set(), corpus, cap=1)
        assert [r["candidate_id"] for r in capped] == ["B"]

    def test_C2_B2_names_are_compared_after_name_norm(self):
        dc = _denorm()
        payload = {
            "S1": {"entities": ["AI-Development"], "topics": []},
            "D1": {"entities": ["ai development"], "topics": []},
            "D2": {"entities": ["AI_development"], "topics": []},
        }
        corpus = {"S1", "D1", "D2"}

        rows = dc.payload_expand([_seed("S1", 1)], payload, set(), corpus)

        assert {r["candidate_id"] for r in rows} == {"D1", "D2"}
        assert all(r["via"]["connector"] == "ai development" for r in rows)

    def test_C2_B3_a_string_carried_by_17_corpus_docs_is_a_hub(self):
        dc = _denorm()
        names17 = [f"d{i}" for i in range(17)]
        payload17 = {n: {"entities": ["hub"], "topics": []} for n in names17}

        rows17 = dc.payload_expand([_seed("d0", 1)], payload17, set(), set(names17))
        assert rows17 == []

        names16 = names17[:16]
        payload16 = {n: {"entities": ["hub"], "topics": []} for n in names16}
        rows16 = dc.payload_expand([_seed("d0", 1)], payload16, set(), set(names16))
        assert len(rows16) == 15

    def test_C2_E1_empty_payload_is_no_paths_and_no_error(self):
        dc = _denorm()
        payload = {
            "S1": {"entities": [], "topics": []},
            "D1": {"entities": [], "topics": []},
        }

        assert dc.payload_expand([_seed("S1", 1)], payload, set(), {"S1", "D1"}) == []

    def test_C2_F1_corpus_doc_absent_from_payload_names_the_doc(self):
        dc = _denorm()
        payload = {"S1": {"entities": [], "topics": []}}

        with pytest.raises(ValueError, match="D1"):
            dc.payload_expand([_seed("S1", 1)], payload, set(), {"S1", "D1"})

    def test_C2_A1_constants_and_name_norm_come_from_graph_expand(self):
        dc = _denorm()
        gx = _graph_expand()

        for name in ("SEEDS_K", "QUERY_NODES_K", "NEIGHBOR_CAP", "HUB_MAX_DEGREE"):
            assert getattr(dc, name) is getattr(gx, name), name
        assert dc.NAME_NORM is gx.NAME_NORM

        # Not redefined: no module-level assignment of any of them in this file.
        source = Path(dc.__file__).read_text(encoding="utf-8")
        for name in ("SEEDS_K", "QUERY_NODES_K", "NEIGHBOR_CAP", "HUB_MAX_DEGREE", "NAME_NORM"):
            assert not re.search(rf"^{name}\s*=", source, re.M), name
