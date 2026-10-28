"""U1: read-only harness for the live stores (spec section 5).

Offline tests use fakes; tests marked ``live`` hit the real Qdrant and Neo4j.
LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import live_guard  # noqa: E402

LIVE_MODULES = ("evals/live_guard.py", "evals/freeze_graph.py", "evals/latency_bench.py")
FORBIDDEN_CALLS = (
    "session.run(", ".run(\"CREATE", "execute_write", "write_transaction",
    ".upsert(", ".delete(", "ensure_collection", "create_collection",
    "recreate_collection", "delete_collection", "create_snapshot", "set_payload",
    "overwrite_payload", "delete_payload", "upload_points", "upload_collection",
    "upsert_message", "upsert_link", "add_message", "add_link",
    "\"QDRANT_API_KEY\"", "'QDRANT_API_KEY'",  # the full key is never used by the harness
)


# ---- fakes -------------------------------------------------------------------

class FakeTx:
    def __init__(self, data_by_marker):
        self.data_by_marker = data_by_marker
        self.queries = []

    def run(self, cypher, **params):
        self.queries.append((cypher, params))
        for marker, data in self.data_by_marker.items():
            if marker in cypher:
                return SimpleNamespace(data=lambda d=data: list(d))
        return SimpleNamespace(data=lambda: [])


class FakeSession:
    def __init__(self, tx):
        self.tx = tx
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_read(self, fn, *args, **kwargs):
        self.read_calls += 1
        return fn(self.tx, *args, **kwargs)

    def execute_write(self, *a, **k):
        raise AssertionError("execute_write must never be called")

    def run(self, *a, **k):  # what production search.py calls
        return self.tx.run(*a, **k)


class FakeDriver:
    def __init__(self, data_by_marker=None):
        self.tx = FakeTx(data_by_marker or {})
        self.session_kwargs = []
        self.sessions = []
        self.closed = False

    def session(self, **kwargs):
        self.session_kwargs.append(kwargs)
        s = FakeSession(self.tx)
        self.sessions.append(s)
        return s

    def close(self):
        self.closed = True


class FakeQdrant:
    def __init__(self, ids):
        self.ids = list(ids)

    def count(self, collection_name, exact=True):
        return SimpleNamespace(count=len(self.ids))

    def scroll(self, collection_name, limit=256, offset=None, with_payload=False,
               with_vectors=False, **_):
        start = offset or 0
        batch = [SimpleNamespace(id=i) for i in self.ids[start:start + limit]]
        nxt = start + limit if start + limit < len(self.ids) else None
        return batch, nxt


NEO4J_COUNTS = {
    "labels(": [{"label": "Message", "n": 148}, {"label": "Link", "n": 44}],
    "type(": [{"type": "CONTAINS", "n": 50}, {"type": "ABOUT", "n": 300}],
}


# ---- Qdrant: read-only key is mandatory ----------------------------------------

class TestReadOnlyQdrant:
    def test_missing_read_only_key_is_refused(self):
        with pytest.raises(live_guard.LiveGuardError):
            live_guard.qdrant_read_only_client(env={})

    def test_blank_read_only_key_is_refused(self):
        with pytest.raises(live_guard.LiveGuardError):
            live_guard.qdrant_read_only_client(env={live_guard.READ_ONLY_KEY_ENV: "  "})

    def test_full_key_alone_is_not_accepted_as_a_substitute(self):
        with pytest.raises(live_guard.LiveGuardError):
            live_guard.qdrant_read_only_client(env={"QDRANT_API_KEY": "placeholder"})

    def test_env_name_is_fixed(self):
        assert live_guard.READ_ONLY_KEY_ENV == "QDRANT_READ_ONLY_API_KEY"


# ---- Neo4j: every session is forced into READ access mode -------------------

class TestReadOnlyNeo4j:
    def test_read_only_driver_forces_read_access_mode(self):
        import neo4j
        inner = FakeDriver()
        ro = live_guard.ReadOnlyDriver(inner)
        with ro.session():
            pass
        assert inner.session_kwargs[-1]["default_access_mode"] == neo4j.READ_ACCESS

    def test_caller_cannot_override_access_mode_to_write(self):
        import neo4j
        inner = FakeDriver()
        ro = live_guard.ReadOnlyDriver(inner)
        with ro.session(default_access_mode=neo4j.WRITE_ACCESS, database="neo4j"):
            pass
        kw = inner.session_kwargs[-1]
        assert kw["default_access_mode"] == neo4j.READ_ACCESS
        assert kw["database"] == "neo4j"  # other kwargs pass through

    def test_read_only_store_quacks_like_knowledge_store(self):
        store = live_guard.ReadOnlyStore(FakeDriver())
        assert isinstance(store.driver, live_guard.ReadOnlyDriver)
        store.close()
        assert store.driver.inner.closed

    def test_neo4j_read_goes_through_execute_read(self):
        driver = FakeDriver({"RETURN 1": [{"x": 1}]})
        rows = live_guard.neo4j_read(driver, "RETURN 1 AS x")
        assert rows == [{"x": 1}]
        assert driver.sessions[-1].read_calls == 1

    def test_neo4j_read_passes_parameters(self):
        driver = FakeDriver()
        live_guard.neo4j_read(driver, "MATCH (m) WHERE m.a = $a RETURN m", a=5)
        assert driver.tx.queries[-1][1] == {"a": 5}


# ---- fingerprint ---------------------------------------------------------------

class TestFingerprint:
    def test_fingerprint_shape(self):
        fp = live_guard.fingerprint(FakeQdrant(["b", "a", "c"]), "col", FakeDriver(NEO4J_COUNTS))
        assert fp["qdrant_points"] == 3
        assert fp["neo4j_nodes"] == {"Link": 44, "Message": 148}
        assert fp["neo4j_rels"] == {"ABOUT": 300, "CONTAINS": 50}
        assert len(fp["qdrant_ids_sha256"]) == 64

    def test_id_hash_ignores_scroll_order(self):
        a = live_guard.fingerprint(FakeQdrant(["a", "b", "c"]), "col", FakeDriver(NEO4J_COUNTS))
        b = live_guard.fingerprint(FakeQdrant(["c", "a", "b"]), "col", FakeDriver(NEO4J_COUNTS))
        assert a == b

    def test_id_hash_pages_through_the_whole_collection(self):
        ids = [f"id{i:04d}" for i in range(600)]  # > one scroll page of 256
        full = live_guard.fingerprint(FakeQdrant(ids), "col", FakeDriver(NEO4J_COUNTS))
        short = live_guard.fingerprint(FakeQdrant(ids[:256]), "col", FakeDriver(NEO4J_COUNTS))
        assert full["qdrant_ids_sha256"] != short["qdrant_ids_sha256"]

    def test_unchanged_fingerprints_pass(self):
        fp = live_guard.fingerprint(FakeQdrant(["a"]), "col", FakeDriver(NEO4J_COUNTS))
        live_guard.assert_unchanged(fp, dict(fp))

    def test_changed_fingerprint_fails_and_names_the_key(self):
        before = live_guard.fingerprint(FakeQdrant(["a"]), "col", FakeDriver(NEO4J_COUNTS))
        after = live_guard.fingerprint(FakeQdrant(["a", "b"]), "col", FakeDriver(NEO4J_COUNTS))
        with pytest.raises(live_guard.LiveGuardError, match="qdrant_points"):
            live_guard.assert_unchanged(before, after)


# ---- Drive listener must be stopped -------------------------------------------

class TestDriveListener:
    def test_listener_process_is_detected(self):
        procs = {101: "python drive_event_listener.py", 202: "python -m uvicorn main:app"}
        assert live_guard.listener_pids(procs) == [101]

    def test_no_listener_passes(self):
        live_guard.assert_listener_stopped({202: "python -m uvicorn main:app"})

    def test_running_listener_fails(self):
        with pytest.raises(live_guard.LiveGuardError, match="101"):
            live_guard.assert_listener_stopped({101: "uv run python drive_event_listener.py"})


# ---- static: live modules never contain a write path ---------------------------

@pytest.mark.parametrize("module", LIVE_MODULES)
def test_live_module_contains_no_write_call(module):
    path = ROOT / module
    if module == "evals/live_guard.py":
        assert path.exists(), "U1 must create evals/live_guard.py"
    if not path.exists():
        return  # owned by a later unit; checked again once it exists
    src = path.read_text(encoding="utf-8")
    hits = [c for c in FORBIDDEN_CALLS if c in src]
    assert hits == [], f"{module} contains write-capable calls: {hits}"


# ---- live ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def live():
    from evals import common
    common.load_env()
    client = live_guard.qdrant_read_only_client()
    store = live_guard.read_only_store()
    yield SimpleNamespace(client=client, store=store,
                          collection=common.qdrant_settings()["collection"])
    client.close()
    store.close()


@pytest.mark.live
def test_live_drive_listener_is_stopped():
    live_guard.assert_listener_stopped()


@pytest.mark.live
def test_live_read_only_key_cannot_create_a_collection(live):
    from qdrant_client.models import Distance, VectorParams
    probe = "ladder_write_probe"
    with pytest.raises(Exception):
        live.client.create_collection(probe, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    assert not live.client.collection_exists(probe)


@pytest.mark.live
def test_live_neo4j_read_rejects_create(live):
    with pytest.raises(Exception):
        live_guard.neo4j_read(live.store.driver, "CREATE (n:LadderWriteProbe) RETURN n")
    rows = live_guard.neo4j_read(live.store.driver,
                                 "MATCH (n:LadderWriteProbe) RETURN count(n) AS n")
    assert rows[0]["n"] == 0


@pytest.mark.live
def test_live_auto_commit_write_through_read_only_driver_is_rejected(live):
    """search._graph_candidates uses session.run; READ mode must block writes there too."""
    with pytest.raises(Exception):
        with live.store.driver.session() as s:
            s.run("CREATE (n:LadderWriteProbe) RETURN n").consume()
    rows = live_guard.neo4j_read(live.store.driver,
                                 "MATCH (n:LadderWriteProbe) RETURN count(n) AS n")
    assert rows[0]["n"] == 0


@pytest.mark.live
def test_live_fingerprint_is_stable(live):
    a = live_guard.fingerprint(live.client, live.collection, live.store.driver)
    b = live_guard.fingerprint(live.client, live.collection, live.store.driver)
    live_guard.assert_unchanged(a, b)
    assert a["qdrant_points"] > 0 and a["neo4j_nodes"]


@pytest.mark.live
def test_live_collection_is_still_the_frozen_corpus(live):
    from evals import common
    docs = common.load_documents(common.DEFAULT_CORPUS_VERSION)
    corpus_ids = sorted(d["qdrant_point_id"] for d in docs)
    ids, offset = [], None
    while True:
        points, offset = live.client.scroll(live.collection, limit=256, offset=offset,
                                            with_payload=False, with_vectors=False)
        ids += [str(p.id) for p in points]
        if offset is None:
            break
    assert sorted(ids) == corpus_ids, "live collection drifted from pilot-001: STOP and escalate"
