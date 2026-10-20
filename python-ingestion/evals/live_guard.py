import hashlib, os
READ_ONLY_KEY_ENV = "QDRANT_READ_ONLY_API_KEY"
class LiveGuardError(RuntimeError): pass
def _key(env):
    env = os.environ if env is None else env
    k = (env.get(READ_ONLY_KEY_ENV) or "").strip()
    if not k: raise LiveGuardError(f"{READ_ONLY_KEY_ENV} is not set")
    return k, env
def qdrant_read_only_client(env=None):
    from qdrant_client import QdrantClient
    k, env = _key(env)
    return QdrantClient(host=env.get("QDRANT_HOST", "localhost"), port=int(env.get("QDRANT_PORT", "6333")), api_key=k, https=False)
def read_only_vector_store(env=None):
    from qdrant_client import QdrantClient
    from vector_store import VectorStore
    k, env = _key(env)
    host = env.get("QDRANT_HOST", "localhost")
    port = int(env.get("QDRANT_PORT", "6333"))
    store = VectorStore(host=host, port=port, collection=env.get("QDRANT_COLLECTION", "manthan"), api_key=k)
    # VectorStore forwards api_key but not the scheme. qdrant-client 1.9.1 infers
    # https=True whenever api_key is set, which breaks this plain-HTTP Qdrant, so
    # rebuild the client with an explicit https=False.
    store.client.close()
    store.client = QdrantClient(host=host, port=port, api_key=k, https=False)
    return store
class ReadOnlyDriver:
    def __init__(self, inner): self.inner = inner
    def session(self, **kw):
        import neo4j
        kw["default_access_mode"] = neo4j.READ_ACCESS
        return self.inner.session(**kw)
    def close(self): self.inner.close()
class ReadOnlyStore:
    def __init__(self, driver): self.driver = ReadOnlyDriver(driver)
    def close(self): self.driver.close()
def read_only_store(env=None):
    from neo4j import GraphDatabase
    env = os.environ if env is None else env
    return ReadOnlyStore(GraphDatabase.driver(env.get("NEO4J_URI", "bolt://localhost:7687"), auth=(env.get("NEO4J_USER", "neo4j"), env.get("NEO4J_PASSWORD", ""))))
def neo4j_read(driver, cypher, **params):
    with driver.session() as s:
        return s.execute_read(lambda tx: tx.run(cypher, **params).data())
def fingerprint(client, collection, driver):
    ids, off = [], None
    while True:
        pts, off = client.scroll(collection_name=collection, limit=256, offset=off, with_payload=False, with_vectors=False)
        ids += [str(p.id) for p in pts]
        if off is None: break
    nodes = neo4j_read(driver, "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n")
    rels = neo4j_read(driver, "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n")
    return {"qdrant_points": client.count(collection_name=collection, exact=True).count, "qdrant_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(), "neo4j_nodes": dict(sorted((r["label"], r["n"]) for r in nodes)), "neo4j_rels": dict(sorted((r["type"], r["n"]) for r in rels))}
def assert_unchanged(before, after):
    diff = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if diff: raise LiveGuardError(f"live stores changed: {diff}")
def listener_pids(procs):
    return sorted(p for p, c in procs.items() if "drive_event_listener" in c)
def _procs():
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try: out[int(d)] = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            except OSError: pass
    return out
def assert_listener_stopped(procs=None):
    p = listener_pids(_procs() if procs is None else procs)
    if p: raise LiveGuardError(f"Drive listener running: pids {p}")
