"""Acceptance gate for the eval corpus. Exits non-zero if any check fails.

  1 Hygiene     0 sender keys, 0 Person nodes, 0 true duplicates, 0 phone matches
                (reposts: same text at a different time, are allowed and reported)
  2 Integrity   manifest size == Qdrant points; run/qrels ids all in the manifest
  3 Parity      re-embedded text matches stored vectors (min cosine >= 0.9999)
  4 Determinism two vector runs give identical (query, candidate, rank)
  6 Blindness   annotation.csv columns/shuffle   (skipped until build_pool has run)
  7 Pool cover  must-pool markers are in the pool (skipped until build_pool has run)
  8 Privacy     evals/data is gitignored; only reviewed code is trackable

Usage (from python-ingestion/):
  uv run python -m evals.check_corpus [--corpus-version pilot-001] [--skip-neo4j]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import common, parity_check, pooling, run_vector  # noqa: E402

ALLOWED_TRACKABLE = {"pool_keywords.json", "predictions.json"}


def _result(name: str, ok: bool | None, detail: str) -> tuple:
    return (name, ok, detail)  # ok=None means SKIP


def check_hygiene(docs: list[dict], collection: str, skip_neo4j: bool) -> tuple:
    from qdrant_client import QdrantClient
    from evals.export_corpus import scroll_all
    cfg = common.qdrant_settings()
    client = QdrantClient(host=cfg["host"], port=cfg["port"])
    try:
        sender_keys = sum(1 for p in scroll_all(client, collection) if "sender" in (p.payload or {}))
    finally:
        client.close()
    dupes = len(parity_check.duplicate_documents(docs))
    repost_count = len(parity_check.reposts(docs))
    phones = sum(len(common.find_phone_numbers(d["embedded_text"])) for d in docs)
    persons = "skipped"
    if not skip_neo4j:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "")))
        try:
            with driver.session() as s:
                persons = s.run("MATCH (p:Person) RETURN count(p) AS c").single()["c"]
        finally:
            driver.close()
    ok = sender_keys == 0 and dupes == 0 and phones == 0 and persons in (0, "skipped")
    return _result("1 hygiene", ok, f"sender_keys={sender_keys} person_nodes={persons} "
                   f"true_duplicates={dupes} reposts={repost_count} phone_matches={phones}")


def check_integrity(docs: list[dict], version: str, run_id: str, live_count: int) -> tuple:
    ids = {d["candidate_id"] for d in docs}
    problems = []
    if len(docs) != live_count:
        problems.append(f"manifest={len(docs)} != qdrant={live_count}")
    run_path = common.data_dir(version, create=False) / "runs" / f"{run_id}.jsonl"
    if run_path.exists():
        stray = {r["candidate_id"] for r in common.read_jsonl(run_path)} - ids
        if stray:
            problems.append(f"{len(stray)} run ids not in manifest")
    qrels_path = common.data_dir(version, create=False) / "qrels.csv"
    note = ""
    if qrels_path.exists():
        try:
            common.load_qrels(qrels_path, ids)
        except ValueError as e:
            problems.append(str(e))
    else:
        note = " (qrels.csv not created yet)"
    return _result("2 integrity", not problems, ("; ".join(problems) or "ok") + note)


def check_parity(version: str, collection: str) -> tuple:
    r = parity_check.run_parity(version, collection)
    return _result("3 parity", r["ok"], f"min_cosine={r['min_cosine']} "
                   f"duplicates={len(r['duplicate_documents'])} reposts={len(r['reposts'])} "
                   f"missing={len(r['missing_in_qdrant'])}")


def check_determinism(docs: list[dict], queries: list[dict], collection: str) -> tuple:
    from vector_store import VectorStore
    cfg = common.qdrant_settings()
    vs = VectorStore(host=cfg["host"], port=cfg["port"], collection=collection)
    try:
        a = run_vector.ranking_signature(run_vector.rank_all(queries, vs, len(docs), "det"))
        b = run_vector.ranking_signature(run_vector.rank_all(queries, vs, len(docs), "det"))
    finally:
        vs.close()
    return _result("4 determinism", a == b and len(a) == len(docs) * len(queries),
                   f"rows={len(a)} identical={a == b}")


def check_pool(version: str, run_id: str, docs: list[dict]) -> list[tuple]:
    out_dir = common.data_dir(version, create=False)
    annotation = out_dir / "annotation.csv"
    if not annotation.exists():
        skip = "annotation.csv not built yet"
        return [_result("6 blindness", None, skip), _result("7 pool coverage", None, skip)]
    from evals.build_pool import cosine_order
    ranked = cosine_order(common.load_run(version, run_id))
    blind = pooling.verify_annotation(annotation, ranked)
    pools: dict[str, dict] = {}
    for row in common.read_jsonl(out_dir / "pool_provenance.jsonl"):
        pools.setdefault(row["query_id"], {})[row["candidate_id"]] = row
    cover = pooling.verify_must_pool(common.load_keywords()["must_pool"], docs, pools)
    return [_result("6 blindness", not blind, "; ".join(blind) or "ok"),
            _result("7 pool coverage", not cover, "; ".join(cover) or "ok")]


def check_privacy() -> tuple:
    root = common.REPO_ROOT
    ignored = subprocess.run(["git", "check-ignore", "-q", "python-ingestion/evals/data/x"],
                             cwd=root).returncode == 0
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "python-ingestion/evals"],
        cwd=root, capture_output=True, text=True).stdout.split()
    stray = [p for p in listed
             if not (p.endswith(".py") or Path(p).name in ALLOWED_TRACKABLE)]
    return _result("8 privacy", ignored and not stray,
                   f"data_ignored={ignored} unexpected_trackable={stray}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    ap.add_argument("--run-id", default=common.DEFAULT_RUN_ID)
    ap.add_argument("--queries", type=Path, default=common.DEFAULT_QUERIES)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--skip-neo4j", action="store_true")
    args = ap.parse_args(argv)

    common.load_env()
    cfg = common.qdrant_settings()
    collection = args.collection or cfg["collection"]
    docs = common.load_documents(args.corpus_version)
    queries = common.load_queries(args.queries)

    from qdrant_client import QdrantClient
    client = QdrantClient(host=cfg["host"], port=cfg["port"])
    try:
        live = client.get_collection(collection).points_count or 0
    finally:
        client.close()

    results = [
        check_hygiene(docs, collection, args.skip_neo4j),
        check_integrity(docs, args.corpus_version, args.run_id, live),
        check_parity(args.corpus_version, collection),
        check_determinism(docs, queries, collection),
        *check_pool(args.corpus_version, args.run_id, docs),
        check_privacy(),
    ]
    failed = False
    for name, ok, detail in results:
        status = "SKIP" if ok is None else "PASS" if ok else "FAIL"
        failed |= ok is False
        print(f"{status:4} {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
