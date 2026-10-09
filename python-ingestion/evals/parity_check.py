"""Stage 2: prove the manifest is the index.

Re-embeds every manifest text and compares it with the vector Qdrant stores for
that point. Fails if any cosine is below 0.9999, if two documents share both
text and timestamp (a true duplicate), or if the manifest and collection sizes
differ. The same text posted at a different time is a repost: allowed, reported.

Usage (from python-ingestion/):
  uv run python -m evals.parity_check [--corpus-version pilot-001] [--collection NAME]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import common  # noqa: E402


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity, robust to un-normalised inputs."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def _identity(d: dict) -> tuple:
    return (d["content_hash"], d.get("sent_at") or d.get("url"))


def duplicate_documents(docs: list[dict]) -> list[str]:
    """Hashes stored more than once at the same time/URL: an indexing bug."""
    counts = Counter(_identity(d) for d in docs)
    return sorted({key[0] for key, c in counts.items() if c > 1})


def reposts(docs: list[dict]) -> list[str]:
    """Hashes whose text genuinely recurs at different times (allowed, reported)."""
    times: dict[str, set] = {}
    for d in docs:
        times.setdefault(d["content_hash"], set()).add(_identity(d)[1])
    return sorted(h for h, t in times.items() if len(t) > 1)


def run_parity(corpus_version: str, collection: str | None = None) -> dict:
    common.load_env()
    cfg = common.qdrant_settings()
    collection = collection or cfg["collection"]
    docs = common.load_documents(corpus_version)

    from qdrant_client import QdrantClient
    import embedder

    client = QdrantClient(host=cfg["host"], port=cfg["port"])
    try:
        live_count = client.get_collection(collection).points_count or 0
        stored: dict[str, list[float]] = {}
        ids = [d["qdrant_point_id"] for d in docs]
        for i in range(0, len(ids), 100):
            for rec in client.retrieve(collection, ids=ids[i:i + 100], with_vectors=True):
                stored[str(rec.id)] = rec.vector
    finally:
        client.close()

    missing = [d["candidate_id"] for d in docs if d["qdrant_point_id"] not in stored]
    present = [d for d in docs if d["qdrant_point_id"] in stored]
    result = {
        "documents": len(docs),
        "qdrant_points": live_count,
        "count_matches": len(docs) == live_count,
        "missing_in_qdrant": missing,
        "duplicate_documents": duplicate_documents(docs),
        "reposts": reposts(docs),
        "min_cosine": None,
        "worst": [],
    }
    if present:
        fresh = np.array(embedder.embed_batch([d["embedded_text"] for d in present]))
        cos = cosine_rows(fresh, np.array([stored[d["qdrant_point_id"]] for d in present]))
        result["min_cosine"] = float(cos.min())
        order = np.argsort(cos)[:5]
        result["worst"] = [(present[i]["candidate_id"], float(cos[i])) for i in order]
    result["ok"] = bool(
        result["count_matches"] and not missing and not result["duplicate_documents"]
        and result["min_cosine"] is not None
        and result["min_cosine"] >= common.MIN_PARITY_COSINE)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    ap.add_argument("--collection", default=None)
    args = ap.parse_args(argv)
    r = run_parity(args.corpus_version, args.collection)
    print(f"documents={r['documents']} qdrant_points={r['qdrant_points']} "
          f"min_cosine={r['min_cosine']} duplicates={len(r['duplicate_documents'])} "
          f"reposts={len(r['reposts'])} "
          f"missing_in_qdrant={len(r['missing_in_qdrant'])}")
    if not r["ok"]:
        print("FAIL: manifest is not the index. Lowest cosines:", r["worst"])
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
