"""Stage 1: export the live Qdrant collection as a frozen corpus manifest.

Writes evals/data/<version>/documents.jsonl and meta.json. Fails (writing
nothing) if any embedded text still contains a phone-number-like string.

Usage (from python-ingestion/):
  uv run python -m evals.export_corpus [--corpus-version pilot-001] [--collection NAME]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import common  # noqa: E402


def points_to_documents(points, count_tokens, max_seq_length: int,
                        corpus_version: str) -> list[dict]:
    """Turn Qdrant points into manifest rows, ordered deterministically."""
    docs = []
    for p in points:
        payload = p.payload or {}
        ctype = payload.get("type")
        if ctype not in ("message", "link"):
            raise ValueError(f"point {p.id} has unexpected type {ctype!r}")
        text = common.embedded_text(payload)
        if not text:
            raise ValueError(f"point {p.id} has empty embedded text")
        tokens = count_tokens(text)
        docs.append({
            "candidate_id": f"{ctype}:{p.id}",
            "candidate_type": ctype,
            "qdrant_point_id": str(p.id),
            "sent_at": payload.get("sent_at") if ctype == "message" else None,
            "title": (payload.get("title") or None) if ctype == "link" else None,
            "url": (payload.get("url") or None) if ctype == "link" else None,
            "embedded_text": text,
            "token_count": tokens,
            "truncated": tokens > max_seq_length,
            "content_hash": common.content_hash(text),
            "corpus_version": corpus_version,
        })
    docs.sort(key=lambda d: (d["candidate_type"], d["qdrant_point_id"]))
    return docs


def pii_findings(docs: list[dict]) -> list[str]:
    """Masked description of every phone-like match (URLs are stripped first)."""
    out = []
    for d in docs:
        for m in common.find_phone_numbers(d["embedded_text"]):
            out.append(f"{d['candidate_id']}: {common.mask_digits(m)!r}")
    return out


def scroll_all(client, collection: str):
    points, offset = [], None
    while True:
        batch, offset = client.scroll(collection_name=collection, limit=256,
                                      with_payload=True, with_vectors=False,
                                      offset=offset)
        points.extend(batch)
        if offset is None:
            return points


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-version", default=common.DEFAULT_CORPUS_VERSION)
    ap.add_argument("--collection", default=None)
    args = ap.parse_args(argv)

    common.load_env()
    cfg = common.qdrant_settings()
    collection = args.collection or cfg["collection"]

    from qdrant_client import QdrantClient
    import embedder

    client = QdrantClient(host=cfg["host"], port=cfg["port"])
    try:
        if not client.collection_exists(collection):
            print(f"FAIL: collection {collection!r} does not exist")
            return 1
        expected = client.get_collection(collection).points_count or 0
        points = scroll_all(client, collection)
    finally:
        client.close()
    if len(points) != expected:
        print(f"FAIL: scrolled {len(points)} points but collection reports {expected}")
        return 1

    model = embedder._get_model()
    max_len = int(model.max_seq_length)
    count = lambda t: len(model.tokenizer(t, truncation=False, verbose=False)["input_ids"])
    docs = points_to_documents(points, count, max_len, args.corpus_version)

    findings = pii_findings(docs)
    if findings:
        print(f"FAIL: PII gate - {len(findings)} phone-like match(es) in embedded text:")
        for f in findings[:20]:
            print("  ", f)
        print("Nothing was written. Fix the source data (see spec Phase 0) and re-export.")
        return 1

    sender_keys = sum(1 for p in points if "sender" in (p.payload or {}))
    out = common.data_dir(args.corpus_version)
    common.write_jsonl(out / "documents.jsonl", docs)
    meta = {
        "corpus_version": args.corpus_version,
        "collection": collection,
        "point_count": len(docs),
        "by_type": dict(Counter(d["candidate_type"] for d in docs)),
        "truncated_documents": sum(d["truncated"] for d in docs),
        "points_with_sender_key": sender_keys,
        "embed_model": embedder.MODEL_NAME,
        "max_seq_length": max_len,
        "created_at": common.utc_now(),
        **common.git_state(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(docs)} documents to {out}")
    print(json.dumps({k: meta[k] for k in ("by_type", "truncated_documents",
                                           "points_with_sender_key")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
