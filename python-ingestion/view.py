"""Inspect local Neo4j and Qdrant contents from the command line."""

from dotenv import load_dotenv

from store import KnowledgeStore
from vector_store import VectorStore


def _short(value, limit=80):
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def inspect_neo4j():
    """Print graph counts, sample nodes, and message-link connectivity."""
    store = KnowledgeStore()
    with store.driver.session() as session:
        counts = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
        ).data()
        print("\n=== Neo4j node counts ===")
        for row in counts:
            print(f"  {row['label']:<10} {row['count']}")

        rel_counts = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count ORDER BY count DESC"
        ).data()
        print("\n=== Neo4j relationship counts ===")
        for row in rel_counts:
            print(f"  {row['rel']:<10} {row['count']}")

        for label in ("Person", "Message", "Link", "Entity", "Topic"):
            rows = session.run(
                f"MATCH (n:{label}) RETURN n LIMIT 5"
            ).data()
            if not rows:
                continue
            print(f"\n=== Sample {label} nodes ===")
            for row in rows:
                props = dict(row["n"])
                line = ", ".join(f"{k}={_short(v)}" for k, v in props.items())
                print(f"  {line}")

        links_to_messages = session.run(
            "MATCH (m:Message)-[:CONTAINS]->(l:Link) RETURN count(*) AS c"
        ).single()["c"]
        print(f"\nMessages connected to links (CONTAINS): {links_to_messages}")
    store.close()


def inspect_qdrant():
    """Print collection metadata and sample vector payloads."""
    vs = VectorStore()
    vs.ensure_collection()
    info = vs.client.get_collection(vs.collection)
    print("\n=== Qdrant collection ===")
    print(f"  name: {vs.collection}")
    print(f"  points: {info.points_count}")
    print(f"  vectors: {info.config.params.vectors.size} dims, "
          f"distance: {info.config.params.vectors.distance}")

    records, _ = vs.client.scroll(vs.collection, limit=10, with_payload=True)
    print("\n=== Sample Qdrant points ===")
    for rec in records:
        p = rec.payload
        if p.get("type") == "message":
            content = f"{p.get('sender')} | {p.get('sent_at')} | {_short(p.get('text'))}"
        else:
            content = f"{p.get('title')} | {_short(p.get('url'))}"
        print(f"  [{p.get('type')}] {content}")
    vs.close()


def main():
    load_dotenv()
    inspect_neo4j()
    inspect_qdrant()


if __name__ == "__main__":
    main()
