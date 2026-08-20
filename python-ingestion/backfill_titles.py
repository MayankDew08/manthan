"""Backfill missing Neo4j link titles with stable URL-derived labels."""

from dotenv import load_dotenv

from store import KnowledgeStore, _url_to_title


def main():
    """Update every title-less link and report the affected URLs."""
    load_dotenv()
    store = KnowledgeStore()
    updated = []
    with store.driver.session() as session:
        rows = session.run(
            "MATCH (l:Link) WHERE l.title IS NULL RETURN l.url AS url"
        ).data()
        for row in rows:
            title = _url_to_title(row["url"])
            session.run(
                "MATCH (l:Link {url: $url}) SET l.title = $title",
                url=row["url"], title=title,
            )
            updated.append((row["url"], title))
    store.close()

    print(f"Backfilled titles for {len(updated)} link(s):")
    for url, title in updated:
        print(f"  {url}  ->  {title}")


if __name__ == "__main__":
    main()
