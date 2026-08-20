from dotenv import load_dotenv

from models import load_list
from pipeline import ASK_USER_FILE, ENRICHED_FILE, MIN_QUALITY, SCRAPED_FILE
from pipeline import push_to_stores


def main():
    load_dotenv()
    enriched = load_list(ENRICHED_FILE)
    scraped = load_list(SCRAPED_FILE)
    ask_user = load_list(ASK_USER_FILE)
    summary = push_to_stores(enriched, scraped, ask_user, MIN_QUALITY)
    print(f"Ingested {summary['messages']} messages -> Neo4j "
          f"({summary['vectored_messages']} quality >= {MIN_QUALITY} -> Qdrant)")
    print(f"Ingested {summary['scraped_links']} scraped links -> Neo4j "
          f"({summary['vectored_links']} with content -> Qdrant)")
    print(f"Marked {summary['pending_links']} ask-user links -> pending_paste")


if __name__ == "__main__":
    main()