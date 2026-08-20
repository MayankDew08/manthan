from typing import List, Optional

import enrich
from dotenv import load_dotenv

from models import load_list
from store import KnowledgeStore
from vector_store import VectorStore

ENRICHED_FILE = "enriched_messages.json"
SCRAPED_FILE = "scraped_links.json"
ASK_USER_FILE = "ask_user_links.json"
MIN_QUALITY = 4


def _link_has_content(link: dict) -> bool:
    return any((link.get(k) or "").strip()
               for k in ("summary", "what_it_is", "problem_solved", "raw_text"))


def push_to_stores(enriched: List[dict], scraped: List[dict],
                   ask_user: Optional[List[dict]] = None,
                   min_quality: int = MIN_QUALITY) -> dict:
    load_dotenv()
    store = KnowledgeStore()
    vs = VectorStore()
    vs.ensure_collection()
    try:
        n_msg = 0
        n_vectored = 0
        for msg in enriched:
            store.add_message(msg)
            n_msg += 1
            if msg.get("quality", 0) >= min_quality:
                vs.upsert_message(msg)
                n_vectored += 1

        n_links = 0
        n_vectored_links = 0
        for link in scraped:
            store.add_link(link)
            n_links += 1
            if _link_has_content(link):
                vs.upsert_link(link)
                n_vectored_links += 1

        n_pending = 0
        for rec in ask_user or []:
            store.add_pending_link(rec)
            n_pending += 1

        return {
            "messages": n_msg,
            "vectored_messages": n_vectored,
            "scraped_links": n_links,
            "vectored_links": n_vectored_links,
            "pending_links": n_pending,
        }
    finally:
        store.close()
        vs.close()


def run_full_ingest(chat_path: str, *, force: bool = False,
                    no_scrape: bool = False,
                    min_quality: int = MIN_QUALITY) -> dict:
    enrich.enrich_chat(chat_path, min_quality=min_quality,
                       force=force, no_scrape=no_scrape)
    enriched = load_list(ENRICHED_FILE)
    scraped = load_list(SCRAPED_FILE)
    ask_user = load_list(ASK_USER_FILE)
    summary = push_to_stores(enriched, scraped, ask_user, min_quality)
    summary["chat_path"] = chat_path
    return summary