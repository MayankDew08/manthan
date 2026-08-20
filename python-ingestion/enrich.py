"""Orchestrate grading, link scraping, summarization, and artifact generation."""

import argparse
import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from parser import parse_chat
from heuristic_filter import heuristic_filter
from grader import GradeResult, grade_messages, grade_pass2
from link_extract import extract_links, parse_github_url
from link_scraper import LinkScraper
from models import load_list, save_list, to_dict, ScrapedLink, BlockedLink, AskUserLink
import enhancer
import llm
import metrics

GRADED_FILE = "graded_messages.json"
SCRAPED_FILE = "scraped_links.json"
BLOCKED_FILE = "blocked_links.json"
ASK_USER_FILE = "ask_user_links.json"
ENRICHED_FILE = "enriched_messages.json"
DEFAULT_MIN_QUALITY = 4
MAX_WORKERS = 4

ASK_USER_REASONS = {"truncated", "paywall", "auth_required"}
REQUIRED_SUMMARY_FIELDS = ("summary", "what_it_is", "problem_solved", "how_useful")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _merge_pass2(graded, verified):
    merged = []
    it = iter(verified)
    for msg in graded:
        merged.append(next(it) if msg.quality == 3 else msg)
    return merged


def _already_seen() -> set:
    urls = set()
    for rec in load_list(SCRAPED_FILE):
        urls.add(rec.get("url"))
    for rec in load_list(BLOCKED_FILE):
        urls.add(rec.get("url"))
    for rec in load_list(ASK_USER_FILE):
        urls.add(rec.get("url"))
    return urls


def _grade(chat_path: str):
    """Parse, filter, grade twice, and save the final ordered grade records."""
    messages = parse_chat(chat_path)
    kept, discarded = heuristic_filter(messages)
    print(f"[enrich] parsed {len(messages)} messages, kept {len(kept)} after heuristic filter")
    graded = grade_messages(kept)
    verified = grade_pass2(graded)
    final = _merge_pass2(graded, verified)
    save_list(GRADED_FILE, [asdict(m) for m in final])
    return kept, final


def _link_preview_index(scraped, blocked, ask_user):
    """Normalize all link outcomes into previews embedded in message records."""
    previews = {}
    for record in ask_user:
        url = record.get("url")
        if url:
            previews[url] = {
                "url": url,
                "status": "ask_user",
                "block_reason": record.get("block_reason", ""),
                "partial_text": record.get("partial_text", ""),
                "resolved": record.get("resolved", False),
            }
    for record in blocked:
        url = record.get("url")
        if url:
            previews[url] = {
                "url": url,
                "status": "blocked",
                "block_reason": record.get("block_reason", ""),
            }
    for record in scraped:
        url = record.get("url")
        if url:
            previews[url] = {
                "url": url,
                "status": "scraped",
                "final_url": record.get("final_url", url),
                "title": record.get("title", ""),
                "summary": record.get("summary", ""),
                "what_it_is": record.get("what_it_is", ""),
                "problem_solved": record.get("problem_solved", ""),
                "how_useful": record.get("how_useful", ""),
                "source": record.get("source", "auto"),
                "topics": record.get("topics", []),
                "entities": record.get("entities", []),
            }
    return previews


def _attach_link_previews(enriched, scraped, blocked, ask_user, current_keys):
    """Attach current previews without rewriting historical message snapshots."""
    previews = _link_preview_index(scraped, blocked, ask_user)
    return [
        {
            **record,
            "link_previews": [
                previews.get(url, {"url": url, "status": "not_scraped"})
                for url in record.get("links", [])
            ],
        }
        if (record.get("sent_at"), record.get("sender"),
            record.get("original_text")) in current_keys
        else record
        for record in enriched
    ]


def _refresh_summary_record(record):
    url = record.get("url", "")
    raw_text = record.get("raw_text", "")
    if not url or not raw_text.strip():
        raise ValueError(f"cannot summarize stored record without URL/raw_text: {url or '(missing URL)'}")
    github = parse_github_url(url)
    if github is not None and github.kind in ("repo", "tree", "blob", "raw"):
        result = enhancer.summarize_repo(url, record.get("title", ""), raw_text)
    else:
        result = enhancer.summarize_content(url, record.get("title", ""), raw_text)
    empty = [field for field in REQUIRED_SUMMARY_FIELDS if not result.get(field)]
    if empty:
        raise ValueError(
            f"Gemma returned incomplete summary for {url}: empty {', '.join(empty)}"
        )
    return {**record, **result}


def _load_required_list(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON list in required store {path}") from exc
    if not isinstance(data, list):
        raise ValueError(f"invalid JSON list in required store {path}")
    return data


def refresh_summaries(max_workers: int = MAX_WORKERS) -> None:
    """Rewrite scraped and enriched summaries from saved text without fetching."""
    scraped = _load_required_list(SCRAPED_FILE)
    enriched = _load_required_list(ENRICHED_FILE)
    if not scraped:
        print(f"[summaries] no stored records in {SCRAPED_FILE}")
        return

    workers = min(max_workers, len(scraped))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        refreshed = list(pool.map(_refresh_summary_record, scraped))

    refreshed_urls = {record["url"] for record in refreshed}
    blocked = load_list(BLOCKED_FILE)
    ask_user = load_list(ASK_USER_FILE)
    preview_keys = {
        (record.get("sent_at"), record.get("sender"), record.get("original_text"))
        for record in enriched
        if any(url in refreshed_urls for url in record.get("links", []))
    }
    refreshed_enriched = _attach_link_previews(
        enriched, refreshed, blocked, ask_user, preview_keys)
    save_list(SCRAPED_FILE, refreshed)
    save_list(ENRICHED_FILE, refreshed_enriched)
    print(f"[summaries] refreshed {len(refreshed)} stored summaries from Gemma; "
          "0 grader calls, 0 URLs fetched")


def _load_existing_grades(chat_path: str):
    messages = parse_chat(chat_path)
    kept, _discarded = heuristic_filter(messages)
    stored = load_list(GRADED_FILE)
    if len(stored) != len(kept):
        raise ValueError(
            f"scrape-only requires {len(kept)} grades in {GRADED_FILE}, "
            f"found {len(stored)}; run the grading pipeline first"
        )

    final = []
    for index, (message, item) in enumerate(zip(kept, stored), 1):
        if not isinstance(item, dict):
            raise ValueError(f"invalid grade {index} in {GRADED_FILE}")
        grade = GradeResult(**item)
        message_text = getattr(message, "text", "") or ""
        if grade.original_text != message_text:
            raise ValueError(
                f"grade {index} in {GRADED_FILE} does not match the current chat; "
                "run the grading pipeline again"
            )
        final.append(grade)
    print(f"[enrich] scrape-only: loaded {len(final)} existing grades from {GRADED_FILE}")
    return kept, final


def process_candidates(candidates, *, force: bool = False, no_scrape: bool = False,
                       max_workers: int = MAX_WORKERS, seen=None, scraped=None,
                       blocked=None, enriched=None, ask_user=None):
    """Enrich candidates and merge their outcomes into cumulative artifacts.

    Existing URLs are skipped unless ``force`` is true. ``no_scrape`` still
    enriches message metadata but leaves link retrieval untouched. The return
    value contains enriched messages, each link outcome list, counters, and the
    updated set of seen URLs.
    """
    if seen is None:
        seen = set() if force else _already_seen()
    else:
        seen = set(seen)
    scraped = list(load_list(SCRAPED_FILE) if scraped is None else scraped)
    blocked = list(load_list(BLOCKED_FILE) if blocked is None else blocked)
    enriched = list(load_list(ENRICHED_FILE) if enriched is None else enriched)
    ask_user = list(load_list(ASK_USER_FILE) if ask_user is None else ask_user)

    # The files are cumulative across chats. Replace only the current messages,
    # keeping previews on historical rows as immutable snapshots.
    candidate_keys = {
        (data.datetime_iso, data.sender, grade.original_text)
        for data, grade in candidates
    }
    enriched = [record for record in enriched
                if (record.get("sent_at"), record.get("sender"),
                    record.get("original_text")) not in candidate_keys]

    stats = {"scraped": 0, "blocked": 0, "ask_user": 0,
             "skipped_existing": 0, "no_links": 0}
    tasks = []
    refresh_urls = set()
    for d, g in candidates:
        links = extract_links(g.original_text)
        if not links:
            stats["no_links"] += 1
            continue
        en = enhancer.enrich_message(g.original_text, links)
        enriched.append({
            "sent_at": d.datetime_iso, "sender": d.sender, "quality": g.quality,
            "original_text": g.original_text, "links": links,
            "link_intent": en["link_intent"], "entities": en["entities"],
            "topics": en["topics"],
        })
        ctx = {"sender": d.sender, "sent_at": d.datetime_iso,
               "original_text": g.original_text, "link_intent": en["link_intent"]}
        for url in links:
            if url in seen:
                stats["skipped_existing"] += 1
                continue
            seen.add(url)
            if no_scrape:
                continue
            tasks.append((url, d, ctx, en))
            refresh_urls.add(url)

    if force and refresh_urls:
        scraped = [record for record in scraped
                   if record.get("url") not in refresh_urls]
        blocked = [record for record in blocked
                   if record.get("url") not in refresh_urls]
        ask_user = [record for record in ask_user
                    if record.get("url") not in refresh_urls]

    if tasks:
        n_workers = min(max_workers, len(tasks))
        chunks = [tasks[i::n_workers] for i in range(n_workers)]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for outcomes in pool.map(_scrape_chunk, chunks):
                for url, kind, srec, brec, arec, raw_len in outcomes:
                    if kind == "scraped":
                        scraped.append(srec)
                        stats["scraped"] += 1
                        print(f"[enrich] scraped {url} ({raw_len} chars)")
                    elif kind == "blocked":
                        blocked.append(brec)
                        stats["blocked"] += 1
                        print(f"[enrich] blocked {url} ({brec['block_reason']})")
                    else:
                        ask_user.append(arec)
                        stats["ask_user"] += 1
                        print(f"[enrich] ask-user {url} ({arec['block_reason']})")

    enriched = _attach_link_previews(
        enriched, scraped, blocked, ask_user, candidate_keys)
    return enriched, scraped, blocked, ask_user, stats, seen


def _scrape_chunk(chunk):
    """Scrape one chunk with a thread-local browser and classify each outcome."""
    scraper = LinkScraper()
    out = []
    try:
        for url, d, ctx, en in chunk:
            result = scraper.scrape(url)
            if result.status == "scraped":
                gh = parse_github_url(url)
                if gh is not None and gh.kind in ("repo", "tree", "blob", "raw"):
                    summ = enhancer.summarize_repo(url, result.title, result.raw_text)
                else:
                    summ = enhancer.summarize_content(url, result.title, result.raw_text)
                rec = to_dict(ScrapedLink(
                    url=url, sent_at=d.datetime_iso, message_context=ctx,
                    status="scraped", final_url=result.final_url or url,
                    title=result.title, source="auto", raw_text=result.raw_text,
                    summary=summ["summary"], what_it_is=summ.get("what_it_is", ""),
                    problem_solved=summ.get("problem_solved", ""),
                    how_useful=summ.get("how_useful", ""),
                    link_intent=en["link_intent"], topics=summ["topics"],
                    entities=summ["entities"], scraped_at=_now()))
                out.append((url, "scraped", rec, None, None, len(result.raw_text)))
            else:
                reason = result.block_reason or "failed"
                if reason in ASK_USER_REASONS:
                    rec = to_dict(AskUserLink(
                        url=url, sent_at=d.datetime_iso, message_context=ctx,
                        status="blocked", block_reason=reason,
                        partial_text=result.raw_text, created_at=_now()))
                    out.append((url, "ask_user", None, None, rec, 0))
                else:
                    rec = to_dict(BlockedLink(
                        url=url, sent_at=d.datetime_iso, message_context=ctx,
                        status="blocked", block_reason=reason, created_at=_now()))
                    out.append((url, "blocked", None, rec, None, 0))
    finally:
        scraper.close()
    return out


def enrich_chat(chat_path: str, min_quality: int = DEFAULT_MIN_QUALITY,
                force: bool = False, no_scrape: bool = False,
                max_workers: int = MAX_WORKERS,
                scrape_only: bool = False) -> None:
    """Run the artifact pipeline, optionally reusing existing message grades."""
    kept, final = (_load_existing_grades(chat_path) if scrape_only
                   else _grade(chat_path))
    candidates = [(d, g) for d, g in zip(kept, final) if g.quality >= min_quality]
    print(f"[enrich] {len(candidates)} messages with quality >= {min_quality}")
    enriched, scraped, blocked, ask_user, stats, _seen = process_candidates(
        candidates, force=force, no_scrape=no_scrape, max_workers=max_workers)
    save_list(SCRAPED_FILE, scraped)
    save_list(BLOCKED_FILE, blocked)
    save_list(ASK_USER_FILE, ask_user)
    save_list(ENRICHED_FILE, enriched)

    print(f"[enrich] done: {stats['scraped']} scraped, {stats['blocked']} blocked, "
          f"{stats['ask_user']} ask-user, {stats['skipped_existing']} existing, "
          f"{stats['no_links']} candidates without links")
    print(f"[enrich] tokens used: {llm.usage_stats.total_tokens} "
          f"({llm.usage_stats.llm_calls} LLM calls)")
    metrics.write_last_run(source="enrich",
                           chat=chat_path,
                           min_quality=min_quality,
                           stats=stats)


def main() -> None:
    ap = argparse.ArgumentParser(description="Manthan link enrichment pipeline")
    ap.add_argument("--chat", help="path to the WhatsApp chat export .txt")
    ap.add_argument("--min-quality", type=int, default=DEFAULT_MIN_QUALITY)
    ap.add_argument("--force", action="store_true", help="re-scrape URLs already on file")
    ap.add_argument("--scrape-only", action="store_true",
                    help=f"reuse {GRADED_FILE}; do not run either grading pass")
    ap.add_argument("--no-scrape", action="store_true",
                    help="only grade + enrich message metadata, do not scrape")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help="parallel scrape workers (default 4)")
    ap.add_argument("--summaries-only", action="store_true",
                    help="regenerate complete summaries from stored raw text; no grading or scraping")
    args = ap.parse_args()
    if args.summaries_only:
        refresh_summaries(max_workers=args.workers)
        return
    if not args.chat:
        ap.error("--chat is required unless --summaries-only is used")
    enrich_chat(args.chat, min_quality=args.min_quality, force=args.force,
                no_scrape=args.no_scrape, max_workers=args.workers,
                scrape_only=args.scrape_only)


if __name__ == "__main__":
    main()
