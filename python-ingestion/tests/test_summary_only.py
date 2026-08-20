import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import enrich
import llm


FAILS = []
CALL_TAGS = []


def check(name, condition, extra=""):
    if not condition:
        FAILS.append(name)
    print(f"[{'ok' if condition else 'FAIL'}] {name} {extra}")


def fake_llm(_system_prompt, _user_content, **kwargs):
    CALL_TAGS.append(kwargs.get("tag"))
    return [{
        "summary": "Gemma summary grounded in the stored page text.",
        "what_it_is": "A saved technical article.",
        "problem_solved": "It explains the implementation tradeoff.",
        "how_useful": "Use it to choose an implementation approach.",
        "entities": ["Qdrant"],
        "topics": ["vector-databases"],
    }]


tmp = Path(tempfile.mkdtemp(prefix="manthan_summary_only_"))
enrich.SCRAPED_FILE = str(tmp / "scraped.json")
enrich.BLOCKED_FILE = str(tmp / "blocked.json")
enrich.ASK_USER_FILE = str(tmp / "ask_user.json")
enrich.ENRICHED_FILE = str(tmp / "enriched.json")

url = "https://example.org/article"
enrich.save_list(enrich.SCRAPED_FILE, [{
    "url": url,
    "final_url": url,
    "title": "Stored article",
    "source": "auto",
    "raw_text": "Stored source text about a vector database implementation tradeoff.",
    "summary": "Old summary",
    "what_it_is": "",
    "problem_solved": "",
    "how_useful": "",
    "topics": [],
    "entities": [],
}])
enrich.save_list(enrich.BLOCKED_FILE, [])
enrich.save_list(enrich.ASK_USER_FILE, [])
enrich.save_list(enrich.ENRICHED_FILE, [{
    "sent_at": "2026-08-16T10:00:00",
    "sender": "Alice",
    "original_text": f"Read this {url}",
    "links": [url],
    "link_previews": [{"url": url, "status": "scraped", "summary": "Old summary"}],
}])

llm.call_completion = fake_llm
enrich.refresh_summaries(max_workers=1)

scraped = enrich.load_list(enrich.SCRAPED_FILE)
enriched = enrich.load_list(enrich.ENRICHED_FILE)
required = ("summary", "what_it_is", "problem_solved", "how_useful")

check("summary-only calls only Gemma summarization", CALL_TAGS == ["summarizer"], CALL_TAGS)
check("all required stored summary fields are populated",
      all(scraped[0].get(field) for field in required), scraped[0])
preview = enriched[0]["link_previews"][0]
check("enriched preview receives complete Gemma data",
      all(preview.get(field) for field in required), preview)
check("summary-only preserves stored raw text",
      scraped[0]["raw_text"].startswith("Stored source text"))

scraped_before = Path(enrich.SCRAPED_FILE).read_bytes()
enriched_before = Path(enrich.ENRICHED_FILE).read_bytes()


def incomplete_llm(_system_prompt, _user_content, **_kwargs):
    return [{
        "summary": "Gemma summary.",
        "what_it_is": "An article.",
        "problem_solved": "A technical question.",
        "how_useful": "",
        "entities": [],
        "topics": [],
    }]


llm.call_completion = incomplete_llm
try:
    enrich.refresh_summaries(max_workers=1)
    rejected_incomplete = False
except ValueError as exc:
    rejected_incomplete = "how_useful" in str(exc)

check("incomplete Gemma data fails instead of saving empty fields", rejected_incomplete)
check("failed summary-only run preserves prior JSON bytes",
      Path(enrich.SCRAPED_FILE).read_bytes() == scraped_before
      and Path(enrich.ENRICHED_FILE).read_bytes() == enriched_before)

llm.call_completion = fake_llm
malformed_enriched = b"{not valid json"
Path(enrich.ENRICHED_FILE).write_bytes(malformed_enriched)
scraped_before_malformed = Path(enrich.SCRAPED_FILE).read_bytes()
try:
    enrich.refresh_summaries(max_workers=1)
    rejected_malformed = False
except ValueError as exc:
    rejected_malformed = "invalid JSON list" in str(exc)

check("malformed enriched store fails fast", rejected_malformed)
check("malformed store is never overwritten",
      Path(enrich.ENRICHED_FILE).read_bytes() == malformed_enriched
      and Path(enrich.SCRAPED_FILE).read_bytes() == scraped_before_malformed)

print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
sys.exit(1 if FAILS else 0)
