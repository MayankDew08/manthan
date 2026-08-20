import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import link_ingest
from models import load_list, save_list

FAILS = []


def check(name, cond, extra=""):
    if not cond:
        FAILS.append(name)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {extra}")


def run_all():
    FAILS.clear()
    tmp = Path(tempfile.mkdtemp(prefix="manthan_test_"))
    link_ingest.ASK_USER_FILE = str(tmp / "ask_user.json")
    link_ingest.SCRAPED_FILE = str(tmp / "scraped.json")

    save_list(link_ingest.ASK_USER_FILE, [
        {"url": "https://a.com/older", "sent_at": "2026-01-01T00:00:00", "block_reason": "truncated",
         "partial_text": "teaser only", "message_context": {"original_text": "older link"},
         "resolved": False},
        {"url": "https://b.com/resolved", "sent_at": "2026-01-02T00:00:00", "block_reason": "paywall",
         "message_context": {}, "resolved": True},
        {"url": "https://c.com/newer", "sent_at": "2026-01-03T00:00:00", "block_reason": "auth_required",
         "message_context": {"original_text": "newer link"}, "resolved": False},
    ])

    pending = link_ingest._pending()
    check("pending excludes resolved + sorted by sent_at",
          [p["url"] for p in pending] == ["https://a.com/older", "https://c.com/newer"], pending)

    def fake_summarize(url, title, content):
        return {"summary": "summary of " + url, "entities": ["x"], "topics": ["y"]}

    original_summarize = link_ingest.enhancer.summarize_content
    try:
        link_ingest.enhancer.summarize_content = fake_summarize

        paste_file = tmp / "paste.txt"
        paste_file.write_text("User pasted article content here.", encoding="utf-8")

        link_ingest.ingest(0, file_path=str(paste_file))
    finally:
        link_ingest.enhancer.summarize_content = original_summarize

    scraped = load_list(link_ingest.SCRAPED_FILE)
    check("scraped entry created", len(scraped) == 1 and scraped[0]["source"] == "manual_paste"
          and scraped[0]["raw_text"] == "User pasted article content here.", scraped[0])
    check("ingest marked resolved",
          load_list(link_ingest.ASK_USER_FILE)[0]["resolved"] is True)
    check("only the target resolved", not load_list(link_ingest.ASK_USER_FILE)[2]["resolved"])

    print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
    return FAILS


def test_all():
    fails = run_all()
    assert not fails, f"failed checks: {fails}"


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)