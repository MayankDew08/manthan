import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Same values as test_worker_config.py: query_worker reads config at import
# time, and app.py's load_dotenv() may already have planted the real
# TELEGRAM_BOT_TOKEN from .env. Direct assignment (not setdefault) matches
# test_worker_config.py and keeps both files green regardless of order.
os.environ["TELEGRAM_BOT_TOKEN"] = "test-token"
os.environ["REDIS_ADDR"] = "redis.internal:6380"
os.environ["MANTHAN_API_URL"] = "http://api.internal:9000/"

import query_worker


FAILS = []


def check(name, condition, extra=""):
    if not condition:
        FAILS.append(name)
    print(f"[{'ok' if condition else 'FAIL'}] {name} {extra}")


def _run_blocked_branch(links):
    """Drive query_worker.main through one blocked-type job with mocked IO."""
    fake_resp = Mock()
    fake_resp.json.return_value = {"count": len(links), "links": links}
    fake_resp.raise_for_status.return_value = None
    sent = {}

    def fake_send(chat_id, text):
        sent["chat_id"] = chat_id
        sent["text"] = text

    job = [("manthan:queries", [("1-0", {"data": json.dumps(
        {"type": "blocked", "chat_id": 7})})])]
    with patch.object(query_worker.requests, "get", return_value=fake_resp), \
         patch.object(query_worker, "send_telegram_message",
                      side_effect=fake_send), \
         patch.object(query_worker, "r") as mock_r:
        mock_r.xreadgroup.side_effect = [job, KeyboardInterrupt()]
        try:
            query_worker.main()
        except KeyboardInterrupt:
            pass
    return sent


def run_all():
    FAILS.clear()

    blocked = {"url": "https://x.com/blocked", "status": "blocked",
               "link_id": "LNK-A", "block_reason": "not_found"}
    pending = {"url": "https://example.com/paywall", "status": "pending_paste",
               "link_id": "LNK-B", "block_reason": "paywall"}

    # Mixed lanes: both shown, count matches listed links.
    sent = _run_blocked_branch([blocked, pending])
    check("mixed blocked+pending_paste both shown",
          "LNK-A" in sent.get("text", "") and "LNK-B" in sent.get("text", ""),
          sent.get("text", "")[:120])
    check("mixed count matches listed links",
          sent.get("text", "").startswith("Blocked links (2)"))

    # Regression: all pending_paste must still be shown (was "No blocked links").
    sent = _run_blocked_branch([pending])
    check("pending_paste shown, not dropped",
          "LNK-B" in sent.get("text", ""), sent.get("text", "")[:120])
    check("pending_paste count is 1",
          sent.get("text", "").startswith("Blocked links (1)"))

    # Empty stays empty.
    sent = _run_blocked_branch([])
    check("empty still reports none",
          sent.get("text", "") == "No blocked links 🎉", sent.get("text", ""))

    print("\n" + ("RESULT: FAILED" if FAILS else "RESULT: ALL PASS"))
    return FAILS


def test_all():
    fails = run_all()
    assert not fails, f"failed checks: {fails}"


def test_paste_over_blocked_hits_both_dbs():
    import app as app_module

    store, vs = MagicMock(), MagicMock()
    app_module.app.state.store = store
    app_module.app.state.vs = vs
    summary = {
        "summary": "pasted summary",
        "what_it_is": "a page",
        "problem_solved": "testing",
        "how_useful": "very",
        "entities": ["e1"],
        "topics": ["t1"],
    }
    with patch.object(app_module, "_existing_title", return_value=""), \
         patch.object(app_module.enhancer, "summarize_content",
                      return_value=summary), \
         patch.object(app_module, "mark_resolved", return_value=1) as mr:
        out = app_module.paste_link(app_module.PasteRequest(
            url="https://x.com/blocked", content="pasted body"))
    assert out["ok"] is True
    # Neo4j: same URL node overwritten to scraped, source marks pasted origin.
    link_arg = store.add_link.call_args[0][0]
    assert store.add_link.call_args[1]["status"] == "scraped"
    assert link_arg["url"] == "https://x.com/blocked"
    assert link_arg["source"] == "paste"
    assert link_arg["summary"] == "pasted summary"
    # Qdrant: vectorized with the same pasted origin marker.
    vs_link = vs.upsert_link.call_args[0][0]
    assert vs_link["source"] == "paste"
    assert vs_link["url"] == "https://x.com/blocked"
    # JSON artifact synced.
    mr.assert_called_once_with("https://x.com/blocked")


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
