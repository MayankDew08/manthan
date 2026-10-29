"""mcp_server.py (docs/mcp-server-spec.md section 3, tests per section 5).

Offline only: an httpx.MockTransport stands in for the FastAPI app, and the
tool functions are called directly (``@mcp.tool`` returns the original
function). ``capsys`` proves nothing reaches stdout -- that channel belongs
to the MCP protocol, not to this process's own output.
"""

import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mcp_server  # noqa: E402


def _use_transport(handler, monkeypatch):
    monkeypatch.setattr(mcp_server, "_client",
                        httpx.Client(transport=httpx.MockTransport(handler)))


def _use_log(tmp_path, monkeypatch):
    log_path = tmp_path / "mcp_calls.jsonl"
    monkeypatch.setattr(mcp_server, "LOG_PATH", log_path)
    return log_path


def _read_log(log_path):
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def _search_result(**over):
    base = {"id": "m1", "type": "message", "sent_at": "2026-01-01T00:00:00",
            "text": "a result", "url": "", "title": "", "related_links": []}
    base.update(over)
    return base


# ---- search -------------------------------------------------------------------

def test_search_snippet_is_cut_to_300_characters(tmp_path, monkeypatch, capsys):
    _use_log(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(200, json={
            "results": [_search_result(text="x" * 1000)], "confidence": "strong"})

    _use_transport(handler, monkeypatch)
    out = json.loads(mcp_server.search("anything"))
    assert len(out["results"][0]["text"]) == 300
    assert capsys.readouterr().out == ""


def test_search_returns_at_most_5_results(tmp_path, monkeypatch, capsys):
    _use_log(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(200, json={
            "results": [_search_result(id=f"m{i}") for i in range(8)],
            "confidence": "weak"})

    _use_transport(handler, monkeypatch)
    out = json.loads(mcp_server.search("anything"))
    assert len(out["results"]) <= 5
    assert capsys.readouterr().out == ""


def test_search_includes_id_date_url_title_related_links_and_confidence(tmp_path, monkeypatch):
    _use_log(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(200, json={"results": [_search_result(
            id="l1", url="https://x", title="T",
            related_links=[{"url": "https://y", "title": "Y"}])], "confidence": "strong"})

    _use_transport(handler, monkeypatch)
    out = json.loads(mcp_server.search("anything"))
    item = out["results"][0]
    for key in ("id", "type", "sent_at", "text", "truncated", "url", "title",
                "related_links"):
        assert key in item
    assert out["confidence"] == "strong"


def test_search_truncated_flag_and_ellipsis_only_on_a_cut_snippet(tmp_path, monkeypatch):
    _use_log(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(200, json={"results": [
            _search_result(id="m1", type="message", text="short"),
            _search_result(id="l1", type="link", text="x" * 1000),
        ], "confidence": "strong"})

    _use_transport(handler, monkeypatch)
    out = json.loads(mcp_server.search("anything"))
    by_id = {r["id"]: r for r in out["results"]}

    assert by_id["m1"]["type"] == "message"
    assert by_id["m1"]["truncated"] is False
    assert by_id["m1"]["text"] == "short"

    assert by_id["l1"]["type"] == "link"
    assert by_id["l1"]["truncated"] is True
    assert by_id["l1"]["text"].endswith("…")
    assert len(by_id["l1"]["text"]) == 300


def test_search_kind_defaults_to_any_and_is_posted_through(tmp_path, monkeypatch):
    _use_log(tmp_path, monkeypatch)
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [], "confidence": "none"})

    _use_transport(handler, monkeypatch)
    mcp_server.search("anything")
    assert seen["body"]["kind"] == "any"

    mcp_server.search("anything", kind="introduction")
    assert seen["body"]["kind"] == "introduction"


def test_search_unreachable_api_returns_instructional_text_not_a_raise(
        tmp_path, monkeypatch, capsys):
    _use_log(tmp_path, monkeypatch)

    def handler(request):
        raise httpx.ConnectError("connection refused")

    _use_transport(handler, monkeypatch)
    out = mcp_server.search("anything")  # must not raise
    assert mcp_server.API_URL in out
    assert "not reachable" in out
    assert capsys.readouterr().out == ""


# ---- get_item -------------------------------------------------------------------

def test_get_item_404_message_is_passed_through(tmp_path, monkeypatch, capsys):
    _use_log(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(404, json={"detail": "unknown item id: bogus"})

    _use_transport(handler, monkeypatch)
    out = mcp_server.get_item("bogus")
    assert out == "unknown item id: bogus"
    assert capsys.readouterr().out == ""


def test_get_item_success_returns_the_full_record(tmp_path, monkeypatch):
    _use_log(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(200, json={"id": "m1", "type": "message", "text": "full text"})

    _use_transport(handler, monkeypatch)
    out = json.loads(mcp_server.get_item("m1"))
    assert out["text"] == "full text"


# ---- list_topics / browse_topic --------------------------------------------------

def test_list_topics_success(tmp_path, monkeypatch):
    _use_log(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(200, json={"topics": [{"name": "rag", "count": 3}]})

    _use_transport(handler, monkeypatch)
    out = json.loads(mcp_server.list_topics())
    assert out["topics"] == [{"name": "rag", "count": 3}]


def test_browse_topic_404_message_is_passed_through(tmp_path, monkeypatch):
    _use_log(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(404, json={
            "detail": "unknown topic: xyz. Did you mean: rag?"})

    _use_transport(handler, monkeypatch)
    out = mcp_server.browse_topic("xyz")
    assert out == "unknown topic: xyz. Did you mean: rag?"


def test_browse_topic_percent_encodes_a_slash_in_the_topic(tmp_path, monkeypatch):
    """"AI/ML" must reach the API as one path segment (%2F), not be sent as
    a literal "/" that FastAPI's :path route would otherwise still accept,
    but a plain (non-:path) route would split into two segments."""
    _use_log(tmp_path, monkeypatch)
    seen_path = {}

    def handler(request):
        seen_path["raw"] = request.url.raw_path.decode()
        return httpx.Response(200, json={"topic": "AI/ML", "total": 0,
                                         "items": [], "neighbors": []})

    _use_transport(handler, monkeypatch)
    mcp_server.browse_topic("AI/ML")
    assert seen_path["raw"] == "/topics/AI%2FML"


# ---- call log ---------------------------------------------------------------

def test_every_call_appends_one_json_line_to_the_call_log(tmp_path, monkeypatch):
    log_path = _use_log(tmp_path, monkeypatch)

    def handler(request):
        return httpx.Response(200, json={"results": [], "confidence": "none"})

    _use_transport(handler, monkeypatch)
    mcp_server.search("q1")
    mcp_server.search("q2")

    rows = _read_log(log_path)
    assert len(rows) == 2
    for row in rows:
        for key in ("ts", "tool", "args", "result_count", "confidence", "error"):
            assert key in row
        assert row["tool"] == "search"


def test_error_calls_are_logged_too(tmp_path, monkeypatch):
    log_path = _use_log(tmp_path, monkeypatch)

    def handler(request):
        raise httpx.ConnectError("down")

    _use_transport(handler, monkeypatch)
    mcp_server.get_item("m1")

    rows = _read_log(log_path)
    assert len(rows) == 1
    assert rows[0]["result_count"] is None
    assert rows[0]["error"] is not None


# ---- descriptions / instructions -------------------------------------------------

def test_search_description_says_never_invent_results():
    assert "Never invent results" in mcp_server.SEARCH_DESCRIPTION


def test_server_instructions_say_cite():
    assert "cite" in mcp_server.INSTRUCTIONS


def test_search_description_now_explains_kind_introduction():
    """Phase 3 (spec section 4, item 3): the sentence deferred in Phase 2
    is added back, exactly, right after the "not single keywords." line."""
    sentence = ('Set kind="introduction" only when the user wants people to connect with: '
               "it searches members' self-introductions.")
    assert sentence in mcp_server.SEARCH_DESCRIPTION
    assert mcp_server.SEARCH_DESCRIPTION.index("not single keywords.") < \
        mcp_server.SEARCH_DESCRIPTION.index(sentence)


# ---- live (deselected unless --live) ---------------------------------------

@pytest.mark.live
def test_live_browse_topic_handles_a_slash_topic_end_to_end():
    """mcp_server.browse_topic must percent-encode the slash so FastAPI's
    /topics/{name:path} route receives "AI/ML" whole, against the real,
    running Manthan API (a real topic in this corpus has a "/" in its name).

    This needs the API process itself, not just Qdrant/Neo4j (the other
    --live tests' only dependency) -- skip rather than fail when it's down.
    """
    try:
        httpx.get(f"{mcp_server.API_URL}/health", timeout=5).raise_for_status()
    except httpx.HTTPError:
        pytest.skip(f"Manthan API not running at {mcp_server.API_URL}")

    out = json.loads(mcp_server.browse_topic("AI/ML"))
    assert out["topic"] == "AI/ML"
    assert out["total"] > 0
