"""The incremental importer re-parses its own temp file; the round trip must be lossless."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import incremental_import as ii
from parser import Data, parse_chat


def _roundtrip(tmp_path, messages):
    p = tmp_path / "batch.txt"
    p.write_text(ii._serialized_text(messages), encoding="utf-8")
    return parse_chat(str(p))


def test_plain_message_survives(tmp_path):
    src = [Data("2026-08-14T10:30:00", "just a short note", False)]
    out = _roundtrip(tmp_path, src)
    assert [(m.datetime_iso, m.text) for m in out] == [("2026-08-14T10:30:00", "just a short note")]


def test_body_containing_colon_space_is_not_truncated(tmp_path):
    src = [Data("2026-08-14T10:30:00", "Note: see https://example.com/a", False)]
    out = _roundtrip(tmp_path, src)
    assert [m.text for m in out] == ["Note: see https://example.com/a"]


def test_multiline_body_survives(tmp_path):
    src = [Data("2026-08-14T10:30:00", "line one\nline two: with colon\nline three", False)]
    out = _roundtrip(tmp_path, src)
    assert [m.text for m in out] == ["line one\nline two: with colon\nline three"]


def test_media_flag_survives(tmp_path):
    src = [Data("2026-08-14T10:30:00", "<Media omitted>", True)]
    out = _roundtrip(tmp_path, src)
    assert [(m.text, m.is_media) for m in out] == [("<Media omitted>", True)]


def test_order_and_count_preserved(tmp_path):
    src = [Data(f"2026-08-14T10:30:0{i}", f"msg {i}", False) for i in range(5)]
    out = _roundtrip(tmp_path, src)
    assert [m.text for m in out] == [f"msg {i}" for i in range(5)]
