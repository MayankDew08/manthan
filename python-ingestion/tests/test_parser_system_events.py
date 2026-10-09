"""WhatsApp system events must never become messages (they carry phone numbers)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import parser


def _parse(tmp_path: Path, lines: list[str]):
    p = tmp_path / "chat.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return parser.parse_chat(str(p))


def test_group_link_join_is_not_a_message(tmp_path):
    msgs = _parse(tmp_path, [
        "14/08/2026, 10:30:15 AM +91 98765 43210 joined using a group link.",
    ])
    assert msgs == []


def test_added_event_is_not_a_message(tmp_path):
    msgs = _parse(tmp_path, [
        "14/08/2026, 10:30:15 AM +1 555 0100 was added",
    ])
    assert msgs == []


def test_encryption_notice_is_not_a_message(tmp_path):
    msgs = _parse(tmp_path, [
        "14/08/2026, 10:00:00 AM Messages and calls are end-to-end encrypted. "
        "No one outside of this chat, not even WhatsApp, can read or listen to them.",
    ])
    assert msgs == []


def test_normal_message_keeps_only_the_body(tmp_path):
    msgs = _parse(tmp_path, ["14/08/2026, 10:30:15 AM Alice: hello"])
    assert [m.text for m in msgs] == ["hello"]


def test_multiline_message_still_joins_continuation_lines(tmp_path):
    msgs = _parse(tmp_path, [
        "14/08/2026, 10:30:15 AM Alice: first line",
        "second line",
        "third line",
    ])
    assert [m.text for m in msgs] == ["first line\nsecond line\nthird line"]


def test_system_event_between_messages_leaves_neighbours_intact(tmp_path):
    msgs = _parse(tmp_path, [
        "14/08/2026, 10:30:15 AM Alice: before",
        "14/08/2026, 10:31:00 AM +91 98765 43210 joined using a group link.",
        "14/08/2026, 10:32:00 AM Bob: after",
    ])
    assert [m.text for m in msgs] == ["before", "after"]


def test_continuation_after_system_event_does_not_leak_into_previous_message(tmp_path):
    msgs = _parse(tmp_path, [
        "14/08/2026, 10:30:15 AM Alice: before",
        "14/08/2026, 10:31:00 AM +1 555 0100 was added",
        "stray continuation of the system event",
    ])
    assert [m.text for m in msgs] == ["before"]


def test_24_hour_header_without_ampm_still_parses(tmp_path):
    msgs = _parse(tmp_path, ["21/08/2024, 21:17 - Carol: late message"])
    assert [m.text for m in msgs] == ["late message"]
    assert msgs[0].datetime_iso == "2024-08-21T21:17:00"
