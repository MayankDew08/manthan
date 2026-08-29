import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import parser
from import_state import ImportStateStore, identify_messages

SRC = "local:test-chat"
ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "tests" / "test_chat.txt"

EXTRA_LINES = [
    "14/08/2026, 11:02:00 PM Eve: Brand new message from Eve",
    "14/08/2026, 11:03:00 PM Alice: Another fresh line https://example.com/new",
]


def _augmented_path(tmp_path: Path) -> str:
    augmented = CHAT.read_text(encoding="utf-8").rstrip("\n") + "\n" + "\n".join(EXTRA_LINES) + "\n"
    p = tmp_path / "chat_aug.txt"
    p.write_text(augmented, encoding="utf-8")
    return str(p)


def _outcome(msg) -> str:
    text = msg.data.text.strip()
    if msg.data.is_media or text == "":
        return "heuristic_drop"
    if "http" in text:
        return "stored"
    return "low_quality"


@pytest.fixture
def base_ids():
    return identify_messages(SRC, parser.parse_chat(str(CHAT)))


@pytest.fixture
def aug_ids(tmp_path):
    return identify_messages(SRC, parser.parse_chat(_augmented_path(tmp_path)))


def test_base_parse_count(base_ids):
    assert len(base_ids) == 44


def test_last_five_base_messages(base_ids):
    last = base_ids[-5:]
    assert last[-1].data.sender == "Dave"
    assert "personal brand" in last[-1].data.text
    # previous vs extras: base tail does NOT contain the new lines
    all_text = " ".join(m.data.text for m in last)
    assert "Brand new message from Eve" not in all_text


def test_augmented_tail_includes_extras(base_ids, aug_ids):
    last = aug_ids[-5:]
    all_text = " ".join(m.data.text for m in last)
    assert "Brand new message from Eve" in all_text
    assert "Another fresh line" in all_text


def test_augmented_only_adds_extras(base_ids, aug_ids):
    base_set = {m.message_id for m in base_ids}
    added = [m for m in aug_ids if m.message_id not in base_set]
    assert len(added) == len(EXTRA_LINES)
    texts = {m.data.text for m in added}
    assert "Brand new message from Eve" in texts
    assert "Another fresh line https://example.com/new" in texts


def test_outcome_tally_on_chat(base_ids, tmp_path):
    store = ImportStateStore(str(tmp_path / "import_state.sqlite"))
    store.mark_processed(SRC, [(m.message_id, _outcome(m)) for m in base_ids])
    rows = store.connection.execute(
        "SELECT outcome, COUNT(*) AS c FROM processed_messages GROUP BY outcome"
    ).fetchall()
    tally = {r["outcome"]: r["c"] for r in rows}
    assert tally == {"low_quality": 34, "stored": 8, "heuristic_drop": 2}
    store.close()


def test_incremental_import_flow(tmp_path):
    base = identify_messages(SRC, parser.parse_chat(str(CHAT)))
    aug = identify_messages(SRC, parser.parse_chat(_augmented_path(tmp_path)))

    store = ImportStateStore(str(tmp_path / "import_state.sqlite"))
    store.upsert_source(SRC, "local", "test_chat.txt")

    first = store.find_unseen_messages(SRC, [m.message_id for m in base])
    assert len(first) == len(base)

    store.mark_processed(SRC, [(m.message_id, "stored") for m in base])
    store.update_source_revision(SRC, "rev-1")

    second = store.find_unseen_messages(SRC, [m.message_id for m in aug])
    base_set = {m.message_id for m in base}
    added = [m for m in aug if m.message_id not in base_set]

    assert set(second) == {m.message_id for m in added}
    assert len(second) == len(EXTRA_LINES)

    store.close()
