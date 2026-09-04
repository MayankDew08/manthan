import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from publisher import FakePublisher
from relay import parse_drive_notification


def _relay_event():
    headers = {
        "X-Goog-Channel-ID": "channel-123",
        "X-Goog-Channel-Token": "test-secret",
        "X-Goog-Resource-ID": "resource-456",
        "X-Goog-Resource-State": "change",
        "X-Goog-Message-Number": "17",
    }
    return parse_drive_notification(headers, "test-secret", "manthan-test")


def test_starts_empty():
    p = FakePublisher()
    assert p.events == []


def test_publish_stores_one():
    p = FakePublisher()
    ev = {"installation_id": "a", "channel_id": "c"}
    p.publish(ev)
    assert len(p.events) == 1
    assert p.events[0] == ev


def test_publish_preserves_order():
    p = FakePublisher()
    ev1 = {"n": 1}
    ev2 = {"n": 2}
    p.publish(ev1)
    p.publish(ev2)
    assert p.events[0] == ev1
    assert p.events[1] == ev2


def test_publish_does_not_mutate_original():
    p = FakePublisher()
    orig = {"x": {"y": 1}}
    p.publish(orig)
    orig["x"]["y"] = 99
    assert p.events[0]["x"]["y"] == 1
    # also mutate stored copy doesn't affect original
    p.events[0]["x"]["y"] = 50
    assert orig["x"]["y"] == 99


def test_relay_event_passes_through():
    p = FakePublisher()
    ev = _relay_event()
    p.publish(ev)
    assert p.events[0] == ev
    assert p.events[0]["installation_id"] == "manthan-test"
    assert p.events[0]["channel_id"] == "channel-123"

    # publishing again keeps both
    ev2 = copy.deepcopy(ev)
    ev2["message_number"] = "18"
    p.publish(ev2)
    assert len(p.events) == 2
    assert p.events[1]["message_number"] == "18"
