import copy
import json
import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relay import parse_drive_notification

valid_headers = {
    "X-Goog-Channel-ID": "channel-123",
    "X-Goog-Channel-Token": "test-secret",
    "X-Goog-Resource-ID": "resource-456",
    "X-Goog-Resource-State": "change",
    "X-Goog-Message-Number": "17",
}

def test_valid_notification():
    ev = parse_drive_notification(valid_headers, "test-secret", "manthan-test")
    assert ev["installation_id"] == "manthan-test"
    assert ev["channel_id"] == "channel-123"
    assert ev["resource_id"] == "resource-456"
    assert ev["resource_state"] == "change"
    assert ev["message_number"] == "17"
    assert "received_at" in ev
    # valid UTC timestamp
    dt = datetime.datetime.fromisoformat(ev["received_at"].replace("Z", "+00:00"))
    assert dt.tzinfo is not None
    print("Test 1 valid: PASS", ev)

def test_lowercase_headers():
    lower = {k.lower(): v for k, v in valid_headers.items()}
    ev = parse_drive_notification(lower, "test-secret", "manthan-test")
    assert ev["channel_id"] == "channel-123"
    assert ev["resource_id"] == "resource-456"
    assert ev["resource_state"] == "change"
    assert ev["message_number"] == "17"
    print("Test 2 lowercase: PASS")

def test_wrong_token():
    h = dict(valid_headers)
    h["X-Goog-Channel-Token"] = "attacker-token"
    try:
        parse_drive_notification(h, "test-secret", "manthan-test")
        assert False, "should have raised"
    except PermissionError as e:
        msg = str(e)
        assert "attacker-token" not in msg
        assert "test-secret" not in msg
        assert "Invalid channel token" in msg
        print("Test 3 wrong token: PASS")

def test_missing_headers_parametrized():
    for hdr in ["X-Goog-Channel-ID", "X-Goog-Channel-Token", "X-Goog-Resource-ID", "X-Goog-Resource-State", "X-Goog-Message-Number"]:
        h = dict(valid_headers)
        del h[hdr]
        try:
            parse_drive_notification(h, "test-secret", "manthan-test")
            assert False, f"should have raised for missing {hdr}"
        except ValueError as e:
            assert hdr in str(e)
        print(f"  missing {hdr}: PASS")
    print("Test 4 missing: PASS")

def test_secret_absent():
    ev = parse_drive_notification(valid_headers, "test-secret", "manthan-test")
    assert "test-secret" not in ev
    assert "test-secret" not in " ".join(str(v) for v in ev.values())
    assert "test-secret" not in json.dumps(ev)
    assert "channel_token" not in ev
    assert "secret" not in json.dumps(ev).lower() or "test-secret" not in json.dumps(ev)
    print("Test 5 secret absent: PASS")

def test_sync_notification():
    h = dict(valid_headers)
    h["X-Goog-Resource-State"] = "sync"
    ev = parse_drive_notification(h, "test-secret", "manthan-test")
    assert ev["resource_state"] == "sync"
    print("Test 6 sync: PASS")

def test_extra_headers_ignored():
    h = dict(valid_headers)
    h["Content-Type"] = "application/json"
    h["User-Agent"] = "Google"
    h["Host"] = "example.com"
    h["Accept"] = "*/*"
    ev = parse_drive_notification(h, "test-secret", "manthan-test")
    assert ev["channel_id"] == "channel-123"
    print("Test 7 extra headers: PASS")

def test_empty_value():
    for hdr in ["X-Goog-Channel-ID", "X-Goog-Resource-ID"]:
        h = dict(valid_headers)
        h[hdr] = ""
        try:
            parse_drive_notification(h, "test-secret", "manthan-test")
            assert False, f"should have raised for empty {hdr}"
        except ValueError:
            pass
        print(f"  empty {hdr}: PASS")
    print("Test 8 empty: PASS")

def test_not_mutated():
    orig = copy.deepcopy(valid_headers)
    copy_headers = copy.deepcopy(valid_headers)
    parse_drive_notification(copy_headers, "test-secret", "manthan-test")
    assert copy_headers == orig
    print("Test 9 not mutated: PASS")

if __name__ == "__main__":
    test_valid_notification()
    test_lowercase_headers()
    test_wrong_token()
    test_missing_headers_parametrized()
    test_secret_absent()
    test_sync_notification()
    test_extra_headers_ignored()
    test_empty_value()
    test_not_mutated()
    print("\nALL 9 TESTS PASSED")
