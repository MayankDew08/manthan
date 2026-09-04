import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import azure.functions as func
import function_app


def _valid_headers(token="test-secret"):
    return {
        "X-Goog-Channel-ID": "channel-123",
        "X-Goog-Channel-Token": token,
        "X-Goog-Resource-ID": "resource-456",
        "X-Goog-Resource-State": "change",
        "X-Goog-Message-Number": "17",
    }


def _req(headers, method="POST", url="/api/drive-webhook"):
    return func.HttpRequest(method=method, url=url, headers=headers, body=b"")


def test_health():
    req = func.HttpRequest(method="GET", url="/api/health", headers={}, body=b"")
    resp = function_app.health(req)
    assert resp.status_code == 200
    assert resp.get_body() == b"OK"


def test_valid_notification_returns_204(monkeypatch):
    monkeypatch.setenv("DRIVE_CHANNEL_TOKEN", "test-secret")
    monkeypatch.setenv("INSTALLATION_ID", "manthan-test")
    mock_pub = MagicMock()
    with patch.object(function_app, "_get_publisher", return_value=mock_pub):
        req = _req(_valid_headers())
        resp = function_app.drive_webhook(req)
        assert resp.status_code == 204
        assert mock_pub.publish.call_count == 1
        ev = mock_pub.publish.call_args[0][0]
        assert ev["channel_id"] == "channel-123"
        assert ev["installation_id"] == "manthan-test"
        assert "test-secret" not in str(ev)


def test_missing_header_returns_400(monkeypatch):
    monkeypatch.setenv("DRIVE_CHANNEL_TOKEN", "test-secret")
    mock_pub = MagicMock()
    with patch.object(function_app, "_get_publisher", return_value=mock_pub):
        headers = _valid_headers()
        del headers["X-Goog-Resource-ID"]
        req = _req(headers)
        resp = function_app.drive_webhook(req)
        assert resp.status_code == 400
        mock_pub.publish.assert_not_called()


def test_wrong_token_returns_401(monkeypatch):
    monkeypatch.setenv("DRIVE_CHANNEL_TOKEN", "test-secret")
    mock_pub = MagicMock()
    with patch.object(function_app, "_get_publisher", return_value=mock_pub):
        headers = _valid_headers(token="bad-token")
        req = _req(headers)
        resp = function_app.drive_webhook(req)
        assert resp.status_code == 401
        mock_pub.publish.assert_not_called()


def test_azure_publish_failure_returns_503(monkeypatch):
    monkeypatch.setenv("DRIVE_CHANNEL_TOKEN", "test-secret")
    mock_pub = MagicMock()
    mock_pub.publish.side_effect = RuntimeError("Service Bus down")
    with patch.object(function_app, "_get_publisher", return_value=mock_pub):
        req = _req(_valid_headers())
        resp = function_app.drive_webhook(req)
        assert resp.status_code == 503
        assert b"Service Bus unavailable" in resp.get_body()


def test_unexpected_failure_returns_500(monkeypatch):
    monkeypatch.setenv("DRIVE_CHANNEL_TOKEN", "test-secret")
    # Make _get_publisher raise unexpected error (e.g., misconfig) or parse raise unexpected
    with patch.object(function_app, "_get_publisher", side_effect=RuntimeError("boom unexpected")):
        req = _req(_valid_headers())
        # Publisher creation failure is treated as 503 in current code (inside try), so force unexpected before
        # Instead patch parse to raise unexpected
        with patch("function_app.parse_drive_notification", side_effect=RuntimeError("unexpected")):
            resp = function_app.drive_webhook(req)
            # This falls through to generic 500
            assert resp.status_code == 500


def test_missing_config_returns_500(monkeypatch):
    monkeypatch.delenv("DRIVE_CHANNEL_TOKEN", raising=False)
    monkeypatch.delenv("CHANNEL_TOKEN", raising=False)
    monkeypatch.delenv("EXPECTED_TOKEN", raising=False)
    req = _req(_valid_headers())
    resp = function_app.drive_webhook(req)
    assert resp.status_code == 500
