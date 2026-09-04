import datetime
import hmac

from publisher import Publisher



_REQUIRED_HEADERS = {
    "x-goog-channel-id",
    "x-goog-channel-token",
    "x-goog-resource-id",
    "x-goog-resource-state",
    "x-goog-message-number",
}

# Preserve canonical header names for error messages
_CANONICAL_NAMES = {
    "x-goog-channel-id": "X-Goog-Channel-ID",
    "x-goog-channel-token": "X-Goog-Channel-Token",
    "x-goog-resource-id": "X-Goog-Resource-ID",
    "x-goog-resource-state": "X-Goog-Resource-State",
    "x-goog-message-number": "X-Goog-Message-Number",
}


def parse_drive_notification(headers: dict, expected_token: str, installation_id: str) -> dict:
    """Convert Google Drive notification headers into a privacy-safe event.

    Steps:
      1. Normalize header names to lowercase (values untouched)
      2. Validate all 5 required headers exist and are non-empty
      3. Constant-time compare received token vs expected_token
      4. Build safe event dict with UTC received_at
    """
    # 1. Normalize — don't mutate input
    normalized = {k.lower(): v for k, v in headers.items()}

    # 2. Validate required headers
    for req in _REQUIRED_HEADERS:
        if req not in normalized or not str(normalized[req]).strip():
            # Empty string or whitespace-only is invalid
            if req in normalized and normalized[req] == "":
                raise ValueError(
                    f"Missing required header: {_CANONICAL_NAMES[req]}"
                )
            # Also catch missing / None / empty
            val = normalized.get(req)
            if val is None or str(val).strip() == "":
                raise ValueError(
                    f"Missing required header: {_CANONICAL_NAMES[req]}"
                )

    # 3. Authenticate token (constant-time)
    received_token = str(normalized["x-goog-channel-token"])
    if not hmac.compare_digest(received_token, expected_token):
        raise PermissionError("Invalid channel token")

    # 4. Build safe event — never include token
    received_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "installation_id": installation_id,
        "channel_id": str(normalized["x-goog-channel-id"]),
        "resource_id": str(normalized["x-goog-resource-id"]),
        "resource_state": str(normalized["x-goog-resource-state"]),
        "message_number": str(normalized["x-goog-message-number"]),
        "received_at": received_at,
    }
    
def handle_drive_notification(headers: dict, expected_token: str, installation_id: str, publisher: Publisher) -> dict:
    event = parse_drive_notification(headers, expected_token, installation_id)
    publisher.publish(event)
    return event

