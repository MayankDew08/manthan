import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

_REQUIRED_FIELDS = {
    "installation_id",
    "channel_id",
    "resource_id",
    "resource_state",
    "message_number",
    "received_at",
}

_ALLOWED_STATES = {"sync", "change"}

# For Stage 4: fail once on dry-run when DRIVE_LISTENER_DRY_RUN_FAIL_ONCE=true
_DRY_RUN_FAIL_TRACKER = {"failed": False}


def handle_drive_event(event: dict, expected_installation_id: str, notify_callback) -> None:
    """Validate event and enqueue local Drive sync.

    Args:
        event: dict from Service Bus JSON body
        expected_installation_id: str this installation should handle
        notify_callback: callable(folder_id, db_path) -> None, e.g. notify_drive_change

    Raises:
        ValueError: malformed event, missing fields, or wrong installation_id -> caller should dead-letter
        Exception: temporary local error -> caller should abandon
    """
    if not isinstance(event, dict):
        raise ValueError("event must be a dict")

    # Validate required fields present and non-empty strings
    for field in _REQUIRED_FIELDS:
        if field not in event or event[field] is None or str(event[field]).strip() == "":
            raise ValueError(f"Missing required field: {field}")

    # Verify installation — single-queue setup, dead-letter others
    if str(event["installation_id"]) != str(expected_installation_id):
        raise ValueError(
            f"Wrong installation_id: expected {expected_installation_id}, got {event['installation_id']}"
        )

    # Validate resource_state allowlist
    if str(event["resource_state"]) not in _ALLOWED_STATES:
        raise ValueError(f"Invalid resource_state: {event['resource_state']}")

    # All good — enqueue local sync (or dry-run print)
    folder_id = os.getenv("DRIVE_FOLDER_ID", "")
    db_path = os.getenv("MANTHAN_IMPORT_DB", "import_state.sqlite")
    if not folder_id:
        raise RuntimeError("DRIVE_FOLDER_ID not configured")

    # Stage 1: safe dry-run without Drive/Gemma
    if os.getenv("DRIVE_LISTENER_DRY_RUN", "").lower() == "true":
        if os.getenv("DRIVE_LISTENER_DRY_RUN_FAIL_ONCE", "").lower() == "true" and not _DRY_RUN_FAIL_TRACKER["failed"]:
            _DRY_RUN_FAIL_TRACKER["failed"] = True
            raise RuntimeError("dry-run simulated temporary failure")
        print(f"Received Drive notification\nfolder_id = {folder_id}\ndb_path = {db_path}", flush=True)
        print(
            f"Received message\nchannel_id: {event['channel_id']}\nresource_state: {event['resource_state']}\nDry run: Drive sync would be requested",
            flush=True,
        )
        return

    notify_callback(folder_id, db_path)


def _decode_message_body(msg) -> dict:
    """Decode Service Bus message body bytes -> dict."""
    try:
        # azure.servicebus ReceivedMessage body is iterable of bytes
        if hasattr(msg, "body"):
            body = msg.body
            if isinstance(body, (bytes, bytearray)):
                raw = bytes(body)
            else:
                # body is iterable[bytes]
                raw = b"".join(bytes(chunk) for chunk in body)
        else:
            raw = bytes(str(msg), "utf-8")
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def run_listener():
    """Continuously receive Service Bus messages and trigger local sync."""
    # Lazy imports so tests and initial startup stay fast (avoid loading HF/Qdrant/Neo4j)
    try:
        from azure.identity import DefaultAzureCredential
        from azure.servicebus import ServiceBusClient
    except ImportError as e:
        raise RuntimeError("azure-servicebus and azure-identity are required") from e

    namespace = os.getenv("SERVICE_BUS_NAMESPACE", "")
    queue = os.getenv("SERVICE_BUS_QUEUE", "manthan-drive-events")
    installation_id = os.getenv("INSTALLATION_ID", "manthan-mayank-01")

    if not namespace:
        raise RuntimeError("SERVICE_BUS_NAMESPACE not configured")

    print("Drive event listener started", flush=True)
    print(f"Installation: {installation_id}", flush=True)
    print(f"Queue: {queue}", flush=True)
    print("Waiting for events...", flush=True)

    credential = DefaultAzureCredential()
    client = ServiceBusClient(
        fully_qualified_namespace=namespace,
        credential=credential,
    )
    receiver = client.get_queue_receiver(queue_name=queue, max_wait_time=5)
    with client:
        with receiver:
            while True:
                messages = receiver.receive_messages(max_message_count=1, max_wait_time=5)
                for msg in messages:
                    try:
                        # Lazy import here so startup stays fast; first message pays HF load cost
                        from drive_changes import notify_drive_change

                        event = _decode_message_body(msg)
                        handle_drive_event(event, installation_id, notify_drive_change)
                        receiver.complete_message(msg)
                        print(f"Completed message for {event.get('channel_id')}", flush=True)
                    except ValueError as e:
                        # Malformed or wrong installation -> dead-letter (don't retry)
                        print(f"Dead-lettering: {e}", flush=True)
                        try:
                            receiver.dead_letter_message(msg, reason=str(e)[:200])
                        except Exception as de:
                            print(f"Dead-letter failed: {de}", flush=True)
                    except Exception as e:
                        # Temporary local error -> abandon for retry
                        print(f"Abandoning (temporary error): {e}", flush=True)
                        try:
                            receiver.abandon_message(msg)
                        except Exception as ae:
                            print(f"Abandon failed: {ae}", flush=True)
                # Small pause to avoid tight loop when idle
                if not messages:
                    time.sleep(1)


if __name__ == "__main__":
    run_listener()
