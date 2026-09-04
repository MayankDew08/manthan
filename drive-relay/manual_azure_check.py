"""Manual integration check — NOT collected by pytest (no test_ prefix).

Run locally with real Azure identity:
  uv run python manual_azure_check.py

Sends one clearly marked test event to manthan-drive-events and prints instructions
to verify it arrived via Azure Portal / Service Bus Explorer / az CLI.

Never run this from CI — requires SERVICE_BUS_NAMESPACE / Azure credentials.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from azure.identity import DefaultAzureCredential

from publisher import AzureServiceBusPublisher

TEST_EVENT = {
    "installation_id": os.getenv("INSTALLATION_ID", "manthan-mayank-01"),
    "channel_id": "manual-integration-test",
    "resource_id": "test-resource",
    "resource_state": "change",
    "message_number": "1",
    "received_at": "2026-09-03T00:00:00Z",
}

def main():
    namespace = os.getenv("SERVICE_BUS_NAMESPACE", "manthan-relay-md-20260902.servicebus.windows.net")
    queue = os.getenv("SERVICE_BUS_QUEUE", "manthan-drive-events")
    print(f"Publishing to {namespace}/{queue} ...")
    credential = DefaultAzureCredential()
    pub = AzureServiceBusPublisher(
        fully_qualified_namespace=namespace,
        queue_name=queue,
        credential=credential,
    )
    pub.publish(TEST_EVENT)
    print("Sent:", json.dumps(TEST_EVENT, indent=2))
    print("\nVerify in Azure:")
    print(f"  Portal → Service Bus → {namespace} → Queues → {queue} → Service Bus Explorer → Peek")
    print("  Or: az servicebus queue show --resource-group <rg> --namespace-name manthan-relay-md-20260902 --name manthan-drive-events --query messageCount")
    print("\nCompletion condition: one message with channel_id=manual-integration-test visible in manthan-drive-events.")

if __name__ == "__main__":
    main()
