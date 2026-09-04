import copy
import json
import os
from abc import ABC, abstractmethod

from azure.core.credentials import AzureNamedKeyCredential, AzureSasCredential, TokenCredential
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from dotenv import load_dotenv

load_dotenv()

class Publisher(ABC):
    @abstractmethod
    def publish(self, event: dict) -> None:
        pass


class FakePublisher(Publisher):
    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, event: dict) -> None:
        self.events.append(copy.deepcopy(event))
        
        
class AzureServiceBusPublisher(Publisher):
    def __init__(
        self,
        fully_qualified_namespace: str | None = None,
        queue_name: str | None = None,
        credential: TokenCredential | AzureSasCredential | AzureNamedKeyCredential | None = None,
    ) -> None:
        self.fully_qualified_namespace = fully_qualified_namespace or os.getenv("SERVICE_BUS_NAMESPACE", "")
        self.queue_name = queue_name or os.getenv("SERVICE_BUS_QUEUE", "manthan-drive-events")
        if credential is None:
            raise ValueError("credential is required: TokenCredential | AzureSasCredential | AzureNamedKeyCredential")
        if not isinstance(credential, (AzureSasCredential, AzureNamedKeyCredential)) and not hasattr(credential, "get_token"):
            raise TypeError(
                "credential must be TokenCredential | AzureSasCredential | AzureNamedKeyCredential, "
                f"got {type(credential).__name__}"
            )
        self.credential = credential

    def publish(self, event: dict) -> None:
        payload = json.dumps(event)
        message = ServiceBusMessage(payload)
        client = ServiceBusClient(
            fully_qualified_namespace=self.fully_qualified_namespace,
            credential=self.credential,
        )
        sender = client.get_queue_sender(queue_name=self.queue_name)
        with client:
            with sender:
                sender.send_messages(message)