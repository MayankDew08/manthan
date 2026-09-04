import copy
import json
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import publisher
from publisher import AzureServiceBusPublisher


def _make_mocks():
    mock_client = MagicMock()
    mock_sender = MagicMock()
    mock_client.get_queue_sender.return_value = mock_sender
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_sender.__enter__ = Mock(return_value=mock_sender)
    mock_sender.__exit__ = Mock(return_value=False)
    return mock_client, mock_sender


def _fake_credential():
    cred = Mock()
    cred.get_token = Mock(return_value=Mock(token="t", expires_on=9999999999))
    return cred


def test_event_converted_to_valid_json():
    mock_client, mock_sender = _make_mocks()
    with patch.object(publisher, "ServiceBusClient", return_value=mock_client), \
         patch.object(publisher, "ServiceBusMessage", side_effect=lambda p: Mock(payload=p)) as MockMessage:
        pub = AzureServiceBusPublisher(
            fully_qualified_namespace="manthan-relay-md-20260902.servicebus.windows.net",
            queue_name="manthan-drive-events",
            credential=_fake_credential(),
        )
        event = {
            "installation_id": "manthan-test",
            "channel_id": "manual-integration-test",
            "resource_id": "test-resource",
            "resource_state": "change",
            "message_number": "1",
            "received_at": "2026-09-03T00:00:00Z",
        }
        pub.publish(event)
        payload = MockMessage.call_args[0][0]
        assert json.loads(payload) == event


def test_one_service_bus_message_created():
    mock_client, mock_sender = _make_mocks()
    with patch.object(publisher, "ServiceBusClient", return_value=mock_client), \
         patch.object(publisher, "ServiceBusMessage", side_effect=lambda p: Mock(payload=p)) as MockMessage:
        pub = AzureServiceBusPublisher(
            fully_qualified_namespace="ns.servicebus.windows.net",
            queue_name="manthan-drive-events",
            credential=_fake_credential(),
        )
        pub.publish({"a": 1})
        assert MockMessage.call_count == 1


def test_queue_sender_requested():
    mock_client, mock_sender = _make_mocks()
    with patch.object(publisher, "ServiceBusClient", return_value=mock_client), \
         patch.object(publisher, "ServiceBusMessage", side_effect=lambda p: Mock(payload=p)):
        pub = AzureServiceBusPublisher(
            fully_qualified_namespace="ns.servicebus.windows.net",
            queue_name="manthan-drive-events",
            credential=_fake_credential(),
        )
        pub.publish({"a": 1})
        mock_client.get_queue_sender.assert_called_once_with(queue_name="manthan-drive-events")


def test_message_sent_once():
    mock_client, mock_sender = _make_mocks()
    with patch.object(publisher, "ServiceBusClient", return_value=mock_client), \
         patch.object(publisher, "ServiceBusMessage", side_effect=lambda p: Mock(payload=p)):
        pub = AzureServiceBusPublisher(
            fully_qualified_namespace="ns.servicebus.windows.net",
            queue_name="q",
            credential=_fake_credential(),
        )
        pub.publish({"a": 1})
        mock_sender.send_messages.assert_called_once()
        # ensure single message, not batch
        assert mock_sender.send_messages.call_args[0][0].payload == json.dumps({"a": 1})


def test_azure_exceptions_not_swallowed():
    mock_client, mock_sender = _make_mocks()
    mock_sender.send_messages.side_effect = RuntimeError("Service Bus down")
    with patch.object(publisher, "ServiceBusClient", return_value=mock_client), \
         patch.object(publisher, "ServiceBusMessage", side_effect=lambda p: Mock(payload=p)):
        pub = AzureServiceBusPublisher(
            fully_qualified_namespace="ns.servicebus.windows.net",
            queue_name="q",
            credential=_fake_credential(),
        )
        try:
            pub.publish({"a": 1})
            assert False, "should have propagated"
        except RuntimeError as e:
            assert "Service Bus down" in str(e)
    # sender/client still closed via context managers even on failure
    assert mock_client.__exit__.called
    assert mock_sender.__exit__.called


def test_input_event_not_changed():
    mock_client, mock_sender = _make_mocks()
    with patch.object(publisher, "ServiceBusClient", return_value=mock_client), \
         patch.object(publisher, "ServiceBusMessage", side_effect=lambda p: Mock(payload=p)):
        pub = AzureServiceBusPublisher(
            fully_qualified_namespace="ns.servicebus.windows.net",
            queue_name="q",
            credential=_fake_credential(),
        )
        orig = {"x": {"y": 1}, "installation_id": "manthan-test"}
        before = copy.deepcopy(orig)
        pub.publish(orig)
        assert orig == before
