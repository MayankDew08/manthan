import logging
import os

import azure.functions as func
from azure.identity import DefaultAzureCredential

from publisher import AzureServiceBusPublisher
from relay import parse_drive_notification

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


def _get_config():
    expected_token = os.getenv("DRIVE_CHANNEL_TOKEN") or os.getenv("CHANNEL_TOKEN") or os.getenv("EXPECTED_TOKEN") or ""
    installation_id = os.getenv("INSTALLATION_ID", "manthan-mayank-01")
    namespace = os.getenv("SERVICE_BUS_NAMESPACE", "manthan-relay-md-20260902.servicebus.windows.net")
    queue = os.getenv("SERVICE_BUS_QUEUE", "manthan-drive-events")
    return expected_token, installation_id, namespace, queue


def _get_publisher(namespace: str, queue: str):
    credential = DefaultAzureCredential()
    return AzureServiceBusPublisher(
        fully_qualified_namespace=namespace,
        queue_name=queue,
        credential=credential,
    )


@app.route(route="drive-webhook", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def drive_webhook(req: func.HttpRequest) -> func.HttpResponse:
    try:
        headers = dict(req.headers)
        expected_token, installation_id, namespace, queue = _get_config()
        if not expected_token:
            logging.error("DRIVE_CHANNEL_TOKEN not configured")
            return func.HttpResponse("Server misconfigured", status_code=500)

        event = parse_drive_notification(headers, expected_token, installation_id)

        try:
            publisher = _get_publisher(namespace, queue)
            publisher.publish(event)
        except Exception as e:
            logging.exception("Service Bus publish failed")
            return func.HttpResponse(f"Service Bus unavailable: {e}", status_code=503)

        return func.HttpResponse(status_code=204)

    except ValueError as e:
        # Use ValueError for validation -> 400 (DriveNotificationValidationError is ValueError)
        return func.HttpResponse(str(e), status_code=400)
    except PermissionError as e:
        return func.HttpResponse(str(e), status_code=401)
    except Exception as e:
        logging.exception("Unexpected error in drive-webhook")
        return func.HttpResponse(f"Internal error: {e}", status_code=500)


@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health(req: func.HttpRequest) -> func.HttpResponse:  # noqa: ARG001
    # Must not touch Drive or Service Bus
    return func.HttpResponse("OK", status_code=200)
