# Manthan Telegram bridge

This optional Go bot accepts messages from one configured Telegram user and publishes query or ingestion jobs to Redis. The Python `query_worker.py` and `ingest_worker.py` processes consume those jobs and call the Manthan API.

Copy the example configuration and provide your bot token and numeric Telegram user ID:

```bash
cp .env.example .env
go run .
```

The root `docker-compose.yaml` supplies Redis. The FastAPI service and both Python workers must also be running. The workers require the same bot token in `python-ingestion/.env`; see the root README for the complete workflow.

Telegram messages pass through Telegram's infrastructure, so this optional integration is not fully local. The bot ignores messages from every user except `ALLOWED_TELEGRAM_USER_ID`.
