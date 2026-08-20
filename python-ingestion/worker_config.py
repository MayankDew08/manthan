import os
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class WorkerConfig:
    bot_token: str
    redis_host: str
    redis_port: int
    api_base_url: str


def load_worker_config() -> WorkerConfig:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    redis_addr = os.environ.get("REDIS_ADDR", "localhost:6379").strip()
    try:
        redis_url = urlsplit(f"redis://{redis_addr}")
        redis_port = redis_url.port or 6379
    except ValueError as exc:
        raise RuntimeError("REDIS_ADDR must be in host:port format") from exc
    if not redis_url.hostname:
        raise RuntimeError("REDIS_ADDR must be in host:port format")

    api_base_url = os.environ.get(
        "MANTHAN_API_URL", "http://localhost:8000"
    ).strip().rstrip("/")
    api_url = urlsplit(api_base_url)
    if api_url.scheme not in {"http", "https"} or not api_url.netloc:
        raise RuntimeError("MANTHAN_API_URL must be an HTTP(S) URL")

    return WorkerConfig(
        bot_token=bot_token,
        redis_host=redis_url.hostname,
        redis_port=redis_port,
        api_base_url=api_base_url,
    )
