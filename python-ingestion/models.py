"""Dataclass schemas and JSON helpers for local ingestion artifacts."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class LinkRef:
    """Shared source context recorded for every extracted link."""

    url: str
    sent_at: str
    message_context: dict
    status: str = "scraped"


@dataclass
class ScrapedLink(LinkRef):
    """Successfully retrieved link content and its generated summary."""

    final_url: str = ""
    title: str = ""
    source: str = "auto"
    raw_text: str = ""
    summary: str = ""
    what_it_is: str = ""
    problem_solved: str = ""
    how_useful: str = ""
    link_intent: str = ""
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    scraped_at: str = ""


@dataclass
class BlockedLink(LinkRef):
    """Link that automated retrieval could not access."""

    block_reason: str = ""
    created_at: str = ""
    resolved: bool = False
    ingested_at: Optional[str] = None


@dataclass
class AskUserLink(LinkRef):
    """Blocked or truncated link that can be resolved by manual paste."""

    block_reason: str = ""
    partial_text: str = ""
    created_at: str = ""
    resolved: bool = False
    ingested_at: Optional[str] = None


def to_dict(obj) -> dict:
    return asdict(obj)


def load_list(path: str) -> list:
    """Load a JSON list, returning an empty list for missing or invalid data."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_list(path: str, items: list) -> None:
    """Write a readable UTF-8 JSON list artifact."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def save_dict(path: str, data: dict) -> None:
    """Write a readable UTF-8 JSON object artifact."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
