"""Store searchable message and link embeddings in Qdrant."""

import os
import uuid
from typing import Callable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Batch 

from embedder import embed_batch, embed_text, embedding_dim


class VectorStore:
    """Qdrant gateway that owns embedding and collection operations."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 collection: Optional[str] = None,
                 vector_size: Optional[int] = None,
                 embed_fn: Optional[Callable[[str], List[float]]] = None,
                 api_key: Optional[str] = None):
        self.client = QdrantClient(
            host=host or os.environ.get("QDRANT_HOST", "localhost"),
            port=int(port or os.environ.get("QDRANT_PORT", "6333")),
            api_key=api_key,
        )
        self.collection = collection or os.environ.get("QDRANT_COLLECTION", "manthan")
        self.embed_fn = embed_fn if embed_fn is not None else embed_text
        self.vector_size = vector_size if vector_size is not None else embedding_dim()

    def close(self):
        self.client.close()

    def ensure_collection(self):
        """Create the configured cosine-similarity collection when absent."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size,
                                            distance=Distance.COSINE),
            )

    def _embed(self, text: str) -> List[float]:
        return self.embed_fn(text)

    def upsert_message(self, msg: dict):
        """Embed a message under a deterministic sender/timestamp point ID."""
        text = msg.get("synthesized_text") or msg.get("original_text") or ""
        sender = msg.get("sender") or ""
        sent_at = msg.get("sent_at") or msg.get("datetime_iso") or ""
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"msg-{sender}-{sent_at}")),
            vector=self._embed(text),
            payload={
                "text": text,
                "sender": sender,
                "sent_at": sent_at,
                "quality": msg.get("quality"),
                "trusted": bool(msg.get("trusted")),
                "topics": msg.get("topics") or [],
                "entities": msg.get("entities") or [],
                "type": "message",
            },
        )
        self.client.upsert(collection_name=self.collection, points=[point])

    def upsert_link(self, link: dict):
        """Embed a link summary when the record contains searchable content."""
        url = link.get("url") or ""
        summary = link.get("summary") or ""
        what_it_is = link.get("what_it_is") or ""
        problem_solved = link.get("problem_solved") or ""
        text = " ".join(p for p in (summary, what_it_is, problem_solved) if p)
        if not text.strip():
            print(f"[vector_store] skipping link with no content: {url}")
            return
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"link-{url}")),
            vector=self._embed(text),
            payload={
                "url": url,
                "summary": summary,
                "what_it_is": what_it_is,
                "problem_solved": problem_solved,
                "title": link.get("title") or "",
                "topics": link.get("topics") or [],
                "entities": link.get("entities") or [],
                "type": "link",
                "source": link.get("source") or "scraped",
            },
        )
        self.client.upsert(collection_name=self.collection, points=[point])

    def add_batch(self, items: List[tuple]) -> None:
        """Embed aligned ``(id, text, payload)`` tuples in one Qdrant upsert."""
        if not items:
            return
        ids = [item[0] for item in items]
        texts = [item[1] for item in items]
        payloads = [item[2] for item in items]
        vectors = embed_batch(texts)
        self.client.upsert(collection_name=self.collection,
                            points=Batch(ids=ids, vectors=vectors,
                                         payloads=payloads))

    def search(self, query: str, limit: int = 10) -> list:
        """Return raw Qdrant hits for an embedded query."""
        vector = self._embed(query)
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
        )
