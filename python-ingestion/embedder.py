"""Lazily load the configured sentence-transformer embedding model."""

import os
from typing import List

from sentence_transformers import SentenceTransformer

# The default favors a small local footprint; larger models can improve
# technical retrieval but require matching Qdrant dimensions and more memory.
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

_model = None


def _get_model() -> SentenceTransformer:
    """Return the process-wide CPU model, loading it on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embedding_dim() -> int:
    """Return the vector width required when creating the Qdrant collection."""
    model = _get_model()
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def embed_text(text: str) -> List[float]:
    """Encode one normalized vector for semantic search."""
    return _get_model().encode(
        text.strip(), normalize_embeddings=True, convert_to_numpy=True
    ).tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Encode non-empty texts as normalized vectors in input order."""
    clean = [t.strip() for t in texts if t.strip()]
    if not clean:
        return []
    vectors = _get_model().encode(
        clean, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
    )
    return vectors.tolist()


def unload() -> None:
    """Release the cached model reference during service shutdown."""
    global _model
    if _model is not None:
        _model = None
