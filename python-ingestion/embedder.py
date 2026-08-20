import os
from typing import List

from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2 = 384 dims, ~80MB, very fast, good enough for most RAG
# nomic-embed-text-v1.5 = 768 dims, ~500MB, better for technical/code content
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embedding_dim() -> int:
    model = _get_model()
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def embed_text(text: str) -> List[float]:
    return _get_model().encode(
        text.strip(), normalize_embeddings=True, convert_to_numpy=True
    ).tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    clean = [t.strip() for t in texts if t.strip()]
    if not clean:
        return []
    vectors = _get_model().encode(
        clean, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
    )
    return vectors.tolist()


def unload() -> None:
    global _model
    if _model is not None:
        _model = None
