"""app.py's lifespan warm-up (cold-start fix): load the cross-encoder, warm
the embedding model, and refresh the corpus cache before serving, so the
first /search isn't the slow one and concurrent first requests don't each
pay to load the model.

Offline: KnowledgeStore/VectorStore/Reranker/Corpus and
evals.rerank.load_model are all faked; lifespan is driven directly as an
async context manager (no pytest-asyncio dependency -- just asyncio.run).
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402


class FakeVS:
    def __init__(self):
        self.ensure_collection_calls = 0
        self.embed_calls = []
        self.closed = False

    def ensure_collection(self):
        self.ensure_collection_calls += 1

    def _embed(self, text):
        self.embed_calls.append(text)
        return [0.0]

    def close(self):
        self.closed = True


class FakeStore:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeCorpus:
    def __init__(self):
        self.refresh_calls = []

    def refresh(self, vs):
        self.refresh_calls.append(vs)


def _run_lifespan(monkeypatch):
    fake_vs = FakeVS()
    fake_store = FakeStore()
    fake_corpus = FakeCorpus()
    load_model_calls = []

    monkeypatch.setattr(app, "KnowledgeStore", lambda: fake_store)
    monkeypatch.setattr(app, "VectorStore", lambda: fake_vs)
    monkeypatch.setattr(app, "Reranker", lambda: SimpleNamespace())
    monkeypatch.setattr(app.retrieval, "Corpus", lambda: fake_corpus)
    monkeypatch.setattr(app, "load_reranker_model", lambda: load_model_calls.append(1))

    async def run():
        async with app.lifespan(app.app):
            pass

    asyncio.run(run())
    return fake_vs, fake_corpus, load_model_calls


def test_lifespan_loads_the_reranker_model_once(monkeypatch):
    _vs, _corpus, load_model_calls = _run_lifespan(monkeypatch)
    assert load_model_calls == [1]


def test_lifespan_refreshes_the_corpus_against_the_real_vector_store(monkeypatch):
    fake_vs, fake_corpus, _ = _run_lifespan(monkeypatch)
    assert fake_corpus.refresh_calls == [fake_vs]


def test_lifespan_embeds_one_warm_up_query(monkeypatch):
    fake_vs, _corpus, _ = _run_lifespan(monkeypatch)
    assert len(fake_vs.embed_calls) == 1
    assert isinstance(fake_vs.embed_calls[0], str) and fake_vs.embed_calls[0].strip()
