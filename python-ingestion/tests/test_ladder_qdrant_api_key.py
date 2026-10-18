"""U0: every existing Qdrant client reads QDRANT_API_KEY (spec section 5.1).

Must land before Qdrant is restarted with an API key (H0), or the app,
pipeline and eval scripts all get 401s.
LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vector_store  # noqa: E402

DIRECT_CLIENT_FILES = (
    "evals/check_corpus.py",
    "evals/export_corpus.py",
    "evals/parity_check.py",
    "scripts/migrate_remove_sender.py",
    "scripts/restore_qdrant_from_neo4j.py",
)


class RecordingClient:
    calls = []

    def __init__(self, **kwargs):
        RecordingClient.calls.append(kwargs)


@pytest.fixture
def recorder(monkeypatch):
    RecordingClient.calls = []
    monkeypatch.setattr(vector_store, "QdrantClient", RecordingClient)
    return RecordingClient


ENV_KEY = "placeholder-from-env"    # dummy values, not credentials
EXPLICIT_KEY = "placeholder-explicit"


def make_store(**kw):
    return vector_store.VectorStore(host="h", port=1, collection="c", vector_size=4,
                                    embed_fn=lambda t: [0.0] * 4, **kw)


def test_vector_store_reads_the_key_from_the_environment(recorder, monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", ENV_KEY)
    make_store()
    assert recorder.calls[-1]["api_key"] == ENV_KEY


def test_explicit_key_wins_over_the_environment(recorder, monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", ENV_KEY)
    make_store(api_key=EXPLICIT_KEY)
    assert recorder.calls[-1]["api_key"] == EXPLICIT_KEY


def test_no_key_anywhere_means_none_not_empty_string(recorder, monkeypatch):
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    make_store()
    assert recorder.calls[-1]["api_key"] is None


def test_blank_env_key_means_none(recorder, monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", "")
    make_store()
    assert recorder.calls[-1]["api_key"] is None


def _client_calls(src: str) -> list[str]:
    calls = []
    for m in re.finditer(r"QdrantClient\(", src):
        depth, i = 1, m.end()
        while depth and i < len(src):
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        calls.append(src[m.start():i])
    return calls


@pytest.mark.parametrize("rel", DIRECT_CLIENT_FILES)
def test_direct_qdrant_clients_pass_the_env_key(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    calls = _client_calls(src)
    assert calls, f"{rel}: expected at least one QdrantClient(...) call"
    for call in calls:
        assert "api_key" in call, f"{rel}: QdrantClient call without api_key: {call!r}"
    assert "QDRANT_API_KEY" in src, f"{rel}: must read QDRANT_API_KEY"
