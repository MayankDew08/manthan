"""U8: latency benchmark helpers (spec section 12).

LOCKED: listed in tests/LOCK.sha256. Do not edit.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import common, latency_bench as L  # noqa: E402


class FakeClock:
    """Each call to fn advances the clock by the next duration (seconds)."""

    def __init__(self, durations):
        self.t = 0.0
        self.durations = list(durations)

    def __call__(self):
        return self.t

    def advance(self):
        self.t += self.durations.pop(0)


def test_constants_are_pre_registered():
    assert L.WARMUP == 1 and L.REPEATS == 5


def test_time_call_discards_warmup_and_returns_milliseconds():
    clock = FakeClock([9.0, 0.001, 0.002, 0.003, 0.004, 0.005])
    calls = []

    def fn():
        calls.append(1)
        clock.advance()

    samples = L.time_call(fn, clock=clock)
    assert len(calls) == 6
    assert samples == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_summarize_nearest_rank_percentiles():
    out = L.summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert out == {"n": 5, "p50": 3.0, "p95": 5.0}


def test_summarize_twenty_values():
    out = L.summarize([float(i) for i in range(1, 21)])
    assert out["p50"] == 10.0 and out["p95"] == 19.0


def test_summarize_empty_fails():
    with pytest.raises(ValueError):
        L.summarize([])


def test_component_summary_is_over_per_query_medians():
    per_query = {"q1": [1.0, 1.0, 9.0], "q2": [2.0, 2.0, 2.0], "q3": [3.0, 3.0, 100.0]}
    out = L.component_summary(per_query)
    assert out == {"n": 3, "p50": 2.0, "p95": 3.0}


def test_machine_info_shape():
    info = L.machine_info()
    assert {"platform", "processor", "cpu_count", "gpu"} <= set(info)


def test_components_are_named():
    assert L.COMPONENTS == ("cosine_qdrant", "graph_neo4j", "bm25", "rrf",
                            "rerank_cpu", "rerank_gpu")


@pytest.mark.artifacts
def test_artifact_latency_file_is_complete():
    path = common.data_dir(common.DEFAULT_CORPUS_VERSION, create=False) / "runs" / "latency_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    qids = {q["query_id"] for q in common.load_queries()}
    for name in ("cosine_qdrant", "graph_neo4j", "bm25", "rrf", "rerank_cpu"):
        comp = data["components"][name]
        assert set(comp["per_query"]) == qids
        assert all(len(v) == L.REPEATS for v in comp["per_query"].values())
        assert comp["summary"]["n"] == len(qids)
    assert data["fingerprint_before"] == data["fingerprint_after"]
    assert data["machine"]["cpu_count"] > 0
