"""U7 latency_expand: query-time cost of graph expansion (spec section 13.7).

Recorded only; no decision may read it (D17). Live pieces are marked ``live``
and run only with ``--live``, where a missing store is a failure, never a skip.
"""

import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import latency_bench  # noqa: E402


def _latency_expand():
    from evals import latency_expand
    return latency_expand


def _score_graph_expand():
    from evals import score_graph_expand
    return score_graph_expand


def _artifact():
    from evals import common
    return common.data_dir(common.DEFAULT_CORPUS_VERSION) / "runs" / "latency_expand_v1.json"


EXPECTED_COMPONENTS = {
    "br_walk",
    "bq_lookup_fetch",
    "rerank_30",
    "rerank_nq",
    "rerank_pool",
    "c1_extra",
}


class TestLatencyExpandUnit:
    def test_U7_B1_uses_latency_bench_and_discards_the_warmup(self, monkeypatch):
        """``measure`` delegates to time_call/summarize and never reports the warm-up."""
        le = _latency_expand()

        # Real time_call path: a deliberately slow first call must not reach p50/p95.
        calls = []

        class Clock:
            t = 0.0

            def __call__(self):
                return self.t

        clock = Clock()

        def fn():
            n = len(calls)
            calls.append(n)
            # Warm-up (call 0) takes one second; each measured call takes 1 ms.
            clock.t += 1.0 if n == 0 else 0.001

        out = le.measure(fn, clock=clock)

        assert len(calls) == latency_bench.WARMUP + latency_bench.REPEATS
        assert out["n"] == latency_bench.REPEATS
        assert out["p50"] == pytest.approx(1.0, rel=1e-9)
        assert out["p95"] == pytest.approx(1.0, rel=1e-9)

        # Delegation: the agreed defaults are passed straight through.
        seen = []

        def fake_time_call(fn, warmup, repeats, clock=None):
            seen.append((warmup, repeats))
            fn()
            return [1.0, 2.0, 3.0]

        monkeypatch.setattr(le.latency_bench, "time_call", fake_time_call)
        monkeypatch.setattr(le.latency_bench, "summarize",
                            lambda samples: {"n": len(samples),
                                             "p50": samples[0], "p95": samples[-1]})

        delegated = le.measure(lambda: None)

        assert seen == [(latency_bench.WARMUP, latency_bench.REPEATS)]
        assert delegated == {"n": 3, "p50": 1.0, "p95": 3.0}


class TestLatencyExpandArtifacts:
    @pytest.mark.artifacts
    def test_U7_B2_components_cover_both_sets_and_c1_is_zero(self):
        """B-r walk, B-q lookup+fetch, re-rank at 30/N_q/pool; C1 extra == 0 ms."""
        le = _latency_expand()
        artifact = _artifact()
        if not artifact.exists():
            pytest.fail(f"missing latency artifact: {artifact}")
        data = json.loads(artifact.read_text(encoding="utf-8"))

        assert set(le.COMPONENTS) == EXPECTED_COMPONENTS
        assert set(data["components"]) == EXPECTED_COMPONENTS

        for name in EXPECTED_COMPONENTS:
            component = data["components"][name]
            per_query = component["per_query"]
            summary = component["summary"]
            assert per_query, name
            assert summary["n"] == len(per_query), name
            assert isinstance(summary["p50"], (int, float)), name
            assert isinstance(summary["p95"], (int, float)), name
            assert all(qid.startswith(("q", "g")) for qid in per_query), name

        # Both query sets are timed for the graph work.
        assert any(q.startswith("q") for q in data["components"]["br_walk"]["per_query"])
        assert any(q.startswith("g") for q in data["components"]["br_walk"]["per_query"])
        assert any(q.startswith("q") for q in data["components"]["bq_lookup_fetch"]["per_query"])
        assert any(q.startswith("g") for q in data["components"]["bq_lookup_fetch"]["per_query"])

        # C1's query-time extra is index-time work, recorded as exactly 0 ms.
        c1 = data["components"]["c1_extra"]
        assert all(sample == 0.0 for samples in c1["per_query"].values() for sample in samples)
        assert c1["summary"]["p50"] == 0.0
        assert c1["summary"]["p95"] == 0.0


class TestLatencyExpandLive:
    @pytest.mark.live
    def test_U7_S1_listener_stopped_and_fingerprints_unchanged(self, monkeypatch):
        """The listener check comes before the first read; the store is unchanged."""
        le = _latency_expand()
        calls = []

        def record(name):
            calls.append(name)

        def boom():
            raise RuntimeError("stop before the benchmark body")

        monkeypatch.setattr(le.live_guard, "assert_listener_stopped",
                            lambda: record("listener"))
        monkeypatch.setattr(le.live_guard, "read_only_store",
                            lambda: (record("store"), boom())[1])

        with pytest.raises(RuntimeError, match="stop before"):
            le.run_benchmark()

        assert calls == ["listener", "store"], (
            "assert_listener_stopped must run before the first store access"
        )

        artifact = _artifact()
        if not artifact.exists():
            pytest.fail(f"missing latency artifact: {artifact}")
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert data["fingerprint_before"] == data["fingerprint_after"]


class TestLatencyExpandAntiCheat:
    def test_U7_A1_decision_function_has_no_latency_input(self):
        """D17: latency is reported, never fed into a branch decision."""
        sge = _score_graph_expand()
        decision_functions = ["decide_branch", "beats_original", "graph_beats"]
        for fn_name in decision_functions:
            fn = getattr(sge, fn_name)
            params = inspect.signature(fn).parameters
            for name in params:
                lowered = name.lower()
                assert "latency" not in lowered, (fn_name, name)
                assert "p50" not in lowered, (fn_name, name)
                assert "p95" not in lowered, (fn_name, name)
                assert not lowered.endswith("_ms"), (fn_name, name)
