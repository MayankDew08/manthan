"""Statistics for the retrieval-ladder experiment (spec section 11).

Every metric value comes from :mod:`evals.metrics`; this module only does the
bookkeeping the experiment needs: which queries are answerable, per-query
scores, win/tie/loss counts, a paired bootstrap interval over the per-query
deltas, unjudged-row accounting and the abstention split.

The only randomness is ``random.Random(seed)`` inside :func:`paired_bootstrap`.
"""

import random
import statistics
from typing import Dict, Iterable, List, Mapping, Sequence

from evals import metrics as M

TIE_EPS = 0.01
N_RESAMPLES = 10000
SEED = 42


def _grade(qrels: Mapping[str, Mapping[str, Mapping[str, int]]], query_id: str,
           candidate_id: str) -> int:
    """Grade of a document; unjudged documents count as grade 0."""
    entry = qrels.get(query_id, {}).get(candidate_id)
    return int(entry["relevance"]) if entry else 0


def _ranked(run_rows: Iterable[Mapping[str, object]], query_id: str) -> List[Mapping[str, object]]:
    """Rows of one query ordered by ``rank`` (file order is not trusted)."""
    rows = [r for r in run_rows if r["query_id"] == query_id]
    return sorted(rows, key=lambda r: r["rank"])


def answerable(qrels: Mapping[str, Mapping[str, Mapping[str, int]]],
               query_ids: Sequence[str]) -> List[str]:
    """Sorted subset of ``query_ids`` with at least one judged grade >= 2."""
    out = []
    for qid in query_ids:
        if any(int(e["relevance"]) >= M.RELEVANT_MIN_GRADE
               for e in qrels.get(qid, {}).values()):
            out.append(qid)
    return sorted(out)


def per_query(run_rows: Sequence[Mapping[str, object]],
              qrels: Mapping[str, Mapping[str, Mapping[str, int]]],
              query_ids: Sequence[str],
              k: int = M.TOP_K, depth: int = M.CANDIDATE_DEPTH
              ) -> Dict[str, Dict[str, object]]:
    """nDCG@k, Success@k, MRR@k and Success@depth for every requested query."""
    out: Dict[str, Dict[str, object]] = {}
    for qid in query_ids:
        ranked = _ranked(run_rows, qid)
        grades = [_grade(qrels, qid, r["candidate_id"]) for r in ranked]
        judged = [int(e["relevance"]) for e in qrels.get(qid, {}).values()]
        ndcg = M.ndcg_at_k(grades, judged, k)
        out[qid] = {
            # metrics yields None when nothing relevant-or-related was judged.
            "ndcg": 0.0 if ndcg is None else float(ndcg),
            "success": M.success_at_k(grades, k),
            "mrr": M.mrr_at_k(grades, k),
            "candidate_success": M.success_at_k(grades, depth),
        }
    return out


def win_tie_loss(base: Mapping[str, float], new: Mapping[str, float]) -> Dict[str, object]:
    """Per-query win/tie/loss using ``TIE_EPS`` as the indifference band."""
    if set(base) != set(new):
        raise ValueError("win/tie/loss needs the same query set in both runs")
    wins, losses = [], []
    ties = 0
    for qid in sorted(base):
        delta = new[qid] - base[qid]
        if delta >= TIE_EPS:
            wins.append(qid)
        elif -delta >= TIE_EPS:
            losses.append(qid)
        else:
            ties += 1
    return {
        "wins": len(wins),
        "ties": ties,
        "losses": len(losses),
        "improved": sorted(wins),
        "regressed": sorted(losses),
    }


def paired_bootstrap(base: Mapping[str, float], new: Mapping[str, float],
                     n_resamples: int = N_RESAMPLES, seed: int = SEED
                     ) -> Dict[str, object]:
    """Percentile bootstrap over the per-query deltas (new - base).

    The algorithm is pre-registered: deltas in sorted-query-id order, a single
    ``random.Random(seed)``, ``n_resamples`` resamples of ``n`` indices drawn one
    at a time with ``rng.randrange(n)``, then the 2.5% and 97.5% percentiles of
    the sorted resample means.
    """
    deltas = [new[qid] - base[qid] for qid in sorted(base)]
    n = len(deltas)
    mean_delta = sum(deltas) / n if n else 0.0
    rng = random.Random(seed)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_resamples))
    return {
        "mean_delta": mean_delta,
        "ci_low": means[int(0.025 * n_resamples)],
        "ci_high": means[int(0.975 * n_resamples) - 1],
        "n": n,
    }


def verdict(ci_low: float, ci_high: float, n: int) -> str:
    """Read the interval: a bound of exactly zero is not a detection."""
    if ci_low > 0:
        return "improved"
    if ci_high < 0:
        return "regressed"
    return f"no detectable difference at n={n}"


def unjudged_at(run_rows: Sequence[Mapping[str, object]],
                qrels: Mapping[str, Mapping[str, Mapping[str, int]]],
                query_ids: Sequence[str], depth: int) -> int:
    """How many of the top-``depth`` rows per query carry no qrels entry."""
    judged = set()
    for qid in query_ids:
        judged.update((qid, cid) for cid in qrels.get(qid, {}))
    return sum(1 for qid in query_ids for r in _ranked(run_rows, qid)[:depth]
               if (qid, r["candidate_id"]) not in judged)


def _top1(run_rows: Sequence[Mapping[str, object]], query_ids: Sequence[str]
          ) -> Dict[str, float]:
    out = {}
    for qid in query_ids:
        ranked = _ranked(run_rows, qid)
        if ranked:
            out[qid] = float(ranked[0]["score"])
    return out


def abstention(run_rows: Sequence[Mapping[str, object]],
               no_answer_ids: Sequence[str],
               answerable_ids: Sequence[str]) -> Dict[str, object]:
    """Top-1 scores and their medians, split by answerability."""
    no_answer_top1 = _top1(run_rows, no_answer_ids)
    answerable_top1 = _top1(run_rows, answerable_ids)
    return {
        "no_answer_top1": no_answer_top1,
        "answerable_top1": answerable_top1,
        "median_no_answer": statistics.median(no_answer_top1.values())
        if no_answer_top1 else None,
        "median_answerable": statistics.median(answerable_top1.values())
        if answerable_top1 else None,
    }