"""Pure ranking metrics for the golden-dataset evaluation.

Inputs are *grades in ranked order* (index 0 = rank 1) on the 0-3 scale from
docs/golden-dataset-evaluation-plan.md. Grades 2 and 3 count as relevant;
grade 1 is related context only. Unjudged documents are grade 0.
"""

import math
from typing import Optional, Sequence

RELEVANT_MIN_GRADE = 2
CANDIDATE_DEPTH = 30
TOP_K = 5
PRODUCT_THRESHOLD = 0.25

FAILURE_CLASSES = (
    "success",
    "no_corpus_answer",
    "not_vectorized",        # not assessable from a Qdrant-sourced corpus
    "cosine_retrieval_miss",
    "ranking_miss",
    "threshold_miss",
    "long_document_miss",
)


def is_relevant(grade: int) -> bool:
    return grade >= RELEVANT_MIN_GRADE


def first_rank(ranked: Sequence[int], min_grade: int = RELEVANT_MIN_GRADE) -> Optional[int]:
    """1-based rank of the first document with grade >= min_grade, else None."""
    for i, g in enumerate(ranked, start=1):
        if g >= min_grade:
            return i
    return None


def dcg_at_k(ranked: Sequence[int], k: int) -> float:
    """Graded DCG with gain 2^grade - 1 and log2(rank + 1) discount."""
    return sum((2 ** g - 1) / math.log2(i + 1)
               for i, g in enumerate(ranked[:k], start=1))


def ndcg_at_k(ranked: Sequence[int], all_grades: Sequence[int], k: int) -> Optional[float]:
    """DCG@k over the ideal ordering of every judged grade for the query.

    Returns None when nothing relevant-or-related was judged (ideal DCG is 0).
    """
    ideal = dcg_at_k(sorted(all_grades, reverse=True), k)
    if ideal == 0:
        return None
    return dcg_at_k(ranked, k) / ideal


def success_at_k(ranked: Sequence[int], k: int) -> bool:
    return any(is_relevant(g) for g in ranked[:k])


def mrr_at_k(ranked: Sequence[int], k: int) -> float:
    rank = first_rank(ranked[:k])
    return 1.0 / rank if rank else 0.0


def recall_at_k(ranked: Sequence[int], total_relevant: int, k: int) -> Optional[float]:
    if total_relevant == 0:
        return None
    return sum(1 for g in ranked[:k] if is_relevant(g)) / total_relevant


def roc_auc(scores: Sequence[float], positives: Sequence[bool]) -> Optional[float]:
    """AUC of ``scores`` as a classifier (Mann-Whitney U, ties get half credit)."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def classify_failure(ranked: Sequence[dict], *, k: int = TOP_K,
                     depth: int = CANDIDATE_DEPTH,
                     threshold: float = PRODUCT_THRESHOLD) -> str:
    """Diagnose one query. ``ranked`` rows: {grade, score, truncated} in rank order.

    Precedence: no answer -> success -> threshold -> long-document -> depth.
    "not_vectorized" needs a corpus wider than Qdrant and is never assigned here.
    """
    relevant = [(i, r) for i, r in enumerate(ranked, start=1) if is_relevant(r["grade"])]
    if not relevant:
        return "no_corpus_answer"
    in_top_k = [r for i, r in relevant if i <= k]
    if in_top_k:
        return "success" if any(r["score"] >= threshold for r in in_top_k) else "threshold_miss"
    if any(r.get("truncated") for _, r in relevant):
        return "long_document_miss"
    best = relevant[0][0]
    return "cosine_retrieval_miss" if best > depth else "ranking_miss"


def threshold_sweep(per_query: Sequence[Sequence[dict]], thresholds: Sequence[float],
                    k: int = TOP_K) -> list[dict]:
    """Pooled precision/recall and Success@k if results below a score were dropped.

    ``per_query`` is a list of ranked row lists ({grade, score}). Every document
    scoring >= threshold is "returned"; precision and recall are pooled over all
    queries; Success@k looks at the first k *surviving* results of each query.
    """
    total_relevant = sum(1 for q in per_query for r in q if is_relevant(r["grade"]))
    out = []
    for t in thresholds:
        returned = tp = successes = 0
        for q in per_query:
            kept = [r for r in q if r["score"] >= t]
            returned += len(kept)
            hits = sum(1 for r in kept if is_relevant(r["grade"]))
            tp += hits
            if any(is_relevant(r["grade"]) for r in kept[:k]):
                successes += 1
        out.append({
            "threshold": round(t, 4),
            "returned": returned,
            "precision": tp / returned if returned else None,
            "recall": tp / total_relevant if total_relevant else None,
            "success_at_k": successes / len(per_query) if per_query else None,
        })
    return out


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
