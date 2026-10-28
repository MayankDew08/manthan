"""Tiny BM25 (Okapi) used only to widen the annotation pool beyond cosine.

Implemented in-repo so the eval needs no extra dependency.
"""

import math
import re
from collections import Counter

_STOP = frozenset("""a an the is are was were be been of to in on at for and or but if
that this these those it its with as by from about into than then so do does did can
could would should will what which who whom how when where why you your i we they he
she them his her our their there here not no yes""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        # Lucene-style IDF: always positive, so common terms never subtract.
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for counts, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if not tf:
                    continue
                norm = 1 - self.b + self.b * (length / self.avg_len if self.avg_len else 0)
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            out.append(s)
        return out

    def top(self, query: str, n: int) -> list[tuple[int, float]]:
        """Indices and scores of the best ``n`` documents with a positive score."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))  # stable, deterministic ties
        return scored[:n]
