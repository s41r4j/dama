"""Small dependency-free BM25, used to mine hard negatives and to build realistic candidate pools."""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict

STOP = set("a an the and or but if then of to in on at for with by from as is are was were be been being it this that "
           "these those i you he she we they me my your our their his her its do does did have has had not no so "
           "what when where which who whom how why can could would should will just about".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        df = Counter(w for c in self.docs for w in c)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.postings = defaultdict(list)
        for i, c in enumerate(self.docs):
            for w, f in c.items():
                self.postings[w].append((i, f))

    def scores(self, query: str) -> list[float]:
        out = [0.0] * len(self.docs)
        for w in set(tokenize(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.postings[w]:
                norm = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg or 1))
                out[i] += idf * f * (self.k1 + 1) / norm
        return out

    def top(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        return sorted(range(len(s)), key=lambda i: -s[i])[:k]
