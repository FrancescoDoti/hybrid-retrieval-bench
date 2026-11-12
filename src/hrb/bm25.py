"""Okapi BM25 over a scipy sparse term-document matrix: scoring a query is one sparse row product."""
import re
from collections import Counter

import numpy as np
from scipy import sparse

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were which with this these those not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 0.9, b: float = 0.4):
        # k1=0.9, b=0.4 are the Anserini defaults used by the BEIR BM25 baseline.
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(docs))
        for j, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            lengths[j] = sum(counts.values())
            for term, c in counts.items():
                rows.append(self.vocab.setdefault(term, len(self.vocab)))
                cols.append(j)
                vals.append(c)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.vocab), len(docs)), dtype=np.float64)
        n = len(docs)
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * lengths / lengths.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.col]) * self.idf[tf.row]
        self.weights = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query: str) -> np.ndarray:
        idx = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not idx:
            return np.zeros(self.weights.shape[1])
        q = np.bincount(idx, minlength=len(self.vocab))  # repeated query terms count again
        return np.asarray(self.weights.T @ q).ravel()
