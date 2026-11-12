"""Combine rankings from different retrievers."""
import numpy as np


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def rrf(rankings: list[np.ndarray], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion (Cormack et al., 2009): sum of 1 / (k + rank). Needs no score calibration."""
    out: dict[int, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking, start=1):
            out[int(doc)] = out.get(int(doc), 0.0) + 1.0 / (k + r)
    return out


def linear(score_lists: list[np.ndarray], weights: list[float], depth: int = 1000) -> dict[int, float]:
    """Weighted sum of min-max normalised scores over each retriever's top `depth` candidates.

    BM25 scores are unbounded and cosine scores live in [-1, 1], so raw sums are meaningless;
    normalising within each query's candidate pool puts them on one scale.
    """
    out: dict[int, float] = {}
    for scores, w in zip(score_lists, weights):
        idx = top_k(scores, depth)
        s = scores[idx]
        lo, hi = s.min(), s.max()
        norm = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
        for d, v in zip(idx, norm):
            out[int(d)] = out.get(int(d), 0.0) + w * float(v)
    return out


def ranked(fused: dict[int, float], k: int) -> list[int]:
    return [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:k]]
