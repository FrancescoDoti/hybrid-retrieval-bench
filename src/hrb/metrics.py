"""IR metrics matching trec_eval conventions (linear gain nDCG, as BEIR reports)."""
import math

import numpy as np


def ndcg_at_k(ranking: list[str], rels: dict[str, int], k: int = 10) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranking: list[str], rels: dict[str, int], k: int = 100) -> float:
    return len(set(ranking[:k]) & rels.keys()) / len(rels) if rels else 0.0


def mrr_at_k(ranking: list[str], rels: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranking[:k]):
        if d in rels:
            return 1.0 / (i + 1)
    return 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (b - a) over queries with a 95% bootstrap CI; resamples queries, keeping pairs together."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(b) - np.asarray(a)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
