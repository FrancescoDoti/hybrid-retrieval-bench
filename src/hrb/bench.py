"""Run the benchmark: python -m hrb.bench --data data --datasets scifact nfcorpus

Fusion weights and the RRF constant are chosen on each dataset's train/dev queries and
then frozen; every number in the report comes from the held-out test queries.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .data import load_beir
from .dense import DenseIndex
from .fusion import linear, ranked, rrf, top_k
from .metrics import mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k

TUNE_SPLIT = {"scifact": "train", "nfcorpus": "dev"}
DENSE_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def evaluate(rankings: dict[str, list[str]], qrels) -> dict[str, np.ndarray]:
    qids = list(qrels)
    return {
        "ndcg@10": np.array([ndcg_at_k(rankings[q], qrels[q], 10) for q in qids]),
        "recall@100": np.array([recall_at_k(rankings[q], qrels[q], 100) for q in qids]),
        "mrr@10": np.array([mrr_at_k(rankings[q], qrels[q], 10) for q in qids]),
    }


def score_matrices(ds, bm25, dense):
    qids = list(ds.qrels)
    texts = [ds.queries[q] for q in qids]
    out, lat = {}, {}
    t = time.perf_counter()
    out["bm25"] = np.stack([bm25.scores(x) for x in texts])
    lat["bm25"] = (time.perf_counter() - t) / len(texts) * 1000
    for name, idx in dense.items():
        t = time.perf_counter()
        qe = idx.encode_queries(texts)
        out[name] = qe @ idx.emb.T
        lat[name] = (time.perf_counter() - t) / len(texts) * 1000
    return qids, out, lat


def run_dataset(root: Path, name: str, rerank: bool) -> dict:
    tune = load_beir(root / name, TUNE_SPLIT[name])
    test = load_beir(root / name, "test")
    ids = test.doc_ids
    print(f"[{name}] {len(ids)} docs, {len(tune.qrels)} tuning queries, {len(test.qrels)} test queries", flush=True)
    t = time.perf_counter()
    bm25 = BM25(test.doc_texts)
    index_s = {"bm25": time.perf_counter() - t}
    dense = {}
    for m in DENSE_MODELS:
        t = time.perf_counter()
        dense[m.split("/")[-1]] = DenseIndex(m, test.doc_texts)
        index_s[m.split("/")[-1]] = time.perf_counter() - t
    best_dense = "bge-small-en-v1.5"

    def runs_for(ds, alpha, rrf_k):
        qids, S, lat = score_matrices(ds, bm25, dense)
        runs = {k: {q: [ids[i] for i in top_k(S[k][j], 100)] for j, q in enumerate(qids)} for k in S}
        runs["rrf(bm25+bge)"] = {q: [ids[i] for i in ranked(rrf([top_k(S["bm25"][j], 1000), top_k(S[best_dense][j], 1000)], rrf_k), 100)] for j, q in enumerate(qids)}
        runs["linear(bm25+bge)"] = {q: [ids[i] for i in ranked(linear([S["bm25"][j], S[best_dense][j]], [alpha, 1 - alpha]), 100)] for j, q in enumerate(qids)}
        return qids, S, lat, runs

    # --- tune on train/dev queries only ---
    qids, S, _, _ = runs_for(tune, 0.5, 60)
    grid = {}
    for alpha in np.round(np.arange(0, 1.01, 0.1), 2):
        r = {q: [ids[i] for i in ranked(linear([S["bm25"][j], S[best_dense][j]], [alpha, 1 - alpha]), 10)] for j, q in enumerate(qids)}
        grid[float(alpha)] = float(evaluate(r, tune.qrels)["ndcg@10"].mean())
    alpha = max(grid, key=grid.get)
    rrf_grid = {}
    for k in (10, 30, 60, 100):
        r = {q: [ids[i] for i in ranked(rrf([top_k(S["bm25"][j], 1000), top_k(S[best_dense][j], 1000)], k), 10)] for j, q in enumerate(qids)}
        rrf_grid[k] = float(evaluate(r, tune.qrels)["ndcg@10"].mean())
    rrf_k = max(rrf_grid, key=rrf_grid.get)
    print(f"[{name}] tuned alpha(BM25 weight)={alpha}, rrf k={rrf_k}", flush=True)

    # --- evaluate on test ---
    qids, S, lat, runs = runs_for(test, alpha, rrf_k)
    if rerank:
        from .rerank import Reranker
        rr = Reranker(RERANKER)
        base = runs["linear(bm25+bge)"]
        id2i = {d: i for i, d in enumerate(ids)}
        t = time.perf_counter()
        runs["linear + cross-encoder rerank@100"] = {
            q: [ids[i] for i in rr.rerank(test.queries[q], [id2i[d] for d in base[q][:100]], test.doc_texts)] for q in qids
        }
        lat["linear + cross-encoder rerank@100"] = (time.perf_counter() - t) / len(qids) * 1000
    per_query = {k: evaluate(v, test.qrels) for k, v in runs.items()}
    systems = {}
    for k, m in per_query.items():
        d, lo, hi = paired_bootstrap(per_query["bm25"]["ndcg@10"], m["ndcg@10"])
        systems[k] = {**{mk: float(mv.mean()) for mk, mv in m.items()}, "ndcg@10_delta_vs_bm25": [d, lo, hi], "ms_per_query": lat.get(k)}
    return {"docs": len(ids), "tune_split": TUNE_SPLIT[name], "tune_queries": len(tune.qrels), "test_queries": len(test.qrels),
            "alpha_grid": grid, "alpha": alpha, "rrf_grid": rrf_grid, "rrf_k": rrf_k, "index_seconds": index_s, "systems": systems}


def report(res: dict) -> str:
    lines = ["# Results", "", "All metrics on held-out **test** queries. Fusion settings were tuned on train/dev queries. Δ is the paired bootstrap 95% CI of the nDCG@10 gain over BM25.", ""]
    for name, r in res.items():
        lines += [f"## {name} ({r['docs']} docs, {r['test_queries']} test queries; tuned on {r['tune_queries']} {r['tune_split']} queries)", "",
                  f"Tuned BM25 weight α = {r['alpha']}, RRF k = {r['rrf_k']}.", "",
                  "| System | nDCG@10 | Recall@100 | MRR@10 | Δ nDCG@10 vs BM25 [95% CI] | ms/query |", "|---|---|---|---|---|---|"]
        for s, m in r["systems"].items():
            d, lo, hi = m["ndcg@10_delta_vs_bm25"]
            ms = f"{m['ms_per_query']:.1f}" if m["ms_per_query"] is not None else "fusion only"
            lines.append(f"| {s} | {m['ndcg@10']:.3f} | {m['recall@100']:.3f} | {m['mrr@10']:.3f} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {ms} |")
        lines.append("")
    return "\n".join(lines)


def plot(res: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(res), figsize=(6 * len(res), 4), squeeze=False)
    for ax, (name, r) in zip(axes[0], res.items()):
        names = list(r["systems"])
        vals = [r["systems"][s]["ndcg@10"] for s in names]
        err = np.array([[r["systems"][s]["ndcg@10_delta_vs_bm25"][0] - r["systems"][s]["ndcg@10_delta_vs_bm25"][1] for s in names],
                        [r["systems"][s]["ndcg@10_delta_vs_bm25"][2] - r["systems"][s]["ndcg@10_delta_vs_bm25"][0] for s in names]])
        ax.barh(names, vals, xerr=err, color="#2563eb", capsize=3)
        ax.invert_yaxis()
        ax.set_xlabel("nDCG@10 (test); whiskers = bootstrap CI of gain vs BM25")
        ax.set_title(name)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(out / "ndcg.png", dpi=140)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    res = {d: run_dataset(Path(a.data), d, not a.no_rerank) for d in a.datasets}
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "report.md").write_text(report(res))
    plot(res, out)
    print(report(res))


if __name__ == "__main__":
    main()
