# Results

All metrics on held-out **test** queries. Fusion settings were tuned on train/dev queries. Δ is the paired bootstrap 95% CI of the nDCG@10 gain over BM25.

## scifact (5183 docs, 300 test queries; tuned on 809 train queries)

Tuned BM25 weight α = 0.3, RRF k = 10.

| System | nDCG@10 | Recall@100 | MRR@10 | Δ nDCG@10 vs BM25 [95% CI] | ms/query |
|---|---|---|---|---|---|
| bm25 | 0.662 | 0.886 | 0.632 | +0.000 [+0.000, +0.000] | 0.4 |
| all-MiniLM-L6-v2 | 0.648 | 0.928 | 0.606 | -0.014 [-0.054, +0.025] | 0.9 |
| bge-small-en-v1.5 | 0.708 | 0.943 | 0.676 | +0.046 [+0.009, +0.084] | 0.9 |
| rrf(bm25+bge) | 0.711 | 0.965 | 0.670 | +0.049 [+0.028, +0.070] | fusion only |
| linear(bm25+bge) | 0.725 | 0.965 | 0.689 | +0.062 [+0.034, +0.092] | fusion only |
| linear + cross-encoder rerank@100 | 0.688 | 0.965 | 0.656 | +0.025 [-0.008, +0.059] | 449.0 |

## nfcorpus (3633 docs, 323 test queries; tuned on 324 dev queries)

Tuned BM25 weight α = 0.2, RRF k = 10.

| System | nDCG@10 | Recall@100 | MRR@10 | Δ nDCG@10 vs BM25 [95% CI] | ms/query |
|---|---|---|---|---|---|
| bm25 | 0.305 | 0.243 | 0.508 | +0.000 [+0.000, +0.000] | 0.2 |
| all-MiniLM-L6-v2 | 0.318 | 0.315 | 0.504 | +0.013 [-0.011, +0.035] | 1.9 |
| bge-small-en-v1.5 | 0.341 | 0.309 | 0.533 | +0.036 [+0.014, +0.058] | 0.6 |
| rrf(bm25+bge) | 0.351 | 0.312 | 0.568 | +0.046 [+0.034, +0.059] | fusion only |
| linear(bm25+bge) | 0.354 | 0.318 | 0.550 | +0.049 [+0.030, +0.068] | fusion only |
| linear + cross-encoder rerank@100 | 0.352 | 0.318 | 0.570 | +0.047 [+0.030, +0.063] | 507.4 |
