import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from hrb.bm25 import BM25, tokenize
from hrb.fusion import linear, ranked, rrf, top_k
from hrb.metrics import mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k

DOCS = ["the cat sat on the mat", "dogs chase cats", "stock markets fell sharply", "a cat and a dog"]


def test_tokenize_drops_stopwords():
    assert tokenize("The Cat, and THE dog!") == ["cat", "dog"]


def test_bm25_ranks_matching_doc_first_and_handles_unknown_terms():
    bm = BM25(DOCS)
    assert int(np.argmax(bm.scores("stock markets"))) == 2
    assert bm.scores("zebra").sum() == 0
    s = bm.scores("cat")
    assert s[0] > 0 and s[2] == 0


def test_bm25_idf_matches_formula():
    bm = BM25(DOCS)
    n, df = 4, 2  # "cat" appears in docs 0 and 3 ("cats" is a different token)
    assert bm.idf[bm.vocab["cat"]] == pytest.approx(math.log(1 + (n - df + 0.5) / (df + 0.5)))


def test_ndcg_perfect_and_worst():
    rels = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rels) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], rels) == 0.0
    assert ndcg_at_k(["b", "a"], rels) < 1.0


def test_recall_and_mrr():
    rels = {"a": 1, "b": 1}
    assert recall_at_k(["a", "x"], rels, 2) == 0.5
    assert mrr_at_k(["x", "y", "b"], rels) == pytest.approx(1 / 3)


def test_rrf_rewards_agreement():
    fused = rrf([np.array([1, 2, 3]), np.array([2, 1, 3])])
    assert ranked(fused, 1)[0] in (1, 2)
    assert fused[3] < fused[1]


def test_linear_fusion_normalises_scales():
    bm = np.array([100.0, 0.0, 50.0])
    cos = np.array([0.1, 0.9, 0.5])
    assert ranked(linear([bm, cos], [1.0, 0.0]), 1) == [0]
    assert ranked(linear([bm, cos], [0.0, 1.0]), 1) == [1]
    assert list(top_k(cos, 2)) == [1, 2]


def test_paired_bootstrap_ci_contains_mean():
    a = np.zeros(200)
    b = np.full(200, 0.1)
    m, lo, hi = paired_bootstrap(a, b)
    assert lo <= m <= hi and m == pytest.approx(0.1)


@pytest.fixture
def client(tmp_path, monkeypatch):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps({"_id": str(i), "title": "", "text": t}) for i, t in enumerate(DOCS)))
    monkeypatch.setenv("HRB_CORPUS", str(corpus))
    monkeypatch.delenv("HRB_DENSE_MODEL", raising=False)
    from fastapi.testclient import TestClient
    from hrb import api
    api.engine.cache_clear()
    return TestClient(api.app), api


def test_search_endpoint(client):
    c, _ = client
    r = c.post("/search", json={"query": "stock markets", "k": 2}).json()
    assert r["results"][0]["id"] == "2"
    assert c.post("/search", json={"query": "", "k": 2}).status_code == 422


def test_answer_without_key_is_503(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert c.post("/answer", json={"query": "cats"}).status_code == 503


def test_answer_flags_invented_citations(client, monkeypatch):
    c, api = client
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Cats sit on mats [0]. Also [999].")])))
    monkeypatch.setattr(api, "anthropic_client", lambda: fake)
    r = c.post("/answer", json={"query": "cat mat"}).json()
    assert r["citations"] == ["0"] and r["unsupported_citations"] == ["999"]
