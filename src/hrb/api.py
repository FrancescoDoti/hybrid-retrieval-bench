"""HTTP service: hybrid search, plus an optional grounded answer endpoint.

HRB_CORPUS=data/scifact/corpus.jsonl [HRB_DENSE_MODEL=BAAI/bge-small-en-v1.5] uvicorn hrb.api:app
/answer needs ANTHROPIC_API_KEY; without it the endpoint returns 503 rather than guessing.
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .bm25 import BM25
from .fusion import linear, ranked

app = FastAPI(title="hybrid retrieval")
ANSWER_MODEL = os.environ.get("HRB_ANSWER_MODEL", "claude-sonnet-5-5")


class Engine:
    def __init__(self, ids: list[str], texts: list[str], dense_model: str | None, alpha: float):
        self.ids, self.texts, self.alpha = ids, texts, alpha
        self.bm25 = BM25(texts)
        self.dense = None
        if dense_model:
            from .dense import DenseIndex
            self.dense = DenseIndex(dense_model, texts)

    def search(self, query: str, k: int) -> list[dict]:
        lists, weights = [self.bm25.scores(query)], [1.0]
        if self.dense is not None:
            lists.append(self.dense.scores(self.dense.encode_queries([query])[0]))
            weights = [self.alpha, 1 - self.alpha]
        fused = linear(lists, weights)
        return [{"id": self.ids[i], "score": round(fused[i], 4), "text": self.texts[i]} for i in ranked(fused, k)]


@lru_cache(maxsize=1)
def engine() -> Engine:
    ids, texts = [], []
    with open(Path(os.environ["HRB_CORPUS"])) as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["_id"])
            texts.append(f"{d.get('title', '')}. {d['text']}".strip(". "))
    # alpha=0.3 is the BM25 weight tuned on SciFact train queries (see results/).
    return Engine(ids, texts, os.environ.get("HRB_DENSE_MODEL"), float(os.environ.get("HRB_ALPHA", "0.3")))


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(10, ge=1, le=100)


@app.post("/search")
def search(req: SearchRequest):
    return {"results": engine().search(req.query, req.k)}


def build_prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{p['id']}] {p['text'][:1500]}" for p in passages)
    return (
        "Answer the question using only the passages below. Cite every claim with the passage id in "
        "square brackets, e.g. [12345]. If the passages do not contain the answer, say so plainly.\n\n"
        f"Passages:\n{ctx}\n\nQuestion: {question}"
    )


def cited_ids(answer: str, allowed: set[str]) -> tuple[list[str], list[str]]:
    """Split the ids the model cited into ones we retrieved and ones it invented."""
    found = list(dict.fromkeys(re.findall(r"\[([^\[\]\s]+)\]", answer)))
    return [c for c in found if c in allowed], [c for c in found if c not in allowed]


def anthropic_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set; /search still works")
    import anthropic
    return anthropic.Anthropic()


@app.post("/answer")
def answer(req: SearchRequest):
    passages = engine().search(req.query, min(req.k, 8))
    msg = anthropic_client().messages.create(
        model=ANSWER_MODEL, max_tokens=800, messages=[{"role": "user", "content": build_prompt(req.query, passages)}]
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    valid, invented = cited_ids(text, {p["id"] for p in passages})
    return {"answer": text, "citations": valid, "unsupported_citations": invented, "passages": [p["id"] for p in passages]}
