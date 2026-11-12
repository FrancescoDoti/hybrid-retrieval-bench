"""Load a BEIR-format dataset: corpus.jsonl, queries.jsonl, qrels/<split>.tsv."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dataset:
    doc_ids: list[str]
    doc_texts: list[str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]   # query_id -> {doc_id: graded relevance}

    def split(self, qrels_split: dict[str, dict[str, int]]) -> "Dataset":
        q = {qid: self.queries[qid] for qid in qrels_split if qid in self.queries}
        return Dataset(self.doc_ids, self.doc_texts, q, {k: v for k, v in qrels_split.items() if k in q})


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rel = int(row["score"])
            if rel > 0:
                out.setdefault(row["query-id"], {})[row["corpus-id"]] = rel
    return out


def load_beir(root: Path, split: str) -> Dataset:
    ids, texts = [], []
    with open(root / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["_id"])
            texts.append(f"{d.get('title', '')}. {d['text']}".strip(". "))
    queries = {}
    with open(root / "queries.jsonl") as f:
        for line in f:
            d = json.loads(line)
            queries[d["_id"]] = d["text"]
    full = Dataset(ids, texts, queries, {})
    return full.split(load_qrels(root / "qrels" / f"{split}.tsv"))
