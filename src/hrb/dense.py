"""Dense retrieval with a sentence-transformers bi-encoder; corpus embeddings cached on disk."""
import hashlib
from pathlib import Path

import numpy as np


class DenseIndex:
    # Some models expect an instruction prefix on queries; documents are encoded as-is.
    QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}

    def __init__(self, model_name: str, docs: list[str], cache_dir: Path = Path("cache"), batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        import torch

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        self.prefix = self.QUERY_PREFIX.get(model_name, "")
        key = hashlib.sha1((model_name + "\x00".join(docs)).encode()).hexdigest()[:16]
        path = cache_dir / f"{model_name.replace('/', '_')}-{key}.npy"
        if path.exists():
            self.emb = np.load(path)
        else:
            self.emb = self.model.encode(docs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, self.emb)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode([self.prefix + q for q in queries], normalize_embeddings=True, show_progress_bar=False)

    def scores(self, query_emb: np.ndarray) -> np.ndarray:
        return self.emb @ query_emb  # cosine similarity: embeddings are L2-normalised
