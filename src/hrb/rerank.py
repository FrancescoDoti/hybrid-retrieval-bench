"""Second-stage reranking with a cross-encoder that reads query and document together."""


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query: str, candidates: list[int], docs: list[str]) -> list[int]:
        scores = self.model.predict([(query, docs[i]) for i in candidates], batch_size=64, show_progress_bar=False)
        return [c for _, c in sorted(zip(scores, candidates), key=lambda t: -t[0])]
