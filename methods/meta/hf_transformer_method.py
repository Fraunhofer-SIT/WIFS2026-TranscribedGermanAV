from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from methods.meta.score_method import ScoreMethod


class HFTransformer(ScoreMethod):
    """Scores a pair by the cosine similarity of two frozen encoder embeddings.

    Texts longer than the encoder's max_seq_length are truncated by the encoder.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        device: str | None = None,
    ):
        super().__init__(name)
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def encode(self, texts):
        return self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def get_similarity_score(self, doc1: str, doc2: str) -> float:
        emb1, emb2 = self.encode(doc1), self.encode(doc2)
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        return cos_sim(emb1, emb2).item()
