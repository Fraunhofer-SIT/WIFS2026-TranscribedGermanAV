import random
from typing import Any, Dict

import numpy as np
import torch

from common import AVMethod, calc_metrics, read_corpus
from methods.css.preprocessing import build_scaler, get_model_input
from methods.css.siamese_network import SiameseNetwork


class CSS(AVMethod):
    """Siamese head on frozen encoder embeddings, compared by cosine similarity.

    The reported runs use gbert-base, which won an encoder comparison against
    xlm-roberta-base; the pipeline default used to be the latter. The style
    features are optional and concatenated to the embedding.
    """

    def __init__(
        self,
        name: str = "CSS",
        seed: int | None = None,
        use_style: bool = False,
        model_name: str = "deepset/gbert-base",
        spacy_model: str = "de_core_news_sm",
        lang: str = "de",
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(name=name)
        self.seed = seed
        self.use_style = use_style
        self.model_name = model_name
        self.spacy_model = spacy_model
        self.lang = lang
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.scaler = None
        self._device = None

    def _dev(self):
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def _pairs(self, corpus_path):
        problems, labels, _ = read_corpus(corpus_path)
        texts1 = [p[0] for p in problems]
        texts2 = [p[2] for p in problems]
        return texts1, texts2, [int(round(float(v))) for v in labels]

    def _encode(self, texts1, texts2, device):
        def stack(texts):
            return torch.stack(
                [
                    get_model_input(
                        t,
                        self.model_name,
                        scaler=self.scaler,
                        spacy_model=self.spacy_model,
                        lang=self.lang,
                        device=device,
                    )
                    for t in texts
                ]
            ).to(device).float()

        return stack(texts1), stack(texts2)

    def train(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        device = self._dev()
        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)

        texts1, texts2, y = self._pairs(corpus_path)
        # Refit per corpus, so that no scaler or head carries over.
        self.scaler = build_scaler(texts1 + texts2, self.spacy_model, self.lang) if self.use_style else None
        X1, X2 = self._encode(texts1, texts2, device)
        Y = torch.tensor(y, dtype=torch.float32, device=device)

        self.model = SiameseNetwork(embedding_dim=X1.shape[1]).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        idx = np.arange(X1.shape[0])
        rng = np.random.RandomState(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for b in range(0, len(idx), self.batch_size):
                batch = idx[b:b + self.batch_size]
                if len(batch) < 2:  # BatchNorm1d needs at least two samples
                    continue
                batch = torch.as_tensor(batch, dtype=torch.long, device=device)
                optimizer.zero_grad()
                loss = criterion(self.model(X1[batch], X2[batch]), Y[batch])
                loss.backward()
                optimizer.step()

        scores = self._scores(X1, X2)
        self.train_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= 0.5).astype(int), y
        )
        return self.train_metrics

    def eval(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("call train() before eval()")
        texts1, texts2, y = self._pairs(corpus_path)
        X1, X2 = self._encode(texts1, texts2, self._dev())
        scores = self._scores(X1, X2)
        self.test_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= 0.5).astype(int), y
        )
        return self.test_metrics

    def _scores(self, X1, X2):
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(X1, X2)).cpu().numpy()
