"""FeVec (Weerasinghe, Singh and Greenstadt, CLEF 2021)

A document pair is represented by the absolute difference of the two feature
vectors, standardized twice, and decided by an SGD classifier.
"""

from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import loguniform
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from common import AVMethod, calc_metrics, read_corpus
from methods.fevec.fevec_features import (
    get_nltk_pos_tag_based_ml_chunker,
    get_simple_chunker,
    get_transformer,
    pos_tag_chunk,
    preprocess_text,
    tokenize,
)


class FeVec(AVMethod):
    def __init__(
        self,
        name: str = "FeVec",
        resources_dir: str | None = None,
        seed: int | None = None,
        tokenizer: str = "casual",
    ) -> None:
        super().__init__(name=name)
        from pathlib import Path

        self.seed = seed
        self.tokenizer = tokenizer
        self.resources_dir = (
            resources_dir if resources_dir is not None else Path(__file__).parent / "resources"
        )
        self._tagger = None
        self.clf = None
        self.transformer = None
        self.scaler = None
        self.secondary_scaler = None

    @property
    def tagger(self):
        if self._tagger is None:
            from nltk.tag.perceptron import PerceptronTagger

            self._tagger = PerceptronTagger()
        return self._tagger

    def prepare_entry(self, text):
        # Documents that repeat one word over and over make the chunker hang, so
        # immediate repetitions are dropped.
        tokens, prev = [], ""
        for t in tokenize(text, self.tokenizer):
            if t != prev:
                tokens.append(t)

        tagger_output = self.tagger.tag(tokens)
        pos_chunks, subtree_expansions = pos_tag_chunk(
            tagger_output, get_nltk_pos_tag_based_ml_chunker(self.resources_dir)
        )
        return {
            "preprocessed": text,
            "pos_tags": [t[1] for t in tagger_output],
            "pos_tag_chunks": pos_chunks,
            "pos_tag_chunk_subtrees": subtree_expansions,
            "tokens": [preprocess_text(t) for t in tokens],
        }

    def preprocess_corpus(self, problems):
        return [(self.prepare_entry(t1), self.prepare_entry(t2)) for t1, _, t2, _ in problems]

    def _build_transformer(self):
        return get_transformer(resources_dir=self.resources_dir)

    def _fit_transformers(self, entries) -> Tuple[Any, Any, Any, np.ndarray]:
        # The transformers take the whole entry, not the plain text.
        docs_1 = [entry[0] for entry in entries]
        docs_2 = [entry[1] for entry in entries]

        transformer = self._build_transformer()
        X = np.asarray(transformer.fit_transform(docs_1 + docs_2).todense())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X1, X2 = X_scaled[: len(docs_1)], X_scaled[len(docs_1):]
        secondary_scaler = StandardScaler()
        secondary_scaler.fit(np.abs(X1 - X2))
        return transformer, scaler, secondary_scaler, secondary_scaler.transform(np.abs(X1 - X2))

    def _vectorize_entries(self, entries) -> np.ndarray:
        docs_1 = [entry[0] for entry in entries]
        docs_2 = [entry[1] for entry in entries]
        x1 = self.scaler.transform(np.asarray(self.transformer.transform(docs_1).todense()))
        x2 = self.scaler.transform(np.asarray(self.transformer.transform(docs_2).todense()))
        return self.secondary_scaler.transform(np.abs(x1 - x2))

    def _tune_classifier(self, XX, Y) -> float:
        search = RandomizedSearchCV(
            SGDClassifier(loss="log_loss", alpha=0.01, random_state=self.seed),
            param_distributions={"alpha": loguniform(1e-4, 1.0)},
            n_iter=15,
            random_state=self.seed,
        )
        search.fit(XX, Y)
        return search.best_params_["alpha"]

    def _train_classifier(self, XX, Y, alpha: float) -> SGDClassifier:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=self.seed)
        batch_size = min(1000, len(Y))
        for _ in range(3):
            for idxs in get_simple_chunker(list(range(len(Y))), batch_size):
                clf.partial_fit(XX[idxs, :], Y[idxs], classes=[0, 1])
        return clf

    def train(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        problems, labels, _ = read_corpus(corpus_path)
        entries = self.preprocess_corpus(problems)

        # Refit per corpus, so that nothing carries over.
        self.transformer, self.scaler, self.secondary_scaler, XX = self._fit_transformers(entries)

        Y = np.array(labels, dtype=int)
        XX_split, _, Y_split, _ = train_test_split(
            XX, Y, test_size=0.2, stratify=Y, random_state=self.seed
        )
        self.clf = self._train_classifier(XX, Y, self._tune_classifier(XX_split, Y_split))

        scores = self.clf.predict_proba(XX)[:, 1]
        self.train_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= 0.5).astype(int), labels
        )
        return self.train_metrics

    def eval(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        if self.clf is None:
            raise ValueError("call train() before eval()")
        problems, labels, _ = read_corpus(corpus_path)
        XX = self._vectorize_entries(self.preprocess_corpus(problems))

        scores = self.clf.predict_proba(XX)[:, 1]
        self.test_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= 0.5).astype(int), labels
        )
        return self.test_metrics
