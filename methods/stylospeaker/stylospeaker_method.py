"""StyloSpeaker (Aggazzotti and Smith 2025). Interpretable stylometry for
verifying speakers in transcripts: features on five linguistic levels per
document, joined per pair by the absolute difference, decided by logistic
regression.

No stochastic component: TF-IDF, StandardScaler and lbfgs are deterministic, and
the pairs come from the corpus unchanged rather than being sampled. A single run
is exact.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer

from common import AVMethod, calc_metrics, read_corpus
from methods.stylospeaker.stylometric_analysis import (
    build_doc_feature_matrix,
    build_pairwise_matrix,
    make_doc_feature_pipeline,
    preprocess_texts,
)

_RESOURCES = str(Path(__file__).parent / "resources")


class StyloSpeaker(AVMethod):
    def __init__(
        self,
        name: str = "StyloSpeaker",
        lang: str = "de",
        seed: int | None = None,
        feat_combos: str = "diffabs",
        spacy_model: str | None = None,
        nlp_backend: str = "spacy",
        char_ngram_range=(3, 6),
        tok_ngram_range=(1, 3),
        pos_ngram_range=(1, 3),
        min_df: float = 0.1,
        max_features: int = 2000,
        max_iter: int = 1000,
        resources_dir: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.lang = lang
        self.seed = seed
        # The original compares concatenation, difference and product; the
        # absolute difference reaches the best AUC there and is reported.
        self.feat_combos = feat_combos
        self.spacy_model = spacy_model or ("en_core_web_sm" if lang == "en" else "de_core_news_sm")
        self.nlp_backend = nlp_backend
        self.char_ngram_range = tuple(char_ngram_range)
        self.tok_ngram_range = tuple(tok_ngram_range)
        self.pos_ngram_range = tuple(pos_ngram_range)
        self.min_df = min_df
        self.max_features = max_features
        self.max_iter = max_iter
        self.resources_dir = resources_dir or _RESOURCES
        self.pipeline = None
        self.clf = None

    def _pairs_labels_texts(self, corpus_path):
        problems, labels, _ = read_corpus(corpus_path)

        # Every text goes through the NLP once, however often it is paired.
        unique_texts, str_to_id = [], {}
        for p in problems:
            for text in (p[0], p[2]):
                if text not in str_to_id:
                    str_to_id[text] = f"doc_{len(unique_texts)}"
                    unique_texts.append(text)

        preprocessed = preprocess_texts(unique_texts, spacy_model=self.spacy_model,
                                        backend=self.nlp_backend, lang=self.lang)
        doc_texts = {str_to_id[t]: preprocessed[i] for i, t in enumerate(unique_texts)}
        doc_ids = [str_to_id[t] for t in unique_texts]
        pairs = [(str_to_id[p[0]], str_to_id[p[2]]) for p in problems]
        return pairs, np.asarray(labels).astype(int), doc_ids, doc_texts

    def _make_pipeline(self):
        union = make_doc_feature_pipeline(self.resources_dir, lang=self.lang,
                                          char_ngram_range=self.char_ngram_range,
                                          tok_ngram_range=self.tok_ngram_range,
                                          pos_ngram_range=self.pos_ngram_range,
                                          min_df=self.min_df,
                                          max_features=self.max_features)
        # build_pairwise_matrix multiplies the two vectors, which needs sparse input.
        return SkPipeline([("union", union),
                           ("to_csr", FunctionTransformer(csr_matrix, validate=False))])

    def train(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        pairs, y, doc_ids, doc_texts = self._pairs_labels_texts(corpus_path)
        # Refit per corpus, so that nothing carries over.
        self.pipeline = self._make_pipeline()
        doc2features, _ = build_doc_feature_matrix(doc_ids, doc_texts, self.pipeline, fit=True)
        X = build_pairwise_matrix(pairs, doc2features, self.feat_combos)

        self.clf = LogisticRegression(max_iter=self.max_iter, random_state=self.seed)
        self.clf.fit(X, y)

        scores = self.clf.predict_proba(X)[:, 1]
        self.train_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= 0.5).astype(int), y
        )
        return self.train_metrics

    def eval(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        if self.clf is None:
            raise ValueError("call train() before eval()")
        pairs, y, doc_ids, doc_texts = self._pairs_labels_texts(corpus_path)
        doc2features, _ = build_doc_feature_matrix(doc_ids, doc_texts, self.pipeline, fit=False)
        X = build_pairwise_matrix(pairs, doc2features, self.feat_combos)

        scores = self.clf.predict_proba(X)[:, 1]
        self.test_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= 0.5).astype(int), y
        )
        return self.test_metrics
