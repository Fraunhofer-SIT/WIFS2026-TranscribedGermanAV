"""DV-Bin (Corbara, Moreo and Sebastiani 2023, https://github.com/AlexMoreo/diff-vectors)

A pair is represented by the absolute difference of the two feature frequency
vectors, and a logistic regression decides same or different author. The method
itself lives in external/diff-vectors; this wrapper only adapts the corpus
format and the AVMethod interface.
"""

from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

# methods/dv/__init__.py puts the vendored source root on sys.path.
from feature_extraction.author_vectorizer import FeatureExtractor
from model.pair_classification import PairSAVClassifier

from common import AVMethod, calc_metrics, read_corpus


class DV(AVMethod):
    def __init__(
        self,
        name: str = "DV-Bin",
        lang: str = "english",
        spacy_model: str | None = None,
        use_raw_frequencies: bool = False,
        use_function_words: bool = True,
        use_word_lengths: bool = True,
        use_sentence_lengths: bool = True,
        use_punctuation: bool = True,
        use_pos_ngrams: bool = True,
        use_word_ngrams: bool = True,
        use_char_ngrams: bool = True,
        max_pairs: int = 50000,
        lr_max_iter: int = 1000,
        seed: int | None = None,
    ):
        super().__init__(name=name)
        self.lang = lang
        self.spacy_model = spacy_model
        self.seed = seed
        self.use_raw_frequencies = use_raw_frequencies
        self.feature_flags = dict(
            function_words=use_function_words,
            word_lengths=use_word_lengths,
            sentence_lengths=use_sentence_lengths,
            punctuation=use_punctuation,
            post_ngrams=use_pos_ngrams,  # upstream spelling
            word_ngrams=use_word_ngrams,
            char_ngrams=use_char_ngrams,
        )
        self.max_pairs = max_pairs
        self.lr_max_iter = lr_max_iter
        self.vectorizer = None
        self.classifier = None

    def _build_vectorizer(self) -> FeatureExtractor:
        return FeatureExtractor(
            self.lang,
            cleaning=False,
            use_raw_frequencies=self.use_raw_frequencies,
            **self.feature_flags,
        )

    def _build_classifier(self) -> PairSAVClassifier:
        base = LogisticRegression(max_iter=self.lr_max_iter, n_jobs=-1)
        # pos=-1 takes every same-author pair, neg=-1 as many different-author ones.
        return PairSAVClassifier(base, -1, -1, self.max_pairs)

    @staticmethod
    def _flatten_corpus(corpus_path):
        """The pair generator samples its own pairs from a flat document list,
        so the corpus pairs are dissolved into distinct (text, author) entries."""
        problems, truths, _ = read_corpus(corpus_path)

        texts, authors, seen = [], [], set()
        for known_text, known_author, unknown_text, unknown_author in problems:
            for text, author in ((known_text, known_author), (unknown_text, unknown_author)):
                if (text, author) in seen:
                    continue
                seen.add((text, author))
                texts.append(text)
                authors.append(author)
        return texts, np.array(authors), problems, truths

    def _predict_problems(self, problems):
        scores = []
        for known_text, _, unknown_text, _ in problems:
            x1 = self.vectorizer.transform([known_text], None)
            x2 = self.vectorizer.transform([unknown_text], None)
            scores.append(self.classifier.h_prob(x1, x2))
        scores = np.array(scores, dtype=np.float32)
        return scores, (scores >= 0.5).astype(np.int32)

    def train(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        if self.seed is not None:
            import random

            random.seed(self.seed)
            np.random.seed(self.seed)

        texts, authors, problems, truths = self._flatten_corpus(corpus_path)
        if len(set(authors)) < 2:
            raise ValueError(f"{corpus_name}: DV needs at least two authors to pair up")

        # Refit both per corpus, so that nothing carries over.
        self.vectorizer = self._build_vectorizer()
        X = self.vectorizer.fit_transform(texts, authors)
        self.classifier = self._build_classifier()
        self.classifier.fit(X, authors)

        scores, preds = self._predict_problems(problems)
        self.train_metrics = calc_metrics(corpus_name, self.name, scores, preds, truths)
        return self.train_metrics

    def eval(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        if self.classifier is None:
            raise ValueError("call train() before eval()")
        problems, truths, _ = read_corpus(corpus_path)
        scores, preds = self._predict_problems(problems)
        self.test_metrics = calc_metrics(corpus_name, self.name, scores, preds, truths)
        return self.test_metrics
