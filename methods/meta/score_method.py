import numpy as np
from sklearn.metrics import roc_curve

from common import AVMethod, calc_metrics, read_corpus


def eer_threshold(scores, labels):
    """Score at which the false positive rate equals the false negative rate."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))

    if idx > 0:
        x1, y1 = fpr[idx - 1], fnr[idx - 1]
        x2, y2 = fpr[idx], fnr[idx]
        t1, t2 = thresholds[idx - 1], thresholds[idx]
        if (x2 - x1) != (y2 - y1):
            alpha = (y1 - x1) / ((x2 - x1) - (y2 - y1))
            # roc_curve prepends an infinite threshold, which would make the
            # interpolation nan and in turn label every pair N.
            if 0 <= alpha <= 1 and np.isfinite(t1):
                return float(t1 + alpha * (t2 - t1))
    return float(thresholds[idx])


class ScoreMethod(AVMethod):
    """Base for methods that reduce a document pair to a similarity score.

    Subclasses implement get_similarity_score(); training only fits the decision
    threshold on the training scores.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.threshold = None

    def get_similarity_score(self, doc1: str, doc2: str) -> float:
        raise NotImplementedError

    def _score_corpus(self, corpus_path):
        problems, labels, _ = read_corpus(corpus_path)
        scores = np.array(
            [self.get_similarity_score(known, unknown) for known, _, unknown, _ in problems]
        )
        return scores, labels

    def train(self, corpus_name, corpus_path, **kwargs):
        scores, labels = self._score_corpus(corpus_path)
        # Refit on every call: the runner reuses one instance across corpora, so
        # a threshold kept from an earlier corpus would leak into the next one.
        self.threshold = eer_threshold(scores, labels)
        self.train_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= self.threshold).astype(int), labels
        )
        return self.train_metrics

    def eval(self, corpus_name, corpus_path, **kwargs):
        if self.threshold is None:
            raise ValueError("call train() before eval()")
        scores, labels = self._score_corpus(corpus_path)
        self.test_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= self.threshold).astype(int), labels
        )
        return self.test_metrics
