import os
from typing import Any, Dict

from common import AVMethod, calc_metrics, read_corpus
from methods.lambdag.custom_lambdag import LambdaGMethod


class LambdaG(AVMethod):
    """Likelihood ratio between the grammar of the known author and a reference
    population, calibrated into a probability by logistic regression.

    The reference population is drawn from the other authors of the same corpus,
    so train() and eval() each need a full corpus rather than a single pair.
    """

    def __init__(
        self,
        *,
        basis: str = "tokens",
        order: int = 10,
        smoothing: str = "kneser_ney",
        lowercasing: bool = True,
        sentenize: bool = True,
        num_references: int = 100,
        random_seed: int | None = None,
        num_cores: int | None = None,
        name: str = "LambdaG",
    ) -> None:
        super().__init__(name=name)

        self.method = LambdaGMethod(
            basis=basis,
            order=order,
            smoothing=smoothing,
            lowercasing=lowercasing,
            sentenize=sentenize,
            num_references=num_references,
            random_seed=random_seed,
            num_cores=os.cpu_count() if num_cores is None else num_cores,
        )
        self.train_metrics: Dict[str, Any] = {}
        self.test_metrics: Dict[str, Any] = {}

    def train(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        problems, labels, author_texts = read_corpus(corpus_path)
        scores = self.method.fit_predict(problems, author_texts, labels)[:, 1]
        self.train_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= 0.5).astype(int), labels
        )
        return self.train_metrics

    def eval(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        if self.method.calibration_model is None:
            raise ValueError("call train() before eval()")
        problems, labels, author_texts = read_corpus(corpus_path)
        scores = self.method.predict_proba(problems, author_texts)[:, 1]
        self.test_metrics = calc_metrics(
            corpus_name, self.name, scores, (scores >= 0.5).astype(int), labels
        )
        return self.test_metrics
