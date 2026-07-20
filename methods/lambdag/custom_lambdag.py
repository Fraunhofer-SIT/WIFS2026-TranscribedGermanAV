"""Nini et al. 2026, https://pypi.org/project/LambdaG

Copy of lambdag.LambdaGMethod, extended by a calibration model, fit_predict and
an optional process pool. The scoring itself is unchanged.
"""

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from lambdag.language_models.kneser_ney import KneserNeyLanguageModel, LanguageModel


def _lambdag_worker(task: dict) -> tuple[int, float]:
    # Module level, because spawn has to pickle the worker. Returns the index so
    # that the caller can restore the original order.
    method = LambdaGMethod(
        basis=task["basis"],
        order=task["order"],
        smoothing=task["smoothing"],
        lowercasing=task["lowercasing"],
        sentenize=task["sentenize"],
        num_references=task["num_references"],
        random_seed=task["seed"],
        num_cores=1,
    )

    known_author = task["known_author"]
    unknown_author = task["unknown_author"]
    reference_sentences = [
        sent
        for name, sents in task["author_sentences"].items()
        if name not in (known_author, unknown_author)
        for sent in sents
    ]

    score = method.lambdag_score(
        task["known_sentences"], task["unknown_sentences"], reference_sentences
    )
    return task["idx"], score


class LambdaGMethod:
    def __init__(
        self,
        basis: str,
        order: int,
        smoothing: str,
        lowercasing: bool,
        sentenize: bool,
        num_references: int,
        random_seed: Any = None,
        num_cores: int = 1,
    ):
        """
        Args:
            basis: element type of the sequences, 'tokens' or 'characters'
            order: language model order
            smoothing: 'kneser_ney' or 'kneser_ney_shb_disabled'
            lowercasing: lowercase all texts
            sentenize: split texts into sentences
            num_references: number of reference sentences that are sampled
            random_seed: seed of the internal generator, None for independent runs
            num_cores: processes used in lambdag(), 1 disables multiprocessing
        """
        if basis not in ["tokens", "characters"]:
            raise ValueError("basis must be 'tokens' or 'characters'")
        if smoothing not in ["kneser_ney", "kneser_ney_shb_disabled"]:
            raise ValueError(
                "smoothing must be 'kneser_ney' or 'kneser_ney_shb_disabled'"
            )
        if num_cores < 1:
            raise ValueError("num_cores must be >= 1")

        self.basis = basis
        self.order = order
        self.smoothing = smoothing
        self.lowercasing = lowercasing
        self.sentenize = sentenize
        self.num_references = num_references
        self.num_cores = int(num_cores)

        self._base_seed = random_seed
        self.random_gen = np.random.default_rng(seed=random_seed)
        self.calibration_model = None

    def fit_new_language_model(self, sentences: Iterable[Iterable[str]]) -> LanguageModel:
        lm = KneserNeyLanguageModel(
            discount=0.75,
            order=self.order,
            special_handling_of_pad_start_element=self.smoothing == "kneser_ney",
        )
        for sentence in sentences:
            lm.fit(sentence)
        return lm

    def preprocess_text(self, text: str) -> Iterable[Iterable[str]]:
        if self.lowercasing:
            text = text.lower()

        if self.sentenize:
            sentences = [
                s[0]
                for s in re.findall(r"(.+? ([.?!\n]+ |$))", text, re.VERBOSE)
                if len(s[0].strip()) > 0
            ]
        else:
            sentences = [text]

        if self.basis == "characters":
            return sentences
        return [
            tuple(t for t in re.split(r"\s+", sent) if len(t) > 0) for sent in sentences
        ]

    def lambdag_score(
        self,
        known_sentences: Iterable[Iterable[str]],
        unknown_sentences: Iterable[Iterable[str]],
        reference_sentences: list[Iterable[str]],
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        rng = rng if rng is not None else self.random_gen

        known_lm = self.fit_new_language_model(known_sentences)
        llrs = []
        for _ in range(self.num_references):
            sampled_reference_sentences = [
                reference_sentences[i]
                for i in rng.choice(
                    len(reference_sentences), size=len(known_sentences), replace=False
                )
            ]
            reference_lm = self.fit_new_language_model(sampled_reference_sentences)

            llr = 0
            for sentence in unknown_sentences:
                llr += np.sum(np.log2(known_lm.probabilities(sentence)))
                llr -= np.sum(np.log2(reference_lm.probabilities(sentence)))
            llrs.append(llr)
        return float(np.mean(llrs))

    def lambdag(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        num_cores: Optional[int] = None,
    ) -> np.ndarray:
        num_cores = self.num_cores if num_cores is None else int(num_cores)
        if num_cores < 1:
            raise ValueError("num_cores must be >= 1")

        av_problems_pp = [
            (
                self.preprocess_text(known_text),
                known_author,
                self.preprocess_text(unknown_text),
                unknown_author,
            )
            for known_text, known_author, unknown_text, unknown_author in av_problems
        ]
        author_sentences = {
            name: [sent for text in texts for sent in self.preprocess_text(text)]
            for name, texts in author_texts.items()
        }

        if num_cores == 1:
            scores = []
            for known_sentences, known_author, unknown_sentences, unknown_author in tqdm(
                av_problems_pp
            ):
                reference_sentences = [
                    sent
                    for name, sents in author_sentences.items()
                    if name not in [known_author, unknown_author]
                    for sent in sents
                ]
                scores.append(
                    self.lambdag_score(
                        known_sentences, unknown_sentences, reference_sentences
                    )
                )
            return np.array(scores, dtype=float)

        tasks = []
        for idx, (
            known_sentences,
            known_author,
            unknown_sentences,
            unknown_author,
        ) in enumerate(av_problems_pp):
            # Without a base seed the workers stay unseeded, otherwise idx gives
            # each of them an independent stream.
            if self._base_seed is None:
                seed = None
            else:
                seed = np.random.SeedSequence([self._base_seed, idx]).generate_state(1)[0]

            tasks.append(
                {
                    "idx": idx,
                    "basis": self.basis,
                    "order": self.order,
                    "smoothing": self.smoothing,
                    "lowercasing": self.lowercasing,
                    "sentenize": self.sentenize,
                    "num_references": self.num_references,
                    "seed": seed,
                    "known_sentences": known_sentences,
                    "known_author": known_author,
                    "unknown_sentences": unknown_sentences,
                    "unknown_author": unknown_author,
                    "author_sentences": author_sentences,
                }
            )

        results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=num_cores) as pool:
            futures = [pool.submit(_lambdag_worker, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, score = future.result()
                results[idx] = score
        return np.array(results, dtype=float)

    def fit_predict(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
        labels: Iterable[int],
    ) -> np.ndarray:
        scores = self.lambdag(av_problems, author_texts)
        self.calibration_model = LogisticRegression(class_weight="balanced")
        self.calibration_model.fit(scores[:, None], labels)
        return self.calibration_model.predict_proba(scores[:, None])

    def predict_proba(
        self,
        av_problems: Iterable[tuple[str, str, str, str]],
        author_texts: dict[str, Iterable[str]],
    ) -> np.ndarray:
        if not self.calibration_model:
            raise RuntimeError("call fit_predict() first")
        scores = self.lambdag(av_problems, author_texts)
        return self.calibration_model.predict_proba(scores[:, None])
