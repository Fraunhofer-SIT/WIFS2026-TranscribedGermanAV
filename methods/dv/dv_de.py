"""German DV-Bin. The vendored upstream is left untouched; the FeatureExtractor
is subclassed at the four language bound places: word tokenization, function
words, sentence splitting and POS tagging. Character and word n-grams, the
length distributions, tf-idf, the chi-square selection and the pair classifier
stay as they are.
"""

from feature_extraction.author_vectorizer import (
    FeatureExtractor,
    function_words_dict,
    pos_tagger_task,
)

from methods.dv.dv_method import DV

_NLTK_LANG = {"german": "german", "english": "english"}
_SPACY_DEFAULT = {"german": "de_core_news_sm", "english": "en_core_web_sm"}


def _register_german_function_words() -> None:
    if "german" in function_words_dict:
        return
    from spacy.lang.de.stop_words import STOP_WORDS

    function_words_dict["german"] = sorted(w.lower() for w in STOP_WORDS)


class GermanFeatureExtractor(FeatureExtractor):
    def __init__(self, *args, spacy_model: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        _register_german_function_words()
        self._nltk_lang = _NLTK_LANG.get(self.lang, "english")
        self._spacy_model = spacy_model or _SPACY_DEFAULT.get(self.lang, "en_core_web_sm")

    def tokenize(self, text):
        import nltk

        tokens = nltk.word_tokenize(text, language=self._nltk_lang)
        return [t.lower() for t in tokens if any(c.isalpha() for c in t)]

    def _features_sentence_lengths(self, documents, downto=3, upto=70):
        import nltk
        import numpy as np

        feats = []
        for doc in documents:
            sentences = [
                t.strip()
                for t in nltk.tokenize.sent_tokenize(doc, language=self._nltk_lang)
                if t.strip()
            ]
            nsent = max(1, len(sentences))
            sent_len = [len(self.tokenize(s)) for s in sentences]
            feats.append([sum(j >= i for j in sent_len) / nsent for i in range(downto, upto)])
        return np.asarray(feats)

    def pos_tagger(self, documents):
        import itertools
        import multiprocessing

        from joblib import Parallel, delayed

        n_jobs = multiprocessing.cpu_count()
        n_docs = len(documents)
        batch = max(1, int(n_docs / n_jobs))
        tags = Parallel(n_jobs=-1)(
            delayed(pos_tagger_task)(
                documents[job * batch:(job + 1) * batch
                          + (n_docs % n_jobs if job == n_jobs - 1 else 0)],
                job,
                self._spacy_model,
            )
            for job in range(n_jobs)
        )
        return list(itertools.chain.from_iterable(tags))


class DVDE(DV):
    def __init__(self, name: str = "DV-Bin-DE", lang: str = "german", **kwargs):
        super().__init__(name=name, lang=lang, **kwargs)

    def _build_vectorizer(self) -> FeatureExtractor:
        return GermanFeatureExtractor(
            self.lang,
            cleaning=False,
            use_raw_frequencies=self.use_raw_frequencies,
            spacy_model=self.spacy_model,
            **self.feature_flags,
        )
