"""German FeVec. POS tagging, tokenization and chunking go through spaCy instead
of the NLTK taggers and the conll2000 chunker, POS tags are UPOS so that the POS
features stay language neutral, and function words come from spaCy. The
misspelling feature is dropped, since its lists are English only. Character
n-grams, special characters, length statistics and vocabulary richness are
unchanged.
"""

import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from methods.fevec.custom_transformers import (
    CustomFreqTransformer,
    CustomFuncTransformer,
    CustomTfIdfTransformer,
    MaskedStopWordsTransformer,
    POSTagStats,
)
from methods.fevec.fevec_features import (
    VOCAB_RICHNESS_FNAMES,
    avg_chars_per_word,
    character_count,
    compute_vocab_richness,
    distr_chars_per_word,
    preprocess_text,
)
from methods.fevec.fevec_method import FeVec

UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]


class POSTagStatsUPOS(POSTagStats):
    POS_TAGS = UPOS_TAGS


def get_transformer_de(stopwords, use_chunks=True):
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{\u00a6}~"

    featuresets = [
        ("char_distr", CustomTfIdfTransformer("preprocessed", "char_wb", n=3)),
        ("pos_tag_distr", CustomTfIdfTransformer("pos_tags", "word", n=3)),
        ("special_char_distr",
         CustomTfIdfTransformer("preprocessed", "char_wb", vocab=punctuation)),
        ("freq_func_words", CustomFreqTransformer("word", vocab=list(stopwords))),
        ("character_count", CustomFuncTransformer(character_count)),
        ("distr_chars_per_word",
         CustomFuncTransformer(distr_chars_per_word, fnames=[str(i) for i in range(10)])),
        ("avg_chars_per_word", CustomFuncTransformer(avg_chars_per_word)),
        ("vocab_richness",
         CustomFuncTransformer(compute_vocab_richness, fnames=VOCAB_RICHNESS_FNAMES)),
        ("masked_stop_words_distr", MaskedStopWordsTransformer(stopwords, 3)),
        ("pos_tag_stats", POSTagStatsUPOS()),
    ]
    if use_chunks:
        featuresets.insert(2, ("pos_tag_chunks_distr",
                               CustomTfIdfTransformer("pos_tag_chunks", "word", n=3)))
        featuresets.insert(3, ("pos_tag_chunks_subtree_distr",
                               CustomTfIdfTransformer("pos_tag_chunk_subtrees", "word", n=1)))
    return FeatureUnion(featuresets)


class FeVecDE(FeVec):
    def __init__(
        self,
        name: str = "FeVec-DE",
        seed: int | None = None,
        spacy_model: str = "de_core_news_lg",
        use_chunks: bool = True,
    ) -> None:
        super().__init__(name=name, seed=seed)
        import spacy
        from spacy.lang.de.stop_words import STOP_WORDS

        # A smaller model tags differently and changes four of the twelve
        # feature blocks, so it is not substituted silently.
        self.spacy_model_name = spacy_model
        # The parser stays on for noun_chunks, tagger and morphologizer supply UPOS.
        self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])

        self.use_chunks = use_chunks
        if self.use_chunks and not self.nlp.has_pipe("parser"):
            raise ValueError(f"{spacy_model} has no parser, needed for the chunk features")
        self.stopwords_de = sorted(w.lower() for w in STOP_WORDS)

    def prepare_entry(self, text):
        doc = self.nlp(text)
        toks = list(doc)
        pos_tags = [t.pos_ for t in toks]

        pos_chunks, subtree_expansions = [], []
        if self.use_chunks:
            chunk_spans = list(doc.noun_chunks)
            starts = {ch.start: ch for ch in chunk_spans}
            covered = {i for ch in chunk_spans for i in range(ch.start, ch.end)}
            i = 0
            while i < len(toks):
                if i in starts:
                    pos_chunks.append("NP")
                    i = starts[i].end
                else:
                    pos_chunks.append("NP" if i in covered else pos_tags[i])
                    i += 1
            subtree_expansions = [
                "NP[" + " ".join(t.pos_ for t in ch) + "]" for ch in chunk_spans
            ]

        return {
            "preprocessed": text,
            "pos_tags": pos_tags,
            "pos_tag_chunks": pos_chunks,
            "pos_tag_chunk_subtrees": subtree_expansions,
            "tokens": [preprocess_text(t.text) for t in toks],
        }

    def _build_transformer(self):
        return get_transformer_de(self.stopwords_de, use_chunks=self.use_chunks)