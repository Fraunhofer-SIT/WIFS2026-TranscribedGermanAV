"""Pipeline blocks for StyloSpeaker. The TF-IDF settings, the StandardScaler on
the stylometric block and the pairwise combinations are the original ones; only
the NLP backend is language dependent.
"""

import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from methods.stylospeaker.stylometric_features import (
    get_feature_names,
    get_stylo_features,
    spacy_to_doc,
    stanza_to_doc,
)

_NLP_CACHE = {}


def _get_spacy(model_name):
    if model_name not in _NLP_CACHE:
        import spacy

        nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
        if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        _NLP_CACHE[model_name] = nlp
    return _NLP_CACHE[model_name]


def preprocess_texts(unique_texts, spacy_model="de_core_news_sm", backend="spacy", lang="de"):
    if backend == "stanza":
        import stanza

        nlp = stanza.Pipeline(lang=lang, processors="tokenize,pos", use_gpu=False,
                              pos_batch_size=500, download_method=None)
        docs = [stanza_to_doc(d) for d in nlp([stanza.Document([], text=t) for t in unique_texts])]
    else:
        nlp = _get_spacy(spacy_model)
        docs = [spacy_to_doc(d) for d in nlp.pipe(unique_texts, batch_size=8)]

    preprocessed_texts = []
    for text, doc in zip(unique_texts, docs):
        tokens = []      # words plus punctuation, numbers and symbols
        lex_words = []   # lexical items only
        pos_tags = []
        for sent in doc.sentences:
            for token in sent.words:
                tokens.append(token.text.lower())
                if token.upos not in ["NUM", "PUNCT", "SYM"]:
                    lex_words.append(token.text.lower())
                if token.upos in ["PUNCT", "SYM"]:
                    pos_tags.append(token.text)
                else:
                    pos_tags.append(token.upos)
        preprocessed_texts.append({"string": text,
                                   "doc": doc,
                                   "tokens": tokens,
                                   "words": lex_words,
                                   "pos_tags": " ".join(pos_tags)})
    return preprocessed_texts


class StylometricFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, resources_dir, lang="de"):
        self.resources_dir = resources_dir
        self.lang = lang

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(get_stylo_features(X, self.resources_dir, self.lang), dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(get_feature_names(self.resources_dir, self.lang))


def extract_string_text(docs):
    return [doc["string"] for doc in docs]


def extract_pos_tag_text(docs):
    return [doc["pos_tags"] for doc in docs]


def make_doc_feature_pipeline(resources_dir, lang="de", char_ngram_range=(3, 6),
                              tok_ngram_range=(1, 3), pos_ngram_range=(1, 3),
                              min_df=0.1, max_features=2000):
    return FeatureUnion([
        ("char_tfidf", Pipeline([
            ("get_string", FunctionTransformer(extract_string_text, validate=False)),
            ("char_tfidf_vec", TfidfVectorizer(analyzer="char", ngram_range=char_ngram_range,
                                               lowercase=True, min_df=min_df, norm="l2",
                                               max_features=max_features)),
        ])),
        ("token_tfidf", Pipeline([
            ("get_string", FunctionTransformer(extract_string_text, validate=False)),
            ("token_tfidf_vec", TfidfVectorizer(analyzer="word", ngram_range=tok_ngram_range,
                                                lowercase=True, min_df=min_df, norm="l2",
                                                max_features=max_features)),
        ])),
        ("pos_tfidf", Pipeline([
            ("get_tags", FunctionTransformer(extract_pos_tag_text, validate=False)),
            ("pos_tfidf_vec", TfidfVectorizer(analyzer="word", ngram_range=pos_ngram_range,
                                              lowercase=True, min_df=min_df, norm="l2",
                                              max_features=max_features)),
        ])),
        ("stylo", Pipeline([
            ("extract", StylometricFeatures(resources_dir, lang)),
            ("scale", StandardScaler()),
        ])),
    ])


def build_doc_feature_matrix(doc_ids, doc_texts, pipeline, fit=False):
    texts = [doc_texts[d] for d in doc_ids]
    X = pipeline.fit_transform(texts) if fit else pipeline.transform(texts)
    return {doc: X[i] for i, doc in enumerate(doc_ids)}, texts


def build_pairwise_matrix(pairs, doc2features, feat_combos):
    pair_features = []
    for d1, d2 in pairs:
        f1, f2 = doc2features[d1], doc2features[d2]
        diff = f1 - f2
        prod = f1.multiply(f2)
        if feat_combos == "feats":
            pair_feat = hstack([f1, f2])
        elif feat_combos == "diff":
            pair_feat = diff
        elif feat_combos == "diffabs":
            pair_feat = abs(diff)
        elif feat_combos == "featsdiffabs":
            pair_feat = hstack([f1, f2, abs(diff)])
        elif feat_combos == "diffprod":
            pair_feat = hstack([diff, prod])
        elif feat_combos == "diffabsprod":
            pair_feat = hstack([abs(diff), prod])
        elif feat_combos == "featsdiffprod":
            pair_feat = hstack([f1, f2, diff, prod])
        else:
            raise ValueError(f"unknown feat_combos: {feat_combos}")
        pair_features.append(pair_feat)
    return vstack(pair_features)
