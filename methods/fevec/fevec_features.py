import os
import pickle
import re
from typing import Generator, List, TypeVar

import nltk
import numpy as np
from sklearn.pipeline import FeatureUnion
from textcomplexity import vocabulary_richness

from methods.fevec.consecutive_np_chunk import ConsecutiveNPChunker
from methods.fevec.custom_transformers import (
    CustomFreqTransformer,
    CustomFuncTransformer,
    CustomTfIdfTransformer,
    MaskedStopWordsTransformer,
    MisspellingsFeatureTransformer,
    POSTagStats,
)

T = TypeVar("T")

def get_simple_chunker(seq: List[T], size: int) -> Generator[List[T], None, None]:
    """
    Generate successive fixed-size chunks from a list.

    Args:
        seq (List[T]): List to be split into chunks.
        size (int): Maximum length of each chunk.

    Returns:
        Iterator[List[T]]: An iterator yielding consecutive list slices,
        each with length up to `size`.
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def preprocess_text(text):
    """Lowercase text and replace URLs with a placeholder."""
    text = text.lower()
    text = re.sub(r"((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))", " URL ", text)
    return text


def chunk_to_str(chunk):
    if type(chunk) is nltk.tree.Tree:
        return chunk.label()
    else:
        return chunk[1]


def extract_subtree_expansions(t, res):
    if type(t) is nltk.tree.Tree:
        expansion = (
            t.label() + "[" + " ".join([chunk_to_str(child) for child in t]) + "]"
        )
        res.append(expansion)
        for child in t:
            extract_subtree_expansions(child, res)


def pos_tag_chunk(pos_tags, chunker):
    parse_tree = chunker.parse(pos_tags)
    subtree_expansions = []
    for subt in parse_tree:
        extract_subtree_expansions(subt, subtree_expansions)
    return list(map(chunk_to_str, parse_tree)), subtree_expansions


def tokenize(text, tokenizer):
    if tokenizer == "treebank":
        return nltk.tokenize.TreebankWordTokenizer().tokenize(text)
    if tokenizer == "casual":
        return nltk.tokenize.casual_tokenize(text)
    if tokenizer == "spacy":
        import spacy

        nlp_spacy = spacy.load("en_core_web_sm", disable=["ner"])
        return map(lambda t: t.text, nlp_spacy(text))
    if tokenizer == "stanza":
        import stanza

        nlp_stanza = stanza.Pipeline(
            lang="en", processors="tokenize", tokenize_no_ssplit=True
        )
        return map(lambda t: t.text, nlp_stanza(text).iter_tokens())
    raise "Unknown tokenizer type. Valid options: [treebank, casual, spacy, stanza]"


def get_nltk_pos_tag_based_ml_chunker(chunker_dir):
    chunker_path = os.path.join(chunker_dir, "chunker.p")
    if os.path.isfile(chunker_path):
        ml_chunker = pickle.load(open(chunker_path, "rb"))
        return ml_chunker
    print("Training new chunker...")
    ml_chunker = train_chunker(chunker_path)
    return ml_chunker


def train_chunker(chunker_path):
    from nltk.corpus import conll2000

    nltk.download("conll2000", quiet=True)
    train_sents = conll2000.chunked_sents("train.txt")
    test_sents = conll2000.chunked_sents("test.txt")
    chunker = ConsecutiveNPChunker(train_sents)
    print(chunker.evaluate(test_sents))
    with open(chunker_path, "wb") as f:
        pickle.dump(chunker, f)
        print(f"Saved chunker to {chunker_path}")
    return chunker


VOCAB_RICHNESS_FNAMES = [
    "type_token_ratio",
    "guiraud_r",
    "herdan_c",
    "dugast_k",
    "maas_a2",
    "dugast_u",
    "tuldava_ln",
    "brunet_w",
    "cttr",
    "summer_s",
    "sichel_s",
    "michea_m",
    "honore_h",
    "herdan_vm",
    "entropy",
    "yule_k",
    "simpson_d",
]


def handle_exceptions(func, *args):
    try:
        return func(*args)
    except:
        # print('Error occured', func, *args)
        return 0.0


def compute_vocab_richness(entry):
    if len(entry["tokens"]) == 0:
        return np.zeros(len(VOCAB_RICHNESS_FNAMES))
    window_size = 1000
    res = []
    for chunk in get_simple_chunker(entry["tokens"], window_size):
        text_length, vocabulary_size, frequency_spectrum = (
            vocabulary_richness.preprocess(chunk, fs=True)
        )
        res.append(
            [
                handle_exceptions(
                    vocabulary_richness.type_token_ratio, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.guiraud_r, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.herdan_c, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.dugast_k, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.maas_a2, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.dugast_u, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.tuldava_ln, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.brunet_w, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.cttr, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.summer_s, text_length, vocabulary_size
                ),
                handle_exceptions(
                    vocabulary_richness.sichel_s, vocabulary_size, frequency_spectrum
                ),
                handle_exceptions(
                    vocabulary_richness.michea_m, vocabulary_size, frequency_spectrum
                ),
                handle_exceptions(
                    vocabulary_richness.honore_h,
                    text_length,
                    vocabulary_size,
                    frequency_spectrum,
                ),
                handle_exceptions(
                    vocabulary_richness.herdan_vm,
                    text_length,
                    vocabulary_size,
                    frequency_spectrum,
                ),
                handle_exceptions(
                    vocabulary_richness.entropy, text_length, frequency_spectrum
                ),
                handle_exceptions(
                    vocabulary_richness.yule_k, text_length, frequency_spectrum
                ),
                handle_exceptions(
                    vocabulary_richness.simpson_d, text_length, frequency_spectrum
                ),
            ]
        )
    return np.array(res).mean(axis=0)


def get_stopwords(dir):
    with open(os.path.join(dir, "stopwords.txt"), "r", encoding="utf-8") as f:
        words = [l.strip() for l in f.readlines()]
        return words


def avg_chars_per_word(entry):
    r = np.mean([len(t) for t in entry["tokens"]])
    return r


def distr_chars_per_word(entry, max_chars=10):
    counts = [0] * max_chars
    if len(entry["tokens"]) == 0:
        return counts
    for t in entry["tokens"]:
        l = len(t)
        if l <= max_chars:
            counts[l - 1] += 1
    r = [c / len(entry["tokens"]) for c in counts]
    #     fnames = ['distr_chars_per_word_' + str(i + 1)  for i in range(max_chars)]
    return r


def character_count(entry):
    r = len(re.sub(r"\s+", "", entry["preprocessed"]))
    return r


def get_transformer(selected_featuresets=None, resources_dir="./resources/"):
    char_distr = CustomTfIdfTransformer("preprocessed", "char_wb", n=3)
    pos_tag_distr = CustomTfIdfTransformer("pos_tags", "word", n=3)
    pos_tag_chunks_distr = CustomTfIdfTransformer("pos_tag_chunks", "word", n=3)
    pos_tag_chunks_subtree_distr = CustomTfIdfTransformer(
        "pos_tag_chunk_subtrees", "word", n=1
    )
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{¦}~"
    special_char_distr = CustomTfIdfTransformer(
        "preprocessed", "char_wb", vocab=punctuation
    )
    freq_func_words = CustomFreqTransformer("word", vocab=get_stopwords(resources_dir))

    featuresets = [
        ("char_distr", char_distr),
        ("pos_tag_distr", pos_tag_distr),
        ("pos_tag_chunks_distr", pos_tag_chunks_distr),
        ("pos_tag_chunks_subtree_distr", pos_tag_chunks_subtree_distr),
        ("special_char_distr", special_char_distr),
        ("freq_func_words", freq_func_words),
        ("character_count", CustomFuncTransformer(character_count)),
        (
            "distr_chars_per_word",
            CustomFuncTransformer(
                distr_chars_per_word, fnames=[str(i) for i in range(10)]
            ),
        ),
        ("avg_chars_per_word", CustomFuncTransformer(avg_chars_per_word)),
        (
            "vocab_richness",
            CustomFuncTransformer(compute_vocab_richness, fnames=VOCAB_RICHNESS_FNAMES),
        ),
        ("misspellings", MisspellingsFeatureTransformer(data_dir=resources_dir)),
        (
            "masked_stop_words_distr",
            MaskedStopWordsTransformer(get_stopwords(resources_dir), 3),
        ),
        ("pos_tag_stats", POSTagStats()),
    ]
    if selected_featuresets is None:
        transformer = FeatureUnion(featuresets)
    else:
        transformer = FeatureUnion(
            [f for f in featuresets if f[0] in selected_featuresets]
        )

    return transformer
