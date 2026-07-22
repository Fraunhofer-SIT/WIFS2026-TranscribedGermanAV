"""Stylometric features for StyloSpeaker (Aggazzotti and Smith 2025).

lang="en" runs the original: 390 function words, 69 phrases, English
contractions, all nine readability measures, substring counting.

lang="de" swaps in 1602 German function words and 391 phrases from the POSNoise
pattern list, 34 contraction pairs, counts phrases on word boundaries, and drops
the two lexicon based readability measures (dale_chall, difficult_words), which
fall back to the English word list whatever set_lang says.
"""

import json
import os
import re
from collections import Counter

import textstat

_CACHE = {}


############## ------------- NLP-Doc-Adapter ------------- ##############
# Adapter so that the feature functions below can stay verbatim: they expect
# doc.sentences -> sent.words -> token.text / token.upos.

class _Tok:
    __slots__ = ("text", "upos")

    def __init__(self, text, upos):
        self.text = text
        self.upos = upos


class _Sent:
    __slots__ = ("words",)

    def __init__(self, words):
        self.words = words


class _Doc:
    __slots__ = ("sentences",)

    def __init__(self, sentences):
        self.sentences = sentences


def spacy_to_doc(spacy_doc):
    sents = []
    for sent in spacy_doc.sents:
        words = [_Tok(t.text, t.pos_) for t in sent if not t.is_space]
        if words:
            sents.append(_Sent(words))
    if not sents:
        words = [_Tok(t.text, t.pos_) for t in spacy_doc if not t.is_space]
        sents = [_Sent(words)] if words else [_Sent([_Tok("", "X")])]
    return _Doc(sents)


def stanza_to_doc(stanza_doc):
    sents = [_Sent([_Tok(w.text, w.upos) for w in s.words]) for s in stanza_doc.sentences]
    if not sents:
        sents = [_Sent([_Tok("", "X")])]
    return _Doc(sents)


############## ------------- CHARACTER properties ------------- ##############

### Get punctuation mark frequencies
def punctuation_freqs(tokens):  # use tokens not str_doc so doesn't count multi-marks (e.g. '...') separately
    punct_marks = ['.', '?', '!', ',', ';', ':', '-', '--', '---', '..', '...',
                   '(', ')', '[', ']', '\'', '"', '`']  # 18 total
    punct_counts = [tokens.count(punct) for punct in punct_marks]
    return punct_counts


############## ------------- TOKEN/WORD/POS properties ------------- ##############

### Get POS tag frequencies and word/token properties
def word_pos_freqs(doc, tokens):
    token_counts = Counter(tokens)
    postag_list = [0] * 16
    word_lengths = []
    word_count = 0
    long_words = 0
    short_words = 0
    capitalized_words = 0
    wordprop_list = [0] * 3
    wordratio_list = [0] * 4

    for sent in doc.sentences:
        for token in sent.words:
            if token.upos in ['ADJ']:
                postag_list[0] += 1
            elif token.upos in ['ADP']:
                postag_list[1] += 1
            elif token.upos in ['ADV']:
                postag_list[2] += 1
            elif token.upos in ['AUX']:
                postag_list[3] += 1
            elif token.upos in ['CCONJ']:
                postag_list[4] += 1
            elif token.upos in ['DET']:
                postag_list[5] += 1
            elif token.upos in ['INTJ']:
                postag_list[6] += 1
            elif token.upos in ['NOUN', 'PROPN']:
                postag_list[7] += 1
            elif token.upos in ['NUM']:
                postag_list[8] += 1
            elif token.upos in ['PART']:
                postag_list[9] += 1
            elif token.upos in ['PRON']:
                postag_list[10] += 1
            elif token.upos in ['PUNCT']:
                postag_list[11] += 1
            elif token.upos in ['SCONJ']:
                postag_list[12] += 1
            elif token.upos in ['SYM']:
                postag_list[13] += 1
            elif token.upos in ['VERB']:
                postag_list[14] += 1
            elif token.upos in ['X']:
                postag_list[15] += 1

            ## Word properties (words only, no punct/numbers/symbols)
            if token.upos not in ['NUM', 'PUNCT', 'SYM']:
                if len(token.text) >= 8:  # long words
                    long_words += 1
                if len(token.text) < 5:  # short words
                    short_words += 1
                elif token.text[0].isupper():  # original only checks words of length >= 5
                    capitalized_words += 1
                word_lengths.append(len(token.text))
                word_count += 1  # word count (W)

    word_count = max(word_count, 1)
    wordprop_list[0] = sum(word_lengths) / word_count   # average word length

    ## Total num tokens
    num_tokens = max(len(tokens), 1)
    wordprop_list[1] = len(tokens)
    num_unique_tokens = len(token_counts)
    wordprop_list[2] = num_unique_tokens

    ## Word ratios
    wordratio_list[0] = short_words / word_count
    wordratio_list[1] = long_words / word_count
    wordratio_list[2] = capitalized_words / word_count
    wordratio_list[3] = num_unique_tokens / num_tokens

    word_features = wordprop_list + wordratio_list + postag_list
    return word_features


############## ------------- Other SYNTAX properties ------------- ##############

### Get properties of sentences
def sentence_props(doc):
    total_sent_lens = []
    for sent in doc.sentences:
        total_sent_lens.append(len(list(sent.words)))
    num_sentences = max(len(list(doc.sentences)), 1)
    avg_sent_length = sum(total_sent_lens) / num_sentences
    return num_sentences, avg_sent_length


# Resources

def _read_list(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_function_lists(resources_dir, lang):
    key = ("func", resources_dir, lang)
    if key not in _CACHE:
        if lang == "en":
            with open(_require(os.path.join(resources_dir, "en",
                                            "function_words_augmented.json")), "r") as f:
                fw = json.load(f)
            _CACHE[key] = (fw["words"], fw["phrases"])
        else:
            w = _require(os.path.join(resources_dir, "de", "function_words_de.txt"))
            p = _require(os.path.join(resources_dir, "de", "function_phrases_de.txt"))
            _CACHE[key] = (_read_list(w), _read_list(p))
    return _CACHE[key]


def load_contraction_lists(resources_dir, lang):
    key = ("contr", resources_dir, lang)
    if key not in _CACHE:
        if lang == "en":
            with open(_require(os.path.join(resources_dir, "en",
                                            "comparison_lists_augmented.json")), "r") as f:
                cd = json.load(f)
            _CACHE[key] = (cd["contractions"][0], cd["contractions"][1])
        else:
            short, long_ = [], []
            for line in _read_list(_require(os.path.join(resources_dir, "de",
                                                         "contractions_de.tsv"))):
                parts = line.split("\t")
                if len(parts) == 2:
                    short.append(parts[0])
                    long_.append(parts[1])
            _CACHE[key] = (short, long_)
    return _CACHE[key]


def _count_phrase(phrase, low, word_boundary):
    """The original counts substrings; German counts on word boundaries, so that
    "im" does not match inside "immer"."""
    if not word_boundary:
        return low.count(phrase)
    return len(re.findall(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", low))


def function_words(str_doc, tokens, resources_dir, lang="de"):
    func_words, func_phrases = load_function_lists(resources_dir, lang)
    token_counts = Counter(tokens)
    func_word_feature = []
    for w in func_words:
        if w in token_counts:
            func_word_feature.append(token_counts[w])
        else:
            func_word_feature.append(0)
    low = str_doc.lower()
    wb = (lang != "en")
    func_phrase_feature = [_count_phrase(p, low, wb) for p in func_phrases]
    return func_word_feature, func_phrase_feature


############## ------------- DISCOURSE properties ------------- ##############

### Vocab richness (higher # = higher diversity/richer vocab)
## Source: [https://gist.github.com/magnusnissel/d9521cb78b9ae0b2c7d6]
def Yules_i(lex_words):
    word_counts = Counter(lex_words)
    m1 = sum(word_counts.values())
    m2 = sum([freq ** 2 for freq in word_counts.values()])
    try:
        i = (m1 * m1) / (m2 - m1)
    except ZeroDivisionError:
        i = 0
    return i




READABILITY_NAMES = {
    "en": ['FleschReadingEase', 'SMOGindex', 'Flesch-KincaidGradeLevel', 'Coleman-LiauIndex',
           'AutomatedReadIndex', 'Dale-ChallReadScore', 'DifficultWords',
           'LinsearWriteFormula', 'GunningfogIndex'],
    "de": ['FleschAmstad', 'SMOGindex', 'Flesch-KincaidGradeLevel', 'Coleman-LiauIndex',
           'AutomatedReadIndex', 'LinsearWriteFormula', 'GunningfogIndex'],
}


### Get readability scores
def readability_features(string_doc, lang="de"):
    textstat.set_lang(lang)  # global singleton, so set it on every call
    if lang == "en":
        textstat_scores = [textstat.flesch_reading_ease(string_doc),
                           textstat.smog_index(string_doc),
                           textstat.flesch_kincaid_grade(string_doc),
                           textstat.coleman_liau_index(string_doc),
                           textstat.automated_readability_index(string_doc),
                           textstat.dale_chall_readability_score(string_doc),
                           textstat.difficult_words(string_doc),
                           textstat.linsear_write_formula(string_doc),
                           textstat.gunning_fog(string_doc)]
        return textstat_scores

    textstat_scores = [textstat.flesch_reading_ease(string_doc),  # Amstad under set_lang(de)
                       textstat.smog_index(string_doc),
                       textstat.flesch_kincaid_grade(string_doc),
                       textstat.coleman_liau_index(string_doc),
                       textstat.automated_readability_index(string_doc),
                       textstat.linsear_write_formula(string_doc),
                       textstat.gunning_fog(string_doc)]
    return textstat_scores


### Get hapax legomena/dislegomena per speaker (not based on whole dataset)
def hapax(lex_words):
    word_counts = Counter(lex_words)
    hl_per_speaker = 0
    hd_per_speaker = 0
    for word in word_counts:
        if word_counts[word] == 1:
            hl_per_speaker += 1
        elif word_counts[word] == 2:
            hd_per_speaker += 1
    n = max(len(lex_words), 1)
    hl_normed = hl_per_speaker / n
    hd_normed = hd_per_speaker / n
    return hl_normed, hd_normed


### Get (augmented) stylistic contraction choices
def contractions(str_doc, resources_dir, lang="de"):
    contracted, expanded = load_contraction_lists(resources_dir, lang)
    comparison_counts = [count_occurence_phrase(contracted, str_doc, lang),
                         count_occurence_phrase(expanded, str_doc, lang)]
    return comparison_counts


### Helper function for contractions
def count_occurence_phrase(phrase_list, str_doc, lang="de"):
    num_count = 0
    low = str_doc.lower()
    wb = (lang != "en")
    for phrase in phrase_list:
        num_count += _count_phrase(phrase, low, wb)
    return num_count


############## ------------- GET ALL STYLOMETRIC FEATURES + NAMES ------------- ##############

def get_stylo_features(preprocessed_texts, resources_dir, lang="de"):
    docs_features = []
    for preprocessed_text in preprocessed_texts:
        string_doc = preprocessed_text['string']
        doc = preprocessed_text['doc']
        tokens = preprocessed_text['tokens']
        lex_words = preprocessed_text['words']

        ## Character properties
        punct_counts = punctuation_freqs(tokens)

        ## Token/word and POS properties
        word_features = word_pos_freqs(doc, tokens)

        ## Other syntax properties
        num_sents, avg_sent_length = sentence_props(doc)
        func_word_feature, func_phrase_feature = function_words(string_doc, tokens,
                                                                resources_dir, lang)

        ## Discourse properties
        yules = Yules_i(lex_words)
        textstat_scores = readability_features(string_doc, lang)
        hap_leg, hap_disleg = hapax(lex_words)
        comparison_counts = contractions(string_doc, resources_dir, lang)

        ### Collect all features
        features = [punct_counts, word_features, num_sents, avg_sent_length, func_word_feature,
                    func_phrase_feature, yules, textstat_scores, hap_leg, hap_disleg,
                    comparison_counts]
        features_flat = []
        for elem in features:
            if type(elem) == list:
                for item in elem:
                    features_flat.append(item)
            else:
                features_flat.append(elem)
        docs_features.append(features_flat)
    return docs_features


### NOTE Must keep the same order as in get_stylo_features()
def get_feature_names(resources_dir, lang="de"):
    ## Character properties
    punct_marks = ['.', '?', '!', ',', ';', ':', '-', '--', '---', '..', '...',
                   '(', ')', '[', ']', '\'', '"', '`']

    ## Token/word and POS properties
    word_properties = ['avgwordlength', '#totaltokens', '#uniqtokens']
    word_ratios = ['shortwords:W', 'longwords:W', 'capitalized:W', 'wordtypes:T']
    pos_categories = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUNs', 'NUM',
                      'PART', 'PRON', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

    ## Other syntax properties
    func_words, func_phrases = load_function_lists(resources_dir, lang)

    ## Discourse properties
    comparison_names = ['contracted', 'not contracted']

    ### Collect all feature names (order matters)
    feature_names = [punct_marks, word_properties, word_ratios, pos_categories, 'num sents',
                     'avg sent length', func_words, func_phrases, 'Yules_i',
                     READABILITY_NAMES[lang], 'hapax legomena', 'hapax dislegomena',
                     comparison_names]
    feature_names_flat = []
    for elem in feature_names:
        if type(elem) == list:
            for item in elem:
                feature_names_flat.append(item)
        else:
            feature_names_flat.append(elem)
    return feature_names_flat
