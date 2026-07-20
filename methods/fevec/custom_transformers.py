from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from collections import defaultdict
import spacy
import os
from nltk.probability import FreqDist
import pickle
import numpy as np

def pass_fn(x):
    return x


class CustomTfIdfTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, key, analyzer, n=1, vocab=None):
        self.key = key
        if self.key == 'pos_tags' or self.key == 'tokens' or self.key == 'pos_tag_chunks' or self.key == 'pos_tag_chunk_subtrees':
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, tokenizer=pass_fn, preprocessor=pass_fn, vocabulary=vocab, norm='l2', ngram_range=(1, n), token_pattern=None)
        else:
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, vocabulary=vocab, norm='l2', ngram_range=(1, n))

    def fit(self, x, y=None):
        self.vectorizer.fit([entry[self.key] for entry in x], y)
        return self

    def transform(self, x):
        return self.vectorizer.transform([entry[self.key] for entry in x])
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
    
    
class CustomFreqTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, analyzer, n=1, vocab=None):
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn, vocabulary=vocab, norm=None, ngram_range=(1, n), token_pattern=None)

    def fit(self, x, y=None):
        self.vectorizer.fit([entry['tokens'] for entry in x], y)
        return self

    def transform(self, x):
        d = np.array([1 + len(entry['tokens']) for entry in x])[:, None]
        return self.vectorizer.transform([entry['tokens'] for entry in x]) / d
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
    
    
class CustomFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_func, fnames=None):
        self.transformer_func = transformer_func
        self.fnames = fnames
        
    def fit(self, x, y=None):
        return self;
    
    def transform(self, x):
        xx = np.array([self.transformer_func(entry) for entry in x])
        if len(xx.shape) == 1:
            return xx[:, None]
        else:
            return xx
    
    def get_feature_names(self):
        if self.fnames is None:
            return ['']
        else:
            return self.fnames

class MaskedStopWordsTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, stopwords, n):
        self.stopwords = set(stopwords)
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn, min_df=0.1, ngram_range=(1, n), token_pattern=None)
    
    def _process(self, entry):
        return [
            entry['tokens'][i] if entry['tokens'][i] in self.stopwords else entry['pos_tags'][i]
            for i in range(len(entry['tokens']))
        ]
    
    def fit(self, X, y=None):
        X = [self._process(entry) for entry in X]
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        X = [self._process(entry) for entry in X]
        return self.vectorizer.transform(X)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
    
    
class POSTagStats(BaseEstimator, TransformerMixin):
    
    POS_TAGS = [
                'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ',
                'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 
                'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
                'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',
                'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
                'WP', 'WP$', 'WRB'
            ]
    
    def __init__(self):
        pass
    
    def _process(self, entry):
        tags_dict = defaultdict(set)
        tags_word_length = defaultdict(list)
        for i in range(len(entry['tokens'])):
            tags_dict[entry['pos_tags'][i]].add(entry['tokens'][i])
            tags_word_length[entry['pos_tags'][i]].append(len(entry['tokens'][i]))
        res_tag_fractions = np.array([len(tags_dict[t]) for t in self.POS_TAGS])
        if res_tag_fractions.sum() > 0:
            res_tag_fractions = res_tag_fractions / res_tag_fractions.sum()
        
        res_tag_word_lengths = np.array([np.mean(tags_word_length[t]) if len(tags_word_length[t]) > 0 else 0 for t in self.POS_TAGS])
        return np.concatenate([res_tag_fractions, res_tag_word_lengths])
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._process(entry) for entry in X]
    
    def get_feature_names(self):
        return ['tag_fraction_' + t for t in self.POS_TAGS] + ['tag_word_length_' + t for t in self.POS_TAGS]
    
class DependencyFeatures(BaseEstimator, TransformerMixin):


    def __init__(self, n=3):
        self.vectorizer = TfidfVectorizer(tokenizer=self.pass_fn, preprocessor=self.pass_fn, token_pattern=None)
        self.nlp = spacy.load("en_core_web_sm")
    
    def pass_fn(self, x):
        return x

    def extract_dep_n_grams(self, node, current_list, output, n, curr_depth=0):
        current_list.append(node.dep_)
        output.append(current_list)
        if curr_depth > 200:
            return
        for c in node.children:
            l = current_list.copy()
            if len(l) > n - 1:
                l = l[1:]
            self.extract_dep_n_grams(c, l, output, n, curr_depth + 1)
        
    def dep_ngrams_to_str(self, ngam_list):
        return ['_'.join(ngrams).lower() for ngrams in ngam_list]

    def process(self, text):
        dep_ngrams = []
        doc = self.nlp(text)
        for s in doc.sents:
            o = []
            self.extract_dep_n_grams(s.root, [], o, 4)
            dep_ngrams.extend(o)
        return self.dep_ngrams_to_str(dep_ngrams)

    def fit(self, x, y=None):
        xx = [self.process(entry['preprocessed']) for entry in x]
        self.vectorizer.fit(xx, y)
        return self

    def transform(self, x):
        xx = [self.process(entry['preprocessed']) for entry in x]
        return self.vectorizer.transform(xx)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


def misspelled_arr(misspelled_text_file_path): 
    """
        Given a texfile of mispelled words, 
        return an array of misspellings.
    """
    f = open(misspelled_text_file_path,"r")
    Lines = f.readlines()
    
    mis_arr_temp = [ ]
    
    for t in Lines:
        w = t.split('//')[1].strip()
        w_arr = w.split(',')
        mis_arr_temp.extend(w_arr)
        
    mis_arr = [w.strip().lower() for w in mis_arr_temp]
    
    return mis_arr

def common_typos(typos_text_file_path):
    """
        Given a textfile of typos,
        return an array of common misspellings.
        
        Reference: https://www.lexico.com/grammar/common-misspellings
    """
    f = open(typos_text_file_path)
    
    Lines = f.readlines()
    
    typos = [t.split()[-1].lower() for t in Lines]
        
    return typos

def brit_spelling(file_path):
    """
        Given a text file of British typos,
        return an array of British spellings of words.
    """
    #https://www.lexico.com/grammar/british-and-spelling
    
    f = open(file_path)
    Lines = f.readlines()
    
    b = []
    
    for l in Lines:
        w = l.split('\t')[0].lower()
        word = w.split()
        for x in word: 
            if x != '\n':
                b.append(x)
            
    return b

def determiner(file_path):
    """
        Given a file of typos with determiners,
        return an array of mistyped determiners.
    """
    f = open(file_path)
    
    L = f.readlines()
    
    d = [w.split()[0].lower() for w in L]
    
    return d


def create_misspellings_dict(data_dir):
    dictionary = {
        'typos': set(common_typos(os.path.join(data_dir, 'common_typos.txt'))),
        'common': set(misspelled_arr(os.path.join(data_dir, 'mis_words.txt'))),
        'british': set(brit_spelling(os.path.join(data_dir, 'brit_spelling.txt'))),
        'determiner': set(determiner(os.path.join(data_dir, 'determiner_err.txt')))
    }
    return dictionary

class MisspellingsFeatureTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, data_dir):
        dict_path = os.path.join(data_dir, 'misspellings_dict.p')
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                self.misspelling_dict = pickle.load(f)
        else:
            self.misspelling_dict = create_misspellings_dict(data_dir)
            with open(dict_path, 'wb') as f:
                pickle.dump(self.misspelling_dict, f)
            

    def fit(self, x, y=None):
        return self
    
    def _process(self, x):
        fdist = FreqDist(x['tokens'])
        result = []
        for k in self.misspelling_dict.keys():
            result.append(sum([fdist[w] for w in self.misspelling_dict[k]])/len(x['tokens']))
        return result
    
    def transform(self, X):
        return list(map(self._process, X))
    
    def get_feature_names(self):
        return list(self.misspelling_dict.keys())