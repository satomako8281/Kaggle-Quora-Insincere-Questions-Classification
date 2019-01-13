import array
import re

from collections import defaultdict
from functools import partial
from multiprocessing.pool import Pool
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _make_int_array, CountVectorizer
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm

from quora.config import (
    MAX_FEATURES, MAXLEN, SEED, logger
)


class LowerCase(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def transform(self, X):
        X['question_text'] = X['question_text'].apply(lambda x: x.lower())
        return X


class NumberCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, *arg):
        return self

    def clean_numbers(self, x):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
        return x

    def transform(self, X):
        X['question_text'] = X['question_text'].apply(lambda x: self.clean_numbers(x))
        return X


class TextCleaner(BaseEstimator, TransformerMixin):
    puncts = [
        ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
        '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
        '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
        '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
        '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√',
    ]

    def fit(self, X, *arg):
        return self

    def clean_text(self, x):
        x = str(x)
        for punct in self.puncts:
            x = x.replace(punct, f' {punct} ')
        return x

    def transform(self, X):
        X['question_text'] = X['question_text'].apply(lambda x: self.clean_text(x))
        return X


class FillEmpty(BaseEstimator, TransformerMixin):

    def fit(self, X, *args):
        return self

    def transform(self, X):
        X['question_text'] = X['question_text'].fillna("_##_").values
        return X


class FeaturesCapsVSLength(BaseEstimator, TransformerMixin):

    def fit(self, X, *args):
        return self

    def transform(self, X):
        out = pd.DataFrame()
        X = X.apply(lambda x: str(x))
        out['total_length'] = X.apply(len)
        out['capitals'] = X.apply(lambda comment: sum(1 for c in comment if c.isupper()))
        out['caps_vs_length'] = out.apply(lambda row: float(row['capitals']) / float(row['total_length']), axis=1)
        return out[['caps_vs_length']].fillna(0).values

class FeaturesWordsVSUnique(BaseEstimator, TransformerMixin):

    def fit(self, X, *args):
        return self

    def transform(self, X):
        out = pd.DataFrame()
        X = X.apply(lambda x: str(x))
        out['total_length'] = X.apply(len)
        out['num_words'] = X.str.count('\S+')
        out['num_unique_words'] = X.apply(lambda comment: len(set(w for w in comment.split())))
        out['words_vs_unique'] = out['num_unique_words'] / out['num_words']

        return out[['words_vs_unique']].fillna(0).values


class QuoraTokenizer(BaseEstimator, TransformerMixin):

    def fit(self, X, *args):
        return self

    def transform(self, X):
        tokenizer = Tokenizer(num_words=MAX_FEATURES)
        tokenizer.fit_on_texts(list(X))
        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen=MAXLEN)

        return X

class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, dtype=None, inverse=False,
                 return_vector=True):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

        if isinstance(self.columns, str):
            self.columns = [self.columns]

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, x, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)

    def transform(self, x):
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')

        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column
        # is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1 and self.return_vector:
            return x[selected_cols[0]]
        else:
            return x[selected_cols]


class ReportShape(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info('=' * 30)
        logger.info("Matrix shape {} min {} max {}".format(X.shape, X.min(), X.max()))
        logger.info('=' * 30)
        return X
