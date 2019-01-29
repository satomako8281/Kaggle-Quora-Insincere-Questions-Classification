from contextlib import contextmanager
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
import keras as K
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import gc
import re
import string

from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, StandardScaler
from sklearn.pipeline import make_pipeline, make_union

from quora.transformers import (
    PandasSelector, LowerCase, TextCleaner, NumberCleaner, FillEmpty, MispellFixer,
    QuoraTokenizer, FeaturesCapsVSLength, FeaturesWordsVSUnique, ReportShape
)


def prepare_vectorizer_1():
    vectorizer = make_pipeline(
        LowerCase(),
        TextCleaner(),
        NumberCleaner(),
        FillEmpty(),

        make_union(
            make_pipeline(
                PandasSelector(columns=['question_text']),
                QuoraTokenizer(),
            ),
            make_pipeline(
                PandasSelector(columns=['question_text']),
                FeaturesCapsVSLength(),
                StandardScaler(),
            ),
            make_pipeline(
                PandasSelector(columns=['question_text']),
                FeaturesWordsVSUnique(),
                StandardScaler(),
            )
        ),
        ReportShape()
    )

    return vectorizer

def prepare_vectorizer_2():
    vectorizer = make_pipeline(
        LowerCase(),
        TextCleaner(),
        NumberCleaner(),
        MispellFixer(),
        FillEmpty(),

        make_union(
            make_pipeline(
                PandasSelector(columns=['question_text']),
                QuoraTokenizer(),
            ),
            make_pipeline(
                PandasSelector(columns=['question_text']),
                FeaturesCapsVSLength(),
                StandardScaler(),
            ),
            make_pipeline(
                PandasSelector(columns=['question_text']),
                FeaturesWordsVSUnique(),
                StandardScaler(),
            )
        ),
        ReportShape()
    )

    return vectorizer


def prepare_vectorizer_3():
    tokenizer = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    def on_field(f: str, *vec):
        return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

    def tokenize(s):
        return tokenizer.sub(r' \1 ', s).split()

    vectorizer = make_pipeline(
        LowerCase(),
        TextCleaner(),
        NumberCleaner(),
        MispellFixer(),
        FillEmpty(),
        make_union(
            on_field(
                'question_text',
                TfidfVectorizer(
                    max_features=13000,
                    token_pattern='\w+',
                    strip_accents='unicode',
                    tokenizer=tokenize,
                    sublinear_tf=True
                )
            ),
            on_field(
                'question_text',
                TfidfVectorizer(
                    ngram_range=(3, 3),
                    analyzer='char',
                    min_df=25
                )
            ),
            make_pipeline(
                PandasSelector(columns=['num_words', 'num_singletons', 'caps_vs_length'], return_vector=False),
                MaxAbsScaler()),
            ),
        ReportShape()
    )

    return vectorizer

