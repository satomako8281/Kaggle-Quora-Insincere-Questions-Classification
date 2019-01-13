import os
import re
from tqdm import tqdm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from quora.config import (
    MAX_FEATURES, MAXLEN, SEED,
    X_TRAIN_ARRAY, X_TEST_ARRAY, Y_TRAIN_ARRAY, FEATURES_ARRAY, TEST_FEATURES_ARRAY, WORD_INDEX_ARRAY
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, make_union

from quora.transformers import (
    PandasSelector, LowerCase, TextCleaner, NumberCleaner, FillEmpty,
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


    # np.save(os.path.join('./input', X_TRAIN_ARRAY), x_train)
    # np.save(os.path.join('./input', X_TEST_ARRAY), x_test)
    # np.save(os.path.join('./input', Y_TRAIN_ARRAY), y_train)
    # np.save(os.path.join('./input', FEATURES_ARRAY), features)
    # np.save(os.path.join('./input', TEST_FEATURES_ARRAY), test_features)
    # np.save(os.path.join('./input', WORD_INDEX_ARRAY), tokenizer.word_index)
    # tokenizer = Tokenizer(num_words=MAX_FEATURES)
    # tokenizer.fit_on_texts(list(x_train))
    # x_train = tokenizer.texts_to_sequences(x_train)
    # x_train = pad_sequences(x_train, maxlen=MAXLEN)
    # x_test = tokenizer.texts_to_sequences(x_test)
    # x_test = pad_sequences(x_test, maxlen=MAXLEN)
