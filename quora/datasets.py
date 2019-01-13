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


##########
def load_and_prec(DEBUG, use_load_files=False):
    if use_load_files:
        x_train = np.load(os.path.join('./input', X_TRAIN_ARRAY))
        x_test = np.load(os.path.join('./input', X_TEST_ARRAY))
        y_train = np.load(os.path.join('./input', Y_TRAIN_ARRAY))
        features = np.load(os.path.join('./input', FEATURES_ARRAY))
        test_features = np.load(os.path.join('./input', TEST_FEATURES_ARRAY))
        word_index = np.load(os.path.join('./input', WORD_INDEX_ARRAY)).item()

        return x_train, x_test, y_train, features, test_features, word_index

    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
    if DEBUG:
        train_df = pd.concat([
            train_df[train_df['target'] == 0].head(100), train_df[train_df['target'] == 1].head(100)
        ]).sample(frac=1).reset_index(drop=True)
        test_df = test_df.head(100)
        train_df.to_pickle('train_df_for_debug.pkl')
        test_df.to_pickle('test_df_for_debug.pkl')

    train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
    tqdm.pandas(desc='Progress clean_text')
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
    tqdm.pandas(desc='Progress clean_numbers')
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    x_train = train_df["question_text"].fillna("_##_").values
    y_train = train_df['target'].values

    test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))
    x_test = test_df["question_text"].fillna("_##_").values

    features = extract_features(train_df)
    test_features = extract_features(test_df)
    ss = StandardScaler()
    ss.fit(np.vstack((features, test_features)))
    features = ss.transform(features)
    test_features = ss.transform(test_features)

    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(x_train))
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=MAXLEN)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen=MAXLEN)

    np.random.seed(SEED)
    trn_idx = np.random.permutation(len(x_train))
    x_train = x_train[trn_idx]
    y_train = y_train[trn_idx]

    np.save(os.path.join('./input', X_TRAIN_ARRAY), x_train)
    np.save(os.path.join('./input', X_TEST_ARRAY), x_test)
    np.save(os.path.join('./input', Y_TRAIN_ARRAY), y_train)
    np.save(os.path.join('./input', FEATURES_ARRAY), features)
    np.save(os.path.join('./input', TEST_FEATURES_ARRAY), test_features)
    np.save(os.path.join('./input', WORD_INDEX_ARRAY), tokenizer.word_index)

    return x_train, x_test, y_train, features, test_features, tokenizer.word_index


def clean_text(x):
    puncts = [
        ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
        '+', '\\', '•', '~', '@', '£',
        '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
        '½', 'à', '…',
        '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
        '▓', '—', '‹', '─',
        '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
        'Ã', '⋅', '‘', '∞',
        '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
        '≤', '‡', '√',
    ]
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def extract_features(df):
    df['question_text'] = df['question_text'].progress_apply(lambda x: str(x))
    df['total_length'] = df['question_text'].progress_apply(len)
    df['capitals'] = df['question_text'].progress_apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.progress_apply(lambda row: float(row['capitals']) / float(row['total_length']), axis=1)

    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].progress_apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']

    return df[['caps_vs_length', 'words_vs_unique']].fillna(0)


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x
