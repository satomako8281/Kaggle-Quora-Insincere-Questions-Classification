import gc
from sklearn.externals import joblib
import os
import re
import string
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

TOKENIZER = re.compile('([{}“”¨«»®´·º½¾¿¡§£₤‘’])'.format(string.punctuation))
SEED = 1029
INPUT_PATH = './input'


@contextmanager
def timer(task_name="timer"):
    # a timer cm from https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    print("----{} started".format(task_name))
    t0 = time.time()
    yield
    print("----{} done in {:.0f} seconds".format(task_name, time.time() - t0))


def tokenize(s):
    return TOKENIZER.sub(r' \1 ', s).split()


class NBTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1):
        self.r = None
        self.alpha = alpha

    def fit(self, X, y):
        # store smoothed log count ratio
        p = self.alpha + X[y==1].sum(0)
        q = self.alpha + X[y==0].sum(0)
        self.r = csr_matrix(np.log(
            (p / (self.alpha + (y==1).sum())) /
            (q / (self.alpha + (y==0).sum()))
        ))
        return self

    def transform(self, X, y=None):
        return X.multiply(self.r)


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

with timer("reading_data"):
    train = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"))
    y = train.target.values
    splits = list(StratifiedKFold(n_splits=20, shuffle=True, random_state=SEED).split(train, y))
    (train_idx, test_idx) = splits[5]
    test = train.loc[test_idx.tolist(), :]
    train = train.loc[train_idx.tolist(), :]
    y = train.target.values
    # test = pd.read_csv(os.path.join(INPUT_PATH, 'test.csv'))

    joblib.dump(train, 'train.pkl', compress=3)
    joblib.dump(test, 'valid_for_emsemble.pkl', compress=3)
    joblib.dump(test['target'], 'valid_for_emsemble_label.pkl', compress=3)

    sub = pd.read_csv(os.path.join(INPUT_PATH, 'sample_submission.csv'))
    y = train.target.values

with timer("getting ngram tfidf"):
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1,4),
        tokenizer=tokenize,
        min_df=3,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    ).fit(pd.concat([train['question_text'], test['question_text']]))
    X = tfidf_vectorizer.transform(train['question_text'])
    X_test = tfidf_vectorizer.transform(test['question_text'])

with timer("get Naive Bayes feature"):
    nb_transformer = NBTransformer(alpha=1).fit(X, y)
    X_nb = nb_transformer.transform(X)
    X_test_nb = nb_transformer.transform(X_test)

models = []
train_meta = np.zeros(y.shape)
test_meta = np.zeros(X_test.shape[0])
splits = list(StratifiedKFold(n_splits=20, shuffle=True, random_state=SEED).split(train, y))
(train_idx, valid_idx) = splits[5]
joblib.dump(train_idx, 'train_idx.pkl', compress=3)
joblib.dump(valid_idx, 'valid_idx.pkl', compress=3)
X_train = X_nb[train_idx]
y_train = y[train_idx]
X_val = X_nb[valid_idx]
y_val = y[valid_idx]
joblib.dump(y_val, 'y_val.pkl', compress=3)
model = LogisticRegression(solver='lbfgs', dual=False, class_weight='balanced', C=0.5, max_iter=40)
model.fit(X_train, y_train)
models.append(model)
valid_pred = model.predict_proba(X_val)
joblib.dump(valid_pred[:, 1], 'valid_pred_tfidf.pkl', compress=3)
train_meta[valid_idx] = valid_pred[:,1]
test_meta += model.predict_proba(X_test_nb)[:,1] / 1
joblib.dump(test_meta, 'test_pred_tfidf.pkl', compress=3)

search_result = threshold_search(y_val, valid_pred[:, 1])
print(search_result)

del BaseEstimator, INPUT_PATH, LogisticRegression, NBTransformer, SEED, StratifiedKFold, TOKENIZER, TfidfVectorizer, TransformerMixin, X, X_nb, X_test, X_test_nb, X_train, X_val, contextmanager, csr_matrix, f1_score, gc, model, models, nb_transformer, np, os, pd, re, search_result, splits, string, tfidf_vectorizer, threshold_search, time, timer, tokenize, train_test_split, valid_pred, y
import gc
gc.collect()


import time
from sklearn.externals import joblib
import random
import pandas as pd
import numpy as np
import gc
import os
import re
import torch
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from collections import Counter

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from sklearn.metrics import f1_score
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer

embed_size = 300 # how big is each word vector
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use
batch_size = 512 # how many samples to process at once
n_epochs = 3 # how many times to iterate over all samples

INPUT_PATH = './input'
SEED = 1029
def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

## FUNCTIONS TAKEN FROM https://www.kaggle.com/gmhost/gru-capsule

def load_glove(word_index):
    EMBEDDING_FILE = os.path.join(INPUT_PATH ,'embeddings/glove.840B.300d/glove.840B.300d.txt')
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    # Why random embedding for OOV? what if use mean?
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    #embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, embed_size)) # std 0
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_fasttext(word_index):
    EMBEDDING_FILE = os.path.join(INPUT_PATH, 'embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    #embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = os.path.join(INPUT_PATH, 'embeddings/paragram_300_sl999/paragram_300_sl999.txt')
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = -0.0053247833,0.49346462
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    #embedding_matrix = np.random.normal(emb_mean, 0, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

# df_train = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"))
# df_test = pd.read_csv(os.path.join(INPUT_PATH, "test.csv"))
df_train = joblib.load('train.pkl')
df_test = joblib.load('valid_for_emsemble.pkl')
df = pd.concat([df_train, df_test], sort=True)

def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab
vocab = build_vocab(df['question_text'])


def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, ' {} '.format(p))

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:
            embedding[word.lower()] = embedding[word]
            count += 1
    print("Added {} words to embedding".format(count))


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', ]


def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, ' {} '.format(punct))
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


mispell_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
                "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center',
                'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2',
                'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best',
                'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist',
                'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'}


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


from sklearn.preprocessing import StandardScaler


def add_features(df):
    df['question_text'] = df['question_text'].progress_apply(lambda x: str(x))
    df['total_length'] = df['question_text'].progress_apply(len)
    df['capitals'] = df['question_text'].progress_apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.progress_apply(lambda row: float(row['capitals']) / float(row['total_length']),
                                             axis=1)
    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].progress_apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']

    return df


def load_and_prec():
    train_df = joblib.load('train.pkl')
    test_df = joblib.load('valid_for_emsemble.pkl')
    # train_df = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"))
    # test_df = pd.read_csv(os.path.join(INPUT_PATH, "test.csv"))
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    # lower
    train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())

    # Clean the text
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

    # Clean numbers
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))

    # Clean speelings
    # train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    # test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))

    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values

    ###################### Add Features ###############################
    #  https://github.com/wongchunghang/toxic-comment-challenge-lstm/blob/master/toxic_comment_9872_model.ipynb
    train = add_features(train_df)
    test = add_features(test_df)

    features = train[['caps_vs_length', 'words_vs_unique']].fillna(0)
    test_features = test[['caps_vs_length', 'words_vs_unique']].fillna(0)

    ss = StandardScaler()
    ss.fit(np.vstack((features, test_features)))
    features = ss.transform(features)
    test_features = ss.transform(test_features)
    ###########################################################################

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values

    #     # Splitting to training and a final test set
    #     train_X, x_test_f, train_y, y_test_f = train_test_split(list(zip(train_X,features)), train_y, test_size=0.2, random_state=SEED)
    #     train_X, features = zip(*train_X)
    #     x_test_f, features_t = zip(*x_test_f)

    # shuffling the data
    # np.random.seed(SEED)
    # trn_idx = np.random.permutation(len(train_X))
    #
    # train_X = train_X[trn_idx]
    # train_y = train_y[trn_idx]
    # features = features[trn_idx]

    return train_X, test_X, train_y, features, test_features, tokenizer.word_index
#     return train_X, test_X, train_y, x_test_f,y_test_f,features, test_features, features_t, tokenizer.word_index
#     return train_X, test_X, train_y, tokenizer.word_index

x_train, x_test, y_train, features, test_features, word_index = load_and_prec()

# missing entries in the embedding are set using np.random.normal so we have to seed here too
seed_everything()

# fasttext_embeddings = load_fasttext(word_index)

glove_embeddings = load_glove(word_index)
paragram_embeddings = load_para(word_index)
embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
joblib.dump(embedding_matrix, 'embedding_matrix.pkl', compress=3)
del glove_embeddings, paragram_embeddings

# embedding_matrix = joblib.load('embedding_matrix.pkl')

# vocab = build_vocab(df['question_text'])
# add_lower(embedding_matrix, vocab)
# del glove_embeddings, paragram_embeddings, fasttext_embeddings
gc.collect()
# code inspired from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

def bestThresshold(y_train,train_preds):
    tmp = [0,0,0] # idx, cur, max
    delta = 0
    for tmp[0] in tqdm(np.arange(0.1, 0.501, 0.01)):
        tmp[1] = f1_score(y_train, np.array(train_preds)>tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
    return delta

train_idx = joblib.load('train_idx.pkl')
valid_idx = joblib.load('valid_idx.pkl')

import torch as t
import torch.nn as nn
import torch.nn.functional as F

embedding_dim = 300
embedding_path = '../save/embedding_matrix.npy'  # or False, not use pre-trained-matrix
use_pretrained_embedding = True

hidden_size = 60
gru_len = hidden_size

Routings = 4 #5
Num_capsule = 5
Dim_capsule = 5#16
dropout_p = 0.25
rate_drop_dense = 0.28
LR = 0.001
T_epsilon = 1e-7
num_classes = 30


class Embed_Layer(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(Embed_Layer, self).__init__()
        self.encoder = nn.Embedding(vocab_size + 1, embedding_dim)
        if use_pretrained_embedding:
            # self.encoder.weight.data.copy_(t.from_numpy(np.load(embedding_path))) # 方法一，加载np.save的npy文件
            self.encoder.weight.data.copy_(t.from_numpy(embedding_matrix))  # 方法二

    def forward(self, x, dropout_p=0.25):
        return nn.Dropout(p=dropout_p)(self.encoder(x))


class GRU_Layer(nn.Module):
    def __init__(self):
        super(GRU_Layer, self).__init__()
        self.gru = nn.GRU(input_size=300,
                          hidden_size=gru_len,
                          bidirectional=True)
        '''
        自己修改GRU里面的激活函数及加dropout和recurrent_dropout
        如果要使用，把rnn_revised import进来，但好像是使用cpu跑的，比较慢
       '''
        # # if you uncomment /*from rnn_revised import * */, uncomment following code aswell
        # self.gru = RNNHardSigmoid('GRU', input_size=300,
        #                           hidden_size=gru_len,
        #                           bidirectional=True)

    # 这步很关键，需要像keras一样用glorot_uniform和orthogonal_uniform初始化参数
    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        return self.gru(x)


# core caps_layer with squash func
class Caps_Layer(nn.Module):
    def __init__(self, input_dim_capsule=gru_len * 2, num_capsule=Num_capsule, dim_capsule=Dim_capsule, \
                 routings=Routings, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Caps_Layer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size  # 暂时没用到
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(t.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                t.randn(BATCH_SIZE, input_dim_capsule, self.num_capsule * self.dim_capsule))  # 64即batch_size

    def forward(self, x):

        if self.share_weights:
            u_hat_vecs = t.matmul(x, self.W)
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  # 转成(batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = t.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(t.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = t.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = t.sqrt(s_squared_norm + T_epsilon)
        return x / scale

class Capsule_Main(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None):
        super(Capsule_Main, self).__init__()
        self.embed_layer = Embed_Layer(embedding_matrix, vocab_size)
        self.gru_layer = GRU_Layer()
        # 【重要】初始化GRU权重操作，这一步非常关键，acc上升到0.98，如果用默认的uniform初始化则acc一直在0.5左右
        self.gru_layer.init_weights()
        self.caps_layer = Caps_Layer()
        self.dense_layer = Dense_Layer()

    def forward(self, content):
        content1 = self.embed_layer(content)
        content2, _ = self.gru_layer(
            content1)  # 这个输出是个tuple，一个output(seq_len, batch_size, num_directions * hidden_size)，一个hn
        content3 = self.caps_layer(content2)
        output = self.dense_layer(content3)
        return output

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        fc_layer = 16
        fc_layer1 = 16

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        self.gru_attention = Attention(hidden_size * 2, maxlen)
        self.bn = nn.BatchNorm1d(16, momentum=0.5)
        self.linear = nn.Linear(hidden_size*8+3, fc_layer1) #643:80 - 483:60 - 323:40
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(fc_layer**2,fc_layer)
        self.out = nn.Linear(fc_layer, 1)
        self.lincaps = nn.Linear(Num_capsule * Dim_capsule, 1)
        self.caps_layer = Caps_Layer()

    def forward(self, x):

        #         Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)

        h_embedding = self.embedding(x[0])
        h_embedding = torch.squeeze(
            self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        ##Capsule Layer
        content3 = self.caps_layer(h_gru)
        content3 = self.dropout(content3)
        batch_size = content3.size(0)
        content3 = content3.view(batch_size, -1)
        content3 = self.relu(self.lincaps(content3))

        ##Attention Layer
        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        # global average pooling
        avg_pool = torch.mean(h_gru, 1)
        # global max pooling
        max_pool, _ = torch.max(h_gru, 1)

        f = torch.tensor(x[1], dtype=torch.float).cuda()

        #[512,160]
        conc = torch.cat((h_lstm_atten, h_gru_atten,content3, avg_pool, max_pool,f), 1)
        conc = self.relu(self.linear(conc))
        conc = self.bn(conc)
        conc = self.dropout(conc)

        out = self.out(conc)

        return out


class MyDataset(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return data, target, index
    def __len__(self):
        return len(self.dataset)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# matrix for the out-of-fold predictions
train_preds = np.zeros((len(x_train)))
# matrix for the predictions on the test set
test_preds = np.zeros((len(df_test)))

# always call this before training for deterministic results
seed_everything()

# x_test_cuda_f = torch.tensor(x_test_f, dtype=torch.long).cuda()
# test_f = torch.utils.data.TensorDataset(x_test_cuda_f)
# test_loader_f = torch.utils.data.DataLoader(test_f, batch_size=batch_size, shuffle=False)


x_test_cuda = torch.tensor(x_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

avg_losses_f = []
avg_val_losses_f = []

# x_train = np.array(x_train)
# y_train = np.array(y_train)
# features = np.array(features)

x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.long).cuda()
y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()

kfold_X_features = features[train_idx.astype(int)]
kfold_X_valid_features = features[valid_idx.astype(int)]
x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.long).cuda()
y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()

model = NeuralNet()
model.cuda()
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
step_size = 300
base_lr, max_lr = 0.001, 0.003
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=max_lr)

################################################################################################
scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                     step_size=step_size, mode='exp_range',
                     gamma=0.99994)
###############################################################################################

train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

train = MyDataset(train)
valid = MyDataset(valid)

##No need to shuffle the data again here. Shuffling happens when splitting for kfolds.
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

for epoch in range(n_epochs):
    # set train mode of the model. This enables operations which are only applied during training like dropout
    start_time = time.time()
    model.train()

    avg_loss = 0.
    for i, (x_batch, y_batch, index) in enumerate(train_loader):
        # Forward pass: compute predicted y by passing x to the model.
        ################################################################################################
        f = kfold_X_features[index]
        y_pred = model([x_batch,f])
        ################################################################################################

        ################################################################################################

        if scheduler:
            scheduler.batch_step()
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

    model.eval()
    # valid_preds_fold = np.zeros((x_val_fold.size(0)))
    valid_preds_fold = np.zeros((len(valid_idx)))
    test_preds_fold = np.zeros((len(df_test)))
    avg_val_loss = 0.
    for i, (x_batch, y_batch, index) in enumerate(valid_loader):
        f = kfold_X_valid_features[index]
        y_pred = model([x_batch,f]).detach()

        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    elapsed_time = time.time() - start_time
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
avg_losses_f.append(avg_loss)
avg_val_losses_f.append(avg_val_loss)
# predict all samples in the test set batch per batch
for i, (x_batch,) in enumerate(test_loader):
    f = test_features[i * batch_size:(i+1) * batch_size]
    y_pred = model([x_batch,f]).detach()

    test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

train_preds[valid_idx] = valid_preds_fold
test_preds += test_preds_fold / 1

joblib.dump(valid_preds_fold, 'valid_pred_bilstm.pkl', compress=3)
joblib.dump(test_preds, 'test_pred_bilstm.pkl', compress=3)

print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f),np.average(avg_val_losses_f)))
print(valid_preds_fold.shape)
delta = bestThresshold(y_train[valid_idx.astype(int)],valid_preds_fold)

# del X_test, X_train, df, df_test, df_train, features, full_text, mispellings_re, sub, test, test_features, test_tokenized
# del train, train_tokenized, vocab, word_index, x_test, x_train, y_train
# gc.collect()
del df, df_test, df_train, features, kfold_X_features, kfold_X_valid_features, mispellings_re, test_features
del test_preds, test_preds_fold, train_idx, train_preds, valid_idx, valid_preds_fold, vocab, word_index, x_test, x_train, y_train
import gc
gc.collect()



import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import time
pd.set_option('max_colwidth',400)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.utils.data
import random
import warnings
warnings.filterwarnings("ignore", message="F-score is ill-defined and being set to 0.0 due to no predicted samples.")
import re

n_epochs = 3
embed_size = 300

INPUT_PATH = './input'

# train = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"))
# test = pd.read_csv(os.path.join(INPUT_PATH, "test.csv"))
train = joblib.load('train.pkl')
test = joblib.load('valid_for_emsemble.pkl')
sub = pd.read_csv(os.path.join(INPUT_PATH, 'sample_submission.csv'))

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, ' {} '.format(punct))
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"aren't" : "are not",
                "can't" : "cannot",
                "couldn't" : "could not",
                "didn't" : "did not",
                "doesn't" : "does not",
                "don't" : "do not",
                "hadn't" : "had not",
                "hasn't" : "has not",
                "haven't" : "have not",
                "he'd" : "he would",
                "he'll" : "he will",
                "he's" : "he is",
                "i'd" : "I would",
                "i'd" : "I had",
                "i'll" : "I will",
                "i'm" : "I am",
                "isn't" : "is not",
                "it's" : "it is",
                "it'll":"it will",
                "i've" : "I have",
                "let's" : "let us",
                "mightn't" : "might not",
                "mustn't" : "must not",
                "shan't" : "shall not",
                "she'd" : "she would",
                "she'll" : "she will",
                "she's" : "she is",
                "shouldn't" : "should not",
                "that's" : "that is",
                "there's" : "there is",
                "they'd" : "they would",
                "they'll" : "they will",
                "they're" : "they are",
                "they've" : "they have",
                "we'd" : "we would",
                "we're" : "we are",
                "weren't" : "were not",
                "we've" : "we have",
                "what'll" : "what will",
                "what're" : "what are",
                "what's" : "what is",
                "what've" : "what have",
                "where's" : "where is",
                "who'd" : "who would",
                "who'll" : "who will",
                "who're" : "who are",
                "who's" : "who is",
                "who've" : "who have",
                "won't" : "will not",
                "wouldn't" : "would not",
                "you'd" : "you would",
                "you'll" : "you will",
                "you're" : "you are",
                "you've" : "you have",
                "'re": " are",
                "wasn't": "was not",
                "we'll":" will",
                "didn't": "did not",
                "tryin'":"trying"}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

# Clean the text
train["question_text"] = train["question_text"].apply(lambda x: clean_text(x.lower()))
test["question_text"] = test["question_text"].apply(lambda x: clean_text(x.lower()))

# Clean numbers
train["question_text"] = train["question_text"].apply(lambda x: clean_numbers(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_numbers(x))

# Clean speelings
train["question_text"] = train["question_text"].apply(lambda x: replace_typical_misspell(x))
test["question_text"] = test["question_text"].apply(lambda x: replace_typical_misspell(x))

max_features = 120000
tk = Tokenizer(lower = True, filters='', num_words=max_features)
full_text = list(train['question_text'].values) + list(test['question_text'].values)
tk.fit_on_texts(full_text)

train_tokenized = tk.texts_to_sequences(train['question_text'].fillna('missing'))
test_tokenized = tk.texts_to_sequences(test['question_text'].fillna('missing'))

max_len = 72
maxlen = 72
X_train = pad_sequences(train_tokenized, maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)

y_train = train['target'].values
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

########
embed_size = 300
embedding_path = os.path.join(INPUT_PATH, "embeddings/glove.840B.300d/glove.840B.300d.txt")
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore'))
# all_embs = np.stack(embedding_index.values())
# emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std = -0.005838499, 0.48782197
word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

embedding_path = os.path.join(INPUT_PATH, "embeddings/paragram_300_sl999/paragram_300_sl999.txt")
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding='utf-8', errors='ignore') if len(o)>100)
# all_embs = np.stack(embedding_index.values())
# emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std = -0.0053247833, 0.49346462
embedding_matrix1 = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix1[i] = embedding_vector

embedding_matrix = np.mean([embedding_matrix, embedding_matrix1], axis=0)
del embedding_matrix1

########

# embedding_matrix = joblib.load('embedding_matrix.pkl')
train_idx = joblib.load('train_idx.pkl')
valid_idx = joblib.load('valid_idx.pkl')


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        hidden_size = 128

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size*2, maxlen)
        self.gru_attention = Attention(hidden_size*2, maxlen)

        self.linear = nn.Linear(1024, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out


x_test_cuda = torch.tensor(X_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
batch_size = 512
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)


def train_model(model, x_train, y_train, x_val, y_val, validate=True):
    optimizer = torch.optim.Adam(model.parameters())

    # scheduler = CosineAnnealingLR(optimizer, T_max=5)
    # scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_val, y_val)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    best_score = -np.inf

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for x_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model(x_batch)


            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()

        valid_preds = np.zeros((x_val_fold.size(0)))

        if validate:
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch).detach()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
            search_result = threshold_search(y_val.cpu().numpy(), valid_preds)

            val_f1, val_threshold = search_result['f1'], search_result['threshold']
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} best_t={:.2f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_f1, val_threshold, elapsed_time))
        else:
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, elapsed_time))

    valid_preds = np.zeros((x_val_fold.size(0)))

    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()

        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    print('Validation loss: ', avg_val_loss)

    test_preds = np.zeros((len(test_loader.dataset)))

    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
    # scheduler.step()

    return valid_preds, test_preds#, test_preds_local



seed=1029

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


train_preds = np.zeros(len(train))
test_preds = np.zeros((len(test)))
from tqdm import tqdm
from sklearn.metrics import f1_score

x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()
y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()
y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()

train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

seed_everything(1030)
model = NeuralNet()
model.cuda()

valid_preds_fold, test_preds_fold = train_model(
    model,
    x_train_fold,
    y_train_fold,
    x_val_fold,
    y_val_fold,
    validate=True
)

train_preds[valid_idx] = valid_preds_fold
test_preds += test_preds_fold / 1
# test_preds_local[:, i] = test_preds_local_fold
joblib.dump(valid_preds_fold, 'valid_pred_pytorch.pkl', compress=3)
joblib.dump(test_preds, 'test_pred_pytorch.pkl', compress=3)

search_result = threshold_search(y_train[valid_idx.astype(int)], valid_preds_fold)
print(search_result)
# sub['prediction'] = test_preds.mean(1) > search_result['threshold']
# sub.to_csv("submission.csv", index=False)

# del df, df_test, df_train, embedding_matrix, features, mispellings_re, test_features
# del test_preds, train_idx, train_preds, valid_idx, vocab, word_index, x_test, x_train
# del y_train
# gc.collect()
del X_test, X_train, embedding_index, embedding_matrix, full_text, test_preds, test_preds_fold, test_tokenized, train_idx, train_preds, train_tokenized, valid_idx, valid_preds_fold, word_index
import gc
gc.collect()


import os
import gc
from contextlib import contextmanager
from sklearn.externals import joblib
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

INPUT_PATH = './input'
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {:.0f} s'.format(name, time.time() - t0))

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

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
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')
        selected_cols = []
        for col in x.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)
        self._check_if_all_columns_present(x)
        if len(selected_cols) == 1 and self.return_vector:
            return list(x[selected_cols[0]])
        else:
            return x[selected_cols]

def fit_predict(xs) -> np.ndarray:
    X_train, y_train, train_ids, dev_ids = xs
    config = tf.ConfigProto()
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        K.backend.set_session(sess)
        model_in = K.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = K.layers.Dense(512, activation='relu')(model_in)
        out = K.layers.Concatenate()([K.layers.Dense(64, activation='tanh')(out),
                                      K.layers.Dense(64, activation='relu')(out)])
        out2= K.layers.Concatenate()([K.layers.Dense(32, activation='tanh')(out),
                                      K.layers.Dense(32, activation='relu')(out)])
        out = K.layers.Concatenate()([K.layers.Dropout(0.2)(out), out2])

        out = K.layers.Add()([K.layers.Dense(1, activation='linear')(out),
                              K.layers.Dense(1, activation='relu')(out),
                              ])
        out = K.layers.Add()([K.layers.Dense(1, activation='linear')(out),
                              K.layers.Dense(1, activation='relu')(out),
                              ])
        out = K.layers.Dense(1)(out)
        model = K.Model(model_in, out)
        model.compile(loss='logcosh', optimizer=K.optimizers.Adam(lr=3e-3), metrics=['accuracy'])
        for i in range(3):
            with timer('epoch {}'.format(i + 1)):
                model.fit(x=X_train[train_ids,:], y=y_train[train_ids], batch_size=2**(11 + i), epochs=1, verbose=0)
        preds = np.zeros(X_train.shape[0])
        preds[dev_ids] = model.predict(X_train[dev_ids, :])[:, 0]
        return preds

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, ' {} '.format(punct))
    return x

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
                "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't":
                    "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will",
                "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am",
                "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've":
                    "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've":
                    "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us",
                "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've":
                    "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not",
                "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've":
                    "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
                "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd":
                    "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're":
                    "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've":
                    "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're":
                    "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've":
                    "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll":
                    "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's":
                    "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've":
                    "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're":
                    "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
                "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour':
                    'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling':
                    'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation':
                    'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura':
                    'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do',
                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo':
                    'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation':
                    'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
                'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18':
                    '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst":
                    'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization':
                    'demonetization', 'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

TOKENIZER = re.compile('([{}“”¨«»®´·º½¾¿¡§£₤‘’])'.format(string.punctuation))

def tokenize(s):
    return TOKENIZER.sub(r' \1 ', s).split()

vectorizer = make_union(
    on_field('question_text', TfidfVectorizer(max_features=13000, token_pattern='\w+', strip_accents='unicode',
                                              tokenizer=tokenize, sublinear_tf=True)),
    on_field('question_text', TfidfVectorizer(ngram_range=(3, 3), analyzer='char', min_df=25)),
    make_pipeline(PandasSelector(columns=['num_words', 'num_singletons', 'caps_vs_length',
                                          ], return_vector=False), MaxAbsScaler()),
)

with timer('process train'):
    # df_train = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"))
    # df_test = pd.read_csv(os.path.join(INPUT_PATH, "test.csv"))
    df_train = joblib.load('train.pkl')
    df_test = joblib.load('valid_for_emsemble.pkl')
    train_count= len(df_train)
    test_count= len(df_test)
    test_qids= df_test['qid']
    df_train= pd.concat([df_train, df_test], sort=False).reset_index(drop=True)
    print(df_train.shape)

    df_train["question_text"] = df_train["question_text"].str.lower() \
        .map(clean_text).map(clean_numbers).map(replace_typical_misspell)

    wft = {}
    for line in df_train['question_text']:
        for word in line.lower().split():
            wft[word]= wft.get(word, 0)+1
    df_train["num_words"] = [len(str(x).split()) for x in df_train["question_text"]]
    df_train["num_singletons"] = [len([w for w in str(x).lower().split() if wft.get(word, 0)<=1]) for x in
                                  df_train['question_text']]
    df_train['capitals'] = [sum(1 for c in x if c.isupper()) for x in df_train['question_text']]
    df_train['caps_vs_length'] = df_train['capitals']/df_train['question_text'].str.len()
    X_train = vectorizer.fit_transform(df_train, df_train['target']).astype(np.float32)

    y_train = df_train['target'][:train_count].values
    del (df_train)
    gc.collect()


n_reps = 8

y_pred = np.zeros(train_count + test_count)
with timer('run kfold'):
    datasets= []
    train_ids = joblib.load('train_idx.pkl')
    dev_ids = joblib.load('valid_idx.pkl')
    dev_ids= list(dev_ids)+list(range(train_count, train_count+test_count))
    for rep in range(n_reps):
        datasets.append([X_train, y_train, train_ids, dev_ids])
    ###with ThreadPool(processes=2) as pool:
    with ThreadPool(processes=1) as pool:
        y_pred+= np.sum(pool.map(fit_predict, datasets), axis=0) / n_reps
y_pred[train_count:]/= 1
scores= []
thresholds= np.arange(0.1, 0.501, 0.01)
dev_ids = joblib.load('valid_idx.pkl')
for thresh in thresholds:
    thresh = np.round(thresh, 2)
    scores.append(f1_score(y_train[dev_ids], (y_pred[:train_count][dev_ids] > thresh).astype(int)))
    print("F1 score at threshold {0} is {1}".format(thresh, scores[-1]))
joblib.dump(y_pred[:train_count][dev_ids], 'valid_pred_mercari.pkl', compress=3)
joblib.dump(y_pred[train_count:], 'test_pred_mercari.pkl', compress=3)

# del dev_ids, df_test, mispellings_re, test_qids, train_ids, wft, y_pred
# gc.collect()

del PandasSelector, Pipeline, StratifiedKFold, TOKENIZER, TfidfVectorizer, ThreadPool, TransformerMixin, X_train, clean_numbers, clean_text, contextmanager, datasets, dev_ids, df_test, fit_predict, itemgetter, joblib, make_pipeline, make_union, mispell_dict, mispellings, mispellings_re, np, on_field, os, pd, pool, puncts, re, rep, replace_typical_misspell, scores, test_count, test_qids, tf, thresh, thresholds, tokenize, train_count, train_ids, vectorizer, y_pred, y_train, wft
import gc
gc.collect()


import sys
import logging
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import f1_score

INPUT_PATH = './input'

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    filepath = os.path.dirname(os.path.join('./output', 'log'))
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    handler = logging.FileHandler(os.path.join(filepath, 'log-{os.getpid()}.txt'), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger



def bestThresshold(y_train, train_preds):
    tmp = [0, 0, 0]  # idx, cur, max
    delta = 0
    for tmp[0] in np.arange(0.00, 0.8, 0.01):
        tmp[1] = f1_score(y_train, np.array(train_preds) > tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    return delta, tmp[2]

def merge_predictions(X_tr, y_tr, X_te=None, est=None, verbose=True):
    if est is None:
        est = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                    positive=True, random_state=1029, selection='random')
    est.fit(X_tr, y_tr)


    return (est.predict(X_tr),
            est.predict(X_te) if X_te is not None else None)

va_preds = []
te_preds = []
va_preds.append(joblib.load("valid_pred_tfidf.pkl")[:, np.newaxis])
va_preds.append(joblib.load("valid_pred_mercari.pkl")[:, np.newaxis])
va_preds.append(joblib.load("valid_pred_bilstm.pkl")[:, np.newaxis])
va_preds.append(joblib.load("valid_pred_pytorch.pkl")[:, np.newaxis])
te_preds.append(joblib.load("test_pred_tfidf.pkl")[:, np.newaxis])
te_preds.append(joblib.load("test_pred_mercari.pkl")[:, np.newaxis])
te_preds.append(joblib.load("test_pred_bilstm.pkl")[:, np.newaxis])
te_preds.append(joblib.load("test_pred_pytorch.pkl")[:, np.newaxis])
va_preds = np.hstack(va_preds)
te_preds = np.hstack(te_preds)
y_va = joblib.load("y_val.pkl")


va_preds_merged, te_preds_merged = merge_predictions(
    X_tr=va_preds, y_tr=y_va, X_te=te_preds
)

delta, f_score = bestThresshold(y_va, va_preds_merged)
print('[validation] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))

y_te = joblib.load('valid_for_emsemble_label.pkl').values
f_score = f1_score(y_te, np.array(te_preds_merged) > delta)
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))

delta, f_score = bestThresshold(y_te, te_preds_merged)
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))

df_test = pd.read_csv(os.path.join(INPUT_PATH, "test.csv"))
submission = df_test[['qid']].copy()
submission['prediction'] = (te_preds_merged > delta).astype(int)
submission.to_csv('submission.csv', index=False)

