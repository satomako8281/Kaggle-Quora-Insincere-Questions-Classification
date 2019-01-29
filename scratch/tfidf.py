from sklearn.externals import joblib
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

TOKENIZER = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
SEED = 1029


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
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv('../input/test.csv')
    sub = pd.read_csv('../input/sample_submission.csv')
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
splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(train, y))
(train_idx, valid_idx) = splits[0]
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

