from functools import partial
import os
import pickle
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import traceback

from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from tqdm import tqdm

from quora.config import (
    seed_everything, N_SPLITS, USE_CACHED_DATASET, DUMP_DATASET, MAX_FEATURES, SEED,DEBUG_N, INPUT_PATH, logger
)
from quora.embeddings import make_embedding_matrix
from quora.eval import bestThresshold
from quora.misc import send_line_notification, Timer
from quora.quora_io import load_test_iter


tqdm.pandas(desc='Progress')

USE_LOAD_CASED_DATASET = False
USE_LOAD_CASHED_EMBEDDINGS = False
HANDLE_TEST = False


def fit_one(est, X, y, embedding_matrix=None):
    print("fitting y min={} max={}".format(y.min(), y.max()))
    if embedding_matrix is not None:
        est.set_embedding_weight(embedding_matrix)
    return est.fit(X, y)


def predict_one(est, X):
    yhat = est.predict(X)
    print("predicting y min={} max={}".format(yhat.min(), yhat.max()))
    return yhat


def predict_models(X, fitted_models, vectorizer=None, parallel='thread'):
    if vectorizer:
        # TODO: parallelize this
        with Timer('Transforming data'):
            X = vectorizer.transform(X)
    predict_one_ = partial(predict_one, X=X)
    preds = map_parallel(predict_one_, fitted_models, parallel)
    return np.vstack(preds).T


def fit_transform_vectorizer(vectorizer):
    df_tr = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
    y_tr = df_tr['target'].values
    X_tr = vectorizer.fit_transform(df_tr, y_tr)
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(df_tr['question_text']))
    embedding_matrix = make_embedding_matrix(tokenizer.word_index)

    return X_tr, y_tr, vectorizer, embedding_matrix


def fit_models(X_tr, y_tr, models, embedding_matrix, parallel='thread'):
    fit_one_ = partial(fit_one, X=X_tr, y=y_tr, embedding_matrix=embedding_matrix)
    return map_parallel(fit_one_, models, parallel)


def map_parallel(fn, lst, parallel, max_processes=4):
    if parallel == 'thread':
        with ThreadPool(processes=max_processes) as pool:
            return pool.map(fn, lst)
    elif parallel == 'mp':
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=max_processes) as pool:
            return pool.map(fn, lst)
    elif parallel is None:
        return list(map(fn, lst))
    else:
        raise ValueError('unexpected parallel value: {}'.format(parallel))


def predict_models_test_batches(models, vectorizer, parallel='thread'):
    chunk_preds = []
    test_idx = []
    for df in load_test_iter():
        test_idx.append(df['qid'])
        print("Predicting batch {} {}".format(df.test_id.min(), df.test_id.max()))
        chunk_preds.append(predict_models(df, models, vectorizer=vectorizer, parallel=parallel))
    predictions = np.vstack(chunk_preds)
    test_idx = np.concatenate(test_idx)
    return test_idx, predictions


def make_submission(te_idx, preds, save_as):
    submission = pd.DataFrame({
        "test_id": te_idx,
        "price": preds
    }, columns=['test_id', 'price'])
    submission.to_csv(save_as, index=False)


def fit_validate(models, vectorizer, name=None, fit_parallel='thread', predict_parallel='thread'):
    cached_path = os.path.join(INPUT_PATH, 'data_{}.pkl'.format(name))
    if USE_CACHED_DATASET:
        assert name is not None
        with open(cached_path, 'rb') as f:
            X_train, y_train, embedding_matrix = pickle.load(f)
        if DEBUG_N:
            X_train, y_train = X_train[:DEBUG_N], y_train[:DEBUG_N]
    else:
        X_train, y_train, fitted_vectorizer, embedding_matrix = fit_transform_vectorizer(vectorizer)
    if DUMP_DATASET:
        assert name is not None
        with open(cached_path, 'wb') as f:
            pickle.dump((X_train, y_train, embedding_matrix), f)

    # 以下、CVのコード。  # pilot study用に train_test_splitのコードも作りたい
    all_y_va_preds = np.zeros((len(X_train)))
    print(all_y_va_preds.shape)
    all_fitted_models = []
    splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X_train, y_train))
    for i, (tr_idx, va_idx) in enumerate(splits):
        print('Fold {}'.format(i+1))

        print("Train with Training Dataset.")
        X_tr = X_train[tr_idx]
        print('222')
        print(y_train)
        y_tr = y_train[tr_idx, np.newaxis]
        print(y_tr.shape)
        fitted_models = fit_models(X_tr, y_tr, models, embedding_matrix, parallel=fit_parallel)

        print("Predict with Validation Dataset.")
        X_va = X_train[va_idx]
        y_va_preds = predict_models(X_va, fitted_models, parallel=predict_parallel)

        all_fitted_models.append(fitted_models)
        all_y_va_preds[va_idx] = y_va_preds.flatten()

    return fitted_vectorizer, all_fitted_models, y_train, all_y_va_preds


def merge_predictions(X_tr, y_tr, X_te=None, est=None, verbose=True):
    if est is None:
        est = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                    positive=True, random_state=9999, selection='random')
    est.fit(np.log1p(X_tr), np.log1p(y_tr))
    if hasattr(est, 'intercept_') and verbose:
        logger.info('merge_predictions = \n{:+.4f}\n{}'.format(
            est.intercept_,
            '\n'.join('{:+.4f} * {}'.format(coef, i) for i, coef in
                      zip(range(X_tr.shape[0]), est.coef_))))
    return (np.expm1(est.predict(np.log1p(X_tr))),
            np.expm1(est.predict(np.log1p(X_te))) if X_te is not None else None)


def main(name, action, arg_map, fit_parallel='thread', predict_parallel='thread'):
    prefix = lambda r: '{}_{}s'.format(name, r)

    if action in ('1'):
        model_round = int(action)
        send_line_notification(
            "We will start with the following conditions!! \n \
            action_no: {}".format(action)
        )
        model_round = int(action)
        models, vectorizer = arg_map[model_round]

        seed_everything(SEED)
        vectorizer, fitted_models, y_va, y_va_preds = fit_validate(
            models, vectorizer, name=model_round, fit_parallel=fit_parallel, predict_parallel=predict_parallel
        )
        joblib.dump(y_va_preds, "{}_va_preds.pkl".format(prefix(model_round)), compress=3)
        for i in range(y_va_preds.shape[1]):
            delta, f1_score = bestThresshold(y_va, y_va_preds[:, i])
            print('[Model {}] best threshold is {:.4f} with F1 score: {:.4f}'.format(i, delta, f1_score))
        delta, f1_score = bestThresshold(y_va, y_va_preds.mean(axis=1))
        print('[Model mean] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f1_score))
        if HANDLE_TEST:
            test_idx, y_te_preds = predict_models_test_batches(
                fitted_models, vectorizer, parallel=predict_parallel)
            joblib.dump(test_idx, "test_idx.pkl", compress=3)
            joblib.dump(y_te_preds, "{}_te_preds.pkl".format(prefix(model_round)), compress=3)
        joblib.dump(y_va, "y_va.pkl", compress=3)

        send_line_notification(
            "Training is Done!!"
        )

    elif action == "merge":
        va_preds = []
        te_preds = []
        for model_round in ("1"):
            try:
                va_preds.append(joblib.load("{}_va_preds.pkl".format(prefix(model_round))))
                if HANDLE_TEST:
                    te_preds.append(joblib.load("{}_te_preds.pkl".format(prefix(model_round))))
            except Exception as e:
                print('Warning: error loading round {}: {}'.format(model_round, e))
                traceback.print_exc()
        va_preds = np.hstack(va_preds)
        if HANDLE_TEST:
            te_preds = np.hstack(te_preds)
        else:
            te_preds = None
        y_va = joblib.load("y_va.pkl")
        va_preds_merged, te_preds_merged = merge_predictions(X_tr=va_preds, y_tr=y_va, X_te=te_preds)
        if HANDLE_TEST:
            test_idx = joblib.load("test_idx.pkl")
            make_submission(test_idx, te_preds_merged, 'submission_merged.csv')

