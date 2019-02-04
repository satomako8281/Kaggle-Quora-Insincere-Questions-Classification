import sys
import logging
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import f1_score

INPUT_PATH = './input'




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

f_score = f1_score(y_te[:len(y_te)/4], np.array(te_preds_merged[:len(te_preds_merged)/4]) > delta)
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))
delta, f_score = bestThresshold(y_te[:len(y_te)/4], te_preds_merged[:len(y_te)/4])
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))

f_score = f1_score(y_te[len(y_te)/4:len(y_te)/4 * 2], np.array(te_preds_merged[len(y_te)/4:len(y_te)/4 * 2]) > delta)
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))
delta, f_score = bestThresshold(y_te[len(y_te)/4:len(y_te)/4 * 2], te_preds_merged[len(y_te)/4:len(y_te)/4 * 2])
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))

f_score = f1_score(y_te[len(y_te)/4 * 2:len(y_te)/4 * 3], np.array(te_preds_merged[len(y_te)/4 * 2:len(y_te)/4 * 3]) > delta)
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))
delta, f_score = bestThresshold(y_te[len(y_te)/4 * 2:len(y_te)/4 * 3], te_preds_merged[len(y_te)/4 * 2:len(y_te)/4 * 3])
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))

f_score = f1_score(y_te[len(y_te)/4 * 3:len(y_te)/4 * 4], np.array(te_preds_merged[len(y_te)/4 * 3:len(y_te)/4 * 4]) > delta)
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))
delta, f_score = bestThresshold(y_te[len(y_te)/4 * 3:len(y_te)/4 * 4], te_preds_merged[len(y_te)/4 * 3:len(y_te)/4 * 4])
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))

