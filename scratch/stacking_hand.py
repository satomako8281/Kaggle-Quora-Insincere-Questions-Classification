import sys
import logging
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score

INPUT_PATH = './input'

def bestThresshold(y_train, train_preds):
    tmp = [0, 0, 0]  # idx, cur, max
    delta = 0
    for tmp[0] in np.arange(0.01, 1, 0.01):
        tmp[1] = f1_score(y_train, np.array(train_preds) > tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    return delta, tmp[2]

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

va_preds_merged = (
    0.21 * va_preds[:, 0] +
    0.30 * va_preds[:, 1] +
    0.11 * va_preds[:, 2] +
    0.36 * va_preds[:, 3]
)

print(va_preds_merged.shape)
print(y_va.shape)
delta, f1_score = bestThresshold(y_va, va_preds_merged)
print('[Model mean] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f1_score))

df_test = pd.read_csv(os.path.join(INPUT_PATH, "test.csv"))
submission = df_test[['qid']].copy()
submission['prediction'] = (te_preds_merged > delta).astype(int)
submission.to_csv('submission.csv', index=False)
