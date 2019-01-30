import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import Lasso

def bestThresshold(y_train, train_preds):
    tmp = [0, 0, 0]  # idx, cur, max
    delta = 0
    for tmp[0] in np.arange(0.01, 0.501, 0.01):
        tmp[1] = f1_score(y_train, np.array(train_preds) > tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]
    return delta, tmp[2]

def merge_predictions(X_tr, y_tr, X_te=None, est=None, verbose=True):
    if est is None:
        est = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                    positive=True, random_state=9999, selection='random')
    est.fit(X_tr, y_tr)
    if hasattr(est, 'intercept_') and verbose:
        print('merge_predictions = \n{:+.4f}\n{}'.format(
            est.intercept_,
            '\n'.join(
                '{:+.4f} * {}'.format(coef, i) for i, coef in zip(range(X_tr.shape[0]), est.coef_)
            )
        ))
    return (est.predict(X_tr),
            est.predict(X_te) if X_te is not None else None)

va_preds = []
te_preds = []
va_preds.append(joblib.load("valid_pred_tfidf.pkl"))
va_preds.append(joblib.load("valid_pred_mercari.pkl"))
va_preds.append(joblib.load("valid_pred_bilstm.pkl"))
va_preds.append(joblib.load("valid_pred_pytorch.pkl"))
te_preds.append(joblib.load("test_pred_tfidf.pkl"))
te_preds.append(joblib.load("test_pred_mercari.pkl"))
te_preds.append(joblib.load("test_pred_bilstm.pkl"))
te_preds.append(joblib.load("test_pred_pytorch.pkl"))
va_preds = np.hstack(va_preds)
te_preds = np.hstack(te_preds)
y_va = joblib.load("./input/y_va.pkl")
print(y_va)
va_preds_merged, te_preds_merged = merge_predictions(X_tr=va_preds, y_tr=y_va, X_te=te_preds)
delta, f1_score = bestThresshold(y_va, va_preds_merged)
print('[Model mean] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f1_score))

df_test = pd.read_csv("../input/test.csv")
submission = df_test[['qid']].copy()
submission['prediction'] = (te_preds_merged > delta).astype(int)
submission.to_csv('submission.csv', index=False)
