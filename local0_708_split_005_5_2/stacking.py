import sys
import logging
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import f1_score
from sklearn import svm

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


logger = setup_custom_logger('quora')

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
        # est = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
        #             positive=True, random_state=1029, selection='random')
        # est = Ridge(alpha=0.0001, max_iter=1000,
        #             random_state=1029)
        est = svm.SVR(kernel='rbf', C=1e3, epsilon=2.0)
    est.fit(X_tr, y_tr)
    # if hasattr(est, 'intercept_') and verbose:
        # logger.info('merge_predictions = \n{:+.4f}\n{}'.format(
        #     est.intercept_,
        #     '\n'.join('{:+.4f} * {}'.format(coef, i) for i, coef in
        #               zip(range(X_tr.shape[0]), est.coef_))))
    # if hasattr(est, 'intercept_') and verbose:

    # Feature Importance
    # fti = est.feature_importances_
    # print('Feature Importances:')
    # for i, feat in enumerate(range(4)):
    #     print('\t{} : {:>.6f}'.format(feat, fti[i]))


    return (est.predict(X_tr),
            est.predict(X_te) if X_te is not None else None)

va_preds = []
te_preds = []
va_preds.append(joblib.load("valid_pred_tfidf.pkl")[:, np.newaxis])
va_preds.append(joblib.load("valid_pred_mercari.pkl")[:, np.newaxis])
va_preds.append(joblib.load("valid_pred_bilstm_0.pkl")[:, np.newaxis])
# va_preds.append(joblib.load("valid_pred_bilstm_1.pkl")[:, np.newaxis])
# va_preds.append(joblib.load("valid_pred_bilstm_2.pkl")[:, np.newaxis])
# va_preds.append((va_pred0+va_pred1+va_pred2)/3)
# va_pred0 = joblib.load("valid_pred_bilstm_0.pkl")[:, np.newaxis]
# va_pred1 = joblib.load("valid_pred_bilstm_1.pkl")[:, np.newaxis]
# va_pred2 = joblib.load("valid_pred_bilstm_2.pkl")[:, np.newaxis]
# va_preds.append((va_pred0+va_pred1+va_pred2)/3)
va_preds.append(joblib.load("valid_pred_pytorch_0.pkl")[:, np.newaxis])
# va_preds.append(joblib.load("valid_pred_pytorch_1.pkl")[:, np.newaxis])
# va_preds.append(joblib.load("valid_pred_pytorch_2.pkl")[:, np.newaxis])
# va_pred0 = joblib.load("valid_pred_pytorch_0.pkl")[:, np.newaxis]
# va_pred1 = joblib.load("valid_pred_pytorch_1.pkl")[:, np.newaxis]
# va_pred2 = joblib.load("valid_pred_pytorch_2.pkl")[:, np.newaxis]
# va_preds.append((va_pred0+va_pred1+va_pred2)/3)
# va_preds.append(joblib.load("valid_pred_pytorch_0.pkl")[:, np.newaxis])
# va_preds.append(joblib.load("valid_pred_pytorch_1.pkl")[:, np.newaxis])

te_preds.append(joblib.load("test_pred_tfidf.pkl")[:, np.newaxis])
te_preds.append(joblib.load("test_pred_mercari.pkl")[:, np.newaxis])
te_preds.append(joblib.load("test_pred_bilstm_0.pkl")[:, np.newaxis])
# te_preds.append(joblib.load("test_pred_bilstm_1.pkl")[:, np.newaxis])
# te_preds.append(joblib.load("test_pred_bilstm_2.pkl")[:, np.newaxis])
# te_pred0 = joblib.load("test_pred_bilstm_0.pkl")[:, np.newaxis]
# te_pred1 = joblib.load("test_pred_bilstm_1.pkl")[:, np.newaxis]
# te_pred2 = joblib.load("test_pred_bilstm_2.pkl")[:, np.newaxis]
# te_preds.append((te_pred0+te_pred1+te_pred2)/3)
te_preds.append(joblib.load("test_pred_pytorch_0.pkl")[:, np.newaxis])
# te_preds.append(joblib.load("test_pred_pytorch_1.pkl")[:, np.newaxis])
# te_preds.append(joblib.load("test_pred_pytorch_2.pkl")[:, np.newaxis])
# te_pred0 = joblib.load("test_pred_pytorch_0.pkl")[:, np.newaxis]
# te_pred1 = joblib.load("test_pred_pytorch_1.pkl")[:, np.newaxis]
# te_pred2 = joblib.load("test_pred_pytorch_2.pkl")[:, np.newaxis]
# te_preds.append((te_pred0+te_pred1+te_pred2)/3)
va_preds = np.hstack(va_preds)
te_preds = np.hstack(te_preds)
y_va = joblib.load("y_val.pkl")

# from lightgbm import LGBMRegressor
# est = LGBMRegressor(
#     max_depth=10, learning_rate=0.0025, random_state=5
# )

import xgboost as xgb
est = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1)

from sklearn.ensemble import RandomForestClassifier
est = RandomForestClassifier()

y_te = joblib.load('valid_for_emsemble_label.pkl').values

va_preds_merged, te_preds_merged = merge_predictions(
    X_tr=va_preds, y_tr=y_va, X_te=te_preds,
    # est=est
)

delta, f_score = bestThresshold(y_va, va_preds_merged)
print('[validation] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))

f_score = f1_score(y_te, np.array(te_preds_merged) > delta)
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))

delta, f_score = bestThresshold(y_te, te_preds_merged)
print('[test] best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f_score))


# df_test = pd.read_csv(os.path.join(INPUT_PATH, "test.csv"))
# submission = df_test[['qid']].copy()
# submission['prediction'] = (te_preds_merged > delta).astype(int)
# submission.to_csv('submission.csv', index=False)
