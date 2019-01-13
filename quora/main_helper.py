import os
import pickle

import numpy as np
import pandas as pd
import torch
from quora.config import (
    STEP_SIZE, BASE_LR, MAX_LR, MODE, GAMMA, set_dataset_file, seed_everything, N_SPLITS, SEED, set_pilot_study_config
)
from quora.embeddings import make_embedding_matrix
from quora.eval import bestThresshold
from quora.layers import NeuralNet
from quora.learning_rate import CyclicLR
from quora.run import train, pred
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from quora.quora_io import load_train_validation
from quora.misc import send_line_notification

tqdm.pandas(desc='Progress')

USE_CACHED_DATASET = False
DEBUG_N = 100
DUMP_DATASET = True

def fit_models(X_tr, y_tr, models):
    return list(map(lambda est: est.fit(X_tr, y_tr), models))

def predict_models(X, fitted_models):
    return list(map(lambda est: est.predict(X), fitted_models))

def fit_transform_vectorizer(vectorizer):
    df_tr, df_va = load_train_validation()
    y_tr = df_tr['target'].values
    y_va = df_va['target'].values
    X_tr = vectorizer.fit_transform(df_tr, y_tr)
    X_va = vectorizer.transform(df_va)
    return X_tr, y_tr, X_va, y_va, vectorizer


def fit_validate(models, vectorizer, name=None,
                 fit_parallel='thread', predict_parallel='thread'):
    cached_path = os.path.join('../input', 'data_{}.pkl'.format(name))
    if USE_CACHED_DATASET:
        assert name is not None
        with open(cached_path, 'rb') as f:
            X_tr, y_tr, X_va, y_va, fitted_vectorizer = pickle.load(f)
        if DEBUG_N:
            X_tr, y_tr = X_tr[:DEBUG_N], y_tr[:DEBUG_N]
    else:
        X_tr, y_tr, X_va, y_va, fitted_vectorizer = fit_transform_vectorizer(vectorizer)
    if DUMP_DATASET:
        assert name is not None
        with open(cached_path, 'wb') as f:
            pickle.dump((X_tr, y_tr, X_va, y_va, fitted_vectorizer), f)

    fitted_models = fit_models(X_tr, y_tr, models)
    y_va_preds = predict_models(X_va, fitted_models, parallel=predict_parallel)
    return fitted_vectorizer, fitted_models, y_va, y_va_preds

def main(name, action, arg_map):
    prefix = lambda r: "{}_{}s".format(name, r)

    if action in ('1'):
        model_round = int(action)
        models, vectorizer = arg_map[model_round]
        vectorizer, fitted_models, y_va, y_va_preds = fit_validate(
            models, vectorizer, name=model_round
        )
    for i in range(y_va_preds.shape[1]):
        delta, f1_score = bestThresshold(y_va, y_va_preds[:, i])
        print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f1_score))
        message = 'test'
        send_line_notification(message)

    # embedding_matrix = make_embedding_matrix(word_index, USE_LOAD_CASHED_EMBEDDINGS)
    #
    # train_preds = np.zeros((len(x_train)))
    # test_preds = np.zeros((len(x_test)))
    # avg_losses_f = []
    # avg_val_losses_f = []
    # splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(x_train, y_train))
    # for i, (train_idx, valid_idx) in enumerate(splits):
    #     print(f'Fold {i + 1}')
    #
    #     model = NeuralNet(embedding_matrix).cuda()
    #     loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=MAX_LR
        )
    #     scheduler = CyclicLR(optimizer, base_lr=BASE_LR, max_lr=MAX_LR, step_size=STEP_SIZE, mode=MODE, gamma=GAMMA)
    #     valid_preds_fold, avg_loss, avg_val_loss = train(
    #         x_train[train_idx.astype(int)], y_train[train_idx.astype(int), np.newaxis], features[train_idx.astype(int)],
    #         x_train[valid_idx.astype(int)], y_train[valid_idx.astype(int), np.newaxis], features[valid_idx.astype(int)],
    #         model, loss_fn, optimizer, scheduler=scheduler
    #     )
    #     train_preds[valid_idx] = valid_preds_fold
    #     avg_losses_f.append(avg_loss)
    #     avg_val_losses_f.append(avg_val_loss)
    #
    #     test_preds_fold = pred(model, x_test, test_features)
    #     test_preds += test_preds_fold / N_SPLITS
    #
    # print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f), np.average(avg_val_losses_f)))
    #
    # if not DEBUG:
    #     delta = bestThresshold(y_train, train_preds)
    #     df_test = pd.read_csv("../input/test.csv")
    #     submission = df_test[['qid']].copy()
    #     submission['prediction'] = (test_preds > delta).astype(int)
    #     submission.to_csv('submission.csv', index=False)

