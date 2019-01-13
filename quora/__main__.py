import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from tqdm import tqdm

from quora.datasets import , prepare_vectorizer_1
from quora.config import (
    STEP_SIZE, BASE_LR, MAX_LR, MODE, GAMMA, set_dataset_file, seed_everything, N_SPLITS, SEED, set_pilot_study_config
)
from quora.embeddings import make_embedding_matrix
from quora.eval import bestThresshold
from quora.layers import NeuralNet
from quora.learning_rate import CyclicLR
from quora.run import train, pred
from quora.misc import send_line_notification
from quora.quora_io import load_test_iter

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from quora.config import (
    MAX_FEATURES, MAXLEN, SEED,
    X_TRAIN_ARRAY, X_TEST_ARRAY, Y_TRAIN_ARRAY, FEATURES_ARRAY, TEST_FEATURES_ARRAY, WORD_INDEX_ARRAY
)

tqdm.pandas(desc='Progress')

USE_LOAD_CASED_DATASET = False
USE_LOAD_CASHED_EMBEDDINGS = False
PILOT_STUDY = False
HANDLE_TEST = False

def fit_transform_vectorizer(vectorizer):
    df_tr = pd.read_csv("./input/train.csv")
    # df_tr, df_va = load_train_validation()
    y_tr = df_tr['target'].values
    # y_va = df_va['target'].values
    X_tr = vectorizer.fit_transform(df_tr, y_tr)
    # X_va = vectorizer.transform(df_va)
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(df_tr['question_text']))

    return X_tr, y_tr, tokenizer.word_index


def fit_validate(vectorizer):
    # X_tr, y_tr, X_va, y_va, fitted_vectorizer = fit_transform_vectorizer(vectorizer)
    x_train, y_train, word_index = fit_transform_vectorizer(vectorizer)

    return x_train[:, :x_train.shape[1]-2], y_train, x_train[:, x_train.shape[1]-2:], word_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DEBUG')
    args = parser.parse_args()

    vectorizer = prepare_vectorizer_1()

    global DEBUG
    DEBUG = True if args.DEBUG else False

    seed_everything(SEED)
    set_dataset_file(DEBUG)
    set_pilot_study_config(PILOT_STUDY)

    x_train, y_train, features, word_index = fit_validate(vectorizer)
    embedding_matrix = make_embedding_matrix(word_index, USE_LOAD_CASHED_EMBEDDINGS)

    train_preds = np.zeros((len(x_train)))
    avg_losses_f = []
    avg_val_losses_f = []
    models = []
    splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(x_train, y_train))
    for i, (train_idx, valid_idx) in enumerate(splits):
        print(f'Fold {i + 1}')

        model = NeuralNet(embedding_matrix).cuda()
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=MAX_LR
        )
        scheduler = CyclicLR(optimizer, base_lr=BASE_LR, max_lr=MAX_LR, step_size=STEP_SIZE, mode=MODE, gamma=GAMMA)
        valid_preds_fold, avg_loss, avg_val_loss = train(
            x_train[train_idx.astype(int)], y_train[train_idx.astype(int), np.newaxis], features[train_idx.astype(int)],
            x_train[valid_idx.astype(int)], y_train[valid_idx.astype(int), np.newaxis], features[valid_idx.astype(int)],
            model, loss_fn, optimizer, scheduler=scheduler
        )
        train_preds[valid_idx] = valid_preds_fold
        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss)
        models.append(model)


    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f), np.average(avg_val_losses_f)))
    delta, f1_score = bestThresshold(y_train, train_preds)
    print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, f1_score))
    message = 'test'
    send_line_notification(message)

    if HANDLE_TEST:
        df_test = pd.read_csv("./input/test.csv")
        x_test = vectorizer.fit_transform(df_test)
        x_test_ = x_test[:, :x_test.shape[1]-2]
        test_features = x_test[:, x_test.shape[1]-2:]
        test_preds = np.zeros((len(x_test)))
        for model in models:
            test_preds_fold = pred(model, x_test_, test_features)
            test_preds += test_preds_fold / N_SPLITS

        submission = df_test[['qid']].copy()
        submission['prediction'] = (test_preds > delta).astype(int)
        submission.to_csv('submission.csv', index=False)

