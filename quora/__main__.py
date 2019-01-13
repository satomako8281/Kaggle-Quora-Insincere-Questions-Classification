import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from tqdm.auto import tqdm

from quora.datasets import load_and_prec
from quora.config import (
    STEP_SIZE, BASE_LR, MAX_LR, MODE, GAMMA, set_dataset_file, seed_everything, N_SPLITS, SEED, set_pilot_study_config
)
from quora.embeddings import make_embedding_matrix
from quora.eval import bestThresshold
from quora.layers import NeuralNet
from quora.learning_rate import CyclicLR
from quora.run import train, pred
from quora.misc import send_line_notification

tqdm.pandas(desc='Progress')

USE_LOAD_CASED_DATASET = False
USE_LOAD_CASHED_EMBEDDINGS = False
DEBUG = True
PILOT_STUDY = False

if __name__ == '__main__':
    seed_everything(SEED)
    set_dataset_file(DEBUG)
    set_pilot_study_config(PILOT_STUDY)

    x_train, x_test, y_train, features, test_features, word_index = load_and_prec(DEBUG, USE_LOAD_CASED_DATASET)
    embedding_matrix = make_embedding_matrix(word_index, USE_LOAD_CASHED_EMBEDDINGS)

    train_preds = np.zeros((len(x_train)))
    test_preds = np.zeros((len(x_test)))
    avg_losses_f = []
    avg_val_losses_f = []
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

        test_preds_fold = pred(model, x_test, test_features)
        test_preds += test_preds_fold / N_SPLITS

    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f), np.average(avg_val_losses_f)))
    message = 'test'
    send_line_notification(message)

    if not DEBUG:
        delta = bestThresshold(y_train, train_preds)
        df_test = pd.read_csv("./input/test.csv")
        submission = df_test[['qid']].copy()
        submission['prediction'] = (test_preds > delta).astype(int)
        submission.to_csv('submission.csv', index=False)
