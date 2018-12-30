import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
import torch

from config import seed_everything, N_SPLITS, SEED
from dataset import load_and_prec
from embeddings import make_embedding_matrix
from eval import bestThresshold
from layers import NeuralNet
from run import train, pred
from config import STEP_SIZE, BASE_LR, MAX_LR, MODE, GAMMA
from learning_rate import CyclicLR

tqdm.pandas(desc='Progress')
seed_everything(SEED)

USE_LOAD_FILES = True
DEBUG = True

X_TRAIN_ARRAY = "x_train_for_debug.npy" if DEBUG else "x_train.npy"
X_TEST_ARRAY = "x_test_for_debug.npy" if DEBUG else "x_test.npy"
Y_TRAIN_ARRAY = "y_train_for_debug.npy" if DEBUG else "y_train.npy"
FEATURES_ARRAY = "features_for_debug.npy" if DEBUG else "features.npy"
TEST_FEATURES_ARRAY = "test_features_for_debug.npy" if DEBUG else "test_features.npy"
WORD_INDEX_ARRAY = "word_index_for_debug.npy" if DEBUG else "word_index.npy"

x_train, x_test, y_train, features, test_features, word_index = load_and_prec(DEBUG, USE_LOAD_FILES)
embedding_matrix = make_embedding_matrix(word_index, USE_LOAD_FILES)

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

if not DEBUG:
    delta = bestThresshold(y_train, train_preds)
    df_test = pd.read_csv("../input/test.csv")
    submission = df_test[['qid']].copy()
    submission['prediction'] = (test_preds > delta).astype(int)
    submission.to_csv('submission.csv', index=False)

