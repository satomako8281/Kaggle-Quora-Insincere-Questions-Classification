import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import time

import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data

N_EPOCHS = 3

train = joblib.load('train.pkl')
test = joblib.load('valid_for_emsemble.pkl')
sub = pd.read_csv(os.path.join(INPUT_PATH, 'sample_submission.csv'))

X_train, X_test, y_train, features, test_features, word_index = load_and_prec(use_misspell=True)
train_idx = joblib.load('train_idx.pkl')
valid_idx = joblib.load('valid_idx.pkl')

x_test_cuda = torch.tensor(X_test, dtype=torch.long).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
batch_size = 512
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)


def train_model(model, x_train, y_train, x_val, y_val, validate=True):
    train = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    valid = torch.utils.data.TensorDataset(x_val, y_val)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for x_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        model.eval()
        valid_preds = np.zeros((x_val_fold.size(0)))
        if validate:
            avg_val_loss = 0.
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch).detach()
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
            search_result = threshold_search(y_val.cpu().numpy(), valid_preds)
            val_f1, val_threshold = search_result['f1'], search_result['threshold']
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} best_t={:.2f} \t time={:.2f}s'.format(
                epoch + 1, N_EPOCHS, avg_loss, avg_val_loss, val_f1, val_threshold, elapsed_time))
        else:
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, N_EPOCHS, avg_loss, elapsed_time))
    valid_preds = np.zeros((x_val_fold.size(0)))
    avg_val_loss = 0.
    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        valid_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
    print('Validation loss: ', avg_val_loss)
    test_preds = np.zeros((len(test_loader.dataset)))
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()
        test_preds[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    return valid_preds, test_preds

train_preds = np.zeros(len(train))
test_preds = np.zeros((len(test)))

x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long).cuda()
y_train_fold = torch.tensor(y_train[train_idx, np.newaxis], dtype=torch.float32).cuda()
x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long).cuda()
y_val_fold = torch.tensor(y_train[valid_idx, np.newaxis], dtype=torch.float32).cuda()

train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

seed_everything(1030)
model = NeuralNetPytorch()
model.cuda()

valid_preds_fold, test_preds_fold = train_model(
    model,
    x_train_fold,
    y_train_fold,
    x_val_fold,
    y_val_fold,
    validate=True
)

train_preds[valid_idx] = valid_preds_fold
test_preds += test_preds_fold / 1
# test_preds_local[:, i] = test_preds_local_fold
joblib.dump(valid_preds_fold, 'valid_pred_pytorch.pkl', compress=3)
joblib.dump(test_preds, 'test_pred_pytorch.pkl', compress=3)

search_result = threshold_search(y_train[valid_idx.astype(int)], valid_preds_fold)
print(search_result)

