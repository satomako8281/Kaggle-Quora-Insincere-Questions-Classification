import time

import numpy as np
import torch
from torch.utils.data import Dataset
from quora.config import BATCH_SIZE, N_EPOCHS
from quora.misc import sigmoid


def train(
    x_train, y_train, kfold_X_features,
    x_valid, y_valid, kfold_X_valid_features,
    model, loss_fn, optimizer, scheduler=None
):
    train_loader = make_loader(x_train, y=y_train)
    valid_loader = make_loader(x_valid, y=y_valid, shuffle=False)
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        model.train()
        avg_loss = 0.
        for i, (x_batch, y_batch) in enumerate(train_loader):
            f = kfold_X_features[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            y_pred = model([x_batch, f])
            if scheduler:
                scheduler.batch_step()
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        valid_preds_fold = np.zeros(len(x_valid))
        avg_val_loss = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            f = kfold_X_valid_features[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            y_pred = model([x_batch, f]).detach()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds_fold[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]

        elapsed_time = time.time() - start_time
        print(
            'Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, N_EPOCHS, avg_loss, avg_val_loss, elapsed_time
            )
        )

    return valid_preds_fold, avg_loss, avg_val_loss


def pred(model, x_test, test_features):
    test_loader = make_loader(x_test, shuffle=False)
    test_preds_fold = np.zeros((len(x_test)))
    for i, (x_batch,) in enumerate(test_loader):
        f = test_features[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        y_pred = model([x_batch, f]).detach()
        test_preds_fold[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]

    return test_preds_fold


def make_loader(X, *, y=None, shuffle=True):
    x_fold = torch.tensor(X, dtype=torch.long).cuda()
    if y is not None:
        y_fold = torch.tensor(y, dtype=torch.float32).cuda()
        dataset = torch.utils.data.TensorDataset(x_fold, y_fold)
    else:
        dataset = torch.utils.data.TensorDataset(x_fold)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    return loader

