import time

import torch
import torch.utils.data as data_utils
import torch.nn.init as init

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_absolute_error

import inspect

from quora.config import BATCH_SIZE, N_EPOCHS, BASE_LR, MAX_LR, STEP_SIZE, MODE, GAMMA
from quora.misc import sigmoid
from quora.learning_rate import CyclicLR


class PytorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None, output_dim=1, input_dim=100, hidden_layer_dims=[100, 100],
                 num_epochs=1, learning_rate=0.01, batch_size=128, shuffle=False,
                 callbacks=[], use_gpu=False, verbose=1):
        """
        Called when initializing the regressor
        """
        self._history = None
        self._model = model
        self._gpu = use_gpu and torch.cuda.is_available()
        self._history = []

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def set_embedding_weight(self, embedding_matrix):
        self._model.set_embedding_weight(embedding_matrix)

    def _build_model(self):
        if self._gpu:
            self._model = self._model.cuda()

    def _train_model(self, X, y):
        torch_x = torch.tensor(X, dtype=torch.long)
        torch_y = torch.tensor(y, dtype=torch.long)
        if self._gpu:
            torch_x = torch_x.cuda()
            torch_y = torch_y.cuda()

        train_loader = make_loader(torch_x, y=torch_y)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()), lr=MAX_LR)
        scheduler = CyclicLR(optimizer, base_lr=BASE_LR, max_lr=MAX_LR, step_size=STEP_SIZE, mode=MODE, gamma=GAMMA)

        self._history.append({"loss": [], "val_loss": []})
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            self._model.train()
            avg_loss = 0.
            for i, (x_batch, y_batch) in enumerate(train_loader):
                print("x_batch: ")
                print(x_batch.shape)
                y_pred = self._model(x_batch)
                if scheduler:
                    scheduler.batch_step()
                loss = self.loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
            elapsed_time = time.time() - start_time

            self._history["loss"].append(avg_loss)
            print("Results for epoch {}, loss {}, time={:.2f}s ".format(epoch + 1, avg_loss, elapsed_time))


    def fit(self, X, y):
        self._build_model()
        self._train_model(X, y)

        return self

    def predict(self, X, split_no=None):
        self._model.eval()
        y_preds = np.zeros(len(X))
        loader = make_loader(X, shuffle=False)
        for i, x_batch in enumerate(loader):
            y_pred = self._model(x_batch).detach()
            y_preds[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = sigmoid(y_pred.cpu().numpy())[:, 0]
        return y_preds


    def score(self, X, y, sample_weight=None):
        """
        Scores the data using the trained pytorch model. Under current implementation
        returns negative mae.
        """
        y_pred = self.predict(X, y)
        return mean_absolute_error(y, y_pred) * -1


def make_loader(X, y=None, shuffle=True):
    if y is not None:
        dataset = torch.utils.data.TensorDataset(X, y)
    else:
        dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    return loader

