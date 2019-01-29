from contextlib import contextmanager
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
import keras as K
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import gc
import re
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_absolute_error
from quora.config import BATCH_SIZE, N_EPOCHS, BASE_LR, MAX_LR, STEP_SIZE, MODE, GAMMA
from quora.misc import sigmoid
from quora.learning_rate import CyclicLR


class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, output_dim=1, input_dim=100, hidden_layer_dims=[100, 100],
                 num_epochs=1, learning_rate=0.01, batch_size=128, shuffle=False,
                 callbacks=[], use_gpu=False, verbose=1):
        pass

    def fit(self, X, y):
        config = tf.ConfigProto()
        with tf.Session(graph=tf.Graph(), config=config) as sess:
            K.backend.set_session(sess)
            model_in = K.Input(shape=(X.shape[1],), dtype='float32', sparse=True)
            out = K.layers.Dense(512, activation='relu')(model_in)
            out = K.layers.Concatenate()([K.layers.Dense(64, activation='tanh')(out),
                                          K.layers.Dense(64, activation='relu')(out)])
            out2= K.layers.Concatenate()([K.layers.Dense(32, activation='tanh')(out),
                                          K.layers.Dense(32, activation='relu')(out)])
            out = K.layers.Concatenate()([K.layers.Dropout(0.2)(out), out2])

            out = K.layers.Add()([K.layers.Dense(1, activation='linear')(out),
                                  K.layers.Dense(1, activation='relu')(out),
                                  ])
            out = K.layers.Add()([K.layers.Dense(1, activation='linear')(out),
                                  K.layers.Dense(1, activation='relu')(out),
                                  ])
            out = K.layers.Dense(1)(out)
            model = K.Model(model_in, out)
            model.compile(loss='logcosh', optimizer=K.optimizers.Adam(lr=3e-3), metrics=['accuracy'])
            self._model.fit(x=X, y=y, batch_size=2**(11 + 1), epochs=1, verbose=0)
            self._model = model

        return self

    def predict(self, X):
        y_preds = self._model.predict(X)[:, 0]
        return y_preds

