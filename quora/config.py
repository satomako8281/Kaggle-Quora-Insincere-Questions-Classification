import os
import random

import numpy as np
import torch

DEBUG = True

SEED = 1029

# use "train" or "pred"
EMBED_SIZE = 300
MAX_FEATURES = 120000 # how many unique words to use (i.e num rows in embedding vector)
MAXLEN = 70 # max number of words in a question to use
BATCH_SIZE = 512 # how many samples to process at once
N_EPOCHS = 5 # how many times to iterate over all samples
N_SPLITS = 5 # Number of K-fold Splits

# use "layers"
EMBEDDING_DIM = 300
HIDDEN_SIZE = 60
GRU_LEN = HIDDEN_SIZE
USE_PRETRAINED_EMBEDDINGS = True
ROUTINGS = 4  # 5
NUM_CAPSULE = 5
DIM_CAPSULE = 5  # 16
DROPOUT_P = 0.25
RATE_DROP_DENSE = 0.28
LR = 0.001
T_EPSILON = 1e-7
NUM_CLASSES = 30

# use "dataset"
X_TRAIN_ARRAY = "x_train_for_debug.npy" if DEBUG else "x_train.npy"
X_TEST_ARRAY = "x_test_for_debug.npy" if DEBUG else "x_test.npy"
Y_TRAIN_ARRAY = "y_train_for_debug.npy" if DEBUG else "y_train.npy"
FEATURES_ARRAY = "features_for_debug.npy" if DEBUG else "features.npy"
TEST_FEATURES_ARRAY = "test_features_for_debug.npy" if DEBUG else "test_features.npy"
WORD_INDEX_ARRAY = "word_index_for_debug.npy" if DEBUG else "word_index.npy"

# use "CLR"
STEP_SIZE = 300
BASE_LR = 0.001
MAX_LR = 0.003
MODE = 'exp_range'
GAMMA = 0.99994

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

