import logging
import os
import random
import sys

import numpy as np
import torch

DEBUG :bool = False

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

def set_dataset_file(DEBUG):
    global X_TRAIN_ARRAY
    global X_TEST_ARRAY
    global Y_TRAIN_ARRAY
    global FEATURES_ARRAY
    global TEST_FEATURES_ARRAY
    global WORD_INDEX_ARRAY
    X_TRAIN_ARRAY = "x_train_for_debug.npy" if DEBUG else "x_train.npy"
    X_TEST_ARRAY = "x_test_for_debug.npy" if DEBUG else "x_test.npy"
    Y_TRAIN_ARRAY = "y_train_for_debug.npy" if DEBUG else "y_train.npy"
    FEATURES_ARRAY = "features_for_debug.npy" if DEBUG else "features.npy"
    TEST_FEATURES_ARRAY = "test_features_for_debug.npy" if DEBUG else "test_features.npy"
    WORD_INDEX_ARRAY = "word_index_for_debug.npy" if DEBUG else "word_index.npy"

def set_pilot_study_config(PILOT_STUDY):
    global N_SPLITS
    global N_EPOCHS
    N_SPLITS = 2 if PILOT_STUDY else 5
    N_EPOCHS = 1 if PILOT_STUDY else 5

USE_CACHED_DATASET = True
DEBUG_N = 100
TEST_SIZE = 100
TEST_CHUNK = 100
VALIDATION_SIZE = 100

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(f'../output/log/log-{os.getpid()}.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger('quora')

