import logging
import os
import random
import sys

import numpy as np
import torch

INPUT_PATH = os.environ.get('INPUT_PATH', './input/')
N_CORES = 4
DEBUG_N = 1500
DEBUG = DEBUG_N > 0
TEST_SIZE = int(os.environ.get('TEST_SIZE', 1))
VALIDATION_SIZE = float(os.environ.get('VALIDATION_SIZE', 0.05))
DUMP_DATASET = False
USE_CACHED_DATASET = True
USE_CASHED_EMBEDDINGS = True
HANDLE_TEST = int(os.environ.get('HANDLE_TEST', 1))
TEST_CHUNK = 350000

# use "train" or "pred"
EMBED_SIZE = 300
MAX_FEATURES = 120000 # how many unique words to use (i.e num rows in embedding vector)
MAXLEN = 70 # max number of words in a question to use
BATCH_SIZE = 512 # how many samples to process at once
N_EPOCHS = 1 # how many times to iterate over all samples
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

# use "CLR"
STEP_SIZE = 300
BASE_LR = 0.001
MAX_LR = 0.003
MODE = 'exp_range'
GAMMA = 0.99994

SEED = 1029


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    filepath = os.path.dirname(os.path.join('./output', 'log'))
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    handler = logging.FileHandler(os.path.join(filepath, 'log-{os.getpid()}.txt'), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


logger = setup_custom_logger('quora')

