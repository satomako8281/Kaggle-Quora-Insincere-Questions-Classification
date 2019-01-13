import pandas as pd
from sklearn.model_selection import train_test_split

from quora.config import DEBUG, DEBUG_N, TEST_SIZE, TEST_CHUNK, VALIDATION_SIZE


def load_train(path='../input/train.csv'):
    if DEBUG:
        return pd.read_csv(path).iloc[:DEBUG_N, :]
    else:
        return pd.read_csv(path)


def load_train_validation():
    return quora_train_test_split(load_train())


def load_test_iter():
    for _ in range(TEST_SIZE):
        for df in pd.read_csv('../input/test.csv', chunksize=TEST_CHUNK):
            if DEBUG:
                yield df.iloc[:DEBUG_N]
            else:
                yield df


def quora_train_test_split(*arrays):
    return train_test_split(*arrays, test_size=VALIDATION_SIZE, random_state=0)

