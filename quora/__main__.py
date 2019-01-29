import sys
from quora.main_helper import main
from quora.datasets import prepare_vectorizer_1, prepare_vectorizer_2

from quora.layers import LSTM_CAPSNET, LSTM_NET
from quora.config import MAX_LR
from quora.classifier_pytorch import PytorchClassifier
from quora.classifier_keras import KerasClassifier

def define_models_1():
    models = [
        PytorchClassifier(
            LSTM_CAPSNET(), use_gpu=True
        )
    ]

    # return models, prepare_vecorizer_1()
    return models, prepare_vectorizer_1()


def define_models_2():
    models = [
        PytorchClassifier(
            LSTM_NET(), use_gpu=True
        )
    ]

    # return models, prepare_vecorizer_1()
    return models, prepare_vectorizer_2()


def define_models_3():
    models = [
        KerasClassifier()
    ]

    # return models, prepare_vecorizer_1()
    return models, prepare_vectorizer_3()


if __name__ == '__main__':
    main(
        'pytorch',
        sys.argv[1],
        {
            1: define_models_1(),
            2: define_models_2(),
            3: define_models_3()
        }
    )

