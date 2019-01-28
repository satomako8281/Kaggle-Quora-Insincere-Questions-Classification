import sys
from quora.main_helper import main
from quora.datasets import prepare_vectorizer_1

from quora.layers import NeuralNet
from quora.config import MAX_LR
from quora.classifier import PytorchClassifier

def define_models_1():
    models = [
        PytorchClassifier(
            NeuralNet(), use_gpu=True
        )
    ]

    # return models, prepare_vecorizer_1()
    return models, prepare_vectorizer_1()

if __name__ == '__main__':
    main(
        'pytorch',
        sys.argv[1],
        {
            1: define_models_1()
        }
    )

