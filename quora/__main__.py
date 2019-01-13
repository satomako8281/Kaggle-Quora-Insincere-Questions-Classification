import sys
from quora.main_helper import main
from quora.datasets import prepare_vectorizer_1
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
import torch

from quora.layers import NeuralNet

def define_models_1():
    # models = [
    #     NeuralNetClassifier(
    #         NeuralNet(embeddings),  #TODO: wrapperを作る
    #         criterion=torch.nn.BCEWithLogitsLoss,
    #         optimizer
    #         optimizer__lr=0.001,
    #         max_epochs=5,
    #         batch_size=512,
    #         iterator_train=
    #     )
    # ]

    # return models, prepare_vecorizer_1()
    return prepare_vectorizer_1()

if __name__ == '__main__':
    arg_map = {
        1: define_models_1
    }

    main(
        sys.argv[1],
        {
            1: define_models_1()
        }
    )
