from main_helper import main
from datasets import prepare_vecorizer_1
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
import torch

from layers import NeuralNet

def define_models_1(embeddings):
    models = [
        NeuralNetClassifier(
            NeuralNet(embeddings),  #TODO: wrapperを作る
            criterion=torch.nn.BCEWithLogitsLoss,
            optimizer=torch.optim.Adam,
            optimizer__lr=0.001,
            max_epochs=5,
            batch_size=512,
            iterator_train=
        )
    ]

    return models, prepare_vecorizer_1()

if __name__ == '__main__':
    name = 'pytorch'
    action = '1'
    arg_map = {
        1: define_models_1
    }

    main(name, action, arg_map)

