from tqdm.auto import tqdm
tqdm.pandas(desc='Progress')
from sklearn.externals import joblib
import numpy as np
import gc
import os

max_features = 120000
embed_size = 300
INPUT_PATH = '../'

def load_glove(word_index):
    EMBEDDING_FILE = os.path.join(INPUT_PATH ,'embeddings/glove.840B.300d/glove.840B.300d.txt')
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore'))

    emb_mean,emb_std = -0.005838499, 0.48782197
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_para(word_index):
    EMBEDDING_FILE = os.path.join(INPUT_PATH, 'embeddings/paragram_300_sl999/paragram_300_sl999.txt')
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

    emb_mean,emb_std = -0.005838499, 0.48782197
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


glove_embeddings = load_glove(word_index)
paragram_embeddings = load_para(word_index)
embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)
joblib.dump(embedding_matrix, 'embedding_matrix.pkl', compress=3)

del glove_embeddings, paragram_embeddings
gc.collect()

