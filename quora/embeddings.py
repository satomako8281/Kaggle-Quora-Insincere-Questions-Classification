import gc

import numpy as np

from quora.config import MAX_FEATURES


def make_embedding_matrix(word_index, use_load_files):
    glove_embeddings = load_glove(word_index, use_load_files)
    paragram_embeddings = load_para(word_index, use_load_files)
    embedding_matrix = np.mean([glove_embeddings, paragram_embeddings], axis=0)

    del glove_embeddings, paragram_embeddings
    gc.collect()

    return embedding_matrix


def load_glove(word_index, use_load_files):
    if use_load_files:
        embedding_matrix = np.load("glove_embedding_matrix.npy")
        return embedding_matrix

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')[:300]

    embedding_file = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    np.save("glove_embedding_matrix.npy", embedding_matrix)
    return embedding_matrix


def load_fasttext(word_index, use_load_files):
    if use_load_files:
        embedding_matrix = np.load("fasttext_embedding_matrix.npy")
        return embedding_matrix

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embedding_file = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file) if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    np.save("fasttext_embedding_matrix.npy", embedding_matrix)
    return embedding_matrix


def load_para(word_index, use_load_files):
    if use_load_files:
        embedding_matrix = np.load("para_embedding_matrix.npy")
        return embedding_matrix

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embedding_file = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore') if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.0053247833, 0.49346462
    embed_size = all_embs.shape[1]

    nb_words = min(MAX_FEATURES, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    np.save("para_embedding_matrix.npy", embedding_matrix)
    return embedding_matrix