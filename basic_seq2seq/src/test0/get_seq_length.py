import numpy as np
import os

from keras.preprocessing.text import Tokenizer

# from thinc.neural.util import to_categorical

EMBEDDING_DIM = 100
MAX_NUM_WORDS = 20000
MAX_SEQ_LEN = 250
MAX_SENTENCES = 1000
batch_size = 64
rnn_size = 200
p_dense_dropout = 0.8
BASE_DATA_DIR = os.path.join("/seq2seq/", "data")
BASIC_PERSISTENT_DIR = '/persistent/gpu2/'
GRAPH_DIR = 'graph_stack1/'
MODEL_DIR = 'model_stack1/'
MODEL_CHECKPOINT_DIR = 'model_chkp_stack1/'

try:
    if os.environ['USERDOMAIN_ROAMINGPROFILE'] == 'DESKTOP-C9M296Q':
        BASE_DATA_DIR = "../../data"
        BASIC_PERSISTENT_DIR = '../../persistent/gpu/'
except Exception:
    pass


def load(file):
    """
    Loads the given file into a list.
    :param file: the file which should be loaded
    :return: list of data
    """
    with(open(file, encoding='utf8')) as file:
        data = file.readlines()
    print('Loaded', len(data), "lines of data.")
    return data


def serve_batch(data_x, data_y):
    counter = 0
    batch_X = np.zeros((batch_size, data_x.shape[1]))
    batch_Y = np.zeros((batch_size, data_y.shape[1]))
    while True:
        for i, _ in enumerate(data_x):
            batch_X[counter] = data_x[i]
            batch_Y[counter] = data_y[i]
            counter += 1
            if counter == batch_size:
                print("counter == batch_size", i)
                counter = 0
                yield batch_X, batch_Y


def tokenize(train_input_data, train_target_data, val_input_data, val_target_data):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    # TODO muss ich fit_on_texts auch auf validation data machen
    tokenizer.fit_on_texts(train_input_data + train_target_data + val_input_data + val_target_data)

    train_input_data = tokenizer.texts_to_sequences(train_input_data)
    train_target_data = tokenizer.texts_to_sequences(train_target_data)
    val_input_data = tokenizer.texts_to_sequences(val_input_data)
    val_target_data = tokenizer.texts_to_sequences(val_target_data)

    return train_input_data, train_target_data, val_input_data, val_target_data, tokenizer.word_index


def load_embedding():
    print('Indexing word vectors.')

    embeddings_index = {}
    filename = os.path.join(BASE_DATA_DIR, 'glove.6B.100d.txt')
    with open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def prepare_embedding_matrix(word_index, embeddings_index):
    print('Preparing embedding matrix.')

    # prepare embedding matrix
    vocab_size = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, vocab_size


TRAIN_EN_FILE = "train.en"
TRAIN_DE_FILE = "train.de"
VAL_EN_FILE = "newstest2014.en"
VAL_DE_FILE = "newstest2014.de"

english_train_file = os.path.join(BASE_DATA_DIR, TRAIN_EN_FILE)
german_train_file = os.path.join(BASE_DATA_DIR, TRAIN_DE_FILE)
english_val_file = os.path.join(BASE_DATA_DIR, VAL_EN_FILE)
german_val_file = os.path.join(BASE_DATA_DIR, VAL_DE_FILE)
train_input_data = load(english_train_file)
train_target_data = load(german_train_file)
val_input_data = load(english_val_file)
val_target_data = load(german_val_file)
train_input_data, train_target_data, val_input_data, val_target_data, word_index = tokenize(train_input_data,
                                                                                            train_target_data,
                                                                                            val_input_data,
                                                                                            val_target_data)


def length_stats(data):
    print("\n\n")
    word_count = 0
    max_len = 0
    min_len = 100
    counter = 0
    lengths = []
    for sentence in data:
        length = len(sentence)
        if length > max_len:
            max_len = length
        if length < min_len:
            min_len = length
        word_count += length
        lengths.append(length)
        counter += 1

    print(word_count, max_len, min_len, (word_count / len(data)))
    print("mean", np.mean(lengths))
    print("std", np.std(lengths))


length_stats(train_input_data)
length_stats(train_target_data)
length_stats(val_input_data)
length_stats(val_target_data)
