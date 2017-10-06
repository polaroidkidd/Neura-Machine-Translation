from keras import Input, callbacks
from keras.layers import TimeDistributed, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
import os
from keras.layers import LSTM, Dense
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

EMBEDDING_DIM = 100
MAX_NUM_WORDS = 20000
MAX_SEQ_LEN = 500
BASE_DATA_DIR = "../data"

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


def add_one_hot_dim(sequences, vocab_size):
    x = (np.ones((sequences.shape[0], sequences.shape[1], vocab_size), dtype='int32'))
    for idx, s in enumerate(sequences):
        for token in s:
            x[idx, :len(sequences)] = to_categorical(token, num_classes=vocab_size)
    return x


def preprocess_data(input_data, target_data):
    encoded_input_data, encoded_target_data, word_index = tokenize(input_data, target_data)

    padded_input_data = pad_sequences(encoded_input_data, maxlen=MAX_SEQ_LEN, padding='post')
    padded_target_data = pad_sequences(encoded_target_data, maxlen=MAX_SEQ_LEN, padding='post')

    embeddings_index = load_embedding()
    embedding_matrix, num_words = prepare_embedding_matrix(word_index, embeddings_index)
    target_data = add_one_hot_dim(padded_target_data, num_words)

    return padded_input_data, target_data, embedding_matrix, num_words


def tokenize(input_data, target_data):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(input_data + target_data)

    encoded_input_data = tokenizer.texts_to_sequences(input_data)
    encoded_target_data = tokenizer.texts_to_sequences(target_data)

    return encoded_input_data, encoded_target_data, tokenizer.word_index


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
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, num_words


TRAIN_EN_FILE = "train.en"
TRAIN_DE_FILE = "train.de"

english_train_file = os.path.join(BASE_DATA_DIR, TRAIN_EN_FILE)
german_train_file = os.path.join(BASE_DATA_DIR, TRAIN_DE_FILE)
data_en = load(english_train_file)
data_de = load(german_train_file)
input_data, target_data, embedding_matrix, num_words = preprocess_data(data_en, data_de)

vocab_size = num_words
batch_size = 33
rnn_size = 10
p_dense_dropout = 0.8

B = batch_size
R = rnn_size
S = MAX_SEQ_LEN
V = vocab_size
E = EMBEDDING_DIM
emb_W = embedding_matrix

## dropout parameters
p_dense = p_dense_dropout

M = Sequential()
M.add(Embedding(V, E, weights=[emb_W], mask_zero=True))

M.add(LSTM(R, return_sequences=True))

M.add(Dropout(p_dense))

M.add(LSTM(R * int(1 / p_dense), return_sequences=True))

M.add(Dropout(p_dense))

M.add(TimeDistributed(Dense(V, activation='softmax')))

print('compiling')

M.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('compiled')
print(len(input_data))
print(len(target_data))
print(embedding_matrix.shape)
print(num_words)
print(input_data.shape)
print(target_data.shape)
print(target_data[0])
tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                   write_graph=True, write_images=True)
M.fit(input_data, target_data, callbacks=[tbCallBack])
