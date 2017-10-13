from __future__ import print_function

from keras import callbacks
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Lambda
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import os

#from keras.utils import to_categorical
from thinc.neural.util import to_categorical as to_cat3
from utils import neg_log_likelihood, CustomLossLayer

batch_size = 64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

MAX_SEQ_LEN = 250
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 100

BASE_DATA_DIR = os.path.join("../../", "data")
BASIC_PERSISTENT_DIR = '../../persistent2gpu2/'
GRAPH_DIR = 'graph_stack2/'
MODEL_DIR = 'model_stack2/'
MODEL_CHECKPOINT_DIR = 'model_chkp_stack2/'


# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load(file):
    """
    Loads the given file into a list.
    :param file: the file which should be loaded
    :return: list of data
    """
    with(open(file, encoding='utf8')) as file:
        data = file.readlines()
        # data = []
        # for i in range(MAX_SENTENCES):
        #    data.append(lines[i])
    print('Loaded', len(data), "lines of data.")
    return data


def preprocess_data(train_input_data, train_target_data, val_input_data, val_target_data):
    train_input_data, train_target_data, val_input_data, val_target_data, word_index = tokenize(train_input_data,
                                                                                                train_target_data,
                                                                                                val_input_data,
                                                                                                val_target_data)

    train_input_data = pad_sequences(train_input_data, maxlen=MAX_SEQ_LEN, padding='post')
    train_target_data = pad_sequences(train_target_data, maxlen=MAX_SEQ_LEN, padding='post')
    val_input_data = pad_sequences(val_input_data, maxlen=MAX_SEQ_LEN, padding='post')
    val_target_data = pad_sequences(val_target_data, maxlen=MAX_SEQ_LEN, padding='post')

    embeddings_index = load_embedding()
    embedding_matrix, num_words = prepare_embedding_matrix(word_index, embeddings_index)

    # target_data = convert_last_dim_to_one_hot_enc(padded_target_data, num_words)

    return train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, num_words


def tokenize(train_input_data, train_target_data, val_input_data, val_target_data):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
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

train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, num_words = preprocess_data(
    train_input_data, train_target_data, val_input_data, val_target_data)


def convert_last_dim_to_one_hot_enc(target, vocab_size):
    """
    :param target: shape: (number of samples, max sentence length)
    :param vocab_size: size of the vocabulary
    :return: transformed target with shape: (number of samples, max sentence length, number of words in vocab)
    """
    x = np.ones((target.shape[0], target.shape[1], vocab_size), dtype='int32')
    for idx, s in enumerate(target):
        for token in s:
            x[idx, :len(target)] = to_categorical(token, num_classes=vocab_size)
    return x


def get_decoder_target_data(decoder_input_data):
    decoder_target_data = np.zeros(decoder_input_data.shape)
    for i in range(decoder_input_data.shape[0]):
        for j in range(decoder_input_data.shape[1] - 1):
            decoder_target_data[i][j + 1] = decoder_input_data[i][j]
        decoder_target_data[i][0] = 0
    return decoder_target_data


def serve_batch_perfomance(data_x, data_z, data_y):
    vocab_size = MAX_NUM_WORDS
    batch_X = np.zeros((batch_size, data_x.shape[1]))
    batch_Y = np.zeros((batch_size, data_y.shape[1], vocab_size))
    batch_Z = np.zeros((batch_size, data_z.shape[1]))
    while True:
        counter = 0
        for i, _ in enumerate(data_x):
            out_Y = np.zeros((1, data_y.shape[1], vocab_size), dtype='int32')

            for token in data_y[i]:
                out_Y[0, :len(data_y)] = to_categorical(token, num_classes=vocab_size)

            batch_X[counter] = data_x[i]
            batch_Y[counter] = out_Y
            batch_Z[counter] = data_z[i]
            counter += 1
            if counter == batch_size:
                print("counter == batch_size", i)
                counter = 0
                yield [batch_X, batch_Z], batch_Y


train_decoder_input_data = get_decoder_target_data(train_target_data)
val_decoder_input_data = get_decoder_target_data(val_target_data)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(MAX_SEQ_LEN,))
x = Embedding(MAX_SEQ_LEN, EMBEDDING_DIM)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,
                           return_state=True)(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(MAX_SEQ_LEN,))
x = Embedding(MAX_SEQ_LEN, EMBEDDING_DIM)(decoder_inputs)
x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)

# decoder_outputs = TimeDistributed(Dense(200, input_shape=(None, MAX_SEQ_LEN, MAX_NUM_WORDS), activation='softmax'))(x)
x = Dense(MAX_NUM_WORDS, activation='softmax')(x)

one_hot = Embedding(MAX_NUM_WORDS, MAX_NUM_WORDS, embeddings_initializer='identity', trainable=False)(decoder_inputs)
xent = Lambda(lambda x: neg_log_likelihood(x[0], x[1]), output_shape=(1,))([one_hot, x])
decoder_outputs = CustomLossLayer()(xent)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
# Compile & run training
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer='rmsprop', loss=None)


normal_epochs = 10

# model.fit_generator(serve_batch_perfomance(train_input_data, train_target_data, train_decoder_input_data),
#                    num_samples / batch_size, epochs=normal_epochs, verbose=2, max_queue_size=5)

tbCallBack = callbacks.TensorBoard(log_dir=os.path.join(BASIC_PERSISTENT_DIR, GRAPH_DIR), histogram_freq=0,
                                   write_graph=True, write_images=True)
modelCallback = callbacks.ModelCheckpoint(BASIC_PERSISTENT_DIR + GRAPH_DIR + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                          monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                                          mode='auto', period=1)
normal_epochs = 10
factor = 10
steps_1 = np.math.floor(num_samples / batch_size / factor)
steps_2 = np.math.floor(len(val_input_data) / batch_size)
epochs = normal_epochs * factor
model.fit_generator(serve_batch_perfomance(train_input_data, train_target_data, train_decoder_input_data), steps_1,
                    epochs=epochs, verbose=2, max_queue_size=5,
                    validation_data=serve_batch_perfomance(val_input_data, val_target_data, val_decoder_input_data),
                    validation_steps=steps_2, callbacks=[tbCallBack, modelCallback])
