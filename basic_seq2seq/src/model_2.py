from keras import callbacks
from keras.layers import TimeDistributed, Dropout
from keras.models import Sequential
from thinc.neural.util import to_categorical
import os
from keras.layers import LSTM, Dense
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

from ParamHandler import ParamHandler

param_handler = ParamHandler("char", additional=['tf'])

BASE_DATA_DIR = "../../DataSets"
BASIC_PERSISTENT_DIR = '../../persistent/'
GRAPH_DIR = 'graph' + param_handler.param_summary()
MODEL_DIR = 'model' + param_handler.param_summary()
MODEL_CHECKPOINT_DIR = 'chkp' + param_handler.param_summary()

TRAIN_EN_FILE = "europarl-v7.de-en.en"
TRAIN_DE_FILE = "europarl-v7.de-en.de"
VAL_EN_FILE = "newstest2013.en"
VAL_DE_FILE = "newstest2013.de"

english_train_file = os.path.join(BASE_DATA_DIR, "Training", TRAIN_EN_FILE)
german_train_file = os.path.join(BASE_DATA_DIR, "Training", TRAIN_DE_FILE)
english_val_file = os.path.join(BASE_DATA_DIR, "Validation", VAL_EN_FILE)
german_val_file = os.path.join(BASE_DATA_DIR, "Validation", VAL_DE_FILE)


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


def serve_batch(data_x, data_y):
    counter = 0
    batch_X = np.zeros((param_handler.params['BATCH_SIZE'], data_x.shape[1]))
    batch_Y = np.zeros((param_handler.params['BATCH_SIZE'], data_y.shape[1], vocab_size))
    while True:
        for i, _ in enumerate(data_x):
            in_X = data_x[i]
            out_Y = np.zeros((1, data_y.shape[1], vocab_size), dtype='int32')

            for token in data_y[i]:
                out_Y[0, :len(data_y)] = to_categorical(token, nb_classes=vocab_size)

            batch_X[counter] = in_X
            batch_Y[counter] = out_Y
            counter += 1
            if counter == param_handler.params['BATCH_SIZE']:
                print("counter == param_handler.params['BATCH_SIZE']", i)
                counter = 0
                yield batch_X, batch_Y


def preprocess_data(train_input_data, train_target_data, val_input_data, val_target_data):
    train_input_data, train_target_data, val_input_data, val_target_data, word_index = tokenize(train_input_data,
                                                                                                train_target_data,
                                                                                                val_input_data,
                                                                                                val_target_data)

    train_input_data = pad_sequences(train_input_data, maxlen=param_handler.params['MAX_SEQ_LEN'], padding='post')
    train_target_data = pad_sequences(train_target_data, maxlen=param_handler.params['MAX_SEQ_LEN'], padding='post')
    val_input_data = pad_sequences(val_input_data, maxlen=param_handler.params['MAX_SEQ_LEN'], padding='post')
    val_target_data = pad_sequences(val_target_data, maxlen=param_handler.params['MAX_SEQ_LEN'], padding='post')

    embeddings_index = load_embedding()
    embedding_matrix, num_words = prepare_embedding_matrix(word_index, embeddings_index)

    # target_data = convert_last_dim_to_one_hot_enc(padded_target_data, num_words)

    return train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, num_words


def tokenize(train_input_data, train_target_data, val_input_data, val_target_data):
    tokenizer = Tokenizer(num_words=param_handler.params['MAX_NUM_WORDS'])
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
    num_words = min(param_handler.params['MAX_NUM_WORDS'], len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, param_handler.params['EMBEDDING_DIM']))
    for word, i in word_index.items():
        if i >= param_handler.params['MAX_NUM_WORDS']:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, num_words


data_en = load(english_train_file)
data_de = load(german_train_file)
val_data_en = load(english_val_file)
val_data_de = load(german_val_file)

train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, vocab_size = preprocess_data(
    data_en, data_de, val_data_en, val_data_en)

if len(train_input_data) != len(train_target_data) or len(val_input_data) != len(val_target_data):
    print("length of input_data and target_data have to be the same")
    exit(-1)
num_samples = len(train_input_data)

print("Number of training data:", num_samples)
print("Number of validation data:", len(val_input_data))

M = Sequential()
M.add(Embedding(vocab_size, param_handler.params['EMBEDDING_DIM'], weights=[embedding_matrix], mask_zero=True))

M.add(LSTM(param_handler.params['LATENT_DIM'], return_sequences=True))

M.add(Dropout(param_handler.params['P_DENSE_DROPOUT']))

M.add(
    LSTM(param_handler.params['LATENT_DIM'] * int(1 / param_handler.params['P_DENSE_DROPOUT']), return_sequences=True))

M.add(Dropout(param_handler.params['P_DENSE_DROPOUT']))

M.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

print('compiling')

M.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('compiled')
tbCallBack = callbacks.TensorBoard(log_dir=os.path.join(BASIC_PERSISTENT_DIR, GRAPH_DIR), histogram_freq=0,
                                   write_graph=True, write_images=True)
modelCallback = callbacks.ModelCheckpoint(BASIC_PERSISTENT_DIR + GRAPH_DIR + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                          monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                                          mode='auto', period=5)
normal_epochs = 10
epochs = np.math.floor(num_samples / param_handler.params['BATCH_SIZE'] * normal_epochs)
M.fit_generator(serve_batch(train_input_data, train_target_data), 1, epochs=epochs, verbose=2,
                validation_data=serve_batch(val_input_data, val_target_data),
                validation_steps=(len(val_input_data) / param_handler.params['BATCH_SIZE']),
                callbacks=[tbCallBack, modelCallback])
M.save_model(os.path.join(BASIC_PERSISTENT_DIR, MODEL_DIR, 'stack2.h5'))
