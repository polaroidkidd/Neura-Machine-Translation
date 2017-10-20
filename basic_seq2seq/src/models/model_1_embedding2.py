import os

import numpy as np
from keras import Input, callbacks
from keras.engine import Model
from keras.layers import Embedding, LSTM, TimeDistributed, Dense, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from utils import CustomLossLayer
from utils import categorical_cross_entropy

from helpers.ParamHandler import ParamHandler

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
    print('Loaded', len(data), "lines of data.")
    return data


def serve_batch(data_x, data_y):
    counter = 0
    batch_X = np.zeros((param_handler.params['BATCH_SIZE'], data_x.shape[1]))
    batch_Y = np.zeros((param_handler.params['BATCH_SIZE'], data_y.shape[1]))
    while True:
        for i, _ in enumerate(data_x):
            batch_X[counter] = data_x[i]
            batch_Y[counter] = data_y[i]
            counter += 1
            if counter == param_handler.params['BATCH_SIZE']:
                print("counter == batch_size", i)
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
    embedding_matrix, vocab_size = prepare_embedding_matrix(word_index, embeddings_index)

    return train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, vocab_size


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
    filename = os.path.join(BASE_DATA_DIR, "glove.6B.100d.txt")
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
    vocab_size = min(param_handler.params['MAX_NUM_WORDS'], len(word_index)) + 1
    embedding_matrix = np.zeros((vocab_size, param_handler.params['EMBEDDING_DIM']))
    for word, i in word_index.items():
        if i >= param_handler.params['MAX_NUM_WORDS']:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, vocab_size


train_input_data = load(english_train_file)
train_target_data = load(german_train_file)
val_input_data = load(english_val_file)
val_target_data = load(german_val_file)

train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, vocab_size = preprocess_data(
    train_input_data, train_target_data, val_input_data, val_target_data)

if len(train_input_data) != len(train_target_data) or len(val_input_data) != len(val_target_data):
    print("length of input_data and target_data have to be the same")
    exit(-1)
num_samples = len(train_input_data)

print("Number of training data:", num_samples)
print("Number of validation data:", len(val_input_data))

xin = Input(batch_shape=(param_handler.params['BATCH_SIZE'], param_handler.params['MAX_SEQ_LEN']), dtype='int32')
y_input = Input(batch_shape=(param_handler.params['BATCH_SIZE'], param_handler.params['MAX_SEQ_LEN']))
out_emb = Embedding(vocab_size, vocab_size, weights=[np.identity(vocab_size)], trainable=False)(y_input)

xemb = Embedding(vocab_size, param_handler.params['EMBEDDING_DIM'])(xin)  # 3 dim (batch,time,feat)
seq = LSTM(param_handler.params['LATENT_DIM'], return_sequences=True)(xemb)
mlp = TimeDistributed(Dense(vocab_size, activation='softmax'))(seq)

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


xent = Lambda(lambda x: categorical_cross_entropy(x[0], x[1]), output_shape=(1,))([out_emb, mlp])
decoder_outputs = CustomLossLayer()(xent)

model = Model(input=xin, output=decoder_outputs)
# model = Model(input=xin, output=decoder_outputs)
model.compile(optimizer='Adam', loss=None, metrics=['accuracy'])

tbCallBack = callbacks.TensorBoard(log_dir=os.path.join(BASIC_PERSISTENT_DIR, GRAPH_DIR), histogram_freq=0,
                                   write_graph=True, write_images=True)
modelCallback = callbacks.ModelCheckpoint(BASIC_PERSISTENT_DIR + GRAPH_DIR + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                          monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                                          mode='auto', period=1)

epochs = np.math.floor(num_samples / param_handler.params['BATCH_SIZE'] * param_handler.params['EPOCHS'])
model.fit_generator(serve_batch(train_input_data, train_target_data), num_samples / param_handler.params['BATCH_SIZE'],
                    epochs=param_handler.params['EPOCHS'], verbose=2, max_queue_size=5,
                    validation_data=serve_batch(val_input_data, val_target_data),
                    validation_steps=len(val_input_data) / param_handler.params['BATCH_SIZE'],
                    callbacks=[tbCallBack, modelCallback])
model.save_model(os.path.join(BASIC_PERSISTENT_DIR, MODEL_DIR, 'stack1.h5'))
