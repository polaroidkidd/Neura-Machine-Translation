from keras import Input, callbacks
from keras.engine import Model
from keras.layers import Embedding, LSTM, TimeDistributed, Dense
import numpy as np
import os

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from thinc.neural.util import to_categorical

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


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


def serve_batch_perfomance(data_x, data_y):
    counter = 0
    # print(data_x.shape)
    # print(data_y.shape)
    batch_X = np.zeros((batch_size, data_x.shape[1]))
    batch_Y = np.zeros((batch_size, data_y.shape[1], vocab_size))
    # print('batch_X.shape', batch_X.shape)
    # print('batch_Y.shape', batch_Y.shape)
    for i, _ in enumerate(data_x):
        in_X = data_x[i]
        out_Y = np.zeros((1, data_y.shape[1], vocab_size), dtype='int32')
        # print('in_X.shape', in_X.shape)
        # print("out_Y.shape", out_Y.shape)


        for token in data_y[i]:
            out_Y[0, :len(data_y)] = to_categorical(token, nb_classes=vocab_size)

        batch_X[counter] = in_X
        batch_Y[counter] = out_Y
        counter += 1
        # print("counter", counter)
        if counter == batch_size:
            print("counter == batch_size", i)
            counter = 0
            yield batch_X, batch_Y


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
data_en = load(english_train_file)
data_de = load(german_train_file)
val_data_en = load(english_val_file)
val_data_de = load(german_val_file)

train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, num_words = preprocess_data(
    data_en, data_de, val_data_en, val_data_en)

if len(train_input_data) != len(train_target_data) or len(val_input_data) != len(val_target_data):
    print("length of input_data and target_data have to be the same")
    exit(-1)
num_samples = len(train_input_data)

print("Number of training data:", num_samples)
print("Number of validation data:", len(val_input_data))

vocab_size = num_words
B = batch_size
R = rnn_size
S = MAX_SEQ_LEN
V = vocab_size
E = EMBEDDING_DIM
emb_W = embedding_matrix
# x, y = next(serve_batch_perfomance(input_data, target_data))
# x, y = next(serve_batch(input_data, target_data,vocab_size, batch_size))


# M.fit(input_data, target_data, callbacks=[tbCallBack])

xin = Input(batch_shape=(batch_size, MAX_SEQ_LEN), dtype='int32')
xemb = Embedding(V, E)(xin)  # 3 dim (batch,time,feat)
seq = LSTM(R, return_sequences=True)(xemb)
mlp = TimeDistributed(Dense(V, activation='softmax'))(seq)
model = Model(input=xin, output=mlp)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
tbCallBack = callbacks.TensorBoard(log_dir=os.path.join(BASIC_PERSISTENT_DIR, GRAPH_DIR), histogram_freq=0,
                                   write_graph=True, write_images=True)
modelCallback = callbacks.ModelCheckpoint(BASIC_PERSISTENT_DIR + GRAPH_DIR + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                          monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                                          mode='auto', period=1)
normal_epochs = 10
epochs = np.math.floor(num_samples / batch_size * normal_epochs)
model.fit_generator(serve_batch_perfomance(train_input_data, train_target_data), num_samples/batch_size, epochs=normal_epochs, verbose=2,
                    validation_data=serve_batch_perfomance(val_input_data, val_target_data),
                    validation_steps=len(val_input_data) / batch_size,
                    callbacks=[tbCallBack, modelCallback])
model.save_model(os.path.join(BASIC_PERSISTENT_DIR, MODEL_DIR, 'stack1.h5'))
