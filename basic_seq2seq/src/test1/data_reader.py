import os

from keras.layers import Input, LSTM, Dense
from keras.layers import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np


def load(file):
    """
    Loads the given file into a list.
    :param file: the file which should be loaded
    :return: list of data
    """
    with(open(file, 'r')) as file:
        data = file.readlines()

    print('Loaded', len(data), "lines of data.")
    return data


def preprocess_data(input_data, target_data):
    encoded_input_data, encoded_target_data, word_index = tokenize(input_data, target_data)
    padded_input_data = pad_sequences(encoded_input_data, maxlen=MAX_SEQ_LEN, padding='post')
    padded_target_data = pad_sequences(encoded_target_data, maxlen=MAX_SEQ_LEN, padding='post')

    embeddings_index = load_embedding()
    embedding_matrix, num_words = prepare_embedding_matrix(word_index, embeddings_index)

    return padded_input_data, padded_target_data, embedding_matrix, num_words


MAX_NUM_WORDS = 20000
MAX_SEQ_LEN = 500


def tokenize(input_data, target_data):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(input_data + target_data)

    encoded_input_data = tokenizer.texts_to_sequences(input_data)
    encoded_target_data = tokenizer.texts_to_sequences(target_data)

    return encoded_input_data, encoded_target_data, tokenizer.word_index


def load_embedding():
    print('Indexing word vectors.')

    embeddings_index = {}
    filename = os.path.join('C:/Users/Nicolas/Dropbox/PA_CODE/glove.6B', 'glove.6B.100d.txt')
    with open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


EMBEDDING_DIM = 100


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


BASE_DATA_DIR = ""
TRAIN_EN_FILE = "train100.en"
TRAIN_DE_FILE = "train100.de"

english_train_file = os.path.join(BASE_DATA_DIR, TRAIN_EN_FILE)
german_train_file = os.path.join(BASE_DATA_DIR, TRAIN_DE_FILE)
data_en = load(english_train_file)
data_de = load(german_train_file)
input_data, target_data, embedding_matrix, num_words = preprocess_data(data_en, data_de)

print(len(input_data))
print(len(target_data))
print(embedding_matrix.shape)
print(num_words)


def build_model():
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQ_LEN,
                                trainable=False)

    hidden_state_size = 100
    input_vocabulary_size = 686  # Autoset in the library
    output_vocabulary_size = 513  # Autoset in the library

    sequence_input = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
    input_embed = embedding_layer(sequence_input)

    rnn_encoded = LSTM(hidden_state_size, return_sequences=True)(input_embed)

    decoded_seq = LSTM(hidden_state_size, return_sequences=True)(rnn_encoded)

    model = Model(inputs=sequence_input, outputs=decoded_seq)
    model.summary()
    return model


def build_model2():
    latent_dim = 200
    encoder_inputs = Input(shape=(None, MAX_SEQ_LEN))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, MAX_SEQ_LEN))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(MAX_SEQ_LEN, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


model = build_model()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print(target_data.shape)
print(target_data[0])
target_data.resize(target_data.shape+(100,))
print(target_data.shape)

model.fit(input_data, target_data, epochs=1)  # , batch_size=128)

prediction = model.predict(input_data[0].reshape(1,-1))
print(prediction)
print(prediction.shape)
np.save("C:/Users/Nicolas/Desktop/pred.npy", prediction)