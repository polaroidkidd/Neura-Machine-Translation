import os

import numpy as np
from keras import callbacks
from keras.layers import Embedding
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from models.BaseModel import BaseModel


class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'model_2'

        self.params['BATCH_SIZE'] = 128
        self.params['EMBEDDING_DIM'] = 100
        self.params['EPOCHS'] = 15
        self.params['LATENT_DIM'] = 256
        self.params['NUM_TOKENS'] = 70
        self.params['MAX_NUM_SAMPLES'] = 1000000
        self.params['MAX_NUM_WORDS'] = 20000
        self.params['MAX_SENTENCES'] = 1000
        self.params['MAX_SEQ_LEN'] = 1800
        self.UNKNOWN_CHAR = '\r'
        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENT_DIR = '../../persistent/' + self.identifier
        self.GRAPH_DIR = os.path.join(self.BASIC_PERSISTENT_DIR, 'Graph')
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.LATEST_MODEL_CHKPT = os.path.join(self.MODEL_CHECKPOINT_DIR,
                                               'chkp2prepro_64_100_15_256_1000000_20000_1000_1800_70_70_0.8_char___tfmodelprepro.35999-54.06.hdf5')
        self.token_idx_file = os.path.join(self.BASIC_PERSISTENT_DIR, "input_token_idx_preprocessed.npy")
        self.train_en_file = os.path.join(self.BASE_DATA_DIR, 'Training/train.en')
        self.train_de_file = os.path.join(self.BASE_DATA_DIR, 'Training/train.de')
        self.encoder_model_file = os.path.join(self.MODEL_DIR, 'encoder_model.h5')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.decoder_model_file = os.path.join(self.MODEL_DIR, 'decoder_model.h5')


        TRAIN_EN_FILE = "europarl-v7.de-en.en"
        TRAIN_DE_FILE = "europarl-v7.de-en.de"
        VAL_EN_FILE = "newstest2013.en"
        VAL_DE_FILE = "newstest2013.de"

        english_train_file = os.path.join(self.BASE_DATA_DIR, "Training", TRAIN_EN_FILE)
        german_train_file = os.path.join(self.BASE_DATA_DIR, "Training", TRAIN_DE_FILE)
        english_val_file = os.path.join(self.BASE_DATA_DIR, "Validation", VAL_EN_FILE)
        german_val_file = os.path.join(self.BASE_DATA_DIR, "Validation", VAL_DE_FILE)

    def start_training(self):
        data_en = self.load(self.english_train_file)
        data_de = self.load(self.german_train_file)
        val_data_en = self.load(self.english_val_file)
        val_data_de = self.load(self.german_val_file)

        train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, vocab_size = self.preprocess_data(
            data_en, data_de, val_data_en, val_data_en)

        if len(train_input_data) != len(train_target_data) or len(val_input_data) != len(val_target_data):
            print("length of input_data and target_data have to be the same")
            exit(-1)
        num_samples = len(train_input_data)

        print("Number of training data:", num_samples)
        print("Number of validation data:", len(val_input_data))

        M = Sequential()
        M.add(Embedding(vocab_size, self.params['EMBEDDING_DIM'], weights=[embedding_matrix], mask_zero=True))

        M.add(LSTM(self.params['LATENT_DIM'], return_sequences=True))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        M.add(
            LSTM(self.params['LATENT_DIM'] * int(1 / self.params['P_DENSE_DROPOUT']), return_sequences=True))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        M.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

        print('compiling')

        M.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print('compiled')
        tbCallBack = callbacks.TensorBoard(log_dir=os.path.join(self.BASIC_PERSISTENT_DIR, self.GRAPH_DIR), histogram_freq=0,
                                           write_graph=True, write_images=True)
        modelCallback = callbacks.ModelCheckpoint(self.BASIC_PERSISTENT_DIR + self.GRAPH_DIR + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                  monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
                                                  mode='auto', period=5)
        normal_epochs = 10
        epochs = np.math.floor(num_samples / self.params['BATCH_SIZE'] * normal_epochs)
        M.fit_generator(self.serve_batch(train_input_data, train_target_data), 1, epochs=epochs, verbose=2,
                        validation_data=self.serve_batch(val_input_data, val_target_data),
                        validation_steps=(len(val_input_data) / self.params['BATCH_SIZE']),
                        callbacks=[tbCallBack, modelCallback])
        M.save_model(os.path.join(self.BASIC_PERSISTENT_DIR, self.MODEL_DIR, 'stack2.h5'))
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


    def convert_last_dim_to_one_hot_enc(self, target, vocab_size):
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


    def serve_batch(self, data_x, data_y):
        counter = 0
        batch_X = np.zeros((self.params['BATCH_SIZE'], data_x.shape[1]))
        batch_Y = np.zeros((self.params['BATCH_SIZE'], data_y.shape[1], self.params['vocab_size']))
        while True:
            for i, _ in enumerate(data_x):
                in_X = data_x[i]
                out_Y = np.zeros((1, data_y.shape[1], self.params['vocab_size']), dtype='int32')

                for token in data_y[i]:
                    out_Y[0, :len(data_y)] = to_categorical(token, num_classes=self.params['vocab_size'])

                batch_X[counter] = in_X
                batch_Y[counter] = out_Y
                counter += 1
                if counter == self.params['BATCH_SIZE']:
                    print("counter == self.params['BATCH_SIZE']", i)
                    counter = 0
                    yield batch_X, batch_Y


    def preprocess_data(self, train_input_data, train_target_data, val_input_data, val_target_data):
        train_input_data, train_target_data, val_input_data, val_target_data, word_index = self.tokenize(train_input_data,
                                                                                                    train_target_data,
                                                                                                    val_input_data,
                                                                                                    val_target_data)

        train_input_data = pad_sequences(train_input_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')
        train_target_data = pad_sequences(train_target_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')
        val_input_data = pad_sequences(val_input_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')
        val_target_data = pad_sequences(val_target_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')

        embeddings_index = self.load_embedding()
        embedding_matrix, num_words = self.prepare_embedding_matrix(word_index, embeddings_index)

        # target_data = convert_last_dim_to_one_hot_enc(padded_target_data, num_words)

        return train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, num_words


    def tokenize(self, train_input_data, train_target_data, val_input_data, val_target_data):
        tokenizer = Tokenizer(num_words=self.params['MAX_NUM_WORDS'])
        tokenizer.fit_on_texts(train_input_data + train_target_data + val_input_data + val_target_data)

        train_input_data = tokenizer.texts_to_sequences(train_input_data)
        train_target_data = tokenizer.texts_to_sequences(train_target_data)
        val_input_data = tokenizer.texts_to_sequences(val_input_data)
        val_target_data = tokenizer.texts_to_sequences(val_target_data)

        return train_input_data, train_target_data, val_input_data, val_target_data, tokenizer.word_index


    def load_embedding(self):
        print('Indexing word vectors.')

        embeddings_index = {}
        filename = os.path.join(self.BASE_DATA_DIR, 'glove.6B.100d.txt')
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        return embeddings_index


    def prepare_embedding_matrix(self, word_index, embeddings_index):
        print('Preparing embedding matrix.')

        # prepare embedding matrix
        num_words = min(self.params['MAX_NUM_WORDS'], len(word_index)) + 1
        embedding_matrix = np.zeros((num_words, self.params['EMBEDDING_DIM']))
        for word, i in word_index.items():
            if i >= self.params['MAX_NUM_WORDS']:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix, num_words



    def predict_one_sentence(self, sentence):
        raise NotImplementedError()

    def predict_batch(self, sentences):
        raise NotImplementedError()

    def calculate_hiddenstate_after_encoder(self, sentence):
        raise NotImplementedError()

    def calculate_every_hiddenstate_after_encoder(self, sentence):
        raise NotImplementedError()

    def calculate_every_hiddenstate(self, sentence):
        raise NotImplementedError()

    def calculate_hiddenstate_after_decoder(self, sentence):
        raise NotImplementedError()

    def setup_inference(self):
        raise NotImplementedError()
