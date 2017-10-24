import os

import numpy as np
from keras import Input, callbacks
from keras.engine import Model
from keras.layers import Embedding, LSTM, TimeDistributed, Dense, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from helpers.utils import CustomLossLayer
from helpers.utils import categorical_cross_entropy
from models.BaseModel import BaseModel


class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'model_1_embedding2'

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

        self.english_train_file = os.path.join(self.BASE_DATA_DIR, "Training", TRAIN_EN_FILE)
        self.german_train_file = os.path.join(self.BASE_DATA_DIR, "Training", TRAIN_DE_FILE)
        self.english_val_file = os.path.join(self.BASE_DATA_DIR, "Validation", VAL_EN_FILE)
        self.german_val_file = os.path.join(self.BASE_DATA_DIR, "Validation", VAL_DE_FILE)

    def start_training(self):
        train_input_data = self.load(self.english_train_file)
        train_target_data = self.load(self.german_train_file)
        val_input_data = self.load(self.english_val_file)
        val_target_data = self.load(self.german_val_file)

        train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, vocab_size = self.preprocess_data(
            train_input_data, train_target_data, val_input_data, val_target_data)

        if len(train_input_data) != len(train_target_data) or len(val_input_data) != len(val_target_data):
            print("length of input_data and target_data have to be the same")
            exit(-1)
        num_samples = len(train_input_data)

        print("Number of training data:", num_samples)
        print("Number of validation data:", len(val_input_data))

        xin = Input(batch_shape=(self.params['BATCH_SIZE'], self.params['MAX_SEQ_LEN']), dtype='int32')
        y_input = Input(batch_shape=(self.params['BATCH_SIZE'], self.params['MAX_SEQ_LEN']))
        out_emb = Embedding(vocab_size, vocab_size, weights=[np.identity(vocab_size)], trainable=False)(y_input)

        xemb = Embedding(vocab_size, self.params['EMBEDDING_DIM'])(xin)  # 3 dim (batch,time,feat)
        seq = LSTM(self.params['LATENT_DIM'], return_sequences=True)(xemb)
        mlp = TimeDistributed(Dense(vocab_size, activation='softmax'))(seq)

        # model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


        xent = Lambda(lambda x: categorical_cross_entropy(x[0], x[1]), output_shape=(1,))([out_emb, mlp])
        decoder_outputs = CustomLossLayer()(xent)

        model = Model(input=xin, output=decoder_outputs)
        # model = Model(input=xin, output=decoder_outputs)
        model.compile(optimizer='Adam', loss=None, metrics=['accuracy'])

        tbCallBack = callbacks.TensorBoard(log_dir=os.path.join(self.BASIC_PERSISTENT_DIR, self.GRAPH_DIR),
                                           histogram_freq=0,
                                           write_graph=True, write_images=True)
        modelCallback = callbacks.ModelCheckpoint(
            self.BASIC_PERSISTENT_DIR + self.GRAPH_DIR + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
            mode='auto', period=1)

        epochs = np.math.floor(num_samples / self.params['BATCH_SIZE'] * self.params['EPOCHS'])
        model.fit_generator(self.serve_batch(train_input_data, train_target_data),
                            num_samples / self.params['BATCH_SIZE'],
                            epochs=self.params['EPOCHS'], verbose=2, max_queue_size=5,
                            validation_data=self.serve_batch(val_input_data, val_target_data),
                            validation_steps=len(val_input_data) / self.params['BATCH_SIZE'],
                            callbacks=[tbCallBack, modelCallback])
        model.save(os.path.join(self.BASIC_PERSISTENT_DIR, self.MODEL_DIR, 'stack1.h5'))

    def load(self, file):
        """
        Loads the given file into a list.
        :param file: the file which should be loaded
        :return: list of data
        """
        with(open(file, encoding='utf8')) as file:
            data = file.readlines()
        print('Loaded', len(data), "lines of data.")
        return data

    def serve_batch(self, data_x, data_y):
        counter = 0
        batch_X = np.zeros((self.params['BATCH_SIZE'], data_x.shape[1]))
        batch_Y = np.zeros((self.params['BATCH_SIZE'], data_y.shape[1]))
        while True:
            for i, _ in enumerate(data_x):
                batch_X[counter] = data_x[i]
                batch_Y[counter] = data_y[i]
                counter += 1
                if counter == self.params['BATCH_SIZE']:
                    print("counter == batch_size", i)
                    counter = 0
                    yield batch_X, batch_Y

    def preprocess_data(self, train_input_data, train_target_data, val_input_data, val_target_data):
        train_input_data, train_target_data, val_input_data, val_target_data, word_index = self.tokenize(
            train_input_data,
            train_target_data,
            val_input_data,
            val_target_data)

        train_input_data = pad_sequences(train_input_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')
        train_target_data = pad_sequences(train_target_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')
        val_input_data = pad_sequences(val_input_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')
        val_target_data = pad_sequences(val_target_data, maxlen=self.params['MAX_SEQ_LEN'], padding='post')

        embeddings_index = self.load_embedding()
        embedding_matrix, vocab_size = self.prepare_embedding_matrix(word_index, embeddings_index)

        return train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, vocab_size

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
        filename = os.path.join(self.BASE_DATA_DIR, "glove.6B.100d.txt")
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
        vocab_size = min(self.params['MAX_NUM_WORDS'], len(word_index)) + 1
        embedding_matrix = np.zeros((vocab_size, self.params['EMBEDDING_DIM']))
        for word, i in word_index.items():
            if i >= self.params['MAX_NUM_WORDS']:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        return embedding_matrix, vocab_size

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
