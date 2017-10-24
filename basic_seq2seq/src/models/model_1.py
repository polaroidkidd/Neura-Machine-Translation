import os

import numpy as np
from keras import Input, callbacks
from keras.engine import Model
from keras.layers import Embedding, LSTM, TimeDistributed, Dense
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from models.BaseModel import BaseModel


class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'model_1'

        self.params['batch_size'] = 256
        self.params['epochs'] = 50
        self.params['latent_dim'] = 256
        self.params['num_samples'] = 150000
        self.params['num_tokens'] = 91
        self.params['max_seq_length'] = 100
        self.params['EMBEDDING_DIM'] = 100
        self.params['MAX_WORDS'] = 20000

        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENT_DIR = '../../persistent/' + self.identifier
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.GRAPH_DIR = os.path.join(self.BASIC_PERSISTENT_DIR, 'Graph')
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        # self.input_token_idx_file = os.path.join(self.BASIC_PERSISTENT_DIR, "input_token_index.npy")
        # self.target_token_idx_file = os.path.join(self.BASIC_PERSISTENT_DIR, "target_token_index.npy")
        self.data_path = os.path.join(self.BASE_DATA_DIR, 'Training/deu.txt')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.PRETRAINED_GLOVE_FILE = os.path.join(self.BASE_DATA_DIR, 'glove.6B.100d.txt')
        self.LATEST_MODELCHKPT = os.path.join(self.MODEL_CHECKPOINT_DIR, 'model.999-0.00.hdf5')
        # self.TRAIN_EN_FILE = "europarl-v7.de-en.en"
        # self.TRAIN_DE_FILE = "europarl-v7.de-en.de"
        # self.VAL_EN_FILE = "newstest2013.en"
        # self.VAL_DE_FILE = "newstest2013.de"

        # english_train_file = os.path.join(BASE_DATA_DIR, "Training", TRAIN_EN_FILE)
        # german_train_file = os.path.join(BASE_DATA_DIR, "Training", TRAIN_DE_FILE)
        # english_val_file = os.path.join(BASE_DATA_DIR, "Validation", VAL_EN_FILE)
        # german_val_file = os.path.join(BASE_DATA_DIR, "Validation", VAL_DE_FILE)

        self.START_TOKEN = "_GO"
        self.END_TOKEN = "_EOS"

    def start_training(self):
        # train_input_data = load(english_train_file)
        # train_target_data = load(german_train_file)
        # val_input_data = load(english_val_file)
        # val_target_data = load(german_val_file)
        self._split_count_data()

        # train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, num_words = preprocess_data(
        #    train_input_data, train_target_data, val_input_data, val_target_data)

        # if len(train_input_data) != len(train_target_data) or len(val_input_data) != len(val_target_data):
        #    print("length of input_data and target_data have to be the same")
        #    exit(-1)
        # num_samples = len(train_input_data)

        # print("Number of training data:", num_samples)
        # print("Number of validation data:", len(val_input_data))

        # vocab_size = num_words

        xin = Input(batch_shape=(self.params['batch_size'], self.params['max_seq_length']),
                    dtype='int32')

        xemb = Embedding(self.params['MAX_WORDS'], self.params['EMBEDDING_DIM'])(xin)  # 3 dim (batch,time,feat)
        seq = LSTM(self.params['latent_dim'], return_sequences=True)(xemb)
        mlp = TimeDistributed(Dense(self.params['MAX_WORDS'], activation='softmax'))(seq)
        model = Model(input=xin, output=mlp)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        steps = 5
        mod_epochs = np.math.floor(self.num_samples / self.params['batch_size'] / steps * self.params['epochs'])
        tbCallBack = callbacks.TensorBoard(log_dir=self.GRAPH_DIR, histogram_freq=0, write_graph=True,
                                           write_images=True)
        modelCallback = callbacks.ModelCheckpoint(self.MODEL_CHECKPOINT_DIR + '/model.{epoch:02d}-{loss:.2f}.hdf5',
                                                  monitor='loss', verbose=1, save_best_only=False,
                                                  save_weights_only=True, mode='auto', period=mod_epochs/self.params['epochs'])

        model.fit_generator(self.serve_batch(), steps, epochs=mod_epochs, verbose=2, max_queue_size=5,
                            callbacks=[tbCallBack, modelCallback])
        model.save(self.model_file)

    def _load(self, file):
        """
        Loads the given file into a list.
        :param file: the file which should be loaded
        :return: list of data
        """
        with(open(file, encoding='utf8')) as file:
            data = file.readlines()
        print('Loaded', len(data), "lines of data.")
        return data

    def _serve_batch(self):
        counter = 0
        self.batch_X = np.zeros((self.params['batch_size'], self.params['max_seq_length']), dtype='int32')
        self.batch_Y = np.zeros((self.params['batch_size'], self.params['max_seq_length'], self.params['MAX_WORDS']),
                                dtype='int32')
        while True:
            for i in range(self.input_texts.shape[0]):
                in_X = self.input_texts[i]
                out_Y = np.zeros((1, self.target_texts.shape[1], self.params['MAX_WORDS']), dtype='int32')
                for token in self.target_texts[i]:
                    out_Y[0, :len(self.target_texts)] = to_categorical(token, num_classes=self.params['MAX_WORDS'])

                self.batch_X[counter] = in_X
                self.batch_Y[counter] = out_Y
                counter += 1
                if counter == self.params['batch_size']:
                    print("counter == batch_size", i)
                    counter = 0
                    yield self.batch_X, self.batch_Y

    def _preprocess_data(self, train_input_data, train_target_data, val_input_data, val_target_data):
        train_input_data, train_target_data, val_input_data, val_target_data, word_index = tokenize(train_input_data,
                                                                                                    train_target_data,
                                                                                                    val_input_data,
                                                                                                    val_target_data)

        train_input_data = pad_sequences(train_input_data, maxlen=self.params['max_seq_length'], padding='post')
        train_target_data = pad_sequences(train_target_data, maxlen=self.params['max_seq_length'], padding='post')
        val_input_data = pad_sequences(val_input_data, maxlen=self.params['max_seq_length'], padding='post')
        val_target_data = pad_sequences(val_target_data, maxlen=self.params['max_seq_length'], padding='post')

        embeddings_index = load_embedding()
        embedding_matrix, num_words = prepare_embedding_matrix(word_index, embeddings_index)

        # target_data = convert_last_dim_to_one_hot_enc(padded_target_data, num_words)

        return train_input_data, train_target_data, val_input_data, val_target_data, embedding_matrix, num_words

    def _tokenize(self, train_input_data, train_target_data, val_input_data, val_target_data):
        tokenizer = Tokenizer(num_words=self.params['MAX_NUM_WORDS'])
        tokenizer.fit_on_texts(train_input_data + train_target_data + val_input_data + val_target_data)

        train_input_data = tokenizer.texts_to_sequences(train_input_data)
        train_target_data = tokenizer.texts_to_sequences(train_target_data)
        val_input_data = tokenizer.texts_to_sequences(val_input_data)
        val_target_data = tokenizer.texts_to_sequences(val_target_data)

        return train_input_data, train_target_data, val_input_data, val_target_data, tokenizer.word_index

    def _load_embedding(self):
        print('Indexing word vectors.')

        embeddings_index = {}
        filename = os.path.join(BASE_DATA_DIR, self.params['GLOVE_FILE'])
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        return embeddings_index

    def _prepare_embedding_matrix(self, word_index, embeddings_index):
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

    def _split_count_data(self):
        self.input_texts = []
        self.target_texts = []
        lines = open(self.data_path, encoding='UTF-8').read().split('\n')
        for line in lines[: min(self.params['num_samples'], len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            self.input_texts.append(input_text)
            target_text = target_text
            self.target_texts.append(target_text)
        self.num_samples = len(self.input_texts)
        tokenizer = Tokenizer(num_words=self.params['MAX_WORDS'])
        tokenizer.fit_on_texts(self.input_texts + self.target_texts)
        self.word_index = tokenizer.word_index
        for word in tokenizer.word_index:
            tokenizer.word_index[word] = tokenizer.word_index[word] + 2
        tokenizer.word_index[self.START_TOKEN] = 1
        tokenizer.word_index[self.END_TOKEN] = 2
        tokenizer.num_words = tokenizer.num_words + 2
        self.word_index = tokenizer.word_index

        try:
            self.word_index[self.START_TOKEN]
            self.word_index[self.END_TOKEN]
        except Exception as e:
            print(e, "why")
            exit()

        # np.save(self.WORD_IDX_FILE, self.word_index)
        # self.map_to(self.word_index)
        # self.map_to(self.word_index)

        self.input_texts = tokenizer.texts_to_sequences(self.input_texts)
        self.target_texts = tokenizer.texts_to_sequences(self.target_texts)
        for idx in range(len(self.target_texts)):
            self.target_texts[idx] = [self.word_index[self.START_TOKEN]] + self.target_texts[idx] + [
                self.word_index[self.END_TOKEN]]
            if self.target_texts[idx][0] != 1:
                print(idx)
                print(self.target_texts[idx])
                exit(-1)

        self.input_texts = pad_sequences(self.input_texts, maxlen=self.params['max_seq_length'], padding='post')
        self.target_texts = pad_sequences(self.target_texts, maxlen=self.params['max_seq_length'], padding='post')

        embeddings_index = {}
        filename = self.PRETRAINED_GLOVE_FILE
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        self.num_words = min(self.params['MAX_WORDS'], len(self.word_index)) + 1
        self.embedding_matrix = np.zeros((self.num_words, self.params['EMBEDDING_DIM']))
        for word, i in self.word_index.items():
            if i >= self.params['MAX_WORDS']:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector




    def predict_one_sentence(self, sentence):
        # from split_and_count_data
        self.input_texts = []
        self.target_texts = []
        lines = open(self.data_path, encoding='UTF-8').read().split('\n')
        for line in lines[: min(self.params['num_samples'], len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            self.input_texts.append(input_text)
            target_text = target_text
            self.target_texts.append(target_text)
        self.num_samples = len(self.input_texts)
        tokenizer = Tokenizer(num_words=self.params['MAX_WORDS'])
        tokenizer.fit_on_texts(self.input_texts + self.target_texts)
        self.word_index = tokenizer.word_index
        for word in tokenizer.word_index:
            tokenizer.word_index[word] = tokenizer.word_index[word] + 2
        tokenizer.word_index[self.START_TOKEN] = 1
        tokenizer.word_index[self.END_TOKEN] = 2
        tokenizer.num_words = tokenizer.num_words + 2
        self.word_index = tokenizer.word_index

        try:
            self.word_index[self.START_TOKEN]
            self.word_index[self.END_TOKEN]
        except Exception as e:
            print(e, "why")
            exit()

        # np.save(self.WORD_IDX_FILE, self.word_index)
        # self.map_to(self.word_index)
        # self.map_to(self.word_index)
        print(sentence)
        sentence = tokenizer.texts_to_sequences([sentence])
        print(sentence)
        sentence = pad_sequences(sentence, maxlen=self.params['max_seq_length'], padding='post')
        print(sentence.shape)
        print(sentence)
        sentence=sentence.reshape(sentence.shape[0],sentence.shape[1])
        print(sentence.shape)




        xin = Input(batch_shape=(1, self.params['max_seq_length']),
                    dtype='int32')

        xemb = Embedding(self.params['MAX_WORDS'], self.params['EMBEDDING_DIM'])(xin)  # 3 dim (batch,time,feat)
        seq = LSTM(self.params['latent_dim'], return_sequences=True)(xemb)
        mlp = TimeDistributed(Dense(self.params['MAX_WORDS'], activation='softmax'))(seq)
        model = Model(input=xin, output=mlp)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.load_weights(self.LATEST_MODELCHKPT)
        prediction = model.predict(sentence, batch_size=1)
        print(prediction)
        print(prediction.shape)
        predicted_sentence = []
        for sentence in prediction:
            for token in sentence:
                print(token)
                print(token.shape)
                max_idx = np.argmax(token)
                print(max_idx)
                if max_idx == 0:
                    print("id of max token = 0")
                else:
                    print(self.word_index[max_idx])
                    predicted_sentence += self.word_index[max_idx]

        return predicted_sentence


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
