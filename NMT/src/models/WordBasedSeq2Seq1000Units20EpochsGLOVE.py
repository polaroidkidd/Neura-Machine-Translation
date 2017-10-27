import os

import numpy as np
from keras import callbacks
from keras.engine import Model
from keras.layers import Embedding
from keras.layers import LSTM, Dense
from keras.layers import TimeDistributed, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from helpers.CustomCallback import CustomCallback
from helpers.Tokenizer import Tokenizer
from models.BaseModel import BaseModel


# TODO: Use Blue metric
class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'WordBasedSeq2Seq1000Units20EpochsGLOVE'

        self.params['batch_size'] = 128
        self.params['epochs'] = 20
        self.params['latent_dim'] = 1000
        self.params['MAX_SEQ_LEN'] = 100
        self.params['EMBEDDING_DIM'] = 300
        self.params['MAX_WORDS_DE'] = 32000
        self.params['MAX_WORDS_EN'] = 16000
        self.params['P_DENSE_DROPOUT'] = 0.2

        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENT_DIR = '../../Persistence/' + self.identifier
        if not os.path.exists("../../Persistence"):
            os.makedirs("../../Persistence")
        if not os.path.exists(self.BASIC_PERSISTENT_DIR):
            os.makedirs(self.BASIC_PERSISTENT_DIR)
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.GRAPH_DIR = os.path.join(self.BASIC_PERSISTENT_DIR, 'Graph')
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)

        self.TRAIN_DATA_FILE = os.path.join(self.BASE_DATA_DIR, 'Training/DE_EN_(tatoeba)_train.txt')
        self.VAL_DATA_FILE = os.path.join(self.BASE_DATA_DIR, 'Validation/DE_EN_(tatoeba)_validation.txt')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.PRETRAINED_GLOVE_FILE = os.path.join(self.BASE_DATA_DIR, 'glove.6B.300d.txt')
        self.LATEST_MODELCHKPT = os.path.join(self.MODEL_CHECKPOINT_DIR, 'model.878-1.90.hdf5')

        self.START_TOKEN = "_GO"
        self.END_TOKEN = "_EOS"
        self.UNK_TOKEN = "_UNK"

    def __create_vocab(self):
        en_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_EN'])
        en_tokenizer.fit_on_texts(self.train_input_texts)
        self.train_input_texts = en_tokenizer.texts_to_sequences(self.train_input_texts)
        self.train_input_texts = pad_sequences(self.train_input_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                               padding='post',
                                               truncating='post')
        self.en_word_index = en_tokenizer.word_index

        de_tokenizer = Tokenizer(self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN,
                                 num_words=self.params['MAX_WORDS_DE'])
        de_tokenizer.fit_on_texts(self.train_target_texts)
        self.train_target_texts = de_tokenizer.texts_to_sequences(self.train_target_texts)
        self.train_target_texts = pad_sequences(self.train_target_texts, maxlen=self.params['MAX_SEQ_LEN'],
                                                padding='post',
                                                truncating='post')
        self.de_word_index = de_tokenizer.word_index

        embeddings_index = {}
        filename = self.PRETRAINED_GLOVE_FILE
        with open(filename, 'r', encoding='utf8') as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        self.num_train_words = self.params['MAX_WORDS_EN'] + 3
        self.en_embedding_matrix = np.zeros((self.num_train_words, self.params['EMBEDDING_DIM']))
        for word, i in self.en_word_index.items():
            if i >= self.params['MAX_WORDS_EN'] + 3:
                continue
            embedding_vector = None
            if word == self.START_TOKEN:
                embedding_vector = self.START_TOKEN_VECTOR
            elif word == self.END_TOKEN:
                embedding_vector = self.END_TOKEN_VECTOR
            else:
                embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                embedding_vector = self.UNK_TOKEN_VECTOR
            self.en_embedding_matrix[i] = embedding_vector
        np.save(self.BASIC_PERSISTENT_DIR + '/en_word_index.npy', self.en_word_index)
        np.save(self.BASIC_PERSISTENT_DIR + '/de_word_index.npy', self.de_word_index)
        np.save(self.BASIC_PERSISTENT_DIR + '/en_embedding_matrix.npy', self.en_embedding_matrix)

    def start_training(self):
        self.START_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])
        self.END_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])
        self.UNK_TOKEN_VECTOR = np.random.rand(self.params['EMBEDDING_DIM'])

        self.train_input_texts, self.train_target_texts = self.__split_data(self.TRAIN_DATA_FILE)
        self.num_train_samples = len(self.train_input_texts)
        self.val_input_texts, self.val_target_texts = self.__split_data(self.VAL_DATA_FILE)
        self.__create_vocab()

        M = Sequential()
        M.add(
            Embedding(self.params['MAX_WORDS_EN'] + 3, self.params['EMBEDDING_DIM'], weights=[self.en_embedding_matrix],
                      mask_zero=True))

        M.add(LSTM(self.params['latent_dim'], return_sequences=True))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        # M.add(LSTM(self.params['latent_dim'] * int(1 / self.params['P_DENSE_DROPOUT']), return_sequences=True))
        M.add(LSTM(self.params['latent_dim'], return_sequences=True))

        M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        M.add(TimeDistributed(Dense(self.params['MAX_WORDS_DE'] + 3,
                                    input_shape=(None, self.params['MAX_SEQ_LEN'], self.params['MAX_WORDS_DE'] + 3),
                                    activation='softmax')))

        print('compiling')

        M.compile(optimizer='Adam', loss='categorical_crossentropy')

        print('compiled')

        steps_per_epoch = 1
        mod_epochs = np.math.floor(
            self.num_train_samples / self.params['batch_size'] / steps_per_epoch * self.params['epochs'])
        tbCallBack = callbacks.TensorBoard(log_dir=self.GRAPH_DIR, histogram_freq=0, write_grads=True, write_graph=True,
                                           write_images=True)
        modelCallback = callbacks.ModelCheckpoint(
            self.MODEL_CHECKPOINT_DIR + '/model.{epoch:03d}-{loss:.3f}-{val-loss:.3f}.hdf5',
            monitor='loss', verbose=1, save_best_only=False,
            save_weights_only=True, mode='auto',
            period=mod_epochs / self.params['epochs'])
        customCallback = CustomCallback()
        M.fit_generator(self.__serve_batch(self.train_input_texts, self.train_target_texts), steps_per_epoch,
                        epochs=mod_epochs, verbose=2, callbacks=[tbCallBack, modelCallback, customCallback],
                        validation_data=self.__serve_batch(self.val_input_texts, self.val_target_texts),
                        validation_steps=len(self.val_input_texts) / self.params['batch_size'])
        M.save(self.model_file)

    def __split_data(self, file):
        """
        Reads the data from the given file.
        The two languages in the file have to be splitted by a tab
        :param file: file which should be read from
        :return: (input_texts, target_texts)
        """
        input_texts = []
        target_texts = []
        lines = open(file, encoding='UTF-8').read().split('\n')
        for line in lines:
            input_text, target_text = line.split('\t')
            input_texts.append(input_text)
            target_text = target_text
            target_texts.append(target_text)

        assert len(input_texts) == len(target_texts)
        return input_texts, target_texts

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

    def __serve_batch(self, input_texts, target_texts):
        counter = 0
        batch_X = np.zeros((self.params['batch_size'], self.params['MAX_SEQ_LEN']), dtype='int32')
        batch_Y = np.zeros(
            (self.params['batch_size'], self.params['MAX_SEQ_LEN'], self.params['MAX_WORDS_DE'] + 3),
            dtype='int32')
        while True:
            for i in range(input_texts.shape[0]):
                in_X = input_texts[i]
                out_Y = np.zeros((1, target_texts.shape[1], self.params['MAX_WORDS_DE'] + 3), dtype='int32')
                token_counter = 0
                for token in target_texts[i]:
                    out_Y[0, token_counter, :] = to_categorical(token, num_classes=self.params['MAX_WORDS_DE'] + 3)
                    token_counter += 1
                batch_X[counter] = in_X
                batch_Y[counter] = out_Y
                counter += 1
                if counter == self.params['batch_size']:
                    print("counter == batch_size", i)
                    counter = 0
                    yield batch_X, batch_Y

    def __setup_model(self):
        try:
            test = self.embedding_matrix
            test = self.M
            return
        except AttributeError:
            pass

        self.embedding_matrix = np.load(self.BASIC_PERSISTENT_DIR + '/embedding_matrix.npy')

        self.M = Sequential()
        self.M.add(
            Embedding(self.params['MAX_WORDS'] + 3, self.params['EMBEDDING_DIM'], weights=[self.embedding_matrix],
                      mask_zero=True))

        self.M.add(LSTM(self.params['latent_dim'], return_sequences=True, name='encoder'))

        self.M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        self.M.add(
            LSTM(self.params['latent_dim'] * int(1 / self.params['P_DENSE_DROPOUT']), return_sequences=True))

        self.M.add(Dropout(self.params['P_DENSE_DROPOUT']))

        self.M.add(TimeDistributed(Dense(self.params['MAX_WORDS'] + 3,
                                         input_shape=(None, self.params['num_tokens'], self.params['MAX_WORDS'] + 3),
                                         activation='softmax')))

        self.M.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.M.load_weights(self.LATEST_MODELCHKPT)

    def predict_one_sentence(self, sentence):
        self.__setup_model()
        tokenizer = Tokenizer()
        self.word_index = np.load(self.BASIC_PERSISTENT_DIR + '/word_index.npy')
        self.word_index = self.word_index.item()
        tokenizer.word_index = self.word_index
        self.num_words = self.params['MAX_WORDS'] + 3
        tokenizer.num_words = self.num_words

        try:
            self.word_index[self.START_TOKEN]
            self.word_index[self.END_TOKEN]
            self.word_index[self.UNK_TOKEN]
        except Exception as e:
            print(e, "why")
            exit()

        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = [self.word_index[self.START_TOKEN]] + sentence[0] + [self.word_index[self.END_TOKEN]]
        sentence = pad_sequences([sentence], maxlen=self.params['max_seq_length'], padding='post')
        sentence = sentence.reshape(sentence.shape[0], sentence.shape[1])
        prediction = self.M.predict(sentence)

        predicted_sentence = ""
        reverse_word_index = dict((i, word) for word, i in self.word_index.items())
        for sentence in prediction:
            for token in sentence:
                max_idx = np.argmax(token)
                if max_idx == 0:
                    print("id of max token = 0")
                    print("second best prediction is ", reverse_word_index[np.argmax(np.delete(token, max_idx))])
                else:
                    next_word = reverse_word_index[max_idx]
                    if next_word == self.END_TOKEN:
                        break
                    elif next_word == self.START_TOKEN:
                        continue
                    predicted_sentence += next_word + " "

        return predicted_sentence

    def predict_batch(self, sentences):
        self.__setup_model()

        tokenizer = Tokenizer()
        self.word_index = np.load(self.BASIC_PERSISTENT_DIR + '/word_index.npy')
        self.word_index = self.word_index.item()
        tokenizer.word_index = self.word_index
        self.num_words = self.params['MAX_WORDS'] + 3
        tokenizer.num_words = self.num_words

        try:
            self.word_index[self.START_TOKEN]
            self.word_index[self.END_TOKEN]
            self.word_index[self.UNK_TOKEN]
        except Exception as e:
            print(e, "why")
            exit()

        sentences = tokenizer.texts_to_sequences(sentences)
        mod_sentences = []
        for sentence in sentences:
            mod_sentences.append([self.word_index[self.START_TOKEN]] + sentence + [self.word_index[self.END_TOKEN]])
        sentences = pad_sequences(mod_sentences, maxlen=self.params['max_seq_length'], padding='post')
        sentences = sentences.reshape(sentences.shape[0], sentences.shape[1])

        batch_size = sentences.shape[0]
        if batch_size > 10:
            batch_size = 10

        reverse_word_index = dict((i, word) for word, i in self.word_index.items())
        predicted_sentences = []
        from_idx = 0
        to_idx = batch_size
        while True:
            print("from_idx, to_idx, hm_sentences", from_idx, to_idx, sentences.shape[0])
            current_batch = sentences[from_idx:to_idx]
            prediction = self.M.predict(current_batch, batch_size=batch_size)

            for sentence in prediction:
                predicted_sent = ""
                for token in sentence:
                    max_idx = np.argmax(token)
                    if max_idx == 0:
                        print("id of max token = 0")
                        print("second best prediction is ", reverse_word_index[np.argmax(np.delete(token, max_idx))])
                    else:
                        next_word = reverse_word_index[max_idx]
                        if next_word == self.END_TOKEN:
                            break
                        elif next_word == self.START_TOKEN:
                            continue
                        predicted_sent += next_word + " "
                predicted_sentences.append(predicted_sent)
            from_idx += batch_size
            to_idx += batch_size
            if to_idx > sentences.shape[0]:
                # todo accept not multiple of batchsize
                break
        return predicted_sentences

    def calculate_hiddenstate_after_encoder(self, sentence):
        self.__setup_model()

        tokenizer = Tokenizer()
        self.word_index = np.load(self.BASIC_PERSISTENT_DIR + '/word_index.npy')
        self.word_index = self.word_index.item()
        tokenizer.word_index = self.word_index
        self.num_words = self.params['MAX_WORDS'] + 3
        tokenizer.num_words = self.num_words

        try:
            self.word_index[self.START_TOKEN]
            self.word_index[self.END_TOKEN]
            self.word_index[self.UNK_TOKEN]
        except Exception as e:
            print(e, "why")
            exit()

        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = [self.word_index[self.START_TOKEN]] + sentence[0] + [self.word_index[self.END_TOKEN]]
        sentence = pad_sequences([sentence], maxlen=self.params['max_seq_length'], padding='post')
        sentence = sentence.reshape(sentence.shape[0], sentence.shape[1])

        encoder_name = 'encoder'

        encoder = Model(inputs=self.M.input, outputs=self.M.get_layer(encoder_name).output)

        prediction = encoder.predict(sentence, batch_size=1)
        print(prediction.shape)
        return prediction

    def calculate_every_hiddenstate_after_encoder(self, sentence):
        raise NotImplementedError()

    def calculate_every_hiddenstate(self, sentence):
        raise NotImplementedError()

    def calculate_hiddenstate_after_decoder(self, sentence):
        raise NotImplementedError()

    def setup_inference(self):
        raise NotImplementedError()
