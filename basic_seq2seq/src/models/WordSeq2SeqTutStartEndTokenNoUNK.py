from __future__ import print_function

import gc
import numpy as np
from keras.layers import Dense, Input, LSTM, Embedding
from keras.models import Model
from keras.models import load_model
from keras import callbacks
import os

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from models.BaseModel import BaseModel


class Seq2Seq2(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.identifier = 'WordSeq2SeqTutStartEndTokenNoUNK'

        self.params['batch_size'] = 128
        self.params['epochs'] = 100
        self.params['latent_dim'] = 256
        self.params['num_samples'] = 150000
        self.params['num_tokens'] = 91
        self.params['max_seq_length'] = 100
        self.params['EMBEDDING_DIM'] = 100
        self.params['MAX_WORDS'] = 20000

        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENT_DIR = '../../persistent/WordSeq2SeqTutStartEndTokenNoUNK'
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.GRAPH_DIR = os.path.join(self.BASIC_PERSISTENT_DIR, 'Graph')
        self.token_idx_file = os.path.join(self.BASIC_PERSISTENT_DIR, "token_index.npy")
        self.data_path = os.path.join(self.BASE_DATA_DIR, 'Training/deu.txt')
        self.encoder_model_file = os.path.join(self.MODEL_DIR, 'encoder_model.h5')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.weights_file = os.path.join(self.MODEL_DIR, 'weights.h5')
        self.decoder_model_file = os.path.join(self.MODEL_DIR, 'decoder_model.h5')
        self.PRETRAINED_GLOVE_FILE = os.path.join(self.BASE_DATA_DIR, 'glove.6B.100d.txt')
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.LATEST_MODEL_CHKPT = os.path.join(self.BASIC_PERSISTENT_DIR, 'model.12999-2774.86.hdf5')
        self.WORD_IDX_FILE = os.path.join(self.BASIC_PERSISTENT_DIR, "word_idx.npy")

        self.START_TOKEN = "_GO"
        self.END_TOKEN = "_EOS"

    def start_training(self):
        self._split_count_data()

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(self.num_words, self.params['EMBEDDING_DIM'], weights=[self.embedding_matrix],
                                      mask_zero=True)
        encoder_embedded = encoder_embedding(encoder_inputs)
        encoder = LSTM(self.params['latent_dim'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_embedded)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(self.num_words, self.params['EMBEDDING_DIM'], weights=[self.embedding_matrix],
                                      mask_zero=True)
        decoder_embedded = decoder_embedding(decoder_inputs)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['latent_dim'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
        decoder_dense = Dense(self.num_words, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        steps = 5
        mod_epochs = np.math.floor(self.num_samples / self.params['batch_size'] / steps * self.params['epochs'])
        tbCallBack = callbacks.TensorBoard(log_dir=self.GRAPH_DIR, histogram_freq=0, write_graph=True,
                                           write_images=True)
        modelCallback = callbacks.ModelCheckpoint(self.MODEL_CHECKPOINT_DIR + '/model.{epoch:02d}-{loss:.2f}.hdf5',
                                                  monitor='loss', verbose=1, save_best_only=False,
                                                  save_weights_only=True, mode='auto',
                                                  period=mod_epochs / self.params['epochs'])
        model.fit_generator(self.serve_batch(), steps, epochs=mod_epochs, verbose=2, max_queue_size=5,
                            callbacks=[tbCallBack, modelCallback])

        #
        # Save model
        model.save(self.model_file)
        model.save_weights(self.weights_file)

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states)
        encoder_model.save(self.encoder_model_file)

        decoder_state_input_h = Input(shape=(self.params['latent_dim'],))
        decoder_state_input_c = Input(shape=(self.params['latent_dim'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        decoder_model.save(self.decoder_model_file)

    def _serve_batch(self):
        # gc.collect()
        self.encoder_input_data = np.zeros((self.params['batch_size'], self.params['max_seq_length']),
                                           dtype='int32')
        self.decoder_input_data = np.zeros((self.params['batch_size'], self.params['max_seq_length']),
                                           dtype='int32')
        self.decoder_target_data = np.zeros((self.params['batch_size'], self.params['max_seq_length'], self.num_words),
                                            dtype='float32')

        from_idx = 0
        while True:
            to_idx = from_idx + self.params['batch_size']
            self.encoder_input_data = self.input_texts[from_idx:to_idx]
            self.decoder_input_data = self.target_texts[from_idx:to_idx]

            for i, (input_text, target_text) in enumerate(
                    zip(self.input_texts[from_idx:to_idx], self.target_texts[from_idx:to_idx])):
                last_t = -1
                for t, token in enumerate(target_text):
                    # decoder_target_data is ahead of decoder_target_data by one timestep
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        self.decoder_target_data[i, t - 1, token] = 1.

            from_idx += self.params['batch_size']
            if from_idx + self.params['batch_size'] > len(self.input_texts):
                from_idx = 0
            yield [self.encoder_input_data, self.decoder_input_data], self.decoder_target_data

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

        np.save(self.WORD_IDX_FILE, self.word_index)

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

        np.save(self.BASIC_PERSISTENT_DIR + '/embedding_matrix.npy', self.embedding_matrix)

    def predict_one_sentence_old(self, sentence):
        self.word_index = np.load(self.WORD_IDX_FILE)
        self.word_index = self.word_index.item()
        tokenizer = Tokenizer(num_words=self.params['MAX_WORDS'])
        tokenizer.word_index = self.word_index
        tokenizer.num_words = len(self.word_index)

        sentence = tokenizer.texts_to_sequences([sentence])

        sentence = pad_sequences(sentence, maxlen=self.params['max_seq_length'], padding='post')

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

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(self.num_words, self.params['EMBEDDING_DIM'], weights=[self.embedding_matrix],
                                      mask_zero=True)
        encoder_embedded = encoder_embedding(encoder_inputs)
        encoder = LSTM(self.params['latent_dim'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_embedded)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(self.num_words, self.params['EMBEDDING_DIM'], weights=[self.embedding_matrix],
                                      mask_zero=True)
        decoder_embedded = decoder_embedding(decoder_inputs)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['latent_dim'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
        decoder_dense = Dense(self.num_words, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        model.load_weights(self.LATEST_MODEL_CHKPT)

        # Define sampling models
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.params['latent_dim'],), name='dec_state_inpu_h')
        decoder_state_input_c = Input(shape=(self.params['latent_dim'],), name='dec_state_input_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedded, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        decoded_sentence = self.decode_sequence(sentence)
        return decoded_sentence

    def _set_up_inference(self):
        self.word_index = np.load(self.WORD_IDX_FILE)
        self.word_index = self.word_index.item()

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

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(self.num_words, self.params['EMBEDDING_DIM'], weights=[self.embedding_matrix],
                                      mask_zero=True)
        encoder_embedded = encoder_embedding(encoder_inputs)
        encoder = LSTM(self.params['latent_dim'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_embedded)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(self.num_words, self.params['EMBEDDING_DIM'], weights=[self.embedding_matrix],
                                      mask_zero=True)
        decoder_embedded = decoder_embedding(decoder_inputs)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['latent_dim'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
        decoder_dense = Dense(self.num_words, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        model.load_weights(self.LATEST_MODEL_CHKPT)

        # Define sampling models
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.params['latent_dim'],), name='dec_state_inpu_h')
        decoder_state_input_c = Input(shape=(self.params['latent_dim'],), name='dec_state_input_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedded, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        #model = load_model(self.model_file)
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        #model.load_weights(self.LATEST_MODEL_CHKPT)
        #self.encoder_model = load_model(self.encoder_model_file)
        #self.decoder_model = load_model(self.decoder_model_file)

    def predict_one_sentence(self, sentence):
        self._set_up_inference()

        tokenizer = Tokenizer(num_words=self.params['MAX_WORDS'])
        tokenizer.word_index = self.word_index
        tokenizer.num_words = len(self.word_index)
        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=self.params['max_seq_length'], padding='post')

        decoded_sentence = self.decode_sequence(sentence)
        return decoded_sentence

    def decode_sequence(self, input_sequence):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_sequence)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, len(self.word_index)))
        # Populate the first character of target sequence with the start character.
        target_seq[0, self.word_index[self.START_TOKEN]] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        p = [target_seq] + states_value
        # print(np.array(p).shape)
        stop_condition = False
        decoded_sentence = ''
        reverse_word_index = dict((i, word) for word, i in self.word_index.items())
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            try:
                sampled_char = reverse_word_index[sampled_token_index]
                print(sampled_char)
            except KeyError:
                print("key error", sampled_token_index)
            decoded_sentence += sampled_char + " "

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == self.END_TOKEN or len(decoded_sentence) > self.params['max_seq_length']):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, len(self.word_index)))
            target_seq[0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

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