from __future__ import print_function

import numpy as np
from keras.layers import Dense, Input, LSTM
from keras.models import Model
from keras.models import load_model
from keras import callbacks
import os
import gc
from models.BaseModel import BaseModel

class KerasTutSeq2Seq(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.params['BATCH_SIZE'] = 256
        self.params['EPOCHS'] = 100
        self.params['LATENT_DIM'] = 256
        self.params['MAX_NUM_SAMPLES'] = 3000000
        self.params['NUM_TOKENS'] = 127
        self.params['MAX_ENCODER_SEQ_LEN'] = 286
        self.params['MAX_DECODER_SEQ_LEN'] = 382

        self.UNKNOWN_CHAR = '\r'
        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENT_DIR = '../../persistent/model_keras_tut_original'
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.LATEST_MODEL_CHKPT = os.path.join(self.MODEL_CHECKPOINT_DIR,
                                               'chkp22_64_100_15_256_1000000_20000_1000_1800_150_150_0.8_char___tfmodel2.6999-47.37.hdf5')
        self.token_idx_file = os.path.join(self.BASIC_PERSISTENT_DIR, "token_index.npy")
        self.train_data_file = os.path.join(self.BASE_DATA_DIR, 'Training/deu.txt')
        self.encoder_model_file = os.path.join(self.MODEL_DIR, 'encoder_model.h5')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.decoder_model_file = os.path.join(self.MODEL_DIR, 'decoder_model.h5')

    def start_training(self):
        input_texts, target_texts, token_index = self.split_count_data()
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        encoder = LSTM(self.params['LATENT_DIM'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['LATENT_DIM'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.params['NUM_TOKENS'], activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        normal_epochs = 10
        steps = 5
        mod_epochs = np.math.floor(len(input_texts) / self.params['BATCH_SIZE'] / steps * normal_epochs)
        model.fit_generator(self.serve_batch(input_texts, target_texts, token_index), steps, epochs=mod_epochs,
                            verbose=2, max_queue_size=3)

        model.save(self.model_file)

        encoder_model = Model(encoder_inputs, encoder_states)
        encoder_model.save(self.encoder_model_file)

        decoder_state_input_h = Input(shape=(self.params['LATENT_DIM'],))
        decoder_state_input_c = Input(shape=(self.params['LATENT_DIM'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        decoder_model.save(self.decoder_model_file)

    def split_count_data(self):
        # Vectorize the data.
        input_texts = []
        target_texts = []
        input_characters = set()
        lines = open(self.train_data_file, encoding='UTF-8').read().split('\n')
        for line in lines[: min(self.params['MAX_NUM_SAMPLES'], len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in input_characters:
                    input_characters.add(char)
            input_text = None
            target_text = None
        lines = None
        input_characters.add('\r')
        input_characters = sorted(list(input_characters))
        num_tokens = len(input_characters)
        if num_tokens != self.params['NUM_TOKENS']:
            print("number of tokens is different. expected", self.params['NUM_TOKENS'], 'got', num_tokens)
            exit()
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])
        if max_encoder_seq_length != self.params['MAX_ENCODER_SEQ_LEN']:
            print("other max encoder seq length. expected", self.params['MAX_ENCODER_SEQ_LEN'], 'got',
                  max_encoder_seq_length)
            exit()
        if max_decoder_seq_length != self.params['MAX_DECODER_SEQ_LEN']:
            print("other max decoder seq length. expected", self.params['MAX_DECODER_SEQ_LEN'], 'got',
                  max_decoder_seq_length)
            exit()

        print('Number of samples:', len(input_texts))
        print('Number of unique input tokens:', self.params['NUM_TOKENS'])
        print('Max sequence length for inputs:', self.params['MAX_ENCODER_SEQ_LEN'])
        print('Max sequence length for outputs:', self.params['MAX_DECODER_SEQ_LEN'])

        token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        input_characters = None

        gc.collect()
        # np.save('token_index.npy', token_index)
        return input_texts, target_texts, token_index

    def setup_inference(self):
        encoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        encoder = LSTM(self.params['LATENT_DIM'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.params['NUM_TOKENS']))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['LATENT_DIM'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.params['NUM_TOKENS'], activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        model = load_model(self.model_file)

        self.encoder_model = Model(encoder_inputs, encoder_states)
        self.encoder_model = load_model(self.encoder_model_file)

        decoder_state_input_h = Input(shape=(self.params['LATENT_DIM'],))
        decoder_state_input_c = Input(shape=(self.params['LATENT_DIM'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        self.decoder_model = load_model(self.decoder_model_file)

        self.char_index = np.load(self.token_idx_file)
        self.char_index = self.char_index.item()
        self.reverse_char_index = dict((i, char) for char, i in self.char_index.items())

    def setup_inferencedd(self):
        model = load_model(self.model_file)
        self.encoder_model = load_model(self.encoder_model_file)
        self.decoder_model = load_model(self.decoder_model_file)
        self.char_index = np.load(self.token_idx_file)
        print(self.char_index)
        self.char_index = self.char_index.item()
        self.reverse_char_index = dict((i, char) for char, i in self.char_index.items())

    def predict(self, sentence):
        input_seq = np.zeros((1, self.params['MAX_ENCODER_SEQ_LEN'], self.params['NUM_TOKENS']))

        index = 0
        for char in sentence:
            try:
                input_seq[0][index][self.char_index[char]] = 1.
            except KeyError:
                input_seq[0][index][self.char_index['\r']] = 1.
            index += 1

        decoded_sentence = self.decode_sequence(input_seq)
        return decoded_sentence

    def decode_sequence(self, input_sequence):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_sequence)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.params['NUM_TOKENS']))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.char_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_sentence) > self.params['MAX_DECODER_SEQ_LEN']):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.params['NUM_TOKENS']))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def serve_batch(self, input_texts, target_texts, token_index):
        gc.collect()
        encoder_input_data = np.zeros(
            (self.params['BATCH_SIZE'], self.params['MAX_ENCODER_SEQ_LEN'], self.params['NUM_TOKENS']), dtype='float32')
        decoder_input_data = np.zeros(
            (self.params['BATCH_SIZE'], self.params['MAX_DECODER_SEQ_LEN'], self.params['NUM_TOKENS']), dtype='float32')
        decoder_target_data = np.zeros(
            (self.params['BATCH_SIZE'], self.params['MAX_DECODER_SEQ_LEN'], self.params['NUM_TOKENS']), dtype='float32')
        start = 0
        print("serve_batch", "start", start)
        while True:
            for i, (input_text, target_text) in enumerate(
                    zip(input_texts[start:start + self.params['BATCH_SIZE']],
                        target_texts[start:start + self.params['BATCH_SIZE']])):
                for t, char in enumerate(input_text):
                    try:
                        encoder_input_data[i, t, token_index[char]] = 1.
                    except KeyError:
                        encoder_input_data[i, t, token_index['\r']] = 1.
                for t, char in enumerate(target_text):
                    # decoder_target_data is ahead of decoder_target_data by one timestep
                    try:
                        decoder_input_data[i, t, token_index[char]] = 1.
                    except KeyError:
                        decoder_input_data[i, t, token_index['\r']] = 1.
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        try:
                            decoder_target_data[i, t - 1, token_index[char]] = 1.
                        except KeyError:
                            decoder_target_data[i, t - 1, token_index['\r']] = 1.
            start += self.params['BATCH_SIZE']
            print("serve_batch", start, start / self.params['BATCH_SIZE'])
            yield [encoder_input_data, decoder_input_data], decoder_target_data


    def get_hidden_state(self, sentence):
        input_seq = np.zeros((1, self.params['MAX_ENCODER_SEQ_LEN'], self.params['NUM_TOKENS']))

        index = 0
        for char in sentence:
            try:
                input_seq[0][index][self.char_index[char]] = 1.
            except KeyError:
                input_seq[0][index][self.char_index['\r']] = 1.
            index += 1

        states_value = self.encoder_model.predict(input_seq)
        return states_value