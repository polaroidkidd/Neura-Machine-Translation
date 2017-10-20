from __future__ import print_function

import gc
import numpy as np
from keras import callbacks
from keras.engine import Model
from keras.layers import Dense, Input, LSTM
from keras.models import load_model
import os

from models.BaseModel import BaseModel


class Seq2Seq1(BaseModel):
    def __init__(self):
        self.params = {}
        self.params['BATCH_SIZE'] = 64
        self.params['EMBEDDING_DIM'] = 100
        self.params['EPOCHS'] = 15
        self.params['LATENT_DIM'] = 256
        self.params['MAX_DECODER_SEQ_LEN'] = 2174
        self.params['NUM_DECODER_TOKENS'] = 267
        self.params['MAX_ENCODER_SEQ_LEN'] = 1871
        self.params['NUM_ENCODER_TOKENS'] = 254
        self.params['MAX_NUM_SAMPLES'] = 1000000
        self.params['MAX_NUM_WORDS'] = 20000
        self.params['MAX_SENTENCES'] = 1000

        self.BASE_DATA_DIR = "../../DataSets"
        self.BASIC_PERSISTENT_DIR = '../../persistent/model_einrdan_seq2seq1'
        self.input_token_idx_file = os.path.join(self.BASIC_PERSISTENT_DIR, "input_token_idx.npy")
        self.target_token_idx_file = os.path.join(self.BASIC_PERSISTENT_DIR, "target_token_idx.npy")
        self.train_data_path = os.path.join(self.BASE_DATA_DIR, 'Training', 'merged_en_de.txt')
        self.GRAPH_DIR = os.path.join(self.BASIC_PERSISTENT_DIR,
                                      'graph_64_100_15_256_2174_1871_1000000_20000_1000_100_267_254_0.8_char___tf')
        self.MODEL_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.MODEL_CHECKPOINT_DIR = os.path.join(self.BASIC_PERSISTENT_DIR)
        self.LATEST_MODEL_CHKPT = os.path.join(self.MODEL_CHECKPOINT_DIR,
                                               'chkp_64_100_15_256_2174_1871_1000000_20000_1000_100_267_254_0.8_char___tfmodel.54999-72.71.hdf5')
        self.encoder_model_file = os.path.join(self.MODEL_DIR, 'encoder_model.h5')
        self.model_file = os.path.join(self.MODEL_DIR, 'model.h5')
        self.decoder_model_file = os.path.join(self.MODEL_DIR, 'decoder_model.h5')

    def start_training(self):
        input_texts, input_token_index, target_token_index, target_texts = self._split_data_and_count(
            self.train_data_path)
        gc.collect()
        np.save(self.input_token_idx_file, input_token_index)
        np.save(self.target_token_idx_file, target_token_index)

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.params['NUM_ENCODER_TOKENS']))
        encoder = LSTM(self.params['LATENT_DIM'], return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.params['NUM_DECODER_TOKENS']))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.params['LATENT_DIM'], return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.params['NUM_DECODER_TOKENS'], activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()
        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        tbCallBack = callbacks.TensorBoard(log_dir=self.GRAPH_DIR, histogram_freq=0, write_graph=True,
                                           write_images=True)
        modelCallback = callbacks.ModelCheckpoint(self.MODEL_CHECKPOINT_DIR + 'modelprepro.{epoch:02d}-{loss:.2f}.hdf5',
                                                  monitor='loss', verbose=1, save_best_only=False,
                                                  save_weights_only=False, mode='auto', period=1000)

        steps = 1
        mod_epochs = np.floor(
            len(input_texts) / self.params['BATCH_SIZE'] / steps * self.params['EPOCHS'])
        print('steps', steps, 'mod_epochs', mod_epochs, 'len(train_input_texts)', len(input_texts), "batch_size",
              self.params['BATCH_SIZE'], 'epochs', self.params['EPOCHS'])
        model.fit_generator(self.serve_batch(input_texts, target_texts, input_token_index, target_token_index),
                            steps, epochs=mod_epochs, verbose=2, max_queue_size=1,
                            # validation_data=serve_batch(val_input_texts, val_target_texts, train_input_token_idx,
                            #                            train_target_token_idx),
                            # validation_steps=len(val_input_texts) / self.params['BATCH_SIZE'],
                            callbacks=[tbCallBack, modelCallback])

        # Save model
        model.save(self.model_file)

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states
        # model = load_model('s2s.h5')

        # Define sampling models
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

    def setup_inference(self):
        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states


        # input_texts, input_token_index, target_token_index, target_texts = self._split_data_and_count(
        #    self.train_data_path)

        # Define sampling models
        model = load_model(self.model_file)
        model.load_weights(self.LATEST_MODEL_CHKPT)
        self.encoder_model = load_model(self.encoder_model_file)
        self.decoder_model = load_model(self.decoder_model_file)

        input_tokens = np.load(self.input_token_idx_file)
        target_tokens = np.load(self.target_token_idx_file)

        self.input_tokens = input_tokens = input_tokens.item()
        target_tokens = target_tokens.item()
        self.reverse_input_char_index = dict((i, char) for char, i in input_tokens.items())
        self.reverse_target_char_index = dict((i, char) for char, i in target_tokens.items())

    def predict(self, sentence):
        input_seq = np.zeros((1, self.params['MAX_ENCODER_SEQ_LEN'], self.params['NUM_ENCODER_TOKENS']))

        index = 0
        for char in sentence:
            try:
                input_seq[0][index][self.input_tokens[char]] = 1.
            except KeyError:
                input_seq[0][index][self.input_tokens['\r']] = 1.
            index += 1

        decoded_sentence = self.decode_sequence(input_seq)
        return decoded_sentence

    def _split_data_and_count(self, data_path):
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        lines = open(data_path, encoding='UTF-8').read().split('\n')
        counter = 0
        for line in lines[: min(self.params['MAX_NUM_SAMPLES'], len(lines) - 1)]:
            try:
                input_text, target_text = line.split('\t')
                counter += 1
            except Exception:
                print(line)
                print(counter)
                print(line.split('\t'))
                exit()
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            if len(input_text) == 0 or len(target_text) == 0:
                continue
            # if len(input_text) > params['MAX_ENCODER_SEQ_LEN']:
            #    input_text = input_text[0:2999]
            # if len(target_text) > params['MAX_DECODER_SEQ_LEN']:
            #    target_text = target_text[0:2999]
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        lines = None
        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        if num_encoder_tokens != self.params['NUM_ENCODER_TOKENS']:
            print("different num_encoder_tokens as expected. Expected:",
                  self.params['NUM_ENCODER_TOKENS'], 'was',
                  num_encoder_tokens)
        if num_decoder_tokens != self.params['NUM_DECODER_TOKENS']:
            print("different num_decoder_tokens as expected. Expected:",
                  self.params['NUM_DECODER_TOKENS'], 'was',
                  num_decoder_tokens)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])
        if max_encoder_seq_length != self.params['MAX_ENCODER_SEQ_LEN']:
            print("different max_encoder_seq_length as expected. Expected:",
                  self.params['MAX_ENCODER_SEQ_LEN'],
                  'was', max_encoder_seq_length)
        if max_decoder_seq_length != self.params['MAX_DECODER_SEQ_LEN']:
            print("different max_decoder_seq_length as expected. Expected:",
                  self.params['MAX_DECODER_SEQ_LEN'],
                  'was', max_decoder_seq_length)

        print('Number of samples:', len(input_texts))
        print('Number of unique input tokens:', num_encoder_tokens)
        print('Number of unique output tokens:', num_decoder_tokens)
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
        return input_texts, input_token_index, target_token_index, target_texts

    def decode_sequence(self, input_sequence):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_sequence)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.params['NUM_DECODER_TOKENS']))
        # Populate the first character of target sequence with the start character.

        # target_seq[0, 0, input_token_index['\t']] = 1.
        target_seq[0, 0, 0] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]

            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_sentence) > self.params['MAX_DECODER_SEQ_LEN']):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.params['NUM_DECODER_TOKENS']))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def serve_batch(self, input_data, target_data, token_index, target_token_index):
        gc.collect()
        encoder_input_data = np.zeros((self.params['BATCH_SIZE'], self.params['MAX_ENCODER_SEQ_LEN'],
                                       self.params['NUM_ENCODER_TOKENS']), dtype='float32')
        decoder_input_data = np.zeros((self.params['BATCH_SIZE'], self.params['MAX_DECODER_SEQ_LEN'],
                                       self.params['NUM_DECODER_TOKENS']), dtype='float32')
        decoder_target_data = np.zeros((self.params['BATCH_SIZE'], self.params['MAX_DECODER_SEQ_LEN'],
                                        self.params['NUM_DECODER_TOKENS']), dtype='float32')
        start = 0
        print("serve_batch", "start", start)
        while True:
            for i, (input_text, target_text) in enumerate(
                    zip(input_data[start:start + self.params['BATCH_SIZE']],
                        target_data[start:start + self.params['BATCH_SIZE']])):
                for t, char in enumerate(input_text):
                    char = char.lower()
                    encoder_input_data[i, t, token_index[char]] = 1.
                for t, char in enumerate(target_text):
                    char = char.lower()
                    # decoder_target_data is ahead of decoder_target_data by one timestep
                    decoder_input_data[i, t, target_token_index[char]] = 1.
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            start += self.params['BATCH_SIZE']
            print("serve_batch", start, start / self.params['BATCH_SIZE'])
            yield [encoder_input_data, decoder_input_data], decoder_target_data

    def get_hidden_state(self, sentence):
        input_seq = np.zeros((1, self.params['MAX_ENCODER_SEQ_LEN'], self.params['NUM_ENCODER_TOKENS']))

        index = 0
        for char in sentence:
            try:
                input_seq[0][index][self.input_tokens[char]] = 1.
            except KeyError:
                input_seq[0][index][self.input_tokens['\r']] = 1.
            index += 1

        states_value = self.encoder_model.predict(input_seq)
        return states_value
