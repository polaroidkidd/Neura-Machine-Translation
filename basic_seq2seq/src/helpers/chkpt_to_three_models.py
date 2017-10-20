from __future__ import print_function

import gc
import os

import numpy as np
from keras.engine import Model
from keras.layers import Dense, Input, LSTM


class Chkpt_to_Models():
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
        self.params['MAX_SEQ_LEN'] = 100
        self.params['P_DENSE_DROPOUT'] = 0.8
        self.BASIC_PERSISTENT_DIR = '../../persistent/'
        self.CHECKPOINT_FILE = 'chkp_64_100_15_256_2174_1871_1000000_20000_1000_100_267_254_0.8_char___tfmodel.40999-72.71.hdf5'

    def start(self):
        self.build_model()

    def build_model(self):
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
        model.load_weights(os.path.join(self.BASIC_PERSISTENT_DIR, 'model_einrdan_seq2seq1', self.CHECKPOINT_FILE))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # Save model
        model.save(os.path.join(self.BASIC_PERSISTENT_DIR, 'model_einrdan_seq2seq1', 'model.h5'))

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
        encoder_model.save(os.path.join(self.BASIC_PERSISTENT_DIR, 'model_einrdan_seq2seq1', 'encoder_model.h5'))
        # encoder_model = load_model('./data/encoder_model.h5')


        decoder_state_input_h = Input(shape=(self.params['LATENT_DIM'],))
        decoder_state_input_c = Input(shape=(self.params['LATENT_DIM'],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        decoder_model.save(os.path.join(self.BASIC_PERSISTENT_DIR, 'model_einrdan_seq2seq1', 'decoder_model.h5'))
