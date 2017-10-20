from __future__ import print_function

import os

from keras.engine import Model
from keras.layers import Dense, Input, LSTM


params = {}
params['BATCH_SIZE'] = 64
params['EMBEDDING_DIM'] = 100
params['EPOCHS'] = 15
params['LATENT_DIM'] = 256
params['NUM_TOKENS'] = 70
params['MAX_TOKENS'] = 70
params['MAX_NUM_SAMPLES'] = 1000000
params['MAX_NUM_WORDS'] = 20000
params['MAX_SENTENCES'] = 1000
params['MAX_SEQ_LEN'] = 1800


BASE_DATA_DIR = "../../../persistent"
MODEL_DIR = 'persistentModelseq2seqbugfree'

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, params['MAX_TOKENS']))
encoder = LSTM(params['LATENT_DIM'], return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, params['MAX_TOKENS']))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(params['LATENT_DIM'], return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(params['MAX_TOKENS'], activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()
# Run training
model.load_weights(os.path.join(BASE_DATA_DIR, MODEL_DIR,
                                'chkp2prepro_64_100_15_256_1000000_20000_1000_1800_70_70_0.8_char___tfmodelprepro.9999-42.87.hdf5'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Save model
model.save(os.path.join(BASE_DATA_DIR, MODEL_DIR, 'model.h5'))

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save(os.path.join(BASE_DATA_DIR, MODEL_DIR, 'encoder_model.h5'))

decoder_state_input_h = Input(shape=(params['LATENT_DIM'],))
decoder_state_input_c = Input(shape=(params['LATENT_DIM'],))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
decoder_model.save(os.path.join(BASE_DATA_DIR, MODEL_DIR, 'decoder_model.h5'))
