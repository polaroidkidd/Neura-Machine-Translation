from __future__ import print_function

import numpy as np
from keras import callbacks
from keras.engine import Model
from keras.layers import Dense, Input, LSTM
import os

from ParamHandler import ParamHandler

param_handler = ParamHandler("char", additional=['tf'])

BASE_DATA_DIR = "../../DataSets"
BASIC_PERSISTENT_DIR = '../../persistent/'
GRAPH_DIR = 'graph' + param_handler.param_summary()
MODEL_DIR = 'model' + param_handler.param_summary()
MODEL_CHECKPOINT_DIR = 'chkp' + param_handler.param_summary()

train_data_path = os.path.join(BASE_DATA_DIR, 'Training', 'merged_en_de.txt')


# TODO: save num_encoder_tokens
# TODO: use validation
def vectorize_data(data_path):
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    lines = open(data_path, encoding='UTF-8').read().split('\n')
    counter = 0
    for line in lines[: min(param_handler.params['MAX_NUM_SAMPLES'], len(lines) - 1)]:
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
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    print(len(input_characters))
    print(len(target_characters))
    print(len(input_texts))
    print(len(target_texts))
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    if num_encoder_tokens != param_handler.params['NUM_ENCODER_TOKENS']:
        print("different num_encoder_tokens as expected")
    if num_decoder_tokens != param_handler.params['NUM_DECODER_TOKENS']:
        print("different num_decoder_tokens as expected")
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    if max_encoder_seq_length != param_handler.params['MAX_ENCODER_SEQ_LEN']:
        print("different max_encoder_seq_length as expected")
    if max_decoder_seq_length != param_handler.params['MAX_DECODER_SEQ_LEN']:
        print("different max_decoder_seq_length as expected")

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    return input_texts, input_token_index, target_token_index, target_texts


def serve_batch():
    encoder_input_data = np.zeros((param_handler.params['BATCH_SIZE'], param_handler.params['MAX_ENCODER_SEQ_LEN'],
                                   param_handler.params['NUM_ENCODER_TOKENS']), dtype='float32')
    decoder_input_data = np.zeros((param_handler.params['BATCH_SIZE'], param_handler.params['MAX_DECODER_SEQ_LEN'],
                                   param_handler.params['NUM_DECODER_TOKENS']), dtype='float32')
    decoder_target_data = np.zeros((param_handler.params['BATCH_SIZE'], param_handler.params['MAX_DECODER_SEQ_LEN'],
                                    param_handler.params['NUM_DECODER_TOKENS']), dtype='float32')
    start = 0
    while True:
        for i, (input_text, target_text) in enumerate(zip(input_texts[start:start + param_handler.params['BATCH_SIZE']],
                                                          target_texts[
                                                          start:start + param_handler.params['BATCH_SIZE']])):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_target_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.
        start += param_handler.params['BATCH_SIZE']
        yield [encoder_input_data, decoder_input_data], decoder_target_data


def build_model():
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, param_handler.params['NUM_ENCODER_TOKENS']))
    encoder = LSTM(param_handler.params['LATENT_DIM'], return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, param_handler.params['NUM_DECODER_TOKENS']))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(param_handler.params['LATENT_DIM'], return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(param_handler.params['NUM_DECODER_TOKENS'], activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


input_texts, input_token_index, target_token_index, target_texts = vectorize_data(train_data_path)

model = build_model()
model.summary()
print(param_handler.param_summary())
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

tbCallBack = callbacks.TensorBoard(log_dir=os.path.join(BASIC_PERSISTENT_DIR, GRAPH_DIR), histogram_freq=1,
                                   write_graph=True, write_images=True)
modelCallback = callbacks.ModelCheckpoint(
    BASIC_PERSISTENT_DIR + MODEL_CHECKPOINT_DIR + 'model.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False,
    mode='auto', period=1)

# model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=param_handler.params['BATCH_SIZE'],
#          epochs=param_handler.params['EPOCHS'],
#         validation_split=0.2, verbose=1, callbacks=[tbCallBack, modelCallback])

model.fit_generator(serve_batch(), len(input_texts) / param_handler.params['BATCH_SIZE'],
                    epochs=param_handler.params['EPOCHS'], verbose=2)
# Save model
model.save(os.path.join(BASE_DATA_DIR, MODEL_DIR, 'model.h5'))
