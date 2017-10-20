# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import os

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

MAX_NUM_WORDS = 20000
MAX_SEQ_LEN = 500

BASE_DATA_DIR = ""
TRAIN_EN_FILE = "train100.en"
TRAIN_DE_FILE = "train100.de"

german_train_file = os.path.join(BASE_DATA_DIR, TRAIN_DE_FILE)
with(open(german_train_file, 'r')) as file:
    target_data = file.readlines()

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(target_data)
encoded_target_data = tokenizer.texts_to_sequences(target_data)
padded_target_data = pad_sequences(encoded_target_data, maxlen=MAX_SEQ_LEN, padding='post')
padded_target_data = add_one_hot_dim(padded_target_data)
print(padded_target_data.shape)
print(padded_target_data)
