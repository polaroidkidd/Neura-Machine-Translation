from keras import Input
from keras.engine import Model
from keras.layers import Embedding, LSTM, TimeDistributed, Dense
import numpy as np
from keras.utils import to_categorical

max_sequence_len = 0
vocab_size = 0
batch_size = 0

def serve_sentence(data):
    for data_i in np.random.choice(len(data), len(data), replace=False):
        in_X = np.zeros(max_sequence_len)
        out_y = np.zeros(max_sequence_len, dtype=np.int32)
        bigram_data = zip(data[data_i][0:-1], data[data_i][1:])
        for datum_j, (datum_in, datum_out) in enumerate(bigram_data):
            in_X[datum_j] = datum_in
            out_y[datum_j] = datum_out
        yield in_X, out_y


def serve_batch(data):
    dataiter = serve_sentence(data)
    V = vocab_size
    S = max_sequence_len
    B = batch_size
    while dataiter:
        in_X = np.zeros((B, S), dtype=np.int32)
        out_Y = np.zeros((B, S, V), dtype=np.int32)
        next_batch = list(np.itertools.isslice(dataiter, 0, batch_size))
        if len(next_batch) < batch_size:
            raise StopIteration
        for d_i, (d_X, d_Y) in enumerate(next_batch):
            in_X[d_i] = d_X
            out_Y[d_i] = to_categorical(d_Y, V)
        yield in_X, out_Y


xin = Input(batch_shape=(batch, timesteps), dytpe='int32')
xemb = Embedding(vocab_size, embedding_size)(xin)  # 3 dim (batch,time,feat)
seq = LSTM(seq_size, return_sequences=True)(xemb)
mlp = TimeDistributed(Dense(mlp_size, activation='softmax'))(seq)
model = Model(input=xin, output=mlp)
model.compile(optimizer='Adam', loss='categorical_crossentropy')
