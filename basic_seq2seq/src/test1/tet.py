from keras.models import Sequential
import numpy as np
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Activation

n_in = 684
n_out = 684
n_hidden = 512
n_samples = 2297
n_timesteps = 87

model = Sequential()
model.add(LSTM(n_hidden, input_shape=(n_timesteps,n_out), return_sequences=True))
model.add(Dense(n_hidden))
model.compile(loss='mse', optimizer='rmsprop')

X = np.random.random((n_samples, n_timesteps, n_in))
Y = np.random.random((n_samples, n_timesteps, n_out))

Xp = model.predict(X)
print(Xp.shape)
print(Y.shape)

model.fit(X, Y, nb_epoch=1)