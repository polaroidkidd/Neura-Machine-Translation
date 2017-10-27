# TODO implement custom callback for keras.
# Callback should be called after each epoch and print or save the results of a prediction of some static sentences.
# To evaluate if model implementation is correct
import keras
import numpy as np

from metrics.Bleu import Bleu


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, generator, steps):
        super(CustomCallback, self).__init__()
        self.generator = generator
        self.steps = steps

    def on_epoch_end(self, epoch, logs={}):
        predictions = []
        for i in range(self.steps):
            batch_X, batch_Y = next(self.generator)
            print("on_epoch_end", i, batch_X.shape)
            print("on_epoch_end", i, batch_Y.shape)
            prediction = self.model.predict(self.validation_data[0])
            print("on_epoch_end", prediction.shape)
            predictions.append(prediction)
        print("on_epoch_end", np.asarray(predictions).shape)

        bleu_score = Bleu().evaluate_hypothesis_corpus(predictions, self.validation_data[1], epoch=epoch)
