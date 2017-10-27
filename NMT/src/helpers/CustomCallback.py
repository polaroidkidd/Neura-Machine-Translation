# TODO implement custom callback for keras.
# Callback should be called after each epoch and print or save the results of a prediction of some static sentences.
# To evaluate if model implementation is correct
import keras
import numpy as np


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print("callback", self.validation_data)
        print("\n\n")
        print("callback", self.validation_data.shape)

        print(self.validation_data[0])
        print(self.validation_data[0].shape)
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        print(predict.shape)
        print(predict)
        print(self.validation_data[1])
        print(self.validation_data[1].shape)

        with(open("../../Persistence/WordBasedSeq2Seq1000Units20EpochsGLOVE/customcallback.txt", 'a')) as out_file:
            out_file.write("test")
