import keras
import numpy as np


class EvalCallback(keras.callbacks.Callback):
    def __init__(self, val_data_generator, val_steps, sm_epochs_for_one_epoch, model_identifier):
        super(EvalCallback, self).__init__()

        self.val_data_generator = val_data_generator
        self.validation_steps = val_steps
        self.sm_epochs_for_one_epoch = sm_epochs_for_one_epoch
        self.model_identifier = model_identifier

    def on_epoch_end(self, epoch, logs={}):
        if epoch > 1 and epoch % self.sm_epochs_for_one_epoch == 0:
            print("now eval callback:")
            losses = 0.
            num = 0
            for i in range(self.validation_steps):
                x, y = next(self.val_data_generator)
                losses += self.model.evaluate(x, y, x.shape[0])
                num += 1
            losses = losses / num

            print("epoch:", epoch, "real epoch:", int(np.floor(epoch / self.sm_epochs_for_one_epoch)),
                  "validation Loss:", losses)

            with(open('../../Persistence/val_data' + self.model_identifier + '.txt', 'a')) as file:
                file.write("epoch" + str(epoch) + "realepoch" + str(
                    int(np.floor(epoch / self.sm_epochs_for_one_epoch))) + "val_loss" + str(losses))
