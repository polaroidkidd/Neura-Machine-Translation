from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        self.params = {}

    @abstractmethod
    def start_training(self):
        pass

    @abstractmethod
    def predict_one_sentence(self, sentence):
        pass

    @abstractmethod
    def predict_batch(self, sentences):
        pass

    @abstractmethod
    def calculate_hiddenstate_after_encoder(self, sentence):
        pass

    @abstractmethod
    def calculate_hiddenstate_after_decoder(self, sentence):
        pass

    @abstractmethod
    def calculate_every_hiddenstate_after_encoder(self, sentence):
        pass

    @abstractmethod
    def calculate_every_hiddenstate(self, sentence):
        pass
