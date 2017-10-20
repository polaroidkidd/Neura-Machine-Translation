from inference.inference import Inference
import numpy as np


class CharBasedInference(Inference):
    def __init__(self, model_file, input_token_idx_file, target_token_idx_file):
        self.model = self._load_model_from_file(model_file)
        self.input_token_index = np.load(input_token_idx_file)
        self.target_token_index = np.load(target_token_idx_file)


def _load_model_from_file(self, model_file):
    """
    Loads an pretrained model from the given file
    :param model_file:
    :return:
    """
    if model_file is None:
        pass
        # TODO how to throw exception
    print("\nModel is loading...\n")
    model = None
    return model


def _preprocess_source_sentence(self, source_sentence):
    preprocessed_sentence = None
    return preprocessed_sentence


def _prediction_to_sentence(self, prediction):
    """
    Converts a prediction to a sentence
    :param prediction:
    :return:
    """
    pass


def translate(self, model, source_sentence):
    preprocessed_src_sentence = self._preprocess_source_sentence(source_sentence)
    prediction = model.predict(preprocessed_src_sentence)
    target_sentence = self._prediction_to_sentence(prediction)
    return target_sentence
