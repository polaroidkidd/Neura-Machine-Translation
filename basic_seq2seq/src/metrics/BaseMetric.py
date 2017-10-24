from abc import ABCMeta, abstractmethod


class BaseMetric(metaclass=ABCMeta):
    def __init__(self):
        self.params = {}

    @abstractmethod
    def evaluate_hypothesis_single(self, hypothesis, reference):
        pass

    @abstractmethod
    def evaluate_hypothesis_batch_single(self, hypothesis, references):
        pass

    @abstractmethod
    def evaluate_hypothesis_corpus(self, hypothesis, references):
        pass
