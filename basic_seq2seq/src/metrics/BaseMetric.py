from abc import ABCMeta, abstractmethod


class BaseMetric(metaclass=ABCMeta):
    """
    Provides access to various metrics
    """

    def __init__(self):
        self.params = {}

    @abstractmethod
    def start_evaluation(self):
        pass
