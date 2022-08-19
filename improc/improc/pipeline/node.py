from abc import ABC, abstractmethod


class Node(ABC):

    @staticmethod
    @abstractmethod
    def transform(experiment):
        pass
